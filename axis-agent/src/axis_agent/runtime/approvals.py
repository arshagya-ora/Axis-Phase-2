"""Server-side approval policy for high-impact browser actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from axis_agent.contracts.actions import (
    ActionCommand,
    ClickElementAction,
    InputTextAction,
    ProductionAction,
    SelectDropdownOptionAction,
    SendKeysAction,
    parse_production_action,
)
from axis_agent.contracts.approvals import (
    ApprovalDecision,
    ApprovalDecisionValue,
    ApprovalRequest,
    ApprovalRisk,
)
from axis_agent.firewall import FirewallService, UrlPurpose
from axis_agent.persistence import AxisDatabase


@dataclass(frozen=True, slots=True)
class ApprovalRequirement:
    """A deterministic policy result, independent of model instructions."""

    risk: ApprovalRisk
    reason_code: str


@runtime_checkable
class ApprovalHandler(Protocol):
    """Ask the authenticated user to approve one in-memory action."""

    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool: ...


@runtime_checkable
class ActionApprovalGate(Protocol):
    """Gate invoked inside the MCP server before dispatcher execution."""

    async def authorize(self, command: ActionCommand) -> bool: ...


class RejectByDefaultApprovalHandler:
    """Production-safe default used when no authenticated UI is connected."""

    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool:
        del request, action
        return False


class TrustedSiteAutoApprovalHandler:
    """Apply the user's pre-authorized trusted-site execution policy.

    This handler does not decide whether a site is trusted.  That decision remains
    authoritative in the URL firewall and the Playwright request interceptor.  It
    only converts the already configured allow-list delegation into the
    action-specific, short-lived approval record required by the dispatcher.
    """

    def __init__(self, firewall: FirewallService) -> None:
        self._firewall = firewall

    async def permits_origin(self, origin: str | None) -> bool:
        """Require the exact current HTTP(S) origin to match an allow rule."""

        if origin is None:
            return False
        decision = await self._firewall.evaluate(origin, purpose=UrlPurpose.NAVIGATION)
        return decision.allowed

    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool:
        del request, action
        return True


class ApprovalPolicy:
    """Classify actions that can enter data or trigger external effects."""

    @staticmethod
    def requirement(action: ProductionAction) -> ApprovalRequirement | None:
        if isinstance(action, InputTextAction):
            return ApprovalRequirement(ApprovalRisk.DATA_ENTRY, "DATA_ENTRY_REQUIRES_APPROVAL")
        if isinstance(action, SelectDropdownOptionAction):
            return ApprovalRequirement(
                ApprovalRisk.DATA_ENTRY,
                "SELECTION_REQUIRES_APPROVAL",
            )
        if isinstance(action, ClickElementAction):
            return ApprovalRequirement(
                ApprovalRisk.EXTERNAL_SIDE_EFFECT,
                "CLICK_REQUIRES_APPROVAL",
            )
        if isinstance(action, SendKeysAction):
            return ApprovalRequirement(
                ApprovalRisk.EXTERNAL_SIDE_EFFECT,
                "KEYSTROKE_REQUIRES_APPROVAL",
            )
        return None


class ApprovalCoordinator:
    """Persist immutable approval metadata and fail closed on any handler error."""

    def __init__(
        self,
        *,
        database: AxisDatabase,
        plan_id: UUID,
        step_key: str,
        handler: ApprovalHandler | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        if timeout_seconds < 1 or timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        self._database = database
        self._plan_id = plan_id
        self._step_key = step_key
        self._handler = handler
        self._timeout_seconds = timeout_seconds

    async def authorize(self, command: ActionCommand) -> bool:
        action = parse_production_action(command.action)
        requirement = ApprovalPolicy.requirement(action)
        if requirement is None:
            return True
        if command.expected_observation_id is None:
            await self._database.record_audit_event(
                event_type="approval.precondition_rejected",
                severity="warning",
                session_id=command.session_id,
                payload={
                    "actionId": command.action_id,
                    "actionType": action.type,
                    "reasonCode": "MISSING_OBSERVATION",
                },
            )
            return False

        if isinstance(
            self._handler, TrustedSiteAutoApprovalHandler
        ) and not await self._handler.permits_origin(command.expected_origin):
            await self._database.record_audit_event(
                event_type="approval.precondition_rejected",
                severity="warning",
                session_id=command.session_id,
                payload={
                    "actionId": command.action_id,
                    "actionType": action.type,
                    "reasonCode": "CURRENT_ORIGIN_NOT_ALLOWED",
                },
            )
            return False

        request = ApprovalRequest.for_action(
            session_id=command.session_id,
            task_id=command.task_id,
            plan_id=self._plan_id,
            step_key=self._step_key,
            action_id=command.action_id,
            observation_id=command.expected_observation_id,
            action=action,
            risk=requirement.risk,
            reason_code=requirement.reason_code,
            ttl_seconds=min(self._timeout_seconds, 300),
        )
        await self._database.record_approval_request(request)

        approved = False
        decision_reason = "NO_APPROVAL_HANDLER"
        actor: Literal["user", "system"] = "system"
        if self._handler is not None:
            decision_reason = "USER_REJECTED"
            actor = "user"
            try:
                handler_decision: object = await asyncio.wait_for(
                    self._handler.approve(request, action),
                    timeout=self._timeout_seconds,
                )
                if type(handler_decision) is not bool:
                    decision_reason = "APPROVAL_HANDLER_INVALID"
                    actor = "system"
                elif handler_decision:
                    approved = True
                    decision_reason = (
                        "TRUSTED_SITE_POLICY_APPROVED"
                        if isinstance(self._handler, TrustedSiteAutoApprovalHandler)
                        else "USER_APPROVED"
                    )
            except TimeoutError:
                decision_reason = "APPROVAL_TIMEOUT"
                actor = "system"
            except Exception:
                decision_reason = "APPROVAL_HANDLER_FAILED"
                actor = "system"

        decision = ApprovalDecision(
            approval_id=request.approval_id,
            decision=(
                ApprovalDecisionValue.APPROVED if approved else ApprovalDecisionValue.REJECTED
            ),
            actor="user" if approved else actor,
            reason_code=decision_reason,
        )
        await self._database.record_approval_decision(decision)
        await self._database.record_audit_event(
            event_type="approval.decision",
            severity="info" if approved else "warning",
            session_id=command.session_id,
            payload={
                "actionId": command.action_id,
                "actionType": action.type,
                "approved": approved,
                "reasonCode": decision_reason,
            },
        )
        return approved


__all__ = [
    "ActionApprovalGate",
    "ApprovalCoordinator",
    "ApprovalHandler",
    "ApprovalPolicy",
    "ApprovalRequirement",
    "RejectByDefaultApprovalHandler",
    "TrustedSiteAutoApprovalHandler",
]
