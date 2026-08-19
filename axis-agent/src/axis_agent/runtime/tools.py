"""Workflow-facing adapter over the secured AXIS MCP service."""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import ValidationError

from axis_agent.browser import BrowserObservation
from axis_agent.contracts import (
    ActionResult,
    ProductionAction,
    ProductionActionType,
    production_action_payload,
)
from axis_agent.mcp import AxisMCPService, MCPProfile
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime.approvals import ApprovalCoordinator, ApprovalHandler


class WorkflowToolError(RuntimeError):
    """Raised when an MCP response violates the trusted workflow contract."""


@runtime_checkable
class WorkflowBrowserTools(Protocol):
    async def bind_step(
        self,
        *,
        session_id: UUID,
        task_id: UUID,
        step_id: UUID,
        plan_id: UUID,
        step_key: str,
        allowed_action_types: tuple[ProductionActionType, ...],
    ) -> None: ...

    async def observe(self) -> BrowserObservation: ...

    async def execute(self, action: ProductionAction) -> ActionResult: ...


class MCPServiceBrowserTools:
    """Use the same MCP tool surface in standalone and production orchestration."""

    def __init__(
        self,
        *,
        service: AxisMCPService,
        database: AxisDatabase,
        approval_handler: ApprovalHandler | None = None,
        approval_timeout_seconds: int = 120,
    ) -> None:
        if service.profile is not MCPProfile.NAVIGATOR:
            raise ValueError("workflow browser tools require a Navigator MCP profile")
        self._service = service
        self._database = database
        self._approval_handler = approval_handler
        self._approval_timeout_seconds = approval_timeout_seconds

    async def bind_step(
        self,
        *,
        session_id: UUID,
        task_id: UUID,
        step_id: UUID,
        plan_id: UUID,
        step_key: str,
        allowed_action_types: tuple[ProductionActionType, ...],
    ) -> None:
        gate = ApprovalCoordinator(
            database=self._database,
            plan_id=plan_id,
            step_key=step_key,
            handler=self._approval_handler,
            timeout_seconds=self._approval_timeout_seconds,
        )
        await self._service.bind_execution_context(
            session_id=session_id,
            task_id=task_id,
            step_id=step_id,
            approval_gate=gate,
            allowed_action_types=allowed_action_types,
        )

    async def observe(self) -> BrowserObservation:
        response = await self._service.invoke("browser_observe")
        payload = response.get("observation")
        try:
            return BrowserObservation.model_validate_json(
                json.dumps(payload, separators=(",", ":")),
                strict=True,
            )
        except (TypeError, ValidationError) as exc:
            raise WorkflowToolError("MCP returned an invalid browser observation") from exc

    async def execute(self, action: ProductionAction) -> ActionResult:
        response = await self._service.invoke(
            "browser_execute",
            {"action": production_action_payload(action)},
        )
        payload = response.get("result")
        try:
            return ActionResult.model_validate_json(
                json.dumps(payload, separators=(",", ":")),
                strict=True,
            )
        except (TypeError, ValidationError) as exc:
            raise WorkflowToolError("MCP returned an invalid action result") from exc


__all__ = ["MCPServiceBrowserTools", "WorkflowBrowserTools", "WorkflowToolError"]
