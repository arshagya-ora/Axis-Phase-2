"""Security boundary from validated action intents to a browser adapter."""

from __future__ import annotations

import asyncio
from urllib.parse import quote_plus, urlsplit

from axis_agent.browser import BrowserAdapter, BrowserObservation
from axis_agent.contracts.actions import (
    APPROVAL_REQUIRED_ACTION_TYPES,
    Action,
    ActionCommand,
    ActionResult,
    GoToUrlAction,
    OpenTabAction,
    ProductionAction,
    SearchGoogleAction,
    parse_production_action,
    production_action_payload,
)
from axis_agent.firewall import (
    FirewallService,
    NavigationPermit,
    NavigationPermitStore,
    UrlPurpose,
)
from axis_agent.persistence import AxisDatabase


class ActionDispatcher:
    """Validate, persist, authorize, and execute every production browser action.

    The dispatcher is the sole production path to ``BrowserAdapter.execute``. URL actions
    additionally require a firewall decision and a single-use navigation permit. Protocol-3
    ``search_google`` remains available only through ``dispatch_url_action`` as a compatibility
    shim and is not part of the model/MCP production action vocabulary.
    """

    DEFAULT_SEARCH_BASE_URL = "https://www.google.com/search?q="

    def __init__(
        self,
        *,
        browser: BrowserAdapter,
        firewall: FirewallService,
        database: AxisDatabase,
        permits: NavigationPermitStore,
        search_base_url: str = DEFAULT_SEARCH_BASE_URL,
        action_timeout_seconds: int = 30,
    ) -> None:
        if action_timeout_seconds < 1 or action_timeout_seconds > 120:
            raise ValueError("action_timeout_seconds must be between 1 and 120")
        self._browser = browser
        self._firewall = firewall
        self._database = database
        self._permits = permits
        self._search_base_url = self._validate_search_base_url(search_base_url)
        self._action_timeout_seconds = action_timeout_seconds

    async def dispatch(
        self,
        command: ActionCommand,
        *,
        observation: BrowserObservation | None = None,
    ) -> ActionResult:
        """Dispatch one canonical production action exactly once."""

        action = parse_production_action(command.action)
        try:
            await self._database.validate_execution_context(
                session_id=command.session_id,
                task_id=command.task_id,
                step_id=command.step_id,
            )
        except ValueError:
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Action execution context is invalid",
                error_code="INVALID_EXECUTION_CONTEXT",
                retryable=False,
            )

        trusted_observation = observation or await self._browser.observe()
        precondition_error = self._validate_preconditions(command, trusted_observation)
        if precondition_error is not None:
            await self._database.record_audit_event(
                event_type="action.precondition_rejected",
                severity="warning",
                session_id=command.session_id,
                payload={
                    "actionId": command.action_id,
                    "actionType": action.type,
                    "reasonCode": precondition_error,
                },
            )
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Browser state changed before action execution",
                error_code=precondition_error,
                retryable=True,
            )

        if isinstance(action, (GoToUrlAction, OpenTabAction)):
            return await self._dispatch_url(command, action)
        return await self._dispatch_non_url(command, action)

    async def dispatch_url_action(self, command: ActionCommand) -> ActionResult:
        """Compatibility entry point for protocol-3 URL and search actions."""

        if isinstance(command.action, SearchGoogleAction):
            return await self._dispatch_legacy_search(command)
        if not isinstance(command.action, (GoToUrlAction, OpenTabAction)):
            raise ValueError("URL dispatcher accepts only go_to_url, open_tab, and search_google")
        return await self.dispatch(command)

    async def _dispatch_legacy_search(self, command: ActionCommand) -> ActionResult:
        action = command.action
        if not isinstance(action, SearchGoogleAction):
            raise ValueError("legacy search dispatch requires search_google")
        try:
            await self._database.validate_execution_context(
                session_id=command.session_id,
                task_id=command.task_id,
                step_id=command.step_id,
            )
        except ValueError:
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Action execution context is invalid",
                error_code="INVALID_EXECUTION_CONTEXT",
                retryable=False,
            )

        observation = await self._browser.observe()
        precondition_error = self._validate_preconditions(command, observation)
        if precondition_error is not None:
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Browser state changed before action execution",
                error_code=precondition_error,
                retryable=True,
            )

        url = f"{self._search_base_url}{quote_plus(action.query)}"
        return await self._dispatch_url(
            command,
            action,
            url=url,
            purpose=UrlPurpose.NAVIGATION,
        )

    async def _dispatch_url(
        self,
        command: ActionCommand,
        action: GoToUrlAction | OpenTabAction | SearchGoogleAction,
        *,
        url: str | None = None,
        purpose: UrlPurpose | None = None,
    ) -> ActionResult:
        if isinstance(action, GoToUrlAction):
            target_url = action.url
            target_purpose = UrlPurpose.NAVIGATION
        elif isinstance(action, OpenTabAction):
            target_url = action.url
            target_purpose = UrlPurpose.POPUP
        else:
            if url is None or purpose is None:
                raise ValueError("legacy search dispatch requires a resolved URL and purpose")
            target_url = url
            target_purpose = purpose

        decision = await self._firewall.evaluate(target_url, purpose=target_purpose)
        stored = await self._database.record_action_with_firewall_decision(
            session_id=command.session_id,
            step_id=command.step_id,
            action_id=command.action_id,
            ordinal=command.ordinal,
            action_type=action.type,
            action_payload=action.model_dump(mode="json", by_alias=True),
            idempotency_class="navigation",
            raw_url=target_url,
            purpose=target_purpose.value,
            allowed=decision.allowed,
            reason_code=decision.reason.value,
            policy_hash=decision.policy_hash,
            matched_rule_id=decision.matched_rule_id,
            resolved_ips=decision.resolved_ips,
            expected_observation_id=command.expected_observation_id,
            expected_page_id=command.expected_page_id,
            expected_origin=command.expected_origin,
            decision_id=decision.decision_id,
        )

        if not decision.allowed:
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Navigation blocked by AXIS firewall",
                error_code=decision.reason.value,
                retryable=False,
            )
        if stored.state != "allowed":
            return self._duplicate_result(command, stored.state, stored.error_code)

        permit = self._permits.issue(decision, action_id=command.action_id)
        try:
            return await self._execute_recorded(command, action, navigation_permit=permit)
        finally:
            self._permits.revoke(permit.permit_id)

    async def _dispatch_non_url(
        self,
        command: ActionCommand,
        action: ProductionAction,
    ) -> ActionResult:
        action_payload = production_action_payload(action)
        if action.type in APPROVAL_REQUIRED_ACTION_TYPES:
            approved = await self._database.has_valid_action_approval(
                session_id=command.session_id,
                task_id=command.task_id,
                action_id=command.action_id,
                observation_id=command.expected_observation_id,
                action_type=action.type,
                action_payload=action_payload,
            )
            if not approved:
                return ActionResult(
                    action_id=command.action_id,
                    success=False,
                    message="Action requires a current user approval",
                    error_code="APPROVAL_REQUIRED",
                    retryable=False,
                )
        stored = await self._database.record_action(
            session_id=command.session_id,
            step_id=command.step_id,
            action_id=command.action_id,
            ordinal=command.ordinal,
            action_type=action.type,
            action_payload=action_payload,
            idempotency_class=self._idempotency_class(action.type),
            expected_observation_id=command.expected_observation_id,
            expected_page_id=command.expected_page_id,
            expected_origin=command.expected_origin,
        )
        if stored.state != "allowed":
            return self._duplicate_result(command, stored.state, stored.error_code)
        return await self._execute_recorded(command, action)

    async def _execute_recorded(
        self,
        command: ActionCommand,
        action: Action,
        *,
        navigation_permit: NavigationPermit | None = None,
    ) -> ActionResult:
        if not await self._database.mark_dispatched(command.action_id):
            refreshed = await self._database.get_action(command.action_id)
            return self._duplicate_result(
                command,
                refreshed.state if refreshed else "unknown",
                refreshed.error_code if refreshed else "ACTION_STATE_UNKNOWN",
            )

        try:
            result = await asyncio.wait_for(
                self._browser.execute(
                    action,
                    action_id=str(command.action_id),
                    navigation_permit=navigation_permit,
                    expected_observation_id=command.expected_observation_id,
                    expected_page_id=command.expected_page_id,
                    expected_origin=command.expected_origin,
                ),
                timeout=self._action_timeout_seconds,
            )
            if result.action_id != command.action_id:
                raise RuntimeError("browser returned a mismatched action ID")
        except TimeoutError:
            await self._database.complete_action(
                command.action_id,
                state="timed_out",
                error_code="ACTION_TIMEOUT",
            )
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Browser action timed out",
                error_code="ACTION_TIMEOUT",
                retryable=False,
            )
        except Exception as exc:
            await self._database.complete_action(
                command.action_id,
                state="failed",
                error_code="BROWSER_EXECUTION_FAILED",
                result={"exceptionType": type(exc).__name__},
            )
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Browser execution failed",
                error_code="BROWSER_EXECUTION_FAILED",
                retryable=False,
            )

        await self._database.complete_action(
            command.action_id,
            state="succeeded" if result.success else "failed",
            result=result.model_dump(mode="json", by_alias=True),
            error_code=result.error_code,
        )
        return result

    @classmethod
    def _validate_preconditions(
        cls,
        command: ActionCommand,
        observation: BrowserObservation,
    ) -> str | None:
        if (
            command.expected_observation_id is not None
            and command.expected_observation_id != observation.observation_id
        ):
            return "STALE_OBSERVATION"
        if command.expected_page_id is not None and command.expected_page_id != observation.page_id:
            return "STALE_PAGE"
        if command.expected_origin is not None:
            actual_origin = cls._origin(observation.url)
            if actual_origin != command.expected_origin:
                return "STALE_ORIGIN"
        return None

    @staticmethod
    def _origin(url: str) -> str | None:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https"} or hostname is None:
            return None
        default_port = 443 if parsed.scheme == "https" else 80
        suffix = "" if port in {None, default_port} else f":{port}"
        return f"{parsed.scheme}://{hostname.lower()}{suffix}"

    @staticmethod
    def _idempotency_class(action_type: str) -> str:
        if action_type in {
            "scroll_to_percent",
            "scroll_to_top",
            "scroll_to_bottom",
            "scroll_to_text",
            "get_dropdown_options",
            "wait",
        }:
            return "read_only"
        if action_type in {"go_back", "switch_tab"}:
            return "navigation_state"
        return "non_retryable"

    @staticmethod
    def _validate_search_base_url(value: str) -> str:
        """Require a deterministic, credential-free HTTPS query prefix."""
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("search_base_url must be a clean absolute HTTPS query prefix") from exc
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.query
            or parsed.fragment
            or not value.endswith("=")
        ):
            raise ValueError("search_base_url must be a clean absolute HTTPS query prefix")
        return value

    @staticmethod
    def _duplicate_result(
        command: ActionCommand, state: str, error_code: str | None
    ) -> ActionResult:
        return ActionResult(
            action_id=command.action_id,
            success=state == "succeeded",
            message=f"Action already recorded with state: {state}",
            error_code=error_code or (None if state == "succeeded" else "ACTION_NOT_REPLAYED"),
            retryable=False,
        )
