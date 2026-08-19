"""Transport-neutral, policy-bounded MCP service core."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Collection, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from axis_agent.browser.base import BrowserAdapter, BrowserObservation
from axis_agent.contracts.actions import (
    APPROVAL_REQUIRED_ACTION_TYPES,
    PRODUCTION_ACTION_TYPES,
    Action,
    ActionCommand,
    ActionResult,
    ClickElementAction,
    InputTextAction,
    ProductionAction,
    ProductionActionType,
    parse_production_action,
)

if TYPE_CHECKING:
    from axis_agent.runtime.approvals import ActionApprovalGate


class MCPGatewayError(RuntimeError):
    """Base class for stable, non-sensitive gateway failures."""


class MCPGatewayConfigurationError(MCPGatewayError):
    """Raised when a required trusted dependency has not been supplied."""


class MCPGatewayExecutionError(MCPGatewayError):
    """Raised when an execution dependency violates the gateway contract."""


class MCPToolNotAllowedError(MCPGatewayError):
    """Raised when a tool is not exposed by the active agent profile."""


class MCPInvalidArgumentsError(MCPGatewayError):
    """Raised when model-controlled tool arguments fail strict validation."""


class MCPProfile(StrEnum):
    PLANNER = "planner"
    NAVIGATOR = "navigator"


@runtime_checkable
class ActionDispatcher(Protocol):
    """The sole action-execution boundary available to the MCP gateway."""

    async def dispatch(
        self,
        command: ActionCommand,
        *,
        observation: BrowserObservation | None = None,
    ) -> ActionResult: ...


class ManagedStepApprovalFactory(Protocol):
    """Build the authoritative approval gate for one runtime-owned plan step."""

    def __call__(
        self,
        *,
        plan_id: UUID,
        step_key: str,
    ) -> ActionApprovalGate: ...


class _StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _EmptyArguments(_StrictArguments):
    pass


class _BrowserExecuteArguments(_StrictArguments):
    action: dict[str, object]


_PLANNER_TOOLS: Final[tuple[str, ...]] = ()

_NAVIGATOR_TOOLS: Final[tuple[str, ...]] = (
    "browser_observe",
    "browser_execute",
)


class AxisMCPService:
    """Expose the smallest useful browser surface for one AXIS agent profile.

    Tool callers provide only task-level arguments.  Session, task, step and action
    identifiers, ordinal ordering, current page ID and current origin are all owned
    by this service and cannot be supplied by a model.
    """

    def __init__(
        self,
        *,
        profile: MCPProfile,
        browser: BrowserAdapter,
        dispatcher: ActionDispatcher | None = None,
        approval_gate: ActionApprovalGate | None = None,
        allowed_action_types: Collection[ProductionActionType] | None = None,
        session_id: UUID | None = None,
        task_id: UUID | None = None,
        step_id: UUID | None = None,
        managed_step_approval_factory: ManagedStepApprovalFactory | None = None,
    ) -> None:
        self.profile = profile
        self._browser = browser
        self._dispatcher = dispatcher
        self._approval_gate = approval_gate
        self._allowed_action_types = self._validate_allowed_action_types(allowed_action_types)
        self._session_id = session_id or uuid4()
        self._task_id = task_id or uuid4()
        self._step_id = step_id or uuid4()
        self._managed_step_approval_factory = managed_step_approval_factory
        self._ordinal = 0
        self._last_observation: BrowserObservation | None = None
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def tool_names(self) -> tuple[str, ...]:
        if self.profile is MCPProfile.PLANNER:
            return _PLANNER_TOOLS
        return _NAVIGATOR_TOOLS

    @property
    def managed_step_binding_enabled(self) -> bool:
        """Whether the trusted stdio parent may rebind this task-scoped gateway."""

        return (
            self.profile is MCPProfile.NAVIGATOR and self._managed_step_approval_factory is not None
        )

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_started()

    async def stop(self) -> None:
        async with self._lock:
            if self._started:
                await self._browser.stop()
                self._started = False
                self._last_observation = None

    async def bind_execution_context(
        self,
        *,
        session_id: UUID,
        task_id: UUID,
        step_id: UUID,
        approval_gate: ActionApprovalGate | None = None,
        allowed_action_types: Collection[ProductionActionType] | None = None,
    ) -> None:
        """Bind trusted runtime IDs for the next plan step.

        This is an in-process control-plane method and is deliberately not exposed as
        an MCP tool. Binding is serialized with observations and actions.
        """

        async with self._lock:
            self._bind_execution_context_unlocked(
                session_id=session_id,
                task_id=task_id,
                step_id=step_id,
                approval_gate=approval_gate,
                allowed_action_types=allowed_action_types,
            )

    async def bind_managed_step(
        self,
        *,
        session_id: str,
        task_id: str,
        step_id: str,
        plan_id: str,
        step_key: str,
        allowed_action_types: list[str],
    ) -> dict[str, object]:
        """Rebind one task-scoped stdio gateway using parent-owned identifiers.

        The direct SDK model is never connected to this administrative method.
        It exists solely so a single ephemeral Chromium context can survive all
        steps in one validated execution plan.
        """

        self._require_profile(MCPProfile.NAVIGATOR)
        factory = self._managed_step_approval_factory
        if factory is None:
            raise MCPGatewayConfigurationError("managed step binding is unavailable")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", step_key) is None:
            raise MCPInvalidArgumentsError("invalid managed step key")
        try:
            parsed_session_id = UUID(session_id)
            parsed_task_id = UUID(task_id)
            parsed_step_id = UUID(step_id)
            parsed_plan_id = UUID(plan_id)
            normalized_actions = tuple(
                cast(ProductionActionType, value) for value in allowed_action_types
            )
            gate = factory(plan_id=parsed_plan_id, step_key=step_key)
            validated_actions = self._validate_allowed_action_types(normalized_actions)
        except (TypeError, ValueError) as exc:
            raise MCPInvalidArgumentsError("invalid managed step binding") from exc
        if validated_actions is None:  # pragma: no cover - defensive invariant
            raise MCPInvalidArgumentsError("managed step action allowlist is required")

        async with self._lock:
            self._bind_execution_context_unlocked(
                session_id=parsed_session_id,
                task_id=parsed_task_id,
                step_id=parsed_step_id,
                approval_gate=gate,
                allowed_action_types=validated_actions,
            )
        return {"bound": True, "stepKey": step_key}

    def _bind_execution_context_unlocked(
        self,
        *,
        session_id: UUID,
        task_id: UUID,
        step_id: UUID,
        approval_gate: ActionApprovalGate | None,
        allowed_action_types: Collection[ProductionActionType] | None,
    ) -> None:
        self._session_id = session_id
        self._task_id = task_id
        self._step_id = step_id
        self._approval_gate = approval_gate
        self._allowed_action_types = self._validate_allowed_action_types(allowed_action_types)
        self._ordinal = 0
        self._last_observation = None

    async def invoke(
        self, tool_name: str, arguments: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Invoke one profile-owned tool without depending on an MCP transport."""

        if tool_name not in self.tool_names:
            raise MCPToolNotAllowedError(f"tool is not allowed for {self.profile.value} profile")
        values = {} if arguments is None else dict(arguments)

        try:
            if tool_name == "browser_observe":
                _EmptyArguments.model_validate(values, strict=True)
                return await self.browser_observe()
            if tool_name == "browser_execute":
                execute_request = _BrowserExecuteArguments.model_validate(values, strict=True)
                return await self.browser_execute(execute_request.action)
        except ValidationError as exc:
            raise MCPInvalidArgumentsError(f"invalid arguments for {tool_name}") from exc

        raise MCPToolNotAllowedError("tool is not implemented")

    async def browser_observe(self) -> dict[str, object]:
        self._require_profile(MCPProfile.NAVIGATOR)
        return await self._observe()

    async def browser_execute(self, action_payload: Mapping[str, object]) -> dict[str, object]:
        self._require_profile(MCPProfile.NAVIGATOR)
        if "xpath" in action_payload:
            raise MCPToolNotAllowedError("model-provided XPath is not allowed")
        try:
            action = parse_production_action(dict(action_payload))
        except ValidationError as exc:
            raise MCPInvalidArgumentsError("invalid AXIS browser action") from exc
        if isinstance(action, (ClickElementAction, InputTextAction)) and action.xpath is not None:
            raise MCPToolNotAllowedError("model-provided XPath is not allowed")
        if self._allowed_action_types is None:
            raise MCPGatewayConfigurationError(
                "Navigator execution context has no plan action allowlist"
            )
        if action.type not in self._allowed_action_types:
            raise MCPToolNotAllowedError("action type is outside the active plan step")
        return await self._run_action(action)

    async def _observe(self) -> dict[str, object]:
        async with self._lock:
            await self._ensure_started()
            observation = await self._browser.observe()
            if self.profile is MCPProfile.NAVIGATOR:
                self._last_observation = observation.model_copy(deep=True)
            return {"observation": observation.model_dump(mode="json", by_alias=True)}

    async def _run_action(self, action: ProductionAction) -> dict[str, object]:
        async with self._lock:
            await self._ensure_started()
            if self.profile is MCPProfile.NAVIGATOR:
                if self._last_observation is None:
                    raise MCPGatewayConfigurationError(
                        "Navigator must observe immediately before browser execution"
                    )
                observation = self._last_observation
            else:
                observation = await self._browser.observe()
            command = self._make_command(action, observation)
            try:
                result = await self._execute_command(command, observation=observation)
                return self._action_response(command, result)
            finally:
                if self.profile is MCPProfile.NAVIGATOR:
                    self._last_observation = None

    def _make_command(self, action: Action, observation: BrowserObservation) -> ActionCommand:
        command = ActionCommand(
            session_id=self._session_id,
            task_id=self._task_id,
            step_id=self._step_id,
            ordinal=self._ordinal,
            expected_observation_id=observation.observation_id,
            expected_page_id=observation.page_id,
            expected_origin=self._origin(observation.url),
            action=action,
        )
        self._ordinal += 1
        return command

    async def _execute_command(
        self,
        command: ActionCommand,
        *,
        observation: BrowserObservation,
    ) -> ActionResult:
        if self._dispatcher is None:
            raise MCPGatewayConfigurationError(
                "browser actions require the AXIS security dispatcher"
            )
        if command.action.type in APPROVAL_REQUIRED_ACTION_TYPES and self._approval_gate is None:
            raise MCPGatewayConfigurationError(
                "high-impact browser action requires the AXIS approval gate"
            )
        if self._approval_gate is not None and not await self._approval_gate.authorize(command):
            return ActionResult(
                action_id=command.action_id,
                success=False,
                message="Action rejected by AXIS approval policy",
                error_code="APPROVAL_REJECTED",
                retryable=False,
            )
        result = await self._dispatcher.dispatch(command, observation=observation)
        if result.action_id != command.action_id:
            raise MCPGatewayExecutionError("execution returned a mismatched action ID")
        return result

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._browser.start()
            self._started = True

    def _require_profile(self, expected: MCPProfile) -> None:
        if self.profile is not expected:
            raise MCPToolNotAllowedError(f"tool is not allowed for {self.profile.value} profile")

    def _validate_allowed_action_types(
        self,
        values: Collection[ProductionActionType] | None,
    ) -> frozenset[ProductionActionType] | None:
        if values is None:
            return None
        normalized = frozenset(values)
        if not normalized or not normalized <= PRODUCTION_ACTION_TYPES:
            raise ValueError("allowed_action_types must be a non-empty production action subset")
        if self.profile is MCPProfile.PLANNER:
            raise ValueError("Planner profile does not accept execution action types")
        return normalized

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
    def _action_response(command: ActionCommand, result: ActionResult) -> dict[str, object]:
        return {
            "command": {
                "actionId": str(command.action_id),
                "actionType": command.action.type,
                "expectedObservationId": command.expected_observation_id,
                "expectedOrigin": command.expected_origin,
                "expectedPageId": command.expected_page_id,
                "ordinal": command.ordinal,
            },
            "result": result.model_dump(mode="json", by_alias=True),
        }
