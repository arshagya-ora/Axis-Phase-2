"""Direct Python MCP client for one task-scoped AXIS Playwright gateway."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError

from axis_agent.browser import BrowserObservation
from axis_agent.contracts import (
    ActionResult,
    ProductionAction,
    ProductionActionType,
    production_action_payload,
)
from axis_agent.mcp.client import ManagedGatewayProcessConfig, build_stdio_launch_params
from axis_agent.mcp.service import MCPProfile

_EXPECTED_TOOLS = frozenset({"axis_bind_step", "browser_observe", "browser_execute"})


class DirectMCPClientError(RuntimeError):
    """A stable, non-sensitive local MCP lifecycle or contract failure."""


class TaskScopedPlaywrightMCP:
    """Keep one isolated MCP/Chromium process alive for an entire AXIS task.

    Process startup is intentionally lazy because the deterministic workflow owns
    session, plan and execution-step identifiers and only creates them immediately
    before the first step is bound.
    """

    def __init__(
        self,
        *,
        database_path: Path,
        firewall_policy_path: Path,
        downloads_path: Path,
        source_environment: Mapping[str, str] | None = None,
        timeout_seconds: int = 130,
    ) -> None:
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        self._database_path = database_path.resolve()
        self._firewall_policy_path = firewall_policy_path.resolve()
        self._downloads_path = downloads_path.resolve()
        self._source_environment = source_environment
        self._timeout_seconds = timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._session_id: UUID | None = None
        self._task_id: UUID | None = None

    async def __aenter__(self) -> TaskScopedPlaywrightMCP:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.close()

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
        """Bind trusted IDs and the step action capability before observation."""

        if self._session_id is not None and (
            self._session_id != session_id or self._task_id != task_id
        ):
            raise DirectMCPClientError("an MCP browser cannot be rebound to another task")
        if self._session is None:
            await self._start(
                ManagedGatewayProcessConfig(
                    profile=MCPProfile.NAVIGATOR,
                    database_path=self._database_path,
                    firewall_policy_path=self._firewall_policy_path,
                    downloads_path=self._downloads_path,
                    session_id=session_id,
                    task_id=task_id,
                    step_id=step_id,
                    allowed_action_types=allowed_action_types,
                    plan_id=plan_id,
                    step_key=step_key,
                )
            )
            self._session_id = session_id
            self._task_id = task_id

        payload = await self._call_tool(
            "axis_bind_step",
            {
                "session_id": str(session_id),
                "task_id": str(task_id),
                "step_id": str(step_id),
                "plan_id": str(plan_id),
                "step_key": step_key,
                "allowed_action_types": list(allowed_action_types),
            },
        )
        if payload.get("bound") is not True or payload.get("stepKey") != step_key:
            raise DirectMCPClientError("MCP rejected the execution-step binding")

    async def observe(self) -> BrowserObservation:
        payload = await self._call_tool("browser_observe", {})
        observation = payload.get("observation")
        try:
            return BrowserObservation.model_validate_json(
                json.dumps(observation, separators=(",", ":")),
                strict=True,
            )
        except (TypeError, ValidationError) as exc:
            raise DirectMCPClientError("MCP returned an invalid browser observation") from exc

    async def execute(self, action: ProductionAction) -> ActionResult:
        payload = await self._call_tool(
            "browser_execute",
            {"action": production_action_payload(action)},
        )
        result = payload.get("result")
        try:
            return ActionResult.model_validate_json(
                json.dumps(result, separators=(",", ":")),
                strict=True,
            )
        except (TypeError, ValidationError) as exc:
            raise DirectMCPClientError("MCP returned an invalid browser action result") from exc

    async def close(self) -> None:
        """Release the stdio session, browser child, and its ephemeral context."""

        stack = self._stack
        self._stack = None
        self._session = None
        self._session_id = None
        self._task_id = None
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException as exc:
                raise DirectMCPClientError("MCP gateway cleanup failed") from exc

    async def _start(self, config: ManagedGatewayProcessConfig) -> None:
        if self._session is not None:
            raise DirectMCPClientError("MCP gateway is already running")
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - optional install boundary
            raise DirectMCPClientError("the Python MCP dependency is unavailable") from exc

        launch = build_stdio_launch_params(
            config,
            source_environment=self._source_environment,
        )
        parameters = StdioServerParameters(
            command=launch["command"],
            args=launch["args"],
            env=launch["env"],
            cwd=launch["cwd"],
            encoding=launch["encoding"],
            encoding_error_handler=launch["encoding_error_handler"],
        )
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                )
            )
            await session.initialize()
            listed = await session.list_tools()
            names = frozenset(tool.name for tool in listed.tools)
            if names != _EXPECTED_TOOLS:
                raise DirectMCPClientError("MCP gateway exposed an unexpected tool surface")
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> dict[str, Any]:
        session = self._session
        if session is None:
            raise DirectMCPClientError("MCP execution step has not been bound")
        try:
            result = await session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
            )
        except BaseException as exc:
            raise DirectMCPClientError("MCP tool call failed") from exc
        if result.isError or not isinstance(result.structuredContent, dict):
            raise DirectMCPClientError("MCP tool returned a failed or unstructured result")
        return cast(dict[str, Any], result.structuredContent)


__all__ = ["DirectMCPClientError", "TaskScopedPlaywrightMCP"]
