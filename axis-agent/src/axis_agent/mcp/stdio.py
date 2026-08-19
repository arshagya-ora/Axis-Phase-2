"""Optional FastMCP stdio binding for :mod:`axis_agent.mcp.service`.

The MCP SDK is imported only when this module's factory is called.  Offline AXIS
tests and installations without the ``agent`` extra can use the service core.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, cast

from axis_agent.mcp.service import AxisMCPService, MCPGatewayConfigurationError, MCPProfile


class _FastMCPServer(Protocol):
    def tool(
        self, *, name: str, description: str
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...

    def run(self, *, transport: str) -> None: ...


def _load_fastmcp_factory() -> Callable[[str], _FastMCPServer]:
    try:
        module = importlib.import_module("mcp.server.fastmcp")
    except ImportError as exc:
        raise MCPGatewayConfigurationError(
            "MCP stdio transport requires the optional MCP SDK"
        ) from exc
    factory = getattr(module, "FastMCP", None)
    if factory is None:
        raise MCPGatewayConfigurationError("installed MCP SDK does not provide FastMCP")
    return cast(Callable[[str], _FastMCPServer], factory)


def create_stdio_server(service: AxisMCPService) -> _FastMCPServer:
    """Bind only the active profile's methods to a lazily loaded FastMCP server."""

    server = _load_fastmcp_factory()(f"AXIS {service.profile.value} gateway")

    if service.profile is MCPProfile.NAVIGATOR:
        if service.managed_step_binding_enabled:

            async def axis_bind_step(
                session_id: str,
                task_id: str,
                step_id: str,
                plan_id: str,
                step_key: str,
                allowed_action_types: list[str],
            ) -> dict[str, object]:
                return await service.bind_managed_step(
                    session_id=session_id,
                    task_id=task_id,
                    step_id=step_id,
                    plan_id=plan_id,
                    step_key=step_key,
                    allowed_action_types=allowed_action_types,
                )

            server.tool(
                name="axis_bind_step",
                description="Internal AXIS control-plane binding; never expose to a model.",
            )(axis_bind_step)

        async def browser_observe() -> dict[str, object]:
            return await service.browser_observe()

        async def browser_execute(action: dict[str, object]) -> dict[str, object]:
            return await service.browser_execute(action)

        server.tool(
            name="browser_observe",
            description="Read the bounded, redacted current browser observation.",
        )(browser_observe)
        server.tool(
            name="browser_execute",
            description="Execute one reviewed AXIS browser action; IDs are server-owned.",
        )(browser_execute)

    return server


def run_stdio(service: AxisMCPService) -> None:
    """Run a configured gateway on stdio until its MCP client disconnects."""

    create_stdio_server(service).run(transport="stdio")
