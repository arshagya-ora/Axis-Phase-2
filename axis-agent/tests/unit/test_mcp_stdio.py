from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from axis_agent.browser.mock import MockBrowserAdapter
from axis_agent.mcp import AxisMCPService, MCPGatewayConfigurationError, MCPProfile, stdio


class FakeFastMCP:
    last_instance: FakeFastMCP | None = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable[..., object]] = {}
        self.transport: str | None = None
        FakeFastMCP.last_instance = self

    def tool(
        self, *, name: str, description: str
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        assert description

        def register(function: Callable[..., object]) -> Callable[..., object]:
            self.tools[name] = function
            return function

        return register

    def run(self, *, transport: str) -> None:
        self.transport = transport


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(FastMCP=FakeFastMCP)
    monkeypatch.setattr(stdio.importlib, "import_module", lambda _name: fake_module)


def test_stdio_import_is_lazy_when_mcp_sdk_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(stdio.importlib, "import_module", missing)
    service = AxisMCPService(profile=MCPProfile.PLANNER, browser=MockBrowserAdapter())

    with pytest.raises(MCPGatewayConfigurationError, match="optional MCP SDK"):
        stdio.create_stdio_server(service)


def test_stdio_registers_only_planner_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch)
    service = AxisMCPService(profile=MCPProfile.PLANNER, browser=MockBrowserAdapter())

    server = stdio.create_stdio_server(service)

    assert isinstance(server, FakeFastMCP)
    assert tuple(server.tools) == service.tool_names


def test_stdio_registers_only_navigator_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch)
    service = AxisMCPService(profile=MCPProfile.NAVIGATOR, browser=MockBrowserAdapter())

    server = stdio.create_stdio_server(service)

    assert isinstance(server, FakeFastMCP)
    assert tuple(server.tools) == service.tool_names


async def test_managed_stdio_adds_parent_only_step_binding_to_exact_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    bindings: list[tuple[UUID, str]] = []

    class AllowingGate:
        async def authorize(self, command: object) -> bool:
            del command
            return True

    def approval_factory(*, plan_id: UUID, step_key: str) -> AllowingGate:
        bindings.append((plan_id, step_key))
        return AllowingGate()

    service = AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=MockBrowserAdapter(),
        managed_step_approval_factory=approval_factory,
    )

    server = stdio.create_stdio_server(service)

    assert isinstance(server, FakeFastMCP)
    assert tuple(server.tools) == (
        "axis_bind_step",
        "browser_observe",
        "browser_execute",
    )
    assert "axis_bind_step" not in service.tool_names

    plan_id = uuid4()
    result = await server.tools["axis_bind_step"](
        session_id=str(uuid4()),
        task_id=str(uuid4()),
        step_id=str(uuid4()),
        plan_id=str(plan_id),
        step_key="open_report",
        allowed_action_types=["wait"],
    )

    assert result == {"bound": True, "stepKey": "open_report"}
    assert bindings == [(plan_id, "open_report")]


def test_run_stdio_uses_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch)
    service = AxisMCPService(profile=MCPProfile.NAVIGATOR, browser=MockBrowserAdapter())

    stdio.run_stdio(service)

    assert FakeFastMCP.last_instance is not None
    assert FakeFastMCP.last_instance.transport == "stdio"
