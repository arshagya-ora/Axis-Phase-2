from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

import pytest

from axis_agent.browser import BrowserObservation
from axis_agent.contracts import ActionResult
from axis_agent.contracts.actions import WaitAction
from axis_agent.mcp.direct_client import DirectMCPClientError, TaskScopedPlaywrightMCP

EXPECTED_TOOLS = ("axis_bind_step", "browser_observe", "browser_execute")


class FakeAsyncContext:
    def __init__(self, value: object, events: list[str], label: str) -> None:
        self.value = value
        self.events = events
        self.label = label

    async def __aenter__(self) -> object:
        self.events.append(f"{self.label}.enter")
        return self.value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.events.append(f"{self.label}.exit")


class FakeParameters:
    instances: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(kwargs)


class FakeSession:
    def __init__(self, tool_names: tuple[str, ...], events: list[str]) -> None:
        self.tool_names = tool_names
        self.events = events
        self.initialized = False
        self.calls: list[tuple[str, dict[str, object], object]] = []
        self.observation = BrowserObservation(
            observation_id="0123456789abcdef0123456789abcdef",
            page_id="mcp-page-1",
            url="https://example.com/report",
            title="Report",
            visible_text="Approved content",
        )

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> object:
        return SimpleNamespace(tools=[SimpleNamespace(name=name) for name in self.tool_names])

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        read_timeout_seconds: object,
    ) -> object:
        self.calls.append((name, arguments, read_timeout_seconds))
        if name == "axis_bind_step":
            content: object = {
                "bound": True,
                "stepKey": arguments["step_key"],
            }
        elif name == "browser_observe":
            content = {"observation": self.observation.model_dump(mode="json", by_alias=True)}
        elif name == "browser_execute":
            content = {
                "result": ActionResult(
                    action_id=uuid4(),
                    success=True,
                    message="wait",
                ).model_dump(mode="json", by_alias=True)
            }
        else:  # pragma: no cover - the direct client has a fixed call surface
            raise AssertionError(name)
        return SimpleNamespace(isError=False, structuredContent=content)


class MCPHarness:
    def __init__(self, tool_names: tuple[str, ...] = EXPECTED_TOOLS) -> None:
        self.events: list[str] = []
        self.session = FakeSession(tool_names, self.events)
        self.client_session_calls: list[dict[str, object]] = []
        self.stdio_parameters: list[FakeParameters] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeParameters.instances.clear()
        mcp_module = ModuleType("mcp")
        mcp_module.__path__ = []  # type: ignore[attr-defined]
        client_module = ModuleType("mcp.client")
        client_module.__path__ = []  # type: ignore[attr-defined]
        stdio_module = ModuleType("mcp.client.stdio")

        def create_session(
            read_stream: object,
            write_stream: object,
            *,
            read_timeout_seconds: object,
        ) -> FakeAsyncContext:
            self.client_session_calls.append(
                {
                    "read_stream": read_stream,
                    "write_stream": write_stream,
                    "read_timeout_seconds": read_timeout_seconds,
                }
            )
            return FakeAsyncContext(self.session, self.events, "session")

        def stdio_client(parameters: FakeParameters) -> FakeAsyncContext:
            self.stdio_parameters.append(parameters)
            return FakeAsyncContext(("read", "write"), self.events, "stdio")

        mcp_module.ClientSession = create_session  # type: ignore[attr-defined]
        mcp_module.StdioServerParameters = FakeParameters  # type: ignore[attr-defined]
        stdio_module.stdio_client = stdio_client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mcp", mcp_module)
        monkeypatch.setitem(sys.modules, "mcp.client", client_module)
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)


def _client(
    tmp_path: Path,
    *,
    source_environment: dict[str, str] | None = None,
) -> TaskScopedPlaywrightMCP:
    policy_path = tmp_path / "firewall.json"
    policy_path.write_text("{}", encoding="utf-8")
    return TaskScopedPlaywrightMCP(
        database_path=tmp_path / "axis.db",
        firewall_policy_path=policy_path,
        downloads_path=tmp_path / "downloads",
        source_environment=source_environment,
        timeout_seconds=7,
    )


async def _bind(
    client: TaskScopedPlaywrightMCP,
    *,
    session_id: UUID,
    task_id: UUID,
    step_id: UUID,
    plan_id: UUID,
    step_key: str,
) -> None:
    await client.bind_step(
        session_id=session_id,
        task_id=task_id,
        step_id=step_id,
        plan_id=plan_id,
        step_key=step_key,
        allowed_action_types=("wait",),
    )


async def test_direct_client_exposes_exact_internal_and_browser_tool_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = MCPHarness()
    harness.install(monkeypatch)
    client = _client(tmp_path)

    await _bind(
        client,
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        plan_id=uuid4(),
        step_key="open_report",
    )

    assert harness.session.initialized is True
    assert len(harness.client_session_calls) == 1
    assert harness.session.calls[0][0] == "axis_bind_step"
    await client.close()


async def test_unexpected_mcp_tool_surface_fails_closed_and_releases_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = MCPHarness((*EXPECTED_TOOLS, "browser_run_code"))
    harness.install(monkeypatch)
    client = _client(tmp_path)

    with pytest.raises(DirectMCPClientError, match="unexpected tool surface"):
        await _bind(
            client,
            session_id=uuid4(),
            task_id=uuid4(),
            step_id=uuid4(),
            plan_id=uuid4(),
            step_key="open_report",
        )

    assert harness.events == [
        "stdio.enter",
        "session.enter",
        "session.exit",
        "stdio.exit",
    ]
    with pytest.raises(DirectMCPClientError, match="has not been bound"):
        await client.observe()


async def test_task_process_rebinds_steps_without_restart_and_rejects_another_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = MCPHarness()
    harness.install(monkeypatch)
    client = _client(tmp_path)
    session_id = uuid4()
    task_id = uuid4()
    plan_id = uuid4()

    await _bind(
        client,
        session_id=session_id,
        task_id=task_id,
        step_id=uuid4(),
        plan_id=plan_id,
        step_key="open_report",
    )
    await _bind(
        client,
        session_id=session_id,
        task_id=task_id,
        step_id=uuid4(),
        plan_id=plan_id,
        step_key="verify_report",
    )

    assert len(harness.client_session_calls) == 1
    binding_calls = [call for call in harness.session.calls if call[0] == "axis_bind_step"]
    assert [call[1]["step_key"] for call in binding_calls] == [
        "open_report",
        "verify_report",
    ]

    with pytest.raises(DirectMCPClientError, match="another task"):
        await _bind(
            client,
            session_id=session_id,
            task_id=uuid4(),
            step_id=uuid4(),
            plan_id=plan_id,
            step_key="foreign_task",
        )
    assert len(harness.session.calls) == 2
    await client.close()


async def test_mcp_child_environment_omits_all_model_and_credential_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = MCPHarness()
    harness.install(monkeypatch)
    source_environment = {
        "PATH": "safe-path",
        "SYSTEMROOT": "safe-system-root",
        "AXIS_OCI_GENAI_API_KEY": "secret-key",
        "OPENAI_API_KEY": "secret-openai-key",
        "OCI_CONFIG_FILE": "C:/secret/oci-config",
        "AXIS_OCI_PROJECT_OCID": "secret-project",
        "AXIS_PLANNER_MODEL": "secret-planner-model",
        "AXIS_NAVIGATOR_MODEL": "secret-navigator-model",
    }
    client = _client(tmp_path, source_environment=source_environment)

    await _bind(
        client,
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        plan_id=uuid4(),
        step_key="open_report",
    )

    assert len(harness.stdio_parameters) == 1
    child_environment = harness.stdio_parameters[0].kwargs["env"]
    assert child_environment == {"PATH": "safe-path", "SYSTEMROOT": "safe-system-root"}
    serialized_parameters = repr(harness.stdio_parameters[0].kwargs)
    for secret in source_environment.values():
        if secret not in {"safe-path", "safe-system-root"}:
            assert secret not in serialized_parameters
    await client.close()


async def test_observe_execute_contracts_are_strict_and_cleanup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = MCPHarness()
    harness.install(monkeypatch)
    client = _client(tmp_path)

    await _bind(
        client,
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        plan_id=uuid4(),
        step_key="open_report",
    )
    observation = await client.observe()
    result = await client.execute(WaitAction(seconds=0))

    assert observation == harness.session.observation
    assert result.success is True
    assert [call[0] for call in harness.session.calls] == [
        "axis_bind_step",
        "browser_observe",
        "browser_execute",
    ]
    execute_payload = harness.session.calls[-1][1]
    assert execute_payload == {"action": {"type": "wait", "seconds": 0}}

    await client.close()
    await client.close()
    assert harness.events.count("session.exit") == 1
    assert harness.events.count("stdio.exit") == 1


@pytest.mark.parametrize("timeout_seconds", [0, 301])
def test_direct_client_rejects_unbounded_timeouts(
    tmp_path: Path,
    timeout_seconds: int,
) -> None:
    policy_path = tmp_path / "firewall.json"
    policy_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="timeout_seconds"):
        TaskScopedPlaywrightMCP(
            database_path=tmp_path / "axis.db",
            firewall_policy_path=policy_path,
            downloads_path=tmp_path / "downloads",
            timeout_seconds=timeout_seconds,
        )
