from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from axis_agent.browser.base import BrowserObservation
from axis_agent.contracts import ExecutionPlan, PlanStep
from axis_agent.contracts.actions import Action, ActionResult
from axis_agent.firewall import NavigationPermit, policy_from_hosts
from axis_agent.mcp import MCPProfile, application
from axis_agent.mcp.application import (
    build_gateway_runtime,
    parse_gateway_arguments,
    serve_gateway_runtime,
)
from axis_agent.mcp.client import ManagedGatewayProcessConfig


def _config(
    tmp_path: Path, profile: MCPProfile = MCPProfile.PLANNER
) -> ManagedGatewayProcessConfig:
    policy_path = tmp_path / "firewall.json"
    policy_path.write_text(
        policy_from_hosts(["93.184.216.34"]).model_dump_json(),
        encoding="utf-8",
    )
    navigator = profile is MCPProfile.NAVIGATOR
    return ManagedGatewayProcessConfig(
        profile=profile,
        database_path=tmp_path / "axis.db",
        firewall_policy_path=policy_path,
        downloads_path=tmp_path / "downloads",
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        allowed_action_types=("wait",) if navigator else (),
        plan_id=uuid4() if navigator else None,
        step_key="managed_step" if navigator else None,
    )


class FakeBrowser:
    instances: list[FakeBrowser] = []

    def __init__(self, settings: object, firewall: object, permits: object) -> None:
        self.settings = settings
        self.firewall = firewall
        self.permits = permits
        self.url = "https://93.184.216.34/report"
        FakeBrowser.instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def observe(self) -> BrowserObservation:
        return BrowserObservation(
            observation_id="0" * 32,
            page_id="page",
            url=self.url,
        )

    async def execute(
        self,
        action: Action,
        *,
        action_id: str,
        navigation_permit: NavigationPermit | None = None,
        expected_observation_id: str | None = None,
        expected_page_id: str | None = None,
        expected_origin: str | None = None,
    ) -> ActionResult:
        del (
            action,
            navigation_permit,
            expected_observation_id,
            expected_page_id,
            expected_origin,
        )
        return ActionResult(action_id=UUID(action_id), success=True)


def test_runtime_wires_ephemeral_headless_browser_and_security_dispatcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeBrowser.instances.clear()
    monkeypatch.setattr(application, "_load_playwright_adapter_factory", lambda: FakeBrowser)
    config = _config(tmp_path, MCPProfile.NAVIGATOR)

    runtime = build_gateway_runtime(config)

    assert runtime.service.profile is MCPProfile.NAVIGATOR
    assert runtime.database.path == config.database_path
    assert runtime.service._session_id == config.session_id
    assert runtime.service._task_id == config.task_id
    assert runtime.service._step_id == config.step_id
    assert runtime.service._allowed_action_types == frozenset({"wait"})
    browser = FakeBrowser.instances[-1]
    settings = cast(Any, browser.settings)
    assert settings.headless is True
    assert settings.user_data_dir is None
    assert settings.downloads_path == config.downloads_path


async def test_trusted_ids_reach_dispatcher_and_existing_database_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeBrowser.instances.clear()
    monkeypatch.setattr(application, "_load_playwright_adapter_factory", lambda: FakeBrowser)
    config = _config(tmp_path, MCPProfile.NAVIGATOR)
    runtime = build_gateway_runtime(config)

    await runtime.database.connect()
    await runtime.database.initialize()
    try:
        await runtime.database.create_session(
            session_id=config.session_id,
            task_id=config.task_id,
            task_summary="Managed MCP step",
            config_hash="test-config",
        )
        await runtime.database.create_step(
            session_id=config.session_id,
            step_id=config.step_id,
            step_number=0,
        )
        assert config.plan_id is not None
        plan = ExecutionPlan(
            plan_id=config.plan_id,
            session_id=config.session_id,
            task_id=config.task_id,
            objective="Managed MCP test",
            completion_criteria=("Wait succeeds",),
            steps=(
                PlanStep(
                    key="managed_step",
                    order=1,
                    objective="Wait",
                    success_criteria=("Wait succeeds",),
                    allowed_action_types=("wait",),
                ),
            ),
        )
        await runtime.database.record_plan_metadata(plan)

        await runtime.service.browser_observe()
        response = await runtime.service.browser_execute({"type": "wait", "seconds": 0})

        assert response["result"]["success"] is True  # type: ignore[index]
        row = await (
            await runtime.database.connection.execute("SELECT session_id, step_id FROM actions")
        ).fetchone()
        assert row is not None
        assert row["session_id"] == str(config.session_id)
        assert row["step_id"] == str(config.step_id)
    finally:
        await runtime.service.stop()
        await runtime.database.close()


async def test_managed_navigator_auto_approves_high_impact_action_on_trusted_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeBrowser.instances.clear()
    monkeypatch.setattr(application, "_load_playwright_adapter_factory", lambda: FakeBrowser)
    config = replace(
        _config(tmp_path, MCPProfile.NAVIGATOR),
        allowed_action_types=("click_element",),
    )
    runtime = build_gateway_runtime(config)

    await runtime.database.connect()
    await runtime.database.initialize()
    try:
        await runtime.database.create_session(
            session_id=config.session_id,
            task_id=config.task_id,
            task_summary="Managed approval test",
            config_hash="test-config",
        )
        await runtime.database.create_step(
            session_id=config.session_id,
            step_id=config.step_id,
            step_number=0,
        )
        assert config.plan_id is not None
        await runtime.database.record_plan_metadata(
            ExecutionPlan(
                plan_id=config.plan_id,
                session_id=config.session_id,
                task_id=config.task_id,
                objective="Click on a trusted site",
                completion_criteria=("Click succeeds",),
                steps=(
                    PlanStep(
                        key="managed_step",
                        order=1,
                        objective="Click the approved control",
                        success_criteria=("Click succeeds",),
                        allowed_action_types=("click_element",),
                    ),
                ),
            )
        )

        await runtime.service.browser_observe()
        response = await runtime.service.browser_execute({"type": "click_element", "index": 0})

        assert response["result"]["success"] is True  # type: ignore[index]
        request_count = await (
            await runtime.database.connection.execute("SELECT COUNT(*) FROM approval_requests")
        ).fetchone()
        action_count = await (
            await runtime.database.connection.execute("SELECT COUNT(*) FROM actions")
        ).fetchone()
        approval = await (
            await runtime.database.connection.execute(
                "SELECT decision, reason_code FROM approval_decisions"
            )
        ).fetchone()
        assert request_count[0] == 1
        assert action_count[0] == 1
        assert approval is not None
        assert approval["decision"] == "approved"
        assert approval["reason_code"] == "TRUSTED_SITE_POLICY_APPROVED"
    finally:
        await runtime.service.stop()
        await runtime.database.close()


async def test_managed_navigator_rejects_auto_approval_on_untrusted_current_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeBrowser.instances.clear()
    monkeypatch.setattr(application, "_load_playwright_adapter_factory", lambda: FakeBrowser)
    config = replace(
        _config(tmp_path, MCPProfile.NAVIGATOR),
        allowed_action_types=("click_element",),
    )
    runtime = build_gateway_runtime(config)
    FakeBrowser.instances[-1].url = "about:blank"

    await runtime.database.connect()
    await runtime.database.initialize()
    try:
        await runtime.database.create_session(
            session_id=config.session_id,
            task_id=config.task_id,
            task_summary="Untrusted origin approval test",
            config_hash="test-config",
        )
        await runtime.database.create_step(
            session_id=config.session_id,
            step_id=config.step_id,
            step_number=0,
        )
        assert config.plan_id is not None
        await runtime.database.record_plan_metadata(
            ExecutionPlan(
                plan_id=config.plan_id,
                session_id=config.session_id,
                task_id=config.task_id,
                objective="Reject click outside the trusted site",
                completion_criteria=("Click remains blocked",),
                steps=(
                    PlanStep(
                        key="managed_step",
                        order=1,
                        objective="Attempt click",
                        success_criteria=("Click is blocked",),
                        allowed_action_types=("click_element",),
                    ),
                ),
            )
        )

        await runtime.service.browser_observe()
        response = await runtime.service.browser_execute({"type": "click_element", "index": 0})

        assert response["result"]["success"] is False  # type: ignore[index]
        assert response["result"]["errorCode"] == "APPROVAL_REJECTED"  # type: ignore[index]
        request_count = await (
            await runtime.database.connection.execute("SELECT COUNT(*) FROM approval_requests")
        ).fetchone()
        action_count = await (
            await runtime.database.connection.execute("SELECT COUNT(*) FROM actions")
        ).fetchone()
        assert request_count[0] == 0
        assert action_count[0] == 0
    finally:
        await runtime.service.stop()
        await runtime.database.close()


def test_argument_parser_accepts_only_the_managed_gateway_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)

    parsed = parse_gateway_arguments(
        [
            "--profile=planner",
            f"--database-path={config.database_path}",
            f"--firewall-policy-path={config.firewall_policy_path}",
            f"--downloads-path={config.downloads_path}",
            f"--session-id={config.session_id}",
            f"--task-id={config.task_id}",
            f"--step-id={config.step_id}",
        ]
    )

    assert parsed == config

    with pytest.raises(SystemExit):
        parse_gateway_arguments(
            [
                "--profile=planner",
                f"--database-path={config.database_path}",
                f"--firewall-policy-path={config.firewall_policy_path}",
                f"--downloads-path={config.downloads_path}",
                f"--session-id={config.session_id}",
                f"--task-id={config.task_id}",
                f"--step-id={config.step_id}",
                "--command=malicious",
            ]
        )


async def test_stdio_server_and_resources_share_one_clean_lifecycle() -> None:
    events: list[str] = []

    class FakeDatabase:
        async def connect(self) -> None:
            events.append("database.connect")

        async def initialize(self) -> None:
            events.append("database.initialize")

        async def close(self) -> None:
            events.append("database.close")

    class FakeService:
        async def stop(self) -> None:
            events.append("service.stop")

    class FakeServer:
        async def run_stdio_async(self) -> None:
            events.append("server.run")

    runtime = SimpleNamespace(service=FakeService(), database=FakeDatabase())
    await serve_gateway_runtime(
        cast(Any, runtime),
        server_factory=lambda _service: FakeServer(),
    )

    assert events == [
        "database.connect",
        "database.initialize",
        "server.run",
        "service.stop",
        "database.close",
    ]


async def test_gateway_closes_browser_and_database_when_stdio_fails() -> None:
    events: list[str] = []

    class FakeDatabase:
        async def connect(self) -> None:
            events.append("database.connect")

        async def initialize(self) -> None:
            events.append("database.initialize")

        async def close(self) -> None:
            events.append("database.close")

    class FakeService:
        async def stop(self) -> None:
            events.append("service.stop")

    class FailingServer:
        async def run_stdio_async(self) -> None:
            raise RuntimeError("transport failed")

    runtime = SimpleNamespace(service=FakeService(), database=FakeDatabase())
    with pytest.raises(RuntimeError, match="transport failed"):
        await serve_gateway_runtime(
            cast(Any, runtime),
            server_factory=lambda _service: FailingServer(),
        )

    assert events[-2:] == ["service.stop", "database.close"]
