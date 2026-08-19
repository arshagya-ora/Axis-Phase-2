from __future__ import annotations

from copy import deepcopy
from uuid import UUID, uuid4

import pytest

from axis_agent.browser.base import BrowserObservation
from axis_agent.contracts.actions import (
    PRODUCTION_ACTION_TYPES,
    Action,
    ActionCommand,
    ActionResult,
)
from axis_agent.firewall import NavigationPermit
from axis_agent.mcp import (
    AxisMCPService,
    MCPGatewayConfigurationError,
    MCPGatewayExecutionError,
    MCPInvalidArgumentsError,
    MCPProfile,
    MCPToolNotAllowedError,
)


class RecordingBrowser:
    def __init__(self) -> None:
        self.started = False
        self.start_count = 0
        self.stop_count = 0
        self.observation = BrowserObservation(
            observation_id="0123456789abcdef0123456789abcdef",
            page_id="page-owned-by-browser",
            url="https://Example.COM:443/research?q=private",
            title="Research",
            visible_text="bounded observation",
        )
        self.calls: list[tuple[str, Action, NavigationPermit | None]] = []

    async def start(self) -> None:
        self.started = True
        self.start_count += 1

    async def stop(self) -> None:
        self.started = False
        self.stop_count += 1

    async def observe(self) -> BrowserObservation:
        assert self.started
        return self.observation.model_copy(deep=True)

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
        assert self.started
        assert expected_observation_id in {None, self.observation.observation_id}
        assert expected_page_id in {None, self.observation.page_id}
        assert expected_origin in {None, "https://example.com"}
        self.calls.append((action_id, deepcopy(action), navigation_permit))
        return ActionResult(action_id=UUID(action_id), success=True, message=action.type)


class RecordingDispatcher:
    def __init__(self) -> None:
        self.commands: list[ActionCommand] = []
        self.observations: list[BrowserObservation | None] = []

    async def dispatch(
        self,
        command: ActionCommand,
        *,
        observation: BrowserObservation | None = None,
    ) -> ActionResult:
        self.commands.append(command.model_copy(deep=True))
        self.observations.append(observation.model_copy(deep=True) if observation else None)
        return ActionResult(
            action_id=command.action_id,
            success=True,
            message=f"dispatched:{command.action.type}",
        )


class RejectingApprovalGate:
    def __init__(self) -> None:
        self.commands: list[ActionCommand] = []

    async def authorize(self, command: ActionCommand) -> bool:
        self.commands.append(command)
        return False


class AllowingApprovalGate:
    async def authorize(self, command: ActionCommand) -> bool:
        del command
        return True


class RecordingApprovalFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str]] = []

    def __call__(self, *, plan_id: UUID, step_key: str) -> AllowingApprovalGate:
        self.calls.append((plan_id, step_key))
        return AllowingApprovalGate()


@pytest.fixture
def browser() -> RecordingBrowser:
    return RecordingBrowser()


@pytest.fixture
def dispatcher() -> RecordingDispatcher:
    return RecordingDispatcher()


def navigator_service(
    browser: RecordingBrowser,
    dispatcher: RecordingDispatcher | None = None,
    approval_gate: RejectingApprovalGate | AllowingApprovalGate | None = None,
) -> AxisMCPService:
    return AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=browser,
        dispatcher=dispatcher,
        approval_gate=approval_gate,
        allowed_action_types=tuple(PRODUCTION_ACTION_TYPES),
    )


def test_profiles_publish_only_their_owned_tool_names(browser: RecordingBrowser) -> None:
    planner = AxisMCPService(profile=MCPProfile.PLANNER, browser=browser)
    navigator = AxisMCPService(profile=MCPProfile.NAVIGATOR, browser=browser)

    assert planner.tool_names == ()
    assert navigator.tool_names == ("browser_observe", "browser_execute")


@pytest.mark.parametrize(
    "tool_name",
    [
        "research_navigate",
        "research_observe",
        "research_back",
        "research_scroll",
        "research_close",
        "browser_observe",
        "browser_execute",
        "axis_bind_step",
    ],
)
async def test_direct_sdk_planner_profile_exposes_no_mcp_tools(
    browser: RecordingBrowser,
    tool_name: str,
) -> None:
    service = AxisMCPService(profile=MCPProfile.PLANNER, browser=browser)

    with pytest.raises(MCPToolNotAllowedError, match="planner profile"):
        await service.invoke(tool_name, {})

    assert browser.start_count == 0


async def test_navigator_observation_is_bounded_to_browser_contract(
    browser: RecordingBrowser,
) -> None:
    service = AxisMCPService(profile=MCPProfile.NAVIGATOR, browser=browser)

    response = await service.invoke("browser_observe")

    observation = response["observation"]
    assert observation["pageId"] == "page-owned-by-browser"  # type: ignore[index]
    assert observation["visibleText"] == "bounded observation"  # type: ignore[index]


async def test_navigator_non_url_action_uses_server_id_and_current_preconditions(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = navigator_service(browser, dispatcher, AllowingApprovalGate())

    await service.browser_observe()
    response = await service.browser_execute({"type": "click_element", "index": 2})

    dispatched = dispatcher.commands[-1]
    command = response["command"]
    assert dispatched.action.type == "click_element"
    assert command["actionId"] == str(dispatched.action_id)  # type: ignore[index]
    assert command["expectedObservationId"] == "0123456789abcdef0123456789abcdef"  # type: ignore[index]
    assert command["expectedPageId"] == "page-owned-by-browser"  # type: ignore[index]
    assert command["expectedOrigin"] == "https://example.com"  # type: ignore[index]


async def test_navigator_url_actions_never_bypass_dispatcher(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = navigator_service(browser, dispatcher)

    await service.browser_observe()
    await service.browser_execute({"type": "open_tab", "url": "https://allowed.example"})

    assert not browser.calls
    assert dispatcher.commands[-1].action.type == "open_tab"


async def test_navigator_rejects_action_outside_active_plan_step(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=browser,
        dispatcher=dispatcher,
        allowed_action_types=("wait",),
    )

    with pytest.raises(MCPToolNotAllowedError, match="outside the active plan step"):
        await service.browser_execute({"type": "click_element", "index": 0})

    assert not dispatcher.commands
    assert not browser.calls


async def test_server_side_approval_rejection_prevents_dispatch(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    approval_gate = RejectingApprovalGate()
    service = navigator_service(browser, dispatcher, approval_gate)

    await service.browser_observe()
    response = await service.browser_execute({"type": "click_element", "index": 0})

    assert len(approval_gate.commands) == 1
    assert dispatcher.commands == []
    assert response["result"]["success"] is False  # type: ignore[index]
    assert response["result"]["errorCode"] == "APPROVAL_REJECTED"  # type: ignore[index]


async def test_high_impact_action_fails_closed_without_approval_gate(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = navigator_service(browser, dispatcher)

    await service.browser_observe()
    with pytest.raises(MCPGatewayConfigurationError, match="approval gate"):
        await service.browser_execute({"type": "click_element", "index": 0})

    assert not dispatcher.commands


async def test_action_fails_closed_without_dispatcher(browser: RecordingBrowser) -> None:
    service = navigator_service(browser)

    await service.browser_observe()
    with pytest.raises(MCPGatewayConfigurationError, match="security dispatcher"):
        await service.browser_execute({"type": "go_to_url", "url": "https://example.com"})


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "done", "text": "finished", "success": True},
        {"type": "cache_content", "content": "data"},
        {"type": "click_element", "index": 0, "xpath": "//button"},
        {"type": "input_text", "index": 0, "text": "value", "xpath": "//input"},
        {"type": "run_javascript", "code": "alert(1)"},
        {"type": "click_element", "index": 0, "actionId": "model-owned"},
    ],
)
async def test_navigator_rejects_control_plane_xpath_unknown_and_id_fields(
    browser: RecordingBrowser, payload: dict[str, object]
) -> None:
    service = AxisMCPService(profile=MCPProfile.NAVIGATOR, browser=browser)

    with pytest.raises((MCPInvalidArgumentsError, MCPToolNotAllowedError)):
        await service.browser_execute(payload)

    assert not browser.calls


async def test_service_start_and_stop_are_idempotent(browser: RecordingBrowser) -> None:
    service = AxisMCPService(profile=MCPProfile.PLANNER, browser=browser)

    await service.start()
    await service.start()
    await service.stop()
    await service.stop()

    assert browser.start_count == 1
    assert browser.stop_count == 1


async def test_gateway_rejects_mismatched_dispatcher_action_id(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = navigator_service(browser, dispatcher)

    async def mismatched_dispatch(
        command: ActionCommand,
        *,
        observation: BrowserObservation | None = None,
    ) -> ActionResult:
        del command, observation
        return ActionResult(action_id=uuid4(), success=True)

    dispatcher.dispatch = mismatched_dispatch  # type: ignore[method-assign]

    await service.browser_observe()
    with pytest.raises(MCPGatewayExecutionError, match="mismatched action ID"):
        await service.browser_execute({"type": "wait", "seconds": 0})


async def test_navigator_requires_a_fresh_observation_for_each_action(
    browser: RecordingBrowser, dispatcher: RecordingDispatcher
) -> None:
    service = navigator_service(browser, dispatcher)

    with pytest.raises(MCPGatewayConfigurationError, match="observe immediately"):
        await service.browser_execute({"type": "wait", "seconds": 0})

    await service.browser_observe()
    response = await service.browser_execute({"type": "wait", "seconds": 0})
    assert response["result"]["success"] is True  # type: ignore[index]
    with pytest.raises(MCPGatewayConfigurationError, match="observe immediately"):
        await service.browser_execute({"type": "wait", "seconds": 0})


async def test_managed_step_rebinding_resets_observation_ordinal_and_runtime_ids(
    browser: RecordingBrowser,
    dispatcher: RecordingDispatcher,
) -> None:
    session_id = uuid4()
    task_id = uuid4()
    first_step_id = uuid4()
    second_step_id = uuid4()
    plan_id = uuid4()
    approvals = RecordingApprovalFactory()
    service = AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=browser,
        dispatcher=dispatcher,
        approval_gate=AllowingApprovalGate(),
        allowed_action_types=("click_element",),
        session_id=session_id,
        task_id=task_id,
        step_id=first_step_id,
        managed_step_approval_factory=approvals,
    )

    await service.browser_observe()
    first = await service.browser_execute({"type": "click_element", "index": 0})
    assert first["result"]["success"] is True  # type: ignore[index]
    assert dispatcher.commands[-1].step_id == first_step_id
    assert dispatcher.commands[-1].ordinal == 0

    response = await service.bind_managed_step(
        session_id=str(session_id),
        task_id=str(task_id),
        step_id=str(second_step_id),
        plan_id=str(plan_id),
        step_key="verify_report",
        allowed_action_types=["wait"],
    )

    assert response == {"bound": True, "stepKey": "verify_report"}
    assert approvals.calls == [(plan_id, "verify_report")]
    with pytest.raises(MCPGatewayConfigurationError, match="observe immediately"):
        await service.browser_execute({"type": "wait", "seconds": 0})

    await service.browser_observe()
    second = await service.browser_execute({"type": "wait", "seconds": 0})
    assert second["result"]["success"] is True  # type: ignore[index]
    rebound = dispatcher.commands[-1]
    assert rebound.session_id == session_id
    assert rebound.task_id == task_id
    assert rebound.step_id == second_step_id
    assert rebound.ordinal == 0


async def test_internal_step_binding_is_not_on_model_invocation_surface(
    browser: RecordingBrowser,
) -> None:
    service = AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=browser,
        managed_step_approval_factory=RecordingApprovalFactory(),
    )

    assert service.managed_step_binding_enabled is True
    assert "axis_bind_step" not in service.tool_names
    with pytest.raises(MCPToolNotAllowedError, match="navigator profile"):
        await service.invoke(
            "axis_bind_step",
            {
                "session_id": str(uuid4()),
                "task_id": str(uuid4()),
                "step_id": str(uuid4()),
                "plan_id": str(uuid4()),
                "step_key": "open_report",
                "allowed_action_types": ["wait"],
            },
        )
