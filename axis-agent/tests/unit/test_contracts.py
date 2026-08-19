from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from axis_agent.browser import BrowserAdapter, MockBrowserAdapter
from axis_agent.contracts import (
    ActionCommand,
    ApprovalDecision,
    ApprovalDecisionValue,
    ApprovalRequest,
    ApprovalRisk,
    ExecutionPlan,
    NavigatorInput,
    NavigatorOutcome,
    NavigatorOutcomeStatus,
    PlannerInput,
    PlanStep,
    TaskRequest,
    WorkflowResult,
    WorkflowStatus,
    parse_action,
    parse_production_action,
    production_action_payload,
)


def _plan(*, session_id: UUID | None = None, task_id: UUID | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        session_id=session_id or uuid4(),
        task_id=task_id or uuid4(),
        objective="Collect the approved report",
        completion_criteria=("The approved report is visible",),
        steps=(
            PlanStep(
                key="open_report",
                order=1,
                objective="Open the report",
                success_criteria=("Report page is visible",),
                allowed_action_types=("go_to_url",),
                required_domains=("EXAMPLE.com.",),
            ),
            PlanStep(
                key="read_report",
                order=2,
                objective="Read the report",
                success_criteria=("Report contents are observed",),
                depends_on=("open_report",),
                allowed_action_types=("scroll_to_text", "wait"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "done", "text": "complete", "success": True},
        {"type": "search_google", "intent": "search", "query": "AXIS"},
        {"type": "go_to_url", "url": "https://example.com"},
        {"type": "go_back"},
        {"type": "click_element", "index": 0},
        {"type": "input_text", "index": 1, "text": "hello"},
        {"type": "switch_tab", "tabId": "page-2"},
        {"type": "open_tab", "url": "https://example.com"},
        {"type": "close_tab", "tabId": "page-2"},
        {"type": "cache_content", "content": "value"},
        {"type": "scroll_to_percent", "yPercent": 50},
        {"type": "scroll_to_top"},
        {"type": "scroll_to_bottom"},
        {"type": "previous_page"},
        {"type": "next_page"},
        {"type": "scroll_to_text", "text": "target", "nth": 1},
        {"type": "send_keys", "keys": "Enter"},
        {"type": "get_dropdown_options", "index": 1},
        {"type": "select_dropdown_option", "index": 1, "text": "Option"},
        {"type": "wait", "seconds": 0},
    ],
)
def test_all_twenty_actions_parse(payload: dict[str, object]) -> None:
    assert parse_action(payload).type == payload["type"]


def test_malformed_or_unknown_actions_fail_closed() -> None:
    with pytest.raises(ValidationError):
        parse_action({"type": "go_to_url", "url": "https://example.com", "extra": True})
    with pytest.raises(ValidationError):
        parse_action({"type": "arbitrary_shell", "command": "whoami"})
    with pytest.raises(ValidationError):
        parse_action({"type": "scroll_to_percent", "yPercent": 101})


def test_production_actions_strip_legacy_reasoning_and_selector_fields() -> None:
    action = parse_production_action(
        {
            "type": "click_element",
            "index": 2,
            "intent": "private model reasoning",
            "xpath": "//button",
        }
    )
    assert action.intent == ""
    assert action.xpath is None
    assert production_action_payload(action) == {"type": "click_element", "index": 2}

    command = ActionCommand(
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        ordinal=0,
        expected_observation_id="f" * 32,
        expected_origin="https://EXAMPLE.com:443",
        action=action,
    )
    assert command.expected_observation_id == "f" * 32
    assert command.expected_origin == "https://example.com"
    with pytest.raises(ValidationError, match="clean HTTP"):
        ActionCommand(
            session_id=uuid4(),
            task_id=uuid4(),
            step_id=uuid4(),
            ordinal=0,
            expected_origin="https://example.com/path?secret=value",
            action=action,
        )


@pytest.mark.parametrize(
    "action_type",
    ["done", "search_google", "cache_content", "previous_page", "next_page"],
)
def test_legacy_only_actions_are_not_in_the_production_surface(action_type: str) -> None:
    payloads: dict[str, dict[str, object]] = {
        "done": {"type": "done", "text": "done", "success": True},
        "search_google": {"type": "search_google", "query": "AXIS"},
        "cache_content": {"type": "cache_content", "content": "private"},
        "previous_page": {"type": "previous_page"},
        "next_page": {"type": "next_page"},
    }
    assert parse_action(payloads[action_type]).type == action_type
    with pytest.raises(ValidationError):
        parse_production_action(payloads[action_type])


def test_execution_plan_enforces_ordered_dag_and_production_actions() -> None:
    plan = _plan()
    assert [step.key for step in plan.steps] == ["open_report", "read_report"]
    assert plan.steps[0].required_domains == ("example.com",)

    with pytest.raises(ValidationError, match="contiguous order"):
        ExecutionPlan(
            session_id=plan.session_id,
            task_id=plan.task_id,
            objective="Invalid order",
            completion_criteria=("Never",),
            steps=(
                PlanStep(
                    key="later",
                    order=2,
                    objective="Later",
                    success_criteria=("Later",),
                    allowed_action_types=("wait",),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="must precede"):
        ExecutionPlan(
            session_id=plan.session_id,
            task_id=plan.task_id,
            objective="Invalid dependency",
            completion_criteria=("Never",),
            steps=(
                PlanStep(
                    key="first",
                    order=1,
                    objective="First",
                    success_criteria=("First",),
                    depends_on=("second",),
                    allowed_action_types=("wait",),
                ),
                PlanStep(
                    key="second",
                    order=2,
                    objective="Second",
                    success_criteria=("Second",),
                    allowed_action_types=("wait",),
                ),
            ),
        )


def test_planner_and_navigator_inputs_bind_runtime_owned_ids_and_order() -> None:
    plan = _plan()
    task = TaskRequest(task_id=plan.task_id, prompt="Read the approved report")
    planner_input = PlannerInput(session_id=plan.session_id, task=task)
    assert planner_input.prior_plan is None

    with pytest.raises(ValidationError, match="supplied together"):
        PlannerInput(
            session_id=plan.session_id,
            task=task,
            observation_id="a" * 32,
        )

    navigator_input = NavigatorInput(
        session_id=plan.session_id,
        task_id=plan.task_id,
        plan=plan,
        active_step_key="open_report",
        observation_id="a" * 32,
        page_id="page-1",
        origin="https://example.com",
    )
    assert navigator_input.active_step_key == "open_report"

    with pytest.raises(ValidationError, match="next eligible"):
        NavigatorInput(
            session_id=plan.session_id,
            task_id=plan.task_id,
            plan=plan,
            active_step_key="read_report",
            observation_id="a" * 32,
            page_id="page-1",
            origin="https://example.com",
        )


def test_navigator_outcome_and_workflow_result_enforce_terminal_invariants() -> None:
    plan = _plan()
    outcome = NavigatorOutcome(
        session_id=plan.session_id,
        task_id=plan.task_id,
        plan_id=plan.plan_id,
        step_key="open_report",
        status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
        actions=({"type": "go_to_url", "url": "https://example.com"},),
    )
    assert outcome.actions[0].type == "go_to_url"

    with pytest.raises(ValidationError, match="at least one action"):
        NavigatorOutcome(
            session_id=plan.session_id,
            task_id=plan.task_id,
            plan_id=plan.plan_id,
            step_key="open_report",
            status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
        )

    result = WorkflowResult(
        session_id=plan.session_id,
        task_id=plan.task_id,
        plan_id=plan.plan_id,
        status=WorkflowStatus.SUCCEEDED,
        completed_step_keys=("open_report", "read_report"),
        total_steps=2,
    )
    assert result.error_code is None


def test_approval_contract_hashes_only_the_canonical_action() -> None:
    plan = _plan()
    request = ApprovalRequest.for_action(
        session_id=plan.session_id,
        task_id=plan.task_id,
        plan_id=plan.plan_id,
        step_key="open_report",
        action_id=uuid4(),
        observation_id="b" * 32,
        action={
            "type": "click_element",
            "index": 1,
            "intent": "do not retain",
            "xpath": "//button",
        },
        risk=ApprovalRisk.INTERACTION,
        reason_code="CLICK_REQUIRES_APPROVAL",
    )
    assert len(request.action_sha256) == 64
    assert "retain" not in request.model_dump_json()
    assert "xpath" not in request.model_dump_json()

    decision = ApprovalDecision(
        approval_id=request.approval_id,
        decision=ApprovalDecisionValue.APPROVED,
        actor="user",
        reason_code="USER_APPROVED",
    )
    assert decision.actor == "user"

    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="five minutes"):
        ApprovalRequest(
            session_id=plan.session_id,
            task_id=plan.task_id,
            plan_id=plan.plan_id,
            step_key="open_report",
            action_id=uuid4(),
            observation_id="b" * 32,
            action_type="click_element",
            action_sha256="c" * 64,
            risk=ApprovalRisk.INTERACTION,
            reason_code="CLICK_REQUIRES_APPROVAL",
            requested_at=now,
            expires_at=now + timedelta(minutes=6),
        )


@pytest.mark.asyncio
async def test_mock_browser_is_a_standalone_adapter() -> None:
    browser = MockBrowserAdapter()
    assert isinstance(browser, BrowserAdapter)
    await browser.start()
    assert (await browser.observe()).observation_id == "0" * 32
    action_id = str(uuid4())
    result = await browser.execute(
        parse_action({"type": "wait", "seconds": 0}), action_id=action_id
    )
    assert result.success is True
    assert len(browser.calls) == 1
    await browser.stop()
