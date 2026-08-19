from __future__ import annotations

import json
from copy import deepcopy
from typing import TypeVar, cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from axis_agent.browser import BrowserObservation
from axis_agent.contracts import (
    ActionResult,
    ExecutionPlan,
    NavigatorDecision,
    NavigatorDecisionStatus,
    NavigatorInput,
    NavigatorOutcomeStatus,
    PlanStep,
    ProductionAction,
    ProductionActionType,
)
from axis_agent.navigator import NAVIGATOR_INSTRUCTIONS, Navigator, NavigatorOutputError
from axis_agent.openai_client import (
    ProviderCallError,
    ProviderErrorCategory,
    StructuredOutputClient,
    StructuredOutputError,
    StructuredOutputIssue,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class RecordingClient:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    async def parse_structured(
        self,
        *,
        model: str,
        input: str,
        output_type: type[ModelT],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelT:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "output_type": output_type,
                "instructions": instructions,
                "max_output_tokens": max_output_tokens,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return cast(ModelT, output)


class RecordingBrowser:
    def __init__(self) -> None:
        self.observation = BrowserObservation(
            observation_id="0123456789abcdef0123456789abcdef",
            page_id="page-1",
            url="https://example.com/report",
            title="Approved report",
            visible_text="The report is ready",
            interactive_elements=[{"index": 3, "tag": "button", "text": "Open"}],
        )
        self.bindings: list[dict[str, object]] = []
        self.actions: list[ProductionAction] = []
        self.close_count = 0

    async def bind_step(
        self,
        *,
        session_id: object,
        task_id: object,
        step_id: object,
        plan_id: object,
        step_key: str,
        allowed_action_types: tuple[ProductionActionType, ...],
    ) -> None:
        self.bindings.append(
            {
                "session_id": session_id,
                "task_id": task_id,
                "step_id": step_id,
                "plan_id": plan_id,
                "step_key": step_key,
                "allowed_action_types": allowed_action_types,
            }
        )

    async def observe(self) -> BrowserObservation:
        return self.observation.model_copy(deep=True)

    async def execute(self, action: ProductionAction) -> ActionResult:
        self.actions.append(deepcopy(action))
        return ActionResult(action_id=uuid4(), success=True, message=action.type)

    async def close(self) -> None:
        self.close_count += 1


def _plan(
    *,
    allowed_action_types: tuple[ProductionActionType, ...] = ("click_element", "wait"),
) -> ExecutionPlan:
    return ExecutionPlan(
        session_id=uuid4(),
        task_id=uuid4(),
        objective="Open the approved report",
        completion_criteria=("The report is visible",),
        steps=(
            PlanStep(
                key="open_report",
                order=1,
                objective="Open the report",
                success_criteria=("The report heading is visible",),
                allowed_action_types=allowed_action_types,
                required_domains=("example.com",),
            ),
        ),
    )


def _navigator_input(plan: ExecutionPlan, observation: BrowserObservation) -> NavigatorInput:
    return NavigatorInput(
        session_id=plan.session_id,
        task_id=plan.task_id,
        plan=plan,
        active_step_key="open_report",
        observation_id=observation.observation_id,
        page_id=observation.page_id,
        origin="https://example.com",
    )


def _navigator(
    client: RecordingClient,
    browser: RecordingBrowser,
) -> Navigator:
    return Navigator(
        client=cast(StructuredOutputClient, client),
        model="xai.grok-4.20-0309-non-reasoning",
        browser=browser,
    )


async def test_navigator_returns_exactly_one_runtime_action_from_direct_structured_output() -> None:
    client = RecordingClient(
        [
            NavigatorDecision(
                status=NavigatorDecisionStatus.ACTION_REQUIRED,
                action={"type": "click_element", "index": 3},
                summary="Open the report",
            )
        ]
    )
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan()

    await service.bind_step(
        session_id=plan.session_id,
        task_id=plan.task_id,
        step_id=uuid4(),
        plan_id=plan.plan_id,
        step_key="open_report",
        allowed_action_types=plan.steps[0].allowed_action_types,
    )
    observation = await service.observe()
    outcome = await service.run_navigator(_navigator_input(plan, observation))

    assert outcome.status is NavigatorOutcomeStatus.ACTIONS_REQUIRED
    assert len(outcome.actions) == 1
    assert outcome.actions[0].type == "click_element"
    assert outcome.session_id == plan.session_id
    assert outcome.task_id == plan.task_id
    assert outcome.plan_id == plan.plan_id

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "xai.grok-4.20-0309-non-reasoning"
    assert call["output_type"] is NavigatorDecision
    assert call["instructions"] == NAVIGATOR_INSTRUCTIONS
    assert call["max_output_tokens"] == 8_000
    request = cast(str, call["input"])
    payload = json.loads(request)
    assert payload["correctionRequired"] is False
    assert payload["plan"]["activeStep"]["key"] == "open_report"
    assert payload["observation"]["observationId"] == observation.observation_id
    for runtime_id in (plan.session_id, plan.task_id, plan.plan_id):
        assert str(runtime_id) not in request


async def test_navigator_retries_one_structured_failure_with_correction_marker() -> None:
    client = RecordingClient(
        [
            StructuredOutputError(StructuredOutputIssue.SCHEMA),
            NavigatorDecision(
                status=NavigatorDecisionStatus.STEP_COMPLETED,
                summary="Step is complete",
            ),
        ]
    )
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan()
    observation = await service.observe()

    outcome = await service.run_navigator(_navigator_input(plan, observation))

    assert outcome.status is NavigatorOutcomeStatus.STEP_COMPLETED
    assert len(client.calls) == 2
    first = json.loads(cast(str, client.calls[0]["input"]))
    second = json.loads(cast(str, client.calls[1]["input"]))
    assert first["correctionRequired"] is False
    assert second["correctionRequired"] is True


async def test_navigator_fails_closed_after_two_invalid_structured_outputs() -> None:
    client = RecordingClient(
        [
            StructuredOutputError(StructuredOutputIssue.EMPTY),
            StructuredOutputError(StructuredOutputIssue.REFUSAL),
        ]
    )
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan()
    observation = await service.observe()

    with pytest.raises(NavigatorOutputError, match="INVALID_STRUCTURED_OUTPUT"):
        await service.run_navigator(_navigator_input(plan, observation))

    assert len(client.calls) == 2


async def test_navigator_does_not_retry_provider_failures() -> None:
    client = RecordingClient([ProviderCallError(ProviderErrorCategory.TRANSPORT)])
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan()
    observation = await service.observe()

    with pytest.raises(ProviderCallError):
        await service.run_navigator(_navigator_input(plan, observation))

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("observation_id", "page_id", "origin"),
    [
        ("f" * 32, "page-1", "https://example.com"),
        ("0123456789abcdef0123456789abcdef", "stale-page", "https://example.com"),
        ("0123456789abcdef0123456789abcdef", "page-1", "https://attacker.example"),
    ],
)
async def test_navigator_rejects_stale_observation_before_calling_model(
    observation_id: str,
    page_id: str,
    origin: str,
) -> None:
    client = RecordingClient([])
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan()
    observation = await service.observe()
    navigator_input = _navigator_input(plan, observation).model_copy(
        update={"observation_id": observation_id, "page_id": page_id, "origin": origin}
    )

    with pytest.raises(NavigatorOutputError, match="latest observation"):
        await service.run_navigator(navigator_input)

    assert client.calls == []


async def test_navigator_retries_then_rejects_action_outside_active_step() -> None:
    disallowed = NavigatorDecision(
        status=NavigatorDecisionStatus.ACTION_REQUIRED,
        action={"type": "click_element", "index": 3},
    )
    client = RecordingClient([disallowed, disallowed])
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan(allowed_action_types=("wait",))
    observation = await service.observe()

    with pytest.raises(NavigatorOutputError, match="DECISION_OUTSIDE_RUNTIME_POLICY"):
        await service.run_navigator(_navigator_input(plan, observation))

    assert len(client.calls) == 2
    assert browser.actions == []


async def test_one_observation_authorizes_at_most_one_action_and_close_cleans_up() -> None:
    decision = NavigatorDecision(
        status=NavigatorDecisionStatus.ACTION_REQUIRED,
        action={"type": "wait", "seconds": 0},
    )
    client = RecordingClient([decision, decision])
    browser = RecordingBrowser()
    service = _navigator(client, browser)
    plan = _plan(allowed_action_types=("wait",))
    observation = await service.observe()
    navigator_input = _navigator_input(plan, observation)

    first = await service.run_navigator(navigator_input)
    result = await service.execute(first.actions[0])

    assert result.success is True
    assert len(browser.actions) == 1
    with pytest.raises(NavigatorOutputError, match="latest observation"):
        await service.run_navigator(navigator_input)
    assert len(client.calls) == 1

    await service.close()
    assert browser.close_count == 1


@pytest.mark.parametrize(
    ("model", "max_output_tokens", "message"),
    [
        ("unsafe model", 8_000, "model"),
        ("xai.grok-4.20-0309-non-reasoning", 128, "max_output_tokens"),
    ],
)
def test_navigator_rejects_invalid_runtime_configuration(
    model: str,
    max_output_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Navigator(
            client=cast(StructuredOutputClient, RecordingClient([])),
            model=model,
            browser=RecordingBrowser(),
            max_output_tokens=max_output_tokens,
        )
