from __future__ import annotations

import json
from typing import TypeVar, cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from axis_agent.contracts import (
    ExecutionPlan,
    PlannerInput,
    PlanStep,
    TaskRequest,
)
from axis_agent.contracts.model_io import PlanDraft, PlanStepDraft
from axis_agent.openai_client import (
    ProviderCallError,
    ProviderErrorCategory,
    StructuredOutputClient,
    StructuredOutputError,
    StructuredOutputIssue,
)
from axis_agent.planner import PLANNER_INSTRUCTIONS, Planner, PlannerOutputError

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


def draft(
    *,
    actions: tuple[str, ...] = ("go_to_url", "click_element", "wait"),
    domains: tuple[str, ...] = ("example.com", "blocked.example"),
    key: str = "open_report",
) -> PlanDraft:
    return PlanDraft(
        objective="Open and inspect the approved report",
        assumptions=("The report is available",),
        completion_criteria=("The report is visible",),
        steps=(
            PlanStepDraft(
                key=key,
                order=1,
                objective="Open the report",
                success_criteria=("The report heading is visible",),
                allowed_action_types=actions,  # type: ignore[arg-type]
                required_domains=domains,
                max_attempts=2,
            ),
        ),
    )


def planner(client: RecordingClient, **overrides: object) -> Planner:
    values: dict[str, object] = {
        "client": cast(StructuredOutputClient, client),
        "model": "xai.grok-4.3",
        "authorized_domains": ("example.com",),
        "authorized_action_types": ("click_element", "wait"),
    }
    values.update(overrides)
    return Planner(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_planner_uses_direct_structured_client_and_stamps_runtime_ids() -> None:
    client = RecordingClient([draft()])
    service = planner(client)
    task_id = uuid4()
    session_id = uuid4()

    result = await service.plan(
        TaskRequest(task_id=task_id, prompt="Open the approved report"),
        session_id=session_id,
    )

    assert result.session_id == session_id
    assert result.task_id == task_id
    assert result.plan_id != task_id
    assert result.revision == 1
    assert result.steps[0].allowed_action_types == ("click_element", "wait")
    assert result.steps[0].required_domains == ("example.com",)
    assert result.steps[0].evidence_refs == ()
    assert result.created_at.utcoffset() is not None

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "xai.grok-4.3"
    assert call["output_type"] is PlanDraft
    assert call["instructions"] == PLANNER_INSTRUCTIONS
    assert call["max_output_tokens"] == 16_000
    payload = json.loads(cast(str, call["input"]))
    assert payload["task"] == {"prompt": "Open the approved report"}
    assert payload["authorizedPolicy"] == {
        "domains": ["example.com"],
        "actionTypes": ["click_element", "wait"],
    }
    assert str(session_id) not in cast(str, call["input"])
    assert str(task_id) not in cast(str, call["input"])


@pytest.mark.asyncio
async def test_planner_retries_one_structured_failure_with_correction_marker() -> None:
    client = RecordingClient(
        [
            StructuredOutputError(StructuredOutputIssue.SCHEMA),
            draft(actions=("wait",), domains=()),
        ]
    )

    result = await planner(client).plan(TaskRequest(prompt="Wait for the page"))

    assert result.steps[0].allowed_action_types == ("wait",)
    assert len(client.calls) == 2
    first = json.loads(cast(str, client.calls[0]["input"]))
    second = json.loads(cast(str, client.calls[1]["input"]))
    assert first["correctionRequired"] is False
    assert second["correctionRequired"] is True


@pytest.mark.asyncio
async def test_planner_redacts_secrets_before_any_model_call() -> None:
    client = RecordingClient([draft(actions=("wait",), domains=())])
    raw_secrets = (
        "Bearer abcdefghijklmnop",
        "sk-" + "abcdefghijklmnop",
        "api_key=super-sensitive-value",
        "password:also-sensitive",
    )

    await planner(client).plan(TaskRequest(prompt="Inspect this request: " + " ".join(raw_secrets)))

    model_input = cast(str, client.calls[0]["input"])
    assert "[REDACTED]" in model_input
    for secret in raw_secrets:
        assert secret not in model_input


@pytest.mark.asyncio
async def test_planner_fails_closed_after_two_invalid_structured_outputs() -> None:
    client = RecordingClient(
        [
            StructuredOutputError(StructuredOutputIssue.EMPTY),
            StructuredOutputError(StructuredOutputIssue.REFUSAL),
        ]
    )

    with pytest.raises(PlannerOutputError, match="INVALID_STRUCTURED_OUTPUT"):
        await planner(client).plan(TaskRequest(prompt="Open the report"))

    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_planner_does_not_retry_provider_failures() -> None:
    client = RecordingClient([ProviderCallError(ProviderErrorCategory.TRANSPORT)])

    with pytest.raises(ProviderCallError):
        await planner(client).plan(TaskRequest(prompt="Open the report"))

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_planner_repairs_a_draft_with_no_policy_permitted_action() -> None:
    client = RecordingClient(
        [
            draft(actions=("go_to_url",)),
            draft(actions=("wait",), domains=()),
        ]
    )

    result = await planner(client).plan(TaskRequest(prompt="Use only approved actions"))

    assert result.steps[0].allowed_action_types == ("wait",)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_plan_input_preserves_ids_and_increments_revision_without_leaking_them() -> None:
    session_id = uuid4()
    task = TaskRequest(prompt="Adapt the existing plan")
    prior = ExecutionPlan(
        session_id=session_id,
        task_id=task.task_id,
        objective="Initial plan",
        completion_criteria=("Initial completion",),
        steps=(
            PlanStep(
                key="initial",
                order=1,
                objective="Try initially",
                success_criteria=("Initial evidence",),
                allowed_action_types=("wait",),
            ),
        ),
    )
    client = RecordingClient([draft(actions=("wait",), domains=(), key="adapted")])

    result = await planner(client).plan_input(
        PlannerInput(
            session_id=session_id,
            task=task,
            prior_plan=prior,
            replan_reason_code="NAVIGATOR_BLOCKED",
        )
    )

    assert result.session_id == session_id
    assert result.task_id == task.task_id
    assert result.revision == 2
    assert result.plan_id != prior.plan_id
    request = cast(str, client.calls[0]["input"])
    payload = json.loads(request)
    assert payload["priorPlan"]["revision"] == 1
    assert payload["replanReasonCode"] == "NAVIGATOR_BLOCKED"
    for runtime_value in (prior.plan_id, prior.session_id, prior.task_id):
        assert str(runtime_value) not in request


@pytest.mark.asyncio
async def test_public_plan_generates_a_new_session_id() -> None:
    client = RecordingClient([draft(actions=("wait",), domains=())])
    result = await planner(client).plan(TaskRequest(prompt="Wait briefly"))

    assert isinstance(result.session_id, UUID)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "unsafe model"}, "model"),
        ({"authorized_domains": ("*.example.com",)}, "domain"),
        ({"authorized_action_types": ()}, "must not be empty"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_output_tokens": 10}, "max_output_tokens"),
    ],
)
def test_planner_rejects_invalid_runtime_policy(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        planner(RecordingClient([]), **overrides)
