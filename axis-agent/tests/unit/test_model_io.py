from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from axis_agent.contracts import (
    NavigatorDecision,
    NavigatorDecisionStatus,
    PlanDraft,
    PlanStepDraft,
)
from axis_agent.contracts.model_io import canonicalize_model_domain


def plan_step(**overrides: object) -> PlanStepDraft:
    values: dict[str, object] = {
        "key": "open_report",
        "order": 1,
        "objective": "Open the approved report",
        "success_criteria": ("The report is visible",),
        "depends_on": (),
        "allowed_action_types": ("go_to_url", "click_element"),
        "required_domains": ("EXAMPLE.com.",),
        "max_attempts": 3,
    }
    values.update(overrides)
    return PlanStepDraft(**values)  # type: ignore[arg-type]


def test_plan_draft_is_runtime_id_free_and_normalizes_domains() -> None:
    draft = PlanDraft(
        objective="Read the approved report",
        completion_criteria=("The requested value is available",),
        steps=(plan_step(),),
    )

    assert draft.steps[0].required_domains == ("example.com",)
    schema = json.dumps(PlanDraft.model_json_schema(), sort_keys=True).lower()
    for forbidden in ("planid", "sessionid", "taskid", "createdat", "timestamp"):
        assert forbidden not in schema


def test_plan_draft_requires_a_contiguous_ordered_dag() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        PlanDraft(
            objective="Invalid order",
            completion_criteria=("Done",),
            steps=(plan_step(order=2),),
        )

    with pytest.raises(ValidationError, match="unknown dependency"):
        PlanDraft(
            objective="Unknown dependency",
            completion_criteria=("Done",),
            steps=(plan_step(depends_on=("missing",)),),
        )

    with pytest.raises(ValidationError, match="precede"):
        PlanDraft(
            objective="Future dependency",
            completion_criteria=("Done",),
            steps=(
                plan_step(depends_on=("finish",)),
                plan_step(key="finish", order=2),
            ),
        )


@pytest.mark.parametrize(
    "domain",
    [
        " https://example.com",
        "https://example.com",
        "*.example.com",
        "user@example.com",
        "example.com/path",
        "bad_domain.example",
    ],
)
def test_model_domains_reject_urls_wildcards_credentials_and_invalid_hosts(domain: str) -> None:
    with pytest.raises(ValueError, match="domain|host"):
        canonicalize_model_domain(domain)


def test_model_domains_are_canonical_idna_and_unique() -> None:
    assert canonicalize_model_domain("BÜCHER.example.") == "xn--bcher-kva.example"
    with pytest.raises(ValidationError, match="unique"):
        plan_step(required_domains=("EXAMPLE.com", "example.com."))


def test_plan_step_rejects_duplicate_or_unsupported_actions() -> None:
    with pytest.raises(ValidationError, match="unique"):
        plan_step(allowed_action_types=("wait", "wait"))

    payload = plan_step().model_dump(mode="json", by_alias=True)
    payload["allowedActionTypes"] = ["browser_run_code"]
    with pytest.raises(ValidationError):
        PlanStepDraft.model_validate_json(json.dumps(payload))


def test_navigator_action_schema_has_no_selector_or_hidden_reasoning_fields() -> None:
    schema = NavigatorDecision.model_json_schema()
    property_names: set[str] = set()

    def collect_properties(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                property_names.update(str(name).lower() for name in properties)
            for nested in value.values():
                collect_properties(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_properties(nested)

    collect_properties(schema)
    for forbidden in ("xpath", "selector", "javascript", "intent", "shell"):
        assert forbidden not in property_names

    decision = NavigatorDecision.model_validate_json(
        json.dumps(
            {
                "status": "action_required",
                "action": {"type": "click_element", "index": 7},
            }
        )
    )
    action = decision.production_action()
    assert action.type == "click_element"
    assert action.index == 7


def test_navigator_decision_contains_exactly_one_action_or_terminal_state() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        NavigatorDecision(status=NavigatorDecisionStatus.ACTION_REQUIRED)

    with pytest.raises(ValidationError, match="terminal"):
        NavigatorDecision.model_validate_json(
            json.dumps(
                {
                    "status": "step_completed",
                    "action": {"type": "wait", "seconds": 1},
                }
            )
        )

    blocked = NavigatorDecision(
        status=NavigatorDecisionStatus.BLOCKED,
        error_code="POLICY_BLOCKED",
        replan_requested=True,
    )
    assert blocked.action is None
    with pytest.raises(ValueError, match="does not contain"):
        blocked.production_action()


def test_navigator_decision_rejects_unsafe_keys_and_invalid_failure_states() -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        NavigatorDecision.model_validate_json(
            json.dumps(
                {
                    "status": "action_required",
                    "action": {"type": "send_keys", "keys": "Control+L"},
                }
            )
        )

    with pytest.raises(ValidationError, match="require error_code"):
        NavigatorDecision(status=NavigatorDecisionStatus.FAILED)
    with pytest.raises(ValidationError, match="only when blocked or failed"):
        NavigatorDecision(
            status=NavigatorDecisionStatus.STEP_COMPLETED,
            replan_requested=True,
        )
