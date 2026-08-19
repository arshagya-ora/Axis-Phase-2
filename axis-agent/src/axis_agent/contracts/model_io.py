"""Minimal structured-output contracts exposed to the Planner and Navigator models.

Runtime identifiers, timestamps, selectors, and tool connection details deliberately do
not appear in these schemas.  The trusted runtime converts validated drafts into the
existing persistence and execution contracts.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from axis_agent.contracts.actions import (
    PRODUCTION_ACTION_TYPES,
    ProductionAction,
    ProductionActionType,
    parse_production_action,
)
from axis_agent.contracts.base import ContractModel

_STEP_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def canonicalize_model_domain(value: str) -> str:
    """Return one canonical plain host name suitable for policy intersection."""

    if not value or value != value.strip() or len(value) > 253:
        raise ValueError("required domains must be non-empty, trimmed, and bounded")
    if any(token in value for token in ("://", "/", "\\", "@", "?", "#", "*")):
        raise ValueError("required domains must be plain host names")

    host = value.lower().rstrip(".")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("required domains must be valid IDNA host names") from exc
    if not ascii_host or len(ascii_host) > 253:
        raise ValueError("required domains must be valid IDNA host names")
    if any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in ascii_host.split(".")):
        raise ValueError("required domains must be valid IDNA host names")
    return ascii_host


def _validate_unique_text(
    value: tuple[str, ...],
    *,
    maximum_length: int = 1_000,
) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError("values must be unique")
    if any(not item or item != item.strip() or len(item) > maximum_length for item in value):
        raise ValueError("values must be non-empty, trimmed, and bounded")
    return value


class PlanStepDraft(ContractModel):
    """One model-authored plan step without runtime-owned metadata."""

    key: str = Field(pattern=_STEP_KEY_PATTERN)
    order: int = Field(ge=1, le=100)
    objective: str = Field(min_length=1, max_length=1_000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=10)
    depends_on: tuple[str, ...] = Field(default=(), max_length=99)
    allowed_action_types: tuple[ProductionActionType, ...] = Field(
        min_length=1,
        max_length=len(PRODUCTION_ACTION_TYPES),
    )
    required_domains: tuple[str, ...] = Field(default=(), max_length=20)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("success_criteria", "depends_on")
    @classmethod
    def require_unique_bounded_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_text(value)

    @field_validator("allowed_action_types")
    @classmethod
    def require_unique_supported_actions(
        cls,
        value: tuple[ProductionActionType, ...],
    ) -> tuple[ProductionActionType, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_action_types must be unique")
        if not set(value) <= PRODUCTION_ACTION_TYPES:
            raise ValueError("allowed_action_types contains an unsupported action")
        return value

    @field_validator("required_domains")
    @classmethod
    def normalize_required_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(canonicalize_model_domain(item) for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("required_domains must be unique")
        return normalized


class PlanDraft(ContractModel):
    """Complete model-authored plan content before runtime IDs are attached."""

    objective: str = Field(min_length=1, max_length=2_000)
    assumptions: tuple[str, ...] = Field(default=(), max_length=10)
    completion_criteria: tuple[str, ...] = Field(min_length=1, max_length=10)
    steps: tuple[PlanStepDraft, ...] = Field(min_length=1, max_length=100)

    @field_validator("assumptions", "completion_criteria")
    @classmethod
    def require_unique_bounded_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_text(value)

    @model_validator(mode="after")
    def validate_ordered_dag(self) -> Self:
        keys = [step.key for step in self.steps]
        if len(set(keys)) != len(keys):
            raise ValueError("plan step keys must be unique")

        orders = [step.order for step in self.steps]
        if orders != list(range(1, len(self.steps) + 1)):
            raise ValueError("plan steps must be in contiguous order starting at one")

        order_by_key = {step.key: step.order for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                dependency_order = order_by_key.get(dependency)
                if dependency_order is None:
                    raise ValueError(f"unknown dependency: {dependency}")
                if dependency_order >= step.order:
                    raise ValueError("every dependency must precede its dependent step")
        return self


class _GoToUrlDraft(ContractModel):
    type: Literal["go_to_url"] = "go_to_url"
    url: str = Field(min_length=1, max_length=2_048)


class _GoBackDraft(ContractModel):
    type: Literal["go_back"] = "go_back"


class _ClickElementDraft(ContractModel):
    type: Literal["click_element"] = "click_element"
    index: int = Field(ge=0, le=100_000)


class _InputTextDraft(ContractModel):
    type: Literal["input_text"] = "input_text"
    index: int = Field(ge=0, le=100_000)
    text: str = Field(max_length=20_000)


class _SwitchTabDraft(ContractModel):
    type: Literal["switch_tab"] = "switch_tab"
    tab_id: str = Field(min_length=1, max_length=128)


class _OpenTabDraft(ContractModel):
    type: Literal["open_tab"] = "open_tab"
    url: str = Field(min_length=1, max_length=2_048)


class _CloseTabDraft(ContractModel):
    type: Literal["close_tab"] = "close_tab"
    tab_id: str = Field(min_length=1, max_length=128)


class _ScrollToPercentDraft(ContractModel):
    type: Literal["scroll_to_percent"] = "scroll_to_percent"
    y_percent: int = Field(alias="yPercent", ge=0, le=100)
    index: int | None = Field(default=None, ge=0, le=100_000)


class _ScrollToTopDraft(ContractModel):
    type: Literal["scroll_to_top"] = "scroll_to_top"
    index: int | None = Field(default=None, ge=0, le=100_000)


class _ScrollToBottomDraft(ContractModel):
    type: Literal["scroll_to_bottom"] = "scroll_to_bottom"
    index: int | None = Field(default=None, ge=0, le=100_000)


class _ScrollToTextDraft(ContractModel):
    type: Literal["scroll_to_text"] = "scroll_to_text"
    text: str = Field(min_length=1, max_length=1_000)
    nth: int = Field(default=1, ge=1, le=10_000)


class _SendKeysDraft(ContractModel):
    type: Literal["send_keys"] = "send_keys"
    keys: str = Field(min_length=1, max_length=128)

    @field_validator("keys")
    @classmethod
    def validate_safe_key_chord(cls, value: str) -> str:
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("keys must be one bounded Playwright key chord")
        parts = value.split("+")
        if any(not part for part in parts):
            raise ValueError("keys contains an empty chord component")
        safe_single_keys = {
            "Backspace",
            "Delete",
            "End",
            "Enter",
            "Escape",
            "Home",
            "Insert",
            "PageDown",
            "PageUp",
            "Space",
            "Tab",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "ArrowUp",
        }
        if len(parts) == 1 and parts[0] in safe_single_keys:
            return value
        if parts == ["Control", "A"] or (
            len(parts) == 2 and parts[0] == "Shift" and parts[1] in safe_single_keys
        ):
            return value
        raise ValueError("keys chord is not allowed by AXIS browser policy")


class _GetDropdownOptionsDraft(ContractModel):
    type: Literal["get_dropdown_options"] = "get_dropdown_options"
    index: int = Field(ge=0, le=100_000)


class _SelectDropdownOptionDraft(ContractModel):
    type: Literal["select_dropdown_option"] = "select_dropdown_option"
    index: int = Field(ge=0, le=100_000)
    text: str = Field(max_length=2_000)


class _WaitDraft(ContractModel):
    type: Literal["wait"] = "wait"
    seconds: int = Field(default=3, ge=0, le=120)


NavigatorAction = Annotated[
    _GoToUrlDraft
    | _GoBackDraft
    | _ClickElementDraft
    | _InputTextDraft
    | _SwitchTabDraft
    | _OpenTabDraft
    | _CloseTabDraft
    | _ScrollToPercentDraft
    | _ScrollToTopDraft
    | _ScrollToBottomDraft
    | _ScrollToTextDraft
    | _SendKeysDraft
    | _GetDropdownOptionsDraft
    | _SelectDropdownOptionDraft
    | _WaitDraft,
    Field(discriminator="type"),
]


class NavigatorDecisionStatus(StrEnum):
    ACTION_REQUIRED = "action_required"
    STEP_COMPLETED = "step_completed"
    TASK_COMPLETED = "task_completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class NavigatorDecision(ContractModel):
    """One selector-free action proposal or one terminal execution decision."""

    status: NavigatorDecisionStatus
    action: NavigatorAction | None = None
    summary: str = Field(default="", max_length=2_000)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=20)
    replan_requested: bool = False
    error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_text(value, maximum_length=128)

    @model_validator(mode="after")
    def validate_decision_state(self) -> Self:
        if self.status is NavigatorDecisionStatus.ACTION_REQUIRED:
            if self.action is None:
                raise ValueError("action_required must contain exactly one action")
        elif self.action is not None:
            raise ValueError("terminal Navigator decisions cannot contain an action")

        failure_statuses = {
            NavigatorDecisionStatus.BLOCKED,
            NavigatorDecisionStatus.FAILED,
        }
        if self.status in failure_statuses:
            if self.error_code is None:
                raise ValueError("blocked and failed decisions require error_code")
        elif self.error_code is not None:
            raise ValueError("error_code is allowed only for blocked or failed decisions")
        if self.replan_requested and self.status not in failure_statuses:
            raise ValueError("replanning can be requested only when blocked or failed")
        return self

    def production_action(self) -> ProductionAction:
        """Convert the single model action to the existing trusted runtime union."""

        if self.action is None:
            raise ValueError("decision does not contain an action")
        return parse_production_action(
            self.action.model_dump(mode="python", by_alias=True, exclude_none=True)
        )
