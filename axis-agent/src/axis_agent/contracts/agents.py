"""Strict contracts for the AXIS Planner/Navigator workflow.

These models contain bounded execution data only.  Browser observations remain
behind the MCP tool boundary; agent inputs carry opaque observation references
instead of serialized DOM or unrestricted page content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from axis_agent.contracts.actions import (
    PRODUCTION_ACTION_TYPES,
    ProductionAction,
    ProductionActionType,
    parse_production_action,
)
from axis_agent.contracts.base import ContractModel
from axis_agent.contracts.tasks import TaskRequest

_STEP_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_OBSERVATION_ID_PATTERN = r"^[0-9a-f]{32}$"


class PlanStep(ContractModel):
    """One bounded, topologically ordered unit of browser work."""

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
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=20)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("success_criteria", "depends_on", "evidence_refs")
    @classmethod
    def require_unique_bounded_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("values must be unique")
        if any(not item or item != item.strip() or len(item) > 1_000 for item in value):
            raise ValueError("values must be non-empty, trimmed, and at most 1000 characters")
        return value

    @field_validator("allowed_action_types")
    @classmethod
    def require_unique_actions(
        cls, value: tuple[ProductionActionType, ...]
    ) -> tuple[ProductionActionType, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_action_types must be unique")
        return value

    @field_validator("required_domains")
    @classmethod
    def normalize_required_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            domain = item.lower().rstrip(".")
            if (
                not domain
                or item != item.strip()
                or len(domain) > 253
                or any(token in domain for token in ("://", "/", "\\", "@", "?", "#", "*"))
            ):
                raise ValueError("required_domains must contain plain host names")
            normalized.append(domain)
        if len(set(normalized)) != len(normalized):
            raise ValueError("required_domains must be unique")
        return tuple(normalized)


class ExecutionPlan(ContractModel):
    """Validated Planner output with one deterministic topological order."""

    plan_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    task_id: UUID
    revision: int = Field(default=1, ge=1, le=3)
    objective: str = Field(min_length=1, max_length=2_000)
    assumptions: tuple[str, ...] = Field(default=(), max_length=10)
    completion_criteria: tuple[str, ...] = Field(min_length=1, max_length=10)
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator("assumptions", "completion_criteria")
    @classmethod
    def require_unique_plan_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("values must be unique")
        if any(not item or item != item.strip() or len(item) > 1_000 for item in value):
            raise ValueError("values must be non-empty, trimmed, and at most 1000 characters")
        return value

    @model_validator(mode="after")
    def validate_ordered_dag(self) -> Self:
        keys = [step.key for step in self.steps]
        if len(set(keys)) != len(keys):
            raise ValueError("plan step keys must be unique")

        orders = [step.order for step in self.steps]
        if orders != list(range(1, len(self.steps) + 1)):
            raise ValueError("plan steps must be supplied in contiguous order starting at one")

        order_by_key = {step.key: step.order for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                dependency_order = order_by_key.get(dependency)
                if dependency_order is None:
                    raise ValueError(f"unknown dependency: {dependency}")
                if dependency_order >= step.order:
                    raise ValueError("every dependency must precede its dependent step")
        return self

    def step(self, key: str) -> PlanStep:
        for candidate in self.steps:
            if candidate.key == key:
                return candidate
        raise KeyError(key)


class PlannerInput(ContractModel):
    """Runtime-owned input for an initial plan or bounded replan."""

    session_id: UUID
    task: TaskRequest
    observation_id: str | None = Field(default=None, pattern=_OBSERVATION_ID_PATTERN)
    page_id: str | None = Field(default=None, min_length=1, max_length=128)
    origin: str | None = Field(default=None, min_length=1, max_length=2_048)
    prior_plan: ExecutionPlan | None = None
    replan_reason_code: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{0,63}$",
    )

    @model_validator(mode="after")
    def validate_replan_and_observation(self) -> Self:
        references = (self.observation_id, self.page_id, self.origin)
        if any(item is not None for item in references) and not all(
            item is not None for item in references
        ):
            raise ValueError("observation_id, page_id, and origin must be supplied together")
        if self.replan_reason_code is not None and self.prior_plan is None:
            raise ValueError("replan_reason_code requires prior_plan")
        if self.prior_plan is not None:
            if self.prior_plan.session_id != self.session_id:
                raise ValueError("prior_plan session_id does not match Planner input")
            if self.prior_plan.task_id != self.task.task_id:
                raise ValueError("prior_plan task_id does not match Planner input")
        return self


class ActionExecutionSummary(ContractModel):
    """Bounded action metadata that may be returned to the Navigator."""

    action_id: UUID
    action_type: ProductionActionType
    success: bool
    error_code: str | None = Field(default=None, max_length=128)
    retryable: bool = False


class NavigatorInput(ContractModel):
    """Runtime-owned input for one deterministic plan-step navigation run."""

    session_id: UUID
    task_id: UUID
    plan: ExecutionPlan
    active_step_key: str = Field(pattern=_STEP_KEY_PATTERN)
    observation_id: str = Field(pattern=_OBSERVATION_ID_PATTERN)
    page_id: str = Field(min_length=1, max_length=128)
    origin: str = Field(min_length=1, max_length=2_048)
    completed_step_keys: tuple[str, ...] = Field(default=(), max_length=100)
    recent_results: tuple[ActionExecutionSummary, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def validate_active_step(self) -> Self:
        if self.plan.session_id != self.session_id or self.plan.task_id != self.task_id:
            raise ValueError("Navigator IDs do not match the execution plan")
        if len(set(self.completed_step_keys)) != len(self.completed_step_keys):
            raise ValueError("completed_step_keys must be unique")

        known_keys = {step.key for step in self.plan.steps}
        completed = set(self.completed_step_keys)
        if not completed <= known_keys:
            raise ValueError("completed_step_keys contains an unknown plan step")
        if self.active_step_key in completed:
            raise ValueError("active step is already completed")

        eligible = next(
            (
                step
                for step in self.plan.steps
                if step.key not in completed and set(step.depends_on) <= completed
            ),
            None,
        )
        if eligible is None or eligible.key != self.active_step_key:
            raise ValueError("active_step_key is not the next eligible ordered step")
        return self


class NavigatorOutcomeStatus(StrEnum):
    ACTIONS_REQUIRED = "actions_required"
    STEP_COMPLETED = "step_completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    TASK_COMPLETED = "task_completed"


class NavigatorOutcome(ContractModel):
    """Structured Navigator output for one plan step."""

    session_id: UUID
    task_id: UUID
    plan_id: UUID
    step_key: str = Field(pattern=_STEP_KEY_PATTERN)
    status: NavigatorOutcomeStatus
    actions: tuple[ProductionAction, ...] = Field(default=(), max_length=10)
    summary: str = Field(default="", max_length=2_000)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=20)
    replan_requested: bool = False
    error_code: str | None = Field(default=None, max_length=128)

    @field_validator("actions", mode="before")
    @classmethod
    def canonicalize_actions(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(parse_production_action(item) for item in value)
        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("evidence_ids must be unique")
        if any(not item or item != item.strip() or len(item) > 128 for item in value):
            raise ValueError("evidence IDs must be non-empty, trimmed, and bounded")
        return value

    @model_validator(mode="after")
    def validate_outcome_state(self) -> Self:
        if self.status is NavigatorOutcomeStatus.ACTIONS_REQUIRED:
            if not self.actions:
                raise ValueError("actions_required must contain at least one action")
        elif self.actions:
            raise ValueError("terminal Navigator outcomes cannot contain actions")

        if self.replan_requested and self.status not in {
            NavigatorOutcomeStatus.BLOCKED,
            NavigatorOutcomeStatus.FAILED,
        }:
            raise ValueError("replanning can be requested only for blocked or failed outcomes")
        if self.status in {NavigatorOutcomeStatus.BLOCKED, NavigatorOutcomeStatus.FAILED}:
            if self.error_code is None:
                raise ValueError("blocked and failed outcomes require error_code")
        elif self.error_code is not None:
            raise ValueError("error_code is allowed only for blocked or failed outcomes")
        return self


class WorkflowStatus(StrEnum):
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowResult(ContractModel):
    """Terminal, persistence-safe result of the two-agent workflow."""

    session_id: UUID
    task_id: UUID
    plan_id: UUID | None = None
    status: WorkflowStatus
    completed_step_keys: tuple[str, ...] = Field(default=(), max_length=100)
    total_steps: int = Field(default=0, ge=0, le=100)
    summary: str = Field(default="", max_length=2_000)
    error_code: str | None = Field(default=None, max_length=128)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("completed_at")
    @classmethod
    def require_aware_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_terminal_result(self) -> Self:
        if len(set(self.completed_step_keys)) != len(self.completed_step_keys):
            raise ValueError("completed_step_keys must be unique")
        if len(self.completed_step_keys) > self.total_steps:
            raise ValueError("completed steps cannot exceed total_steps")
        if self.status is WorkflowStatus.SUCCEEDED:
            if len(self.completed_step_keys) != self.total_steps:
                raise ValueError("successful workflows must complete every plan step")
            if self.error_code is not None:
                raise ValueError("successful workflows cannot have error_code")
        elif self.error_code is None:
            raise ValueError("non-successful workflows require error_code")
        return self
