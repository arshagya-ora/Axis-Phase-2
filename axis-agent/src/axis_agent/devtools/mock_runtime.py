"""Deterministic two-agent runtime used only by explicit offline commands/tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from axis_agent.contracts import (
    ExecutionPlan,
    NavigatorInput,
    NavigatorOutcome,
    NavigatorOutcomeStatus,
    PlannerInput,
    PlanStep,
)
from axis_agent.contracts.actions import WaitAction

_CREATED_AT = datetime(1970, 1, 1, tzinfo=UTC)


class MockPlannerNavigatorRuntime:
    """Return valid deterministic contracts with zero provider calls."""

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        del run_config
        revision = 1 if planner_input.prior_plan is None else planner_input.prior_plan.revision + 1
        plan_id = uuid5(
            NAMESPACE_URL,
            f"axis:mock-plan:{planner_input.session_id}:{planner_input.task.task_id}:{revision}",
        )
        return ExecutionPlan(
            plan_id=plan_id,
            session_id=planner_input.session_id,
            task_id=planner_input.task.task_id,
            revision=min(revision, 3),
            objective="Complete the explicit offline AXIS verification task",
            assumptions=("Offline mock mode performs no external calls",),
            completion_criteria=("The deterministic wait action succeeds",),
            steps=(
                PlanStep(
                    key="mock_wait",
                    order=1,
                    objective="Perform one safe deterministic wait action",
                    success_criteria=("The wait action succeeds",),
                    allowed_action_types=("wait",),
                    max_attempts=1,
                ),
            ),
            created_at=_CREATED_AT,
        )

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        if not navigator_input.recent_results:
            return NavigatorOutcome(
                session_id=navigator_input.session_id,
                task_id=navigator_input.task_id,
                plan_id=navigator_input.plan.plan_id,
                step_key=navigator_input.active_step_key,
                status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
                actions=(WaitAction(seconds=0),),
                summary="Explicit offline wait action is required",
            )
        latest = navigator_input.recent_results[-1]
        if not latest.success:
            return NavigatorOutcome(
                session_id=navigator_input.session_id,
                task_id=navigator_input.task_id,
                plan_id=navigator_input.plan.plan_id,
                step_key=navigator_input.active_step_key,
                status=NavigatorOutcomeStatus.FAILED,
                summary="Explicit offline action failed",
                error_code=latest.error_code or "MOCK_ACTION_FAILED",
            )
        completed_after_active = len(navigator_input.completed_step_keys) + 1
        status = (
            NavigatorOutcomeStatus.TASK_COMPLETED
            if completed_after_active == len(navigator_input.plan.steps)
            else NavigatorOutcomeStatus.STEP_COMPLETED
        )
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=status,
            summary="Explicit offline wait action completed",
            evidence_ids=(str(latest.action_id),),
        )


__all__ = ["MockPlannerNavigatorRuntime"]
