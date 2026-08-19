"""Deterministic orchestration for the fixed AXIS Planner and Navigator."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from axis_agent.contracts import (
    ActionExecutionSummary,
    ExecutionPlan,
    NavigatorInput,
    NavigatorOutcome,
    NavigatorOutcomeStatus,
    PlannerInput,
    PlanStep,
    TaskRequest,
    WorkflowResult,
    WorkflowStatus,
)
from axis_agent.contracts.actions import ProductionAction
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime.tools import WorkflowBrowserTools


class PlannerNavigatorRuntime(Protocol):
    """The only model-facing interface required by deterministic orchestration."""

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan: ...

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome: ...


@dataclass(frozen=True, slots=True)
class WorkflowLimits:
    """Hard bounds owned by the runtime rather than either model."""

    max_steps: int = 100
    max_actions_per_step: int = 10
    max_failures: int = 3
    max_plan_revisions: int = 3
    model_timeout_seconds: int = 120
    workflow_timeout_seconds: int = 1_800

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 100:
            raise ValueError("max_steps must be between 1 and 100")
        if not 1 <= self.max_actions_per_step <= 10:
            raise ValueError("max_actions_per_step must be between 1 and 10")
        if not 1 <= self.max_failures <= 10:
            raise ValueError("max_failures must be between 1 and 10")
        if not 1 <= self.max_plan_revisions <= 3:
            raise ValueError("max_plan_revisions must be between 1 and 3")
        if not 1 <= self.model_timeout_seconds <= 600:
            raise ValueError("model_timeout_seconds must be between 1 and 600")
        if not 1 <= self.workflow_timeout_seconds <= 86_400:
            raise ValueError("workflow_timeout_seconds must be between 1 and 86400")


@dataclass(slots=True)
class _WorkflowProgress:
    session_id: UUID
    task_id: UUID
    plan_id: UUID | None = None
    total_steps: int = 0
    completed_step_keys: tuple[str, ...] = ()


class TwoAgentWorkflow:
    """Sequence Planner then Navigator while retaining deterministic control.

    Models may propose plans and actions, but this runtime owns identifiers, limits,
    persistence, status transitions, action execution, approvals, and replanning.
    """

    def __init__(
        self,
        *,
        runner: PlannerNavigatorRuntime,
        database: AxisDatabase,
        browser_tools: WorkflowBrowserTools,
        config_hash: str,
        provider_name: str,
        planner_model: str,
        navigator_model: str,
        limits: WorkflowLimits | None = None,
        run_config: object | None = None,
    ) -> None:
        # Retained as a source-compatibility keyword for Phase-3 embedders.  The
        # direct SDK runtime deliberately ignores Agents-SDK run configuration.
        del run_config
        for name, value, maximum in (
            ("config_hash", config_hash, 128),
            ("provider_name", provider_name, 128),
            ("planner_model", planner_model, 256),
            ("navigator_model", navigator_model, 256),
        ):
            if not value or value != value.strip() or len(value) > maximum:
                raise ValueError(f"{name} must be non-empty, trimmed, and bounded")
        self._runner = runner
        self._database = database
        self._browser_tools = browser_tools
        self._config_hash = config_hash
        self._provider_name = provider_name
        self._planner_model = planner_model
        self._navigator_model = navigator_model
        self._limits = limits or WorkflowLimits()
        # A workflow owns one mutable browser execution context.  Serializing runs
        # prevents a second task from rebinding that context while the first task
        # still holds observation/action leases.
        self._run_lock = asyncio.Lock()

    async def run(self, task: TaskRequest) -> WorkflowResult:
        """Run exclusively with a hard deadline and cancellation-safe cleanup."""

        async with self._run_lock:
            return await self._run_with_deadline(task)

    async def _run_with_deadline(self, task: TaskRequest) -> WorkflowResult:
        """Apply the execution deadline after this instance becomes available."""

        progress = _WorkflowProgress(session_id=uuid4(), task_id=task.task_id)
        try:
            return await asyncio.wait_for(
                self._run_core(task, progress),
                timeout=self._limits.workflow_timeout_seconds,
            )
        except TimeoutError:
            return WorkflowResult(
                session_id=progress.session_id,
                task_id=progress.task_id,
                plan_id=progress.plan_id,
                status=WorkflowStatus.CANCELLED,
                completed_step_keys=progress.completed_step_keys,
                total_steps=progress.total_steps,
                summary="Workflow exceeded its runtime deadline",
                error_code="WORKFLOW_TIMEOUT",
            )

    async def _run_core(
        self,
        task: TaskRequest,
        progress: _WorkflowProgress,
    ) -> WorkflowResult:
        session_id = UUID(
            await self._database.create_session(
                session_id=progress.session_id,
                task_id=task.task_id,
                task_summary=task.prompt,
                client_request_id=task.client_request_id,
                config_hash=self._config_hash,
            )
        )
        await self._database.transition_session_status(session_id, "running")
        planning_step_id = UUID(
            await self._database.create_step(session_id=session_id, step_number=0)
        )
        await self._database.transition_step_status(planning_step_id, "running")

        plan: ExecutionPlan | None = None
        completed: list[str] = []
        active_runtime_step: UUID | None = planning_step_id
        try:
            planner_input = PlannerInput(session_id=session_id, task=task)
            plan = await self._run_planner_with_repair(
                planner_input,
                session_id=session_id,
                step_id=planning_step_id,
                previous=None,
                completed=completed,
            )
            progress.plan_id = plan.plan_id
            progress.total_steps = len(plan.steps)
            await self._database.record_plan_metadata(plan)
            await self._database.transition_step_status(planning_step_id, "succeeded")
            active_runtime_step = None

            execution_number = 1
            while len(completed) < len(plan.steps):
                step = self._next_eligible_step(plan, completed)
                if step is None:
                    return await self._finish_failed(
                        session_id=session_id,
                        task_id=task.task_id,
                        plan=plan,
                        completed=completed,
                        status=WorkflowStatus.FAILED,
                        error_code="PLAN_HAS_NO_ELIGIBLE_STEP",
                    )

                runtime_step_id = UUID(
                    await self._database.create_step(
                        session_id=session_id,
                        step_number=execution_number,
                    )
                )
                execution_number += 1
                active_runtime_step = runtime_step_id
                await self._database.transition_step_status(runtime_step_id, "running")
                await self._browser_tools.bind_step(
                    session_id=session_id,
                    task_id=task.task_id,
                    step_id=runtime_step_id,
                    plan_id=plan.plan_id,
                    step_key=step.key,
                    allowed_action_types=step.allowed_action_types,
                )

                disposition, replan_reason = await self._run_plan_step(
                    task_id=task.task_id,
                    session_id=session_id,
                    runtime_step_id=runtime_step_id,
                    plan=plan,
                    step_key=step.key,
                    completed=completed,
                )
                if disposition == "completed":
                    await self._database.transition_step_status(runtime_step_id, "succeeded")
                    completed.append(step.key)
                    progress.completed_step_keys = tuple(completed)
                    active_runtime_step = None
                    continue

                await self._database.transition_step_status(runtime_step_id, "failed")
                active_runtime_step = None
                if disposition == "replan" and plan.revision < self._limits.max_plan_revisions:
                    observation = await self._browser_tools.observe()
                    replan_input = PlannerInput(
                        session_id=session_id,
                        task=task,
                        observation_id=observation.observation_id,
                        page_id=observation.page_id,
                        origin=self._observation_origin(observation.url),
                        prior_plan=plan,
                        replan_reason_code=replan_reason or "NAVIGATOR_REPLAN_REQUESTED",
                    )
                    replanned = await self._run_planner_with_repair(
                        replan_input,
                        session_id=session_id,
                        step_id=runtime_step_id,
                        previous=plan,
                        completed=completed,
                    )
                    await self._database.record_plan_metadata(replanned)
                    plan = replanned
                    progress.plan_id = plan.plan_id
                    progress.total_steps = len(plan.steps)
                    continue

                status = (
                    WorkflowStatus.BLOCKED if disposition == "blocked" else WorkflowStatus.FAILED
                )
                return await self._finish_failed(
                    session_id=session_id,
                    task_id=task.task_id,
                    plan=plan,
                    completed=completed,
                    status=status,
                    error_code=replan_reason or "NAVIGATOR_STEP_FAILED",
                )

            await self._database.transition_session_status(session_id, "succeeded")
            return WorkflowResult(
                session_id=session_id,
                task_id=task.task_id,
                plan_id=plan.plan_id,
                status=WorkflowStatus.SUCCEEDED,
                completed_step_keys=tuple(completed),
                total_steps=len(plan.steps),
                summary="All validated plan steps completed",
            )
        except asyncio.CancelledError:
            if active_runtime_step is not None:
                await self._safe_step_transition(active_runtime_step, "cancelled")
            await self._safe_session_transition(session_id, "cancelled")
            raise
        except Exception as exc:
            if active_runtime_step is not None:
                await self._safe_step_transition(active_runtime_step, "failed")
            await self._database.record_audit_event(
                event_type="workflow.failed",
                severity="error",
                session_id=session_id,
                payload={"reasonCode": type(exc).__name__.upper()[:64]},
            )
            await self._safe_session_transition(session_id, "failed")
            return WorkflowResult(
                session_id=session_id,
                task_id=task.task_id,
                plan_id=plan.plan_id if plan else None,
                status=WorkflowStatus.FAILED,
                completed_step_keys=tuple(completed),
                total_steps=len(plan.steps) if plan else 0,
                summary="Workflow stopped at a trusted runtime boundary",
                error_code="WORKFLOW_RUNTIME_FAILED",
            )

    async def _run_planner_with_repair(
        self,
        planner_input: PlannerInput,
        *,
        session_id: UUID,
        step_id: UUID,
        previous: ExecutionPlan | None,
        completed: list[str],
    ) -> ExecutionPlan:
        # The direct Planner owns exactly one schema-correction retry so it can
        # tell the provider that the second response must repair its first typed
        # output.  The workflow validates the resulting trusted contract once and
        # never adds an unbounded outer retry layer.
        plan = await self._run_planner(
            planner_input,
            session_id=session_id,
            step_id=step_id,
        )
        if previous is None:
            self._validate_plan(
                plan,
                session_id=session_id,
                task_id=planner_input.task.task_id,
            )
        else:
            self._validate_replan(plan, previous=previous, completed=completed)
        return plan

    async def _run_plan_step(
        self,
        *,
        task_id: UUID,
        session_id: UUID,
        runtime_step_id: UUID,
        plan: ExecutionPlan,
        step_key: str,
        completed: list[str],
    ) -> tuple[str, str | None]:
        recent: list[ActionExecutionSummary] = []
        action_count = 0
        attempt_count = 0
        failure_count = 0
        active_step = plan.step(step_key)
        # Each action attempt consumes one freshly observed browser lease.  One
        # additional Navigator turn lets it provide terminal evidence after its
        # final permitted action.
        max_turns = active_step.max_attempts + 1

        for _ in range(max_turns):
            observation = await self._browser_tools.observe()
            outcome = await self._run_navigator(
                NavigatorInput(
                    session_id=session_id,
                    task_id=task_id,
                    plan=plan,
                    active_step_key=step_key,
                    observation_id=observation.observation_id,
                    page_id=observation.page_id,
                    origin=self._observation_origin(observation.url),
                    completed_step_keys=tuple(completed),
                    recent_results=tuple(recent[-20:]),
                ),
                session_id=session_id,
                step_id=runtime_step_id,
            )
            self._validate_outcome(
                outcome,
                plan=plan,
                step_key=step_key,
                observation_id=observation.observation_id,
                recent=recent,
            )

            if outcome.status is NavigatorOutcomeStatus.ACTIONS_REQUIRED:
                # A browser observation is a one-action lease.  Rejecting the
                # complete batch before dispatch prevents later actions from
                # operating on DOM state invalidated by an earlier action.
                if len(outcome.actions) != 1:
                    return "failed", "MULTI_ACTION_BATCH_REJECTED"
                if action_count >= self._limits.max_actions_per_step:
                    return "failed", "STEP_ACTION_LIMIT_EXCEEDED"
                if attempt_count >= active_step.max_attempts:
                    return "failed", "STEP_ATTEMPT_LIMIT_EXCEEDED"

                action = outcome.actions[0]
                if action.type not in active_step.allowed_action_types:
                    return "failed", "ACTION_OUTSIDE_PLAN"
                if not self._action_within_required_domains(action, active_step):
                    return "failed", "ACTION_OUTSIDE_REQUIRED_DOMAINS"

                result = await self._browser_tools.execute(action)
                action_count += 1
                attempt_count += 1
                recent.append(
                    ActionExecutionSummary(
                        action_id=result.action_id,
                        action_type=action.type,
                        success=result.success,
                        error_code=result.error_code,
                        retryable=result.retryable,
                    )
                )
                if not result.success:
                    # Do not execute another action from this observation after
                    # the first failure.  A retry, when permitted, starts with a
                    # fresh observation and a new Navigator turn.
                    failure_count += 1
                    if result.error_code == "APPROVAL_REJECTED":
                        return "blocked", "APPROVAL_REJECTED"
                    if failure_count >= self._limits.max_failures:
                        return "failed", "STEP_FAILURE_LIMIT_EXCEEDED"
                    if attempt_count >= active_step.max_attempts:
                        return "failed", "STEP_ATTEMPT_LIMIT_EXCEEDED"
                continue

            if outcome.status is NavigatorOutcomeStatus.STEP_COMPLETED:
                return "completed", None
            if outcome.status is NavigatorOutcomeStatus.TASK_COMPLETED:
                remaining = (
                    {candidate.key for candidate in plan.steps} - set(completed) - {step_key}
                )
                if remaining:
                    return "failed", "EARLY_TASK_COMPLETION"
                return "completed", None
            if outcome.replan_requested:
                return "replan", outcome.error_code or "NAVIGATOR_REPLAN_REQUESTED"
            if outcome.status is NavigatorOutcomeStatus.BLOCKED:
                return "blocked", outcome.error_code
            return "failed", outcome.error_code or "NAVIGATOR_STEP_FAILED"
        return "failed", "NAVIGATOR_TURN_LIMIT_EXCEEDED"

    async def _run_planner(
        self,
        planner_input: PlannerInput,
        *,
        session_id: UUID,
        step_id: UUID,
    ) -> ExecutionPlan:
        started = perf_counter()
        status: Literal["succeeded", "failed", "cancelled", "timed_out"] = "succeeded"
        error_code: str | None = None
        try:
            return await asyncio.wait_for(
                self._runner.run_planner(planner_input, run_config=None),
                timeout=self._limits.model_timeout_seconds,
            )
        except TimeoutError:
            status = "timed_out"
            error_code = "MODEL_TIMEOUT"
            raise
        except asyncio.CancelledError:
            status = "cancelled"
            error_code = "MODEL_CANCELLED"
            raise
        except Exception:
            status = "failed"
            error_code = "MODEL_CALL_FAILED"
            raise
        finally:
            await self._database.record_model_call(
                session_id=session_id,
                step_id=step_id,
                agent_name="planner",
                provider=self._provider_name,
                model=self._planner_model,
                status=status,
                latency_ms=int((perf_counter() - started) * 1_000),
                error_code=error_code,
            )

    async def _run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        session_id: UUID,
        step_id: UUID,
    ) -> NavigatorOutcome:
        started = perf_counter()
        status: Literal["succeeded", "failed", "cancelled", "timed_out"] = "succeeded"
        error_code: str | None = None
        try:
            return await asyncio.wait_for(
                self._runner.run_navigator(navigator_input, run_config=None),
                timeout=self._limits.model_timeout_seconds,
            )
        except TimeoutError:
            status = "timed_out"
            error_code = "MODEL_TIMEOUT"
            raise
        except asyncio.CancelledError:
            status = "cancelled"
            error_code = "MODEL_CANCELLED"
            raise
        except Exception:
            status = "failed"
            error_code = "MODEL_CALL_FAILED"
            raise
        finally:
            await self._database.record_model_call(
                session_id=session_id,
                step_id=step_id,
                agent_name="navigator",
                provider=self._provider_name,
                model=self._navigator_model,
                status=status,
                latency_ms=int((perf_counter() - started) * 1_000),
                error_code=error_code,
            )

    def _validate_plan(self, plan: ExecutionPlan, *, session_id: UUID, task_id: UUID) -> None:
        if plan.session_id != session_id or plan.task_id != task_id:
            raise ValueError("Planner returned mismatched workflow identifiers")
        if plan.revision != 1:
            raise ValueError("initial plan revision must be one")
        if len(plan.steps) > self._limits.max_steps:
            raise ValueError("Planner exceeded the runtime step limit")

    def _validate_replan(
        self,
        plan: ExecutionPlan,
        *,
        previous: ExecutionPlan,
        completed: list[str],
    ) -> None:
        if plan.session_id != previous.session_id or plan.task_id != previous.task_id:
            raise ValueError("replan changed workflow identifiers")
        if plan.revision != previous.revision + 1:
            raise ValueError("replan revision must increase by exactly one")
        if len(plan.steps) > self._limits.max_steps:
            raise ValueError("replan exceeded the runtime step limit")
        keys = {step.key for step in plan.steps}
        if not set(completed) <= keys:
            raise ValueError("replan removed an already completed step")
        for step_key in completed:
            if plan.step(step_key) != previous.step(step_key):
                raise ValueError("replan changed an already completed step")

    @staticmethod
    def _validate_outcome(
        outcome: NavigatorOutcome,
        *,
        plan: ExecutionPlan,
        step_key: str,
        observation_id: str,
        recent: list[ActionExecutionSummary],
    ) -> None:
        if (
            outcome.session_id != plan.session_id
            or outcome.task_id != plan.task_id
            or outcome.plan_id != plan.plan_id
            or outcome.step_key != step_key
        ):
            raise ValueError("Navigator returned mismatched workflow identifiers")
        if outcome.status in {
            NavigatorOutcomeStatus.STEP_COMPLETED,
            NavigatorOutcomeStatus.TASK_COMPLETED,
        }:
            permitted_evidence = {observation_id}
            permitted_evidence.update(str(result.action_id) for result in recent if result.success)
            if not outcome.evidence_ids or not set(outcome.evidence_ids) <= permitted_evidence:
                raise ValueError("Navigator completion evidence is missing, stale, or unsuccessful")

    @staticmethod
    def _action_within_required_domains(
        action: ProductionAction,
        step: PlanStep,
    ) -> bool:
        """Require URL-bearing actions to stay in the Planner's exact host scope."""

        if action.type not in {"go_to_url", "open_tab"}:
            return True
        try:
            parsed = urlsplit(action.url)
            hostname = parsed.hostname
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or hostname is None:
            return False
        return hostname.lower().rstrip(".") in step.required_domains

    @staticmethod
    def _next_eligible_step(
        plan: ExecutionPlan,
        completed: list[str],
    ) -> PlanStep | None:
        complete = set(completed)
        return next(
            (
                step
                for step in plan.steps
                if step.key not in complete and set(step.depends_on) <= complete
            ),
            None,
        )

    @staticmethod
    def _observation_origin(url: str) -> str:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return "about:blank"
        if parsed.scheme not in {"http", "https"} or hostname is None:
            return "about:blank"
        default_port = 443 if parsed.scheme == "https" else 80
        suffix = "" if port in {None, default_port} else f":{port}"
        return f"{parsed.scheme}://{hostname.lower()}{suffix}"

    async def _finish_failed(
        self,
        *,
        session_id: UUID,
        task_id: UUID,
        plan: ExecutionPlan,
        completed: list[str],
        status: WorkflowStatus,
        error_code: str,
    ) -> WorkflowResult:
        await self._database.transition_session_status(session_id, "failed")
        return WorkflowResult(
            session_id=session_id,
            task_id=task_id,
            plan_id=plan.plan_id,
            status=status,
            completed_step_keys=tuple(completed),
            total_steps=len(plan.steps),
            summary="Workflow stopped before all validated steps completed",
            error_code=error_code,
        )

    async def _safe_step_transition(
        self,
        step_id: UUID,
        status: Literal["failed", "cancelled"],
    ) -> None:
        with suppress(RuntimeError, ValueError):
            await self._database.transition_step_status(step_id, status)

    async def _safe_session_transition(
        self,
        session_id: UUID,
        status: Literal["failed", "cancelled"],
    ) -> None:
        with suppress(RuntimeError, ValueError):
            await self._database.transition_session_status(session_id, status)


__all__ = ["PlannerNavigatorRuntime", "TwoAgentWorkflow", "WorkflowLimits"]
