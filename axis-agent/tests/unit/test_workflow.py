from __future__ import annotations

import asyncio
from uuid import uuid4

from axis_agent.browser import MockBrowserAdapter
from axis_agent.contracts import (
    ExecutionPlan,
    NavigatorInput,
    NavigatorOutcome,
    NavigatorOutcomeStatus,
    PlannerInput,
    PlanStep,
    TaskRequest,
    WorkflowStatus,
)
from axis_agent.contracts.actions import ClickElementAction, GoToUrlAction, WaitAction
from axis_agent.devtools import MockPlannerNavigatorRuntime
from axis_agent.firewall import FirewallPolicy, FirewallService, NavigationPermitStore
from axis_agent.mcp import AxisMCPService, MCPProfile
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import (
    ActionDispatcher,
    MCPServiceBrowserTools,
    PlannerNavigatorRuntime,
    TwoAgentWorkflow,
    WorkflowLimits,
)


async def build_workflow(
    database: AxisDatabase,
    runner: PlannerNavigatorRuntime,
) -> tuple[TwoAgentWorkflow, AxisMCPService, MockBrowserAdapter]:
    browser = MockBrowserAdapter()
    permits = NavigationPermitStore()
    dispatcher = ActionDispatcher(
        browser=browser,
        firewall=FirewallService(FirewallPolicy()),
        database=database,
        permits=permits,
    )
    service = AxisMCPService(
        profile=MCPProfile.NAVIGATOR,
        browser=browser,
        dispatcher=dispatcher,
    )
    tools = MCPServiceBrowserTools(service=service, database=database)
    workflow = TwoAgentWorkflow(
        runner=runner,
        database=database,
        browser_tools=tools,
        run_config={},
        config_hash="0" * 64,
        provider_name="mock",
        planner_model="mock-planner",
        navigator_model="mock-navigator",
        limits=WorkflowLimits(model_timeout_seconds=5),
    )
    return workflow, service, browser


async def test_mock_two_agent_workflow_runs_standalone_and_audits(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(
            database,
            MockPlannerNavigatorRuntime(),
        )

        result = await workflow.run(TaskRequest(prompt="Run the standalone check"))
        await service.stop()

        assert result.status is WorkflowStatus.SUCCEEDED
        assert result.completed_step_keys == ("mock_wait",)
        assert [call[1].type for call in browser.calls] == ["wait"]
        session = await (
            await database.connection.execute(
                "SELECT status, task_summary, length(task_summary_sha256) FROM sessions"
            )
        ).fetchone()
        assert tuple(session) == ("succeeded", "[HASHED]", 64)
        model_calls = await database.connection.execute_fetchall(
            "SELECT agent_name, status FROM model_calls ORDER BY created_at, rowid"
        )
        assert [tuple(row) for row in model_calls] == [
            ("planner", "succeeded"),
            ("navigator", "succeeded"),
            ("navigator", "succeeded"),
        ]


class ClickRunner:
    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object,
    ) -> ExecutionPlan:
        del run_config
        return ExecutionPlan(
            session_id=planner_input.session_id,
            task_id=planner_input.task.task_id,
            objective="Click only with approval",
            completion_criteria=("Approved click succeeds",),
            steps=(
                PlanStep(
                    key="click",
                    order=1,
                    objective="Click one element",
                    success_criteria=("Click succeeds",),
                    allowed_action_types=("click_element",),
                ),
            ),
        )

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object,
    ) -> NavigatorOutcome:
        del run_config
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
            actions=(ClickElementAction(index=0),),
        )


async def test_high_impact_action_is_blocked_without_user_handler(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(database, ClickRunner())

        result = await workflow.run(TaskRequest(prompt="Click the button"))
        await service.stop()

        assert result.status is WorkflowStatus.BLOCKED
        assert result.error_code == "APPROVAL_REJECTED"
        assert browser.calls == []
        approval = await database.connection.execute_fetchall(
            """
            SELECT decision, actor
            FROM approval_decisions JOIN approval_requests
              ON approval_requests.id=approval_decisions.approval_id
            """
        )
        assert [tuple(row) for row in approval] == [("rejected", "system")]


class ReplanningRunner:
    def __init__(self) -> None:
        self.planner_calls = 0
        self.navigator_calls = 0

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object,
    ) -> ExecutionPlan:
        del run_config
        self.planner_calls += 1
        revision = self.planner_calls
        key = "first" if revision == 1 else "replacement"
        return ExecutionPlan(
            plan_id=uuid4(),
            session_id=planner_input.session_id,
            task_id=planner_input.task.task_id,
            revision=revision,
            objective="Bounded replan",
            completion_criteria=("Replacement succeeds",),
            steps=(
                PlanStep(
                    key=key,
                    order=1,
                    objective="Run bounded action",
                    success_criteria=("Action succeeds",),
                    allowed_action_types=("wait",),
                ),
            ),
        )

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object,
    ) -> NavigatorOutcome:
        del run_config
        self.navigator_calls += 1
        common = {
            "session_id": navigator_input.session_id,
            "task_id": navigator_input.task_id,
            "plan_id": navigator_input.plan.plan_id,
            "step_key": navigator_input.active_step_key,
        }
        if navigator_input.plan.revision == 1:
            return NavigatorOutcome(
                **common,
                status=NavigatorOutcomeStatus.BLOCKED,
                replan_requested=True,
                error_code="TARGET_CHANGED",
            )
        if not navigator_input.recent_results:
            return NavigatorOutcome(
                **common,
                status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
                actions=(WaitAction(seconds=0),),
            )
        return NavigatorOutcome(
            **common,
            status=NavigatorOutcomeStatus.TASK_COMPLETED,
            evidence_ids=(str(navigator_input.recent_results[-1].action_id),),
        )


async def test_workflow_performs_only_bounded_validated_replan(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        runner = ReplanningRunner()
        workflow, service, _ = await build_workflow(database, runner)

        result = await workflow.run(TaskRequest(prompt="Adapt once"))
        await service.stop()

        assert result.status is WorkflowStatus.SUCCEEDED
        assert result.completed_step_keys == ("replacement",)
        assert runner.planner_calls == 2
        plans = await database.connection.execute_fetchall(
            "SELECT revision FROM execution_plans ORDER BY revision"
        )
        assert [row[0] for row in plans] == [1, 2]


class InvalidPlannerRunner(ClickRunner):
    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object,
    ) -> ExecutionPlan:
        valid = await super().run_planner(planner_input, run_config=run_config)
        return valid.model_copy(update={"task_id": uuid4()})


async def test_invalid_planner_output_stops_before_navigator(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(database, InvalidPlannerRunner())

        result = await workflow.run(TaskRequest(prompt="Invalid identifiers"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "WORKFLOW_RUNTIME_FAILED"
        assert browser.calls == []
        calls = await database.connection.execute_fetchall("SELECT agent_name FROM model_calls")
        assert [row[0] for row in calls] == ["planner"]


class RepairingRunner(MockPlannerNavigatorRuntime):
    def __init__(self) -> None:
        self.planner_calls = 0

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        self.planner_calls += 1
        plan = await super().run_planner(planner_input, run_config=run_config)
        if self.planner_calls == 1:
            return plan.model_copy(update={"task_id": uuid4()})
        return plan


async def test_workflow_does_not_add_a_second_planner_repair_layer(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        runner = RepairingRunner()
        workflow, service, _ = await build_workflow(database, runner)

        result = await workflow.run(TaskRequest(prompt="Repair once"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert runner.planner_calls == 1
        repair_events = await database.connection.execute_fetchall(
            "SELECT event_type FROM audit_events WHERE event_type='planner.output_repair'"
        )
        assert repair_events == []


class SlowPlannerRunner(MockPlannerNavigatorRuntime):
    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        await asyncio.sleep(2)
        return await super().run_planner(planner_input, run_config=run_config)


async def test_whole_workflow_deadline_cancels_and_persists_status(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        browser = MockBrowserAdapter()
        permits = NavigationPermitStore()
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(FirewallPolicy()),
            database=database,
            permits=permits,
        )
        service = AxisMCPService(
            profile=MCPProfile.NAVIGATOR,
            browser=browser,
            dispatcher=dispatcher,
        )
        workflow = TwoAgentWorkflow(
            runner=SlowPlannerRunner(),
            database=database,
            browser_tools=MCPServiceBrowserTools(service=service, database=database),
            run_config={},
            config_hash="0" * 64,
            provider_name="mock",
            planner_model="mock-planner",
            navigator_model="mock-navigator",
            limits=WorkflowLimits(
                model_timeout_seconds=5,
                workflow_timeout_seconds=1,
            ),
        )

        result = await workflow.run(TaskRequest(prompt="Time out safely"))

        assert result.status is WorkflowStatus.CANCELLED
        assert result.error_code == "WORKFLOW_TIMEOUT"
        session = await (
            await database.connection.execute("SELECT status FROM sessions")
        ).fetchone()
        assert session[0] == "cancelled"


class SerializingRunner(MockPlannerNavigatorRuntime):
    def __init__(self) -> None:
        self.first_planner_entered = asyncio.Event()
        self.release_first_planner = asyncio.Event()
        self.planner_calls = 0
        self.active_planners = 0
        self.max_active_planners = 0

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        self.planner_calls += 1
        self.active_planners += 1
        self.max_active_planners = max(self.max_active_planners, self.active_planners)
        try:
            if self.planner_calls == 1:
                self.first_planner_entered.set()
                await self.release_first_planner.wait()
            return await super().run_planner(planner_input, run_config=run_config)
        finally:
            self.active_planners -= 1


async def test_one_workflow_instance_serializes_concurrent_runs(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        runner = SerializingRunner()
        workflow, service, _ = await build_workflow(database, runner)
        first = asyncio.create_task(workflow.run(TaskRequest(prompt="First task")))
        await runner.first_planner_entered.wait()
        second = asyncio.create_task(workflow.run(TaskRequest(prompt="Second task")))
        try:
            await asyncio.sleep(0)
            assert runner.planner_calls == 1
            assert runner.max_active_planners == 1
        finally:
            runner.release_first_planner.set()

        first_result, second_result = await asyncio.gather(first, second)
        await service.stop()

        assert first_result.status is WorkflowStatus.SUCCEEDED
        assert second_result.status is WorkflowStatus.SUCCEEDED
        assert runner.planner_calls == 2
        assert runner.max_active_planners == 1


class MultiActionRunner(MockPlannerNavigatorRuntime):
    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
            actions=(WaitAction(seconds=0), WaitAction(seconds=0)),
        )


async def test_multi_action_batch_is_rejected_before_any_execution(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(database, MultiActionRunner())

        result = await workflow.run(TaskRequest(prompt="Reject stale batch"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "MULTI_ACTION_BATCH_REJECTED"
        assert browser.calls == []


class OutOfDomainRunner(MockPlannerNavigatorRuntime):
    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        del run_config
        return ExecutionPlan(
            session_id=planner_input.session_id,
            task_id=planner_input.task.task_id,
            objective="Stay inside the planned domain",
            completion_criteria=("Navigation is scoped",),
            steps=(
                PlanStep(
                    key="navigate",
                    order=1,
                    objective="Navigate within scope",
                    success_criteria=("The planned site is open",),
                    allowed_action_types=("go_to_url",),
                    required_domains=("example.com",),
                    max_attempts=1,
                ),
            ),
        )

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
            actions=(GoToUrlAction(url="https://outside.example/"),),
        )


async def test_navigation_must_match_plan_step_required_domains(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(database, OutOfDomainRunner())

        result = await workflow.run(TaskRequest(prompt="Navigate safely"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "ACTION_OUTSIDE_REQUIRED_DOMAINS"
        assert browser.calls == []


class DirectCompletionRunner(MockPlannerNavigatorRuntime):
    def __init__(self, *, evidence: bool) -> None:
        self._evidence = evidence

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        evidence_ids = (navigator_input.observation_id,) if self._evidence else ()
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=NavigatorOutcomeStatus.TASK_COMPLETED,
            evidence_ids=evidence_ids,
        )


async def test_terminal_completion_accepts_current_observation_evidence(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(
            database,
            DirectCompletionRunner(evidence=True),
        )

        result = await workflow.run(TaskRequest(prompt="Already complete"))
        await service.stop()

        assert result.status is WorkflowStatus.SUCCEEDED
        assert browser.calls == []


async def test_terminal_completion_without_current_success_evidence_fails_closed(
    tmp_path,
) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(
            database,
            DirectCompletionRunner(evidence=False),
        )

        result = await workflow.run(TaskRequest(prompt="Unsupported completion"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "WORKFLOW_RUNTIME_FAILED"
        assert browser.calls == []


class AttemptBoundRunner(MockPlannerNavigatorRuntime):
    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        plan = await super().run_planner(planner_input, run_config=run_config)
        bounded_step = plan.steps[0].model_copy(update={"max_attempts": 2})
        return plan.model_copy(update={"steps": (bounded_step,)})

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        return NavigatorOutcome(
            session_id=navigator_input.session_id,
            task_id=navigator_input.task_id,
            plan_id=navigator_input.plan.plan_id,
            step_key=navigator_input.active_step_key,
            status=NavigatorOutcomeStatus.ACTIONS_REQUIRED,
            actions=(WaitAction(seconds=0),),
        )


async def test_plan_step_max_attempts_is_an_execution_bound(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        workflow, service, browser = await build_workflow(database, AttemptBoundRunner())

        result = await workflow.run(TaskRequest(prompt="Bound attempts"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "STEP_ATTEMPT_LIMIT_EXCEEDED"
        assert [call[1].type for call in browser.calls] == ["wait", "wait"]


class CompletedStepMutationRunner:
    def __init__(self) -> None:
        self.planner_calls = 0

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object,
    ) -> ExecutionPlan:
        del run_config
        self.planner_calls += 1
        changed = planner_input.prior_plan is not None
        return ExecutionPlan(
            plan_id=uuid4(),
            session_id=planner_input.session_id,
            task_id=planner_input.task.task_id,
            revision=2 if changed else 1,
            objective="Preserve completed work",
            completion_criteria=("Both steps complete",),
            steps=(
                PlanStep(
                    key="first",
                    order=1,
                    objective="Changed after completion" if changed else "Original first step",
                    success_criteria=("First complete",),
                    allowed_action_types=("wait",),
                ),
                PlanStep(
                    key="second",
                    order=2,
                    objective="Second step",
                    success_criteria=("Second complete",),
                    depends_on=("first",),
                    allowed_action_types=("wait",),
                ),
            ),
        )

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object,
    ) -> NavigatorOutcome:
        del run_config
        common = {
            "session_id": navigator_input.session_id,
            "task_id": navigator_input.task_id,
            "plan_id": navigator_input.plan.plan_id,
            "step_key": navigator_input.active_step_key,
        }
        if navigator_input.active_step_key == "first":
            return NavigatorOutcome(
                **common,
                status=NavigatorOutcomeStatus.STEP_COMPLETED,
                evidence_ids=(navigator_input.observation_id,),
            )
        return NavigatorOutcome(
            **common,
            status=NavigatorOutcomeStatus.BLOCKED,
            replan_requested=True,
            error_code="SECOND_STEP_CHANGED",
        )


async def test_replan_cannot_change_a_completed_step_definition(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        runner = CompletedStepMutationRunner()
        workflow, service, _ = await build_workflow(database, runner)

        result = await workflow.run(TaskRequest(prompt="Keep completed definitions"))
        await service.stop()

        assert result.status is WorkflowStatus.FAILED
        assert result.error_code == "WORKFLOW_RUNTIME_FAILED"
        assert result.completed_step_keys == ("first",)
        assert runner.planner_calls == 2
