from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from axis_agent.contracts import (
    ActionCommand,
    ApprovalRequest,
    ExecutionPlan,
    PlanStep,
    ProductionAction,
)
from axis_agent.contracts.actions import (
    ClickElementAction,
    WaitAction,
    production_action_payload,
)
from axis_agent.firewall import FirewallService, policy_from_hosts
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import ApprovalCoordinator, TrustedSiteAutoApprovalHandler


class RecordingApprovalHandler:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.requests: list[ApprovalRequest] = []

    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool:
        assert action.type == "click_element"
        self.requests.append(request)
        return self.approved


class FailingApprovalHandler:
    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool:
        del request, action
        raise RuntimeError("unavailable")


class InvalidApprovalHandler:
    async def approve(
        self,
        request: ApprovalRequest,
        action: ProductionAction,
    ) -> bool:
        del request, action
        return "yes"  # type: ignore[return-value]


async def build_context(database: AxisDatabase) -> tuple[ExecutionPlan, UUID, UUID]:
    task_id = uuid4()
    session_id = UUID(
        await database.create_session(
            task_id=task_id,
            task_summary="Approval test task",
            config_hash="config-hash",
        )
    )
    step_id = UUID(await database.create_step(session_id=session_id, step_number=1))
    plan = ExecutionPlan(
        session_id=session_id,
        task_id=task_id,
        objective="Test approval",
        completion_criteria=("Approved action is handled",),
        steps=(
            PlanStep(
                key="approve",
                order=1,
                objective="Test one action",
                success_criteria=("Action decision exists",),
                allowed_action_types=("click_element", "wait"),
            ),
        ),
    )
    await database.record_plan_metadata(plan)
    return plan, step_id, task_id


def command_for(
    plan: ExecutionPlan,
    step_id: UUID,
    task_id: UUID,
    action: ClickElementAction | WaitAction,
    *,
    observation_id: str | None = "0" * 32,
    expected_origin: str | None = None,
) -> ActionCommand:
    return ActionCommand(
        session_id=plan.session_id,
        task_id=task_id,
        step_id=step_id,
        ordinal=0,
        expected_observation_id=observation_id,
        expected_page_id="page-1",
        expected_origin=expected_origin,
        action=action,
    )


@pytest.mark.parametrize("approved", [True, False])
async def test_high_impact_decision_is_persisted(tmp_path, approved: bool) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)
        handler = RecordingApprovalHandler(approved)
        coordinator = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=handler,
        )

        command = command_for(plan, step_id, task_id, ClickElementAction(index=0))
        allowed = await coordinator.authorize(command)

        assert allowed is approved
        assert len(handler.requests) == 1
        stored = await database.get_approval(handler.requests[0].approval_id)
        assert stored is not None
        assert stored.decision == ("approved" if approved else "rejected")
        assert stored.decision_actor == "user"
        assert (
            await database.has_valid_action_approval(
                session_id=command.session_id,
                task_id=command.task_id,
                action_id=command.action_id,
                observation_id=command.expected_observation_id,
                action_type="click_element",
                action_payload=production_action_payload(command.action),
            )
        ) is approved
        assert (
            await database.has_valid_action_approval(
                session_id=command.session_id,
                task_id=command.task_id,
                action_id=command.action_id,
                observation_id=command.expected_observation_id,
                action_type="click_element",
                action_payload=production_action_payload(ClickElementAction(index=1)),
            )
        ) is False


async def test_low_risk_action_does_not_create_approval(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)
        coordinator = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
        )

        allowed = await coordinator.authorize(
            command_for(plan, step_id, task_id, WaitAction(seconds=0))
        )

        assert allowed is True
        count = await (
            await database.connection.execute("SELECT COUNT(*) FROM approval_requests")
        ).fetchone()
        assert count[0] == 0


async def test_missing_observation_and_handler_failure_fail_closed(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)
        missing = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=RecordingApprovalHandler(True),
        )
        assert (
            await missing.authorize(
                command_for(
                    plan,
                    step_id,
                    task_id,
                    ClickElementAction(index=0),
                    observation_id=None,
                )
            )
            is False
        )

        failing = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=FailingApprovalHandler(),
        )
        command = command_for(plan, step_id, task_id, ClickElementAction(index=0))
        assert await failing.authorize(command) is False
        row = await database.connection.execute_fetchall(
            "SELECT decision, actor, reason_code FROM approval_decisions"
        )
        assert tuple(row[-1]) == ("rejected", "system", "APPROVAL_HANDLER_FAILED")


async def test_non_boolean_approval_handler_result_fails_closed(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)
        coordinator = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=InvalidApprovalHandler(),
        )

        allowed = await coordinator.authorize(
            command_for(plan, step_id, task_id, ClickElementAction(index=0))
        )

        assert allowed is False
        rows = await database.connection.execute_fetchall(
            "SELECT decision, actor, reason_code FROM approval_decisions"
        )
        assert tuple(rows[-1]) == ("rejected", "system", "APPROVAL_HANDLER_INVALID")


async def test_trusted_site_policy_creates_exact_action_approval(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)

        async def public_resolver(_host: str, _port: int) -> tuple[str, ...]:
            return ("93.184.216.34",)

        coordinator = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=TrustedSiteAutoApprovalHandler(
                FirewallService(
                    policy_from_hosts(["example.com"]),
                    resolver=public_resolver,
                )
            ),
        )
        command = command_for(
            plan,
            step_id,
            task_id,
            ClickElementAction(index=0),
            expected_origin="https://example.com",
        )

        assert await coordinator.authorize(command) is True
        rows = await database.connection.execute_fetchall(
            "SELECT decision, actor, reason_code FROM approval_decisions"
        )
        assert [tuple(row) for row in rows] == [
            ("approved", "user", "TRUSTED_SITE_POLICY_APPROVED")
        ]
        assert await database.has_valid_action_approval(
            session_id=command.session_id,
            task_id=command.task_id,
            action_id=command.action_id,
            observation_id=command.expected_observation_id,
            action_type="click_element",
            action_payload=production_action_payload(command.action),
        )


async def test_trusted_site_policy_rejects_non_allowlisted_origin(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        plan, step_id, task_id = await build_context(database)

        async def public_resolver(_host: str, _port: int) -> tuple[str, ...]:
            return ("93.184.216.34",)

        coordinator = ApprovalCoordinator(
            database=database,
            plan_id=plan.plan_id,
            step_key="approve",
            handler=TrustedSiteAutoApprovalHandler(
                FirewallService(
                    policy_from_hosts(["example.com"]),
                    resolver=public_resolver,
                )
            ),
        )
        command = command_for(
            plan,
            step_id,
            task_id,
            ClickElementAction(index=0),
            expected_origin="https://blocked.example",
        )

        assert await coordinator.authorize(command) is False
        count = await (
            await database.connection.execute("SELECT COUNT(*) FROM approval_requests")
        ).fetchone()
        assert count[0] == 0
