from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import aiosqlite
import pytest

from axis_agent.contracts import (
    ApprovalDecision,
    ApprovalDecisionValue,
    ApprovalRequest,
    ApprovalRisk,
    ExecutionPlan,
    PlanStep,
)
from axis_agent.persistence.database import AxisDatabase, sanitize_url_for_audit


def _execution_plan(*, session_id: UUID, task_id: UUID) -> ExecutionPlan:
    return ExecutionPlan(
        session_id=session_id,
        task_id=task_id,
        objective="Sensitive customer objective that must not be stored",
        completion_criteria=("Approved page was inspected",),
        steps=(
            PlanStep(
                key="inspect_page",
                order=1,
                objective="Sensitive step prose",
                success_criteria=("Page was inspected",),
                allowed_action_types=("click_element", "wait"),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_migrations_are_idempotent_and_enable_safety_pragmas(tmp_path) -> None:
    database_path = tmp_path / "axis.db"
    database = AxisDatabase(database_path)
    await database.connect()
    await database.initialize()
    await database.initialize()

    health = await database.healthcheck()
    assert health.connected is True
    assert health.journal_mode == "wal"
    assert health.foreign_keys is True
    assert health.migration_version == 2

    await database.close()


@pytest.mark.asyncio
async def test_concurrent_database_instances_apply_each_migration_once(tmp_path) -> None:
    database_path = tmp_path / "axis.db"
    databases = [AxisDatabase(database_path) for _ in range(8)]
    for database in databases:
        await database.connect()

    try:
        await asyncio.gather(*(database.initialize() for database in databases))
        rows = await databases[0].connection.execute_fetchall(
            "SELECT version, COUNT(*) FROM schema_migrations GROUP BY version ORDER BY version"
        )
        assert [tuple(row) for row in rows] == [(1, 1), (2, 1)]
    finally:
        await asyncio.gather(*(database.close() for database in databases))


@pytest.mark.asyncio
async def test_session_stores_only_a_hash_of_task_summary(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        raw_summary = "Customer private task summary"
        session_id = await database.create_session(task_summary=raw_summary, config_hash="cfg")
        row = await (
            await database.connection.execute(
                "SELECT task_summary, task_summary_sha256 FROM sessions WHERE id=?",
                (session_id,),
            )
        ).fetchone()

        assert row["task_summary"] == "[HASHED]"
        assert len(row["task_summary_sha256"]) == 64
        assert raw_summary not in tuple(row)


@pytest.mark.asyncio
async def test_plan_model_call_and_approval_metadata_are_append_only(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        task_id = uuid4()
        session_uuid = uuid4()
        session_id = await database.create_session(
            task_summary="private task",
            config_hash="cfg",
            task_id=task_id,
            session_id=session_uuid,
        )
        plan = _execution_plan(session_id=session_uuid, task_id=task_id)
        stored_plan = await database.record_plan_metadata(plan)
        duplicate_plan = await database.record_plan_metadata(plan)

        assert stored_plan == duplicate_plan
        assert stored_plan.step_count == 1
        assert len(stored_plan.plan_sha256) == 64

        plan_rows = await (
            await database.connection.execute(
                """
                SELECT plans.plan_sha256, steps.step_key, steps.allowed_action_types_json
                FROM execution_plans AS plans
                JOIN execution_plan_steps AS steps ON steps.plan_id=plans.id
                """
            )
        ).fetchall()
        serialized_rows = repr([tuple(row) for row in plan_rows])
        assert "Sensitive customer objective" not in serialized_rows
        assert "Sensitive step prose" not in serialized_rows
        assert "click_element" in serialized_rows

        call_id = await database.record_model_call(
            session_id=session_id,
            agent_name="planner",
            provider="oci-openai",
            model="xai.grok-4.3",
            status="succeeded",
            input_tokens=10,
            output_tokens=5,
            latency_ms=20,
        )
        model_row = await (
            await database.connection.execute(
                "SELECT agent_name, provider, model, input_tokens FROM model_calls WHERE id=?",
                (call_id,),
            )
        ).fetchone()
        assert tuple(model_row) == ("planner", "oci-openai", "xai.grok-4.3", 10)

        approval = ApprovalRequest.for_action(
            session_id=plan.session_id,
            task_id=plan.task_id,
            plan_id=plan.plan_id,
            step_key="inspect_page",
            action_id=uuid4(),
            observation_id="d" * 32,
            action={
                "type": "click_element",
                "index": 4,
                "intent": "private reasoning",
                "xpath": "//private-selector",
            },
            risk=ApprovalRisk.INTERACTION,
            reason_code="CLICK_REQUIRES_APPROVAL",
        )
        pending = await database.record_approval_request(approval)
        assert pending.decision is None

        decision = ApprovalDecision(
            approval_id=approval.approval_id,
            decision=ApprovalDecisionValue.APPROVED,
            actor="user",
            reason_code="USER_APPROVED",
            decided_at=datetime.now(UTC),
        )
        approved = await database.record_approval_decision(decision)
        assert approved.decision == "approved"
        assert approved.decision_actor == "user"

        approval_row = await (
            await database.connection.execute(
                "SELECT action_sha256, action_type FROM approval_requests WHERE id=?",
                (str(approval.approval_id),),
            )
        ).fetchone()
        assert len(approval_row["action_sha256"]) == 64
        assert tuple(approval_row)[1] == "click_element"
        assert "private reasoning" not in repr(tuple(approval_row))
        assert "private-selector" not in repr(tuple(approval_row))

        for table in ("execution_plans", "approval_requests", "model_calls"):
            with pytest.raises(aiosqlite.DatabaseError, match="immutable"):
                await database.connection.execute(f"DELETE FROM {table}")  # noqa: S608
                await database.connection.commit()
            await database.connection.rollback()


@pytest.mark.asyncio
async def test_approval_expiry_uses_trusted_server_clock(tmp_path) -> None:
    requested_at = datetime(2030, 1, 1, tzinfo=UTC)
    expires_at = requested_at + timedelta(seconds=30)
    trusted_now = expires_at + timedelta(milliseconds=1)

    async with AxisDatabase(
        tmp_path / "axis.db",
        trusted_clock=lambda: trusted_now,
    ) as database:
        task_id = uuid4()
        session_uuid = uuid4()
        await database.create_session(
            task_summary="private task",
            config_hash="cfg",
            task_id=task_id,
            session_id=session_uuid,
        )
        plan = _execution_plan(session_id=session_uuid, task_id=task_id)
        await database.record_plan_metadata(plan)
        request = ApprovalRequest.for_action(
            session_id=plan.session_id,
            task_id=plan.task_id,
            plan_id=plan.plan_id,
            step_key="inspect_page",
            action_id=uuid4(),
            observation_id="d" * 32,
            action={"type": "click_element", "index": 4},
            risk=ApprovalRisk.INTERACTION,
            reason_code="CLICK_REQUIRES_APPROVAL",
        ).model_copy(update={"requested_at": requested_at, "expires_at": expires_at})
        await database.record_approval_request(request)

        spoofed_pre_expiry_decision = ApprovalDecision(
            approval_id=request.approval_id,
            decision=ApprovalDecisionValue.APPROVED,
            actor="user",
            reason_code="USER_APPROVED",
            decided_at=requested_at + timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="expired approval"):
            await database.record_approval_decision(spoofed_pre_expiry_decision)

        assert database.connection.in_transaction is False
        count = await (
            await database.connection.execute("SELECT COUNT(*) FROM approval_decisions")
        ).fetchone()
        assert count[0] == 0


@pytest.mark.asyncio
async def test_action_and_firewall_decision_are_atomic_redacted_and_idempotent(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(
            task_summary="Open the approved site", config_hash="cfg"
        )
        step_id = await database.create_step(session_id=session_id, step_number=0)
        action_id = uuid4()

        first = await database.record_action_with_firewall_decision(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="go_to_url",
            action_payload={
                "url": "https://example.com/report?token=raw-secret",
                "authorization": "Bearer secret-value",
                "intent": "private user instruction",
                "text": "customer credential value",
            },
            idempotency_class="navigation",
            raw_url="https://example.com/report?token=raw-secret#private",
            purpose="navigation",
            allowed=True,
            reason_code="ALLOW_RULE_MATCH",
            policy_hash="policy",
            resolved_ips=("93.184.216.34",),
            expected_observation_id="e" * 32,
        )
        duplicate = await database.record_action_with_firewall_decision(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="go_to_url",
            action_payload={
                "url": "https://example.com/report?token=raw-secret",
                "authorization": "Bearer secret-value",
                "intent": "private user instruction",
                "text": "customer credential value",
            },
            idempotency_class="navigation",
            raw_url="https://example.com/report?token=raw-secret#private",
            purpose="navigation",
            allowed=True,
            reason_code="ALLOW_RULE_MATCH",
            policy_hash="policy",
            resolved_ips=("93.184.216.34",),
            expected_observation_id="e" * 32,
        )

        with pytest.raises(ValueError, match="different context, payload, or preconditions"):
            await database.record_action_with_firewall_decision(
                session_id=session_id,
                step_id=step_id,
                action_id=action_id,
                ordinal=0,
                action_type="go_to_url",
                action_payload={"url": "https://changed.invalid"},
                idempotency_class="navigation",
                raw_url="https://changed.invalid",
                purpose="navigation",
                allowed=False,
                reason_code="DENY_NO_ALLOW_MATCH",
                policy_hash="different-policy",
            )
        with pytest.raises(ValueError, match="different firewall decision"):
            await database.record_action_with_firewall_decision(
                session_id=session_id,
                step_id=step_id,
                action_id=action_id,
                ordinal=0,
                action_type="go_to_url",
                action_payload={
                    "url": "https://example.com/report?token=raw-secret",
                    "authorization": "Bearer secret-value",
                    "intent": "private user instruction",
                    "text": "customer credential value",
                },
                idempotency_class="navigation",
                raw_url="https://example.com/report?token=raw-secret#private",
                purpose="navigation",
                allowed=True,
                reason_code="ALLOW_RULE_MATCH",
                policy_hash="changed-policy",
                resolved_ips=("93.184.216.34",),
                expected_observation_id="e" * 32,
            )

        assert first == duplicate
        assert first.state == "allowed"

        action_row = await (
            await database.connection.execute(
                "SELECT request_json, expected_observation_id FROM actions WHERE id=?",
                (str(action_id),),
            )
        ).fetchone()
        decision_row = await (
            await database.connection.execute(
                "SELECT sanitized_url, raw_url_hash FROM firewall_decisions WHERE action_id=?",
                (str(action_id),),
            )
        ).fetchone()
        counts = await (
            await database.connection.execute(
                "SELECT (SELECT COUNT(*) FROM actions), (SELECT COUNT(*) FROM firewall_decisions)"
            )
        ).fetchone()

        assert "secret-value" not in action_row["request_json"]
        assert "raw-secret" not in action_row["request_json"]
        assert "private user instruction" not in action_row["request_json"]
        assert "customer credential value" not in action_row["request_json"]
        assert "[REDACTED]" in action_row["request_json"]
        assert action_row["expected_observation_id"] == "e" * 32
        assert decision_row["sanitized_url"] == "https://example.com/report"
        assert "raw-secret" not in decision_row["raw_url_hash"]
        assert tuple(counts) == (1, 1)


@pytest.mark.asyncio
async def test_dispatch_is_persisted_once_and_completion_is_not_replayed(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(task_summary="task", config_hash="cfg")
        step_id = await database.create_step(session_id=session_id, step_number=0)
        action_id = uuid4()
        await database.record_action_with_firewall_decision(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="go_to_url",
            action_payload={"url": "https://example.com"},
            idempotency_class="navigation",
            raw_url="https://example.com",
            purpose="navigation",
            allowed=True,
            reason_code="ALLOW_RULE_MATCH",
            policy_hash="policy",
        )

        assert await database.mark_dispatched(action_id) is True
        assert await database.mark_dispatched(action_id) is False
        assert (
            await database.complete_action(action_id, state="succeeded", result={"ok": True})
            is True
        )
        assert (
            await database.complete_action(action_id, state="succeeded", result={"ok": True})
            is False
        )

        stored = await database.get_action(action_id)
        assert stored is not None
        assert stored.state == "succeeded"
        assert stored.result == {"ok": True}


@pytest.mark.asyncio
async def test_non_url_action_is_redacted_idempotent_and_has_no_firewall_row(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(task_summary="task", config_hash="cfg")
        step_id = await database.create_step(session_id=session_id, step_number=0)
        action_id = uuid4()

        first = await database.record_action(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="input_text",
            action_payload={
                "type": "input_text",
                "index": 1,
                "text": "private customer value",
            },
            idempotency_class="non_retryable",
            expected_observation_id="a" * 32,
            expected_page_id="page-1",
            expected_origin="https://example.com",
        )
        duplicate = await database.record_action(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="input_text",
            action_payload={
                "type": "input_text",
                "index": 1,
                "text": "private customer value",
            },
            idempotency_class="non_retryable",
            expected_observation_id="a" * 32,
            expected_page_id="page-1",
            expected_origin="https://example.com",
        )

        with pytest.raises(ValueError, match="different context, payload, or preconditions"):
            await database.record_action(
                session_id=session_id,
                step_id=step_id,
                action_id=action_id,
                ordinal=0,
                action_type="wait",
                action_payload={"type": "wait", "seconds": 1},
                idempotency_class="retryable",
            )

        assert duplicate == first
        assert first.state == "allowed"
        row = await (
            await database.connection.execute(
                """
                SELECT request_json, expected_observation_id, expected_page_id, expected_origin
                FROM actions WHERE id=?
                """,
                (str(action_id),),
            )
        ).fetchone()
        firewall_count = await (
            await database.connection.execute(
                "SELECT COUNT(*) FROM firewall_decisions WHERE action_id=?",
                (str(action_id),),
            )
        ).fetchone()

        assert "private customer value" not in row["request_json"]
        assert "[REDACTED]" in row["request_json"]
        assert row["expected_observation_id"] == "a" * 32
        assert row["expected_page_id"] == "page-1"
        assert row["expected_origin"] == "https://example.com"
        assert firewall_count[0] == 0

        with pytest.raises(ValueError, match="non-URL"):
            await database.record_action(
                session_id=session_id,
                step_id=step_id,
                action_id=uuid4(),
                ordinal=1,
                action_type="go_to_url",
                action_payload={"type": "go_to_url", "url": "https://example.com"},
                idempotency_class="navigation",
            )


@pytest.mark.asyncio
async def test_action_insert_rejects_a_step_owned_by_another_session(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        owning_session_id = await database.create_session(task_summary="owner", config_hash="cfg")
        other_session_id = await database.create_session(task_summary="other", config_hash="cfg")
        step_id = await database.create_step(session_id=owning_session_id, step_number=0)

        with pytest.raises(ValueError, match="does not belong"):
            await database.record_action(
                session_id=other_session_id,
                step_id=step_id,
                action_id=uuid4(),
                ordinal=0,
                action_type="wait",
                action_payload={"type": "wait", "seconds": 1},
                idempotency_class="retryable",
            )
        with pytest.raises(ValueError, match="does not belong"):
            await database.record_action_with_firewall_decision(
                session_id=other_session_id,
                step_id=step_id,
                action_id=uuid4(),
                ordinal=0,
                action_type="go_to_url",
                action_payload={"type": "go_to_url", "url": "https://example.com"},
                idempotency_class="navigation",
                raw_url="https://example.com",
                purpose="navigation",
                allowed=True,
                reason_code="ALLOW_RULE_MATCH",
                policy_hash="policy",
            )

        count = await (await database.connection.execute("SELECT COUNT(*) FROM actions")).fetchone()
        assert count[0] == 0


@pytest.mark.asyncio
async def test_cancelled_action_transaction_is_rolled_back(tmp_path, monkeypatch) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(task_summary="task", config_hash="cfg")
        step_id = await database.create_step(session_id=session_id, step_number=0)

        async def cancel_after_begin(*, step_id: str, session_id: str) -> None:
            del step_id, session_id
            raise asyncio.CancelledError

        monkeypatch.setattr(database, "_require_step_session_ownership", cancel_after_begin)
        with pytest.raises(asyncio.CancelledError):
            await database.record_action(
                session_id=session_id,
                step_id=step_id,
                action_id=uuid4(),
                ordinal=0,
                action_type="wait",
                action_payload={"type": "wait", "seconds": 1},
                idempotency_class="retryable",
            )

        assert database.connection.in_transaction is False
        count = await (await database.connection.execute("SELECT COUNT(*) FROM actions")).fetchone()
        assert count[0] == 0


@pytest.mark.asyncio
async def test_crash_recovery_marks_dispatched_action_unknown(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(task_summary="task", config_hash="cfg")
        step_id = await database.create_step(session_id=session_id, step_number=0)
        action_id = uuid4()
        await database.record_action_with_firewall_decision(
            session_id=session_id,
            step_id=step_id,
            action_id=action_id,
            ordinal=0,
            action_type="open_tab",
            action_payload={"url": "https://example.com"},
            idempotency_class="non_retryable",
            raw_url="https://example.com",
            purpose="popup",
            allowed=True,
            reason_code="ALLOW_RULE_MATCH",
            policy_hash="policy",
        )
        await database.mark_dispatched(action_id)

        assert await database.recover_dispatched_actions() == 1
        stored = await database.get_action(action_id)
        assert stored is not None
        assert stored.state == "unknown"


@pytest.mark.asyncio
async def test_audit_and_firewall_records_are_immutable(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        event_id = await database.record_audit_event(
            event_type="configuration.loaded", payload={"token": "never-store-this"}
        )
        row = await (
            await database.connection.execute(
                "SELECT payload_json FROM audit_events WHERE id=?", (event_id,)
            )
        ).fetchone()
        assert "never-store-this" not in row["payload_json"]

        with pytest.raises(aiosqlite.DatabaseError, match="immutable"):
            await database.connection.execute(
                "UPDATE audit_events SET severity='error' WHERE id=?", (event_id,)
            )
            await database.connection.commit()
        await database.connection.rollback()


@pytest.mark.asyncio
async def test_failed_action_transaction_leaves_no_partial_rows(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        with pytest.raises(ValueError, match="does not belong"):
            await database.record_action_with_firewall_decision(
                session_id=uuid4(),
                step_id=uuid4(),
                action_id=uuid4(),
                ordinal=0,
                action_type="go_to_url",
                action_payload={"url": "https://example.com"},
                idempotency_class="navigation",
                raw_url="https://example.com",
                purpose="navigation",
                allowed=True,
                reason_code="ALLOW_RULE_MATCH",
                policy_hash="policy",
            )
        actions = await (
            await database.connection.execute("SELECT COUNT(*) FROM actions")
        ).fetchone()
        decisions = await (
            await database.connection.execute("SELECT COUNT(*) FROM firewall_decisions")
        ).fetchone()
        assert actions[0] == 0
        assert decisions[0] == 0


@pytest.mark.asyncio
async def test_concurrent_session_creation_is_serialized(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_ids = await asyncio.gather(
            *(
                database.create_session(
                    task_summary=f"task-{index}",
                    config_hash="cfg",
                    client_request_id=f"request-{index}",
                )
                for index in range(20)
            )
        )
        assert len(set(session_ids)) == 20
        count = await (
            await database.connection.execute("SELECT COUNT(*) FROM sessions")
        ).fetchone()
        assert count[0] == 20


def test_audit_url_sanitizer_removes_query_fragment_credentials_and_default_port() -> None:
    sanitized, raw_hash = sanitize_url_for_audit(
        "https://user:password@EXAMPLE.com:443/a/report?access_token=secret#fragment"
    )
    assert sanitized == "https://example.com/a/report"
    assert len(raw_hash) == 64
    assert "secret" not in raw_hash


@pytest.mark.asyncio
async def test_session_and_step_status_transitions_are_validated_and_audited(tmp_path) -> None:
    async with AxisDatabase(tmp_path / "axis.db") as database:
        session_id = await database.create_session(task_summary="task", config_hash="cfg")
        step_id = await database.create_step(session_id=session_id, step_number=1)

        assert await database.transition_session_status(session_id, "running") is True
        assert await database.transition_session_status(session_id, "running") is False
        assert await database.transition_step_status(step_id, "running") is True
        assert await database.transition_step_status(step_id, "succeeded") is True
        assert await database.transition_session_status(session_id, "succeeded") is True

        with pytest.raises(ValueError, match="invalid session"):
            await database.transition_session_status(session_id, "running")
        with pytest.raises(ValueError, match="invalid step"):
            await database.transition_step_status(step_id, "failed")

        rows = await database.connection.execute_fetchall(
            "SELECT event_type FROM audit_events ORDER BY sequence"
        )
        assert [row[0] for row in rows] == [
            "session.status",
            "step.status",
            "step.status",
            "session.status",
        ]
