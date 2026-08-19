"""Real-process qualification for the task-scoped AXIS Playwright MCP path."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from axis_agent.contracts.actions import WaitAction
from axis_agent.mcp.direct_client import TaskScopedPlaywrightMCP
from axis_agent.persistence import AxisDatabase

pytestmark = pytest.mark.playwright


@pytest.mark.asyncio
async def test_task_scoped_mcp_process_observes_and_dispatches_wait(tmp_path) -> None:
    """Exercise stdio, the real gateway, SQLite, dispatcher, and Chromium together."""

    database_path = tmp_path / "axis.db"
    policy_path = tmp_path / "firewall.json"
    downloads_path = tmp_path / "downloads"
    policy_path.write_text('{"rules":[]}', encoding="utf-8")

    session_id = uuid4()
    task_id = uuid4()
    step_id = uuid4()
    plan_id = uuid4()
    async with AxisDatabase(database_path) as database:
        assert (
            UUID(
                await database.create_session(
                    session_id=session_id,
                    task_id=task_id,
                    task_summary="controlled MCP process qualification",
                    config_hash="controlled-test-config",
                )
            )
            == session_id
        )
        await database.transition_session_status(session_id, "running")
        assert (
            UUID(
                await database.create_step(
                    session_id=session_id,
                    step_id=step_id,
                    step_number=1,
                )
            )
            == step_id
        )
        await database.transition_step_status(step_id, "running")

    browser = TaskScopedPlaywrightMCP(
        database_path=database_path,
        firewall_policy_path=policy_path,
        downloads_path=downloads_path,
        timeout_seconds=60,
    )
    try:
        await browser.bind_step(
            session_id=session_id,
            task_id=task_id,
            step_id=step_id,
            plan_id=plan_id,
            step_key="wait_briefly",
            allowed_action_types=("wait",),
        )
        observation = await browser.observe()
        result = await browser.execute(WaitAction(seconds=0))

        assert observation.url == "about:blank"
        assert observation.page_id
        assert result.success is True
    finally:
        await browser.close()

    async with AxisDatabase(database_path) as database:
        row = await (
            await database.connection.execute(
                "SELECT action_type, state FROM actions WHERE step_id=?",
                (str(step_id),),
            )
        ).fetchone()
        assert row is not None
        assert tuple(row) == ("wait", "succeeded")
