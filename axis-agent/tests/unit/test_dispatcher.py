from __future__ import annotations

import hashlib
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from axis_agent.browser import MockBrowserAdapter
from axis_agent.contracts.actions import (
    APPROVAL_REQUIRED_ACTION_TYPES,
    ActionCommand,
    GoToUrlAction,
    SearchGoogleAction,
    parse_production_action,
)
from axis_agent.firewall import (
    FirewallService,
    NavigationPermitStore,
    RuleEffect,
    policy_from_hosts,
)
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import ActionDispatcher

PUBLIC_IP = "93.184.216.34"


async def public_resolver(_: str, __: int) -> Sequence[str]:
    return (PUBLIC_IP,)


async def build_command(database: AxisDatabase, url: str) -> ActionCommand:
    task_id = uuid4()
    requested_session_id = uuid4()
    session_id = await database.create_session(
        session_id=requested_session_id,
        task_id=task_id,
        task_summary="Standalone dispatcher test",
        config_hash="config-hash",
    )
    step_id = await database.create_step(session_id=session_id, step_number=0)
    return ActionCommand(
        session_id=UUID(session_id),
        task_id=task_id,
        step_id=UUID(step_id),
        ordinal=0,
        action=GoToUrlAction(url=url),
    )


async def build_search_command(database: AxisDatabase, query: str) -> ActionCommand:
    task_id = uuid4()
    requested_session_id = uuid4()
    session_id = await database.create_session(
        session_id=requested_session_id,
        task_id=task_id,
        task_summary="Standalone search dispatcher test",
        config_hash="config-hash",
    )
    step_id = await database.create_step(session_id=session_id, step_number=0)
    return ActionCommand(
        session_id=UUID(session_id),
        task_id=task_id,
        step_id=UUID(step_id),
        ordinal=0,
        action=SearchGoogleAction(query=query),
    )


async def build_production_command(
    database: AxisDatabase, action_payload: dict[str, object]
) -> ActionCommand:
    task_id = uuid4()
    session_id = await database.create_session(
        task_id=task_id,
        task_summary="Production action dispatcher test",
        config_hash="config-hash",
    )
    step_id = await database.create_step(session_id=session_id, step_number=0)
    return ActionCommand(
        session_id=UUID(session_id),
        task_id=task_id,
        step_id=UUID(step_id),
        ordinal=0,
        expected_observation_id="0" * 32,
        expected_page_id="mock-page-1",
        action=parse_production_action(action_payload),
    )


@pytest.mark.asyncio
async def test_blocked_action_is_audited_without_browser_invocation(tmp_path) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_command(database, "https://blocked.example")
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy_from_hosts([]), resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
        )

        result = await dispatcher.dispatch_url_action(command)

        assert result.success is False
        assert result.error_code == "DENY_NO_ALLOW_MATCH"
        assert browser.calls == []
        counts = await (
            await database.connection.execute(
                "SELECT (SELECT COUNT(*) FROM actions), "
                "(SELECT COUNT(*) FROM firewall_decisions), "
                "(SELECT COUNT(*) FROM audit_events)"
            )
        ).fetchone()
        assert tuple(counts) == (1, 1, 1)
    await browser.stop()


@pytest.mark.asyncio
async def test_allowed_action_executes_exactly_once_with_bound_permit(tmp_path) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_command(database, "https://example.com")
        policy = policy_from_hosts(["example.com"])
        assert policy.rules[0].effect is RuleEffect.ALLOW
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy, resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
        )

        first = await dispatcher.dispatch_url_action(command)
        duplicate = await dispatcher.dispatch_url_action(command)

        assert first.success is True
        assert duplicate.success is True
        assert len(browser.calls) == 1
        assert len(browser.navigation_permits) == 1
        permit = browser.navigation_permits[0]
        assert permit is not None
        assert permit.action_id == command.action_id
        stored = await database.get_action(command.action_id)
        assert stored is not None
        assert stored.state == "succeeded"
    await browser.stop()


@pytest.mark.asyncio
async def test_allowed_search_authorizes_constructed_url_and_executes_once(
    tmp_path,
) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_search_command(database, "AXIS browser agent & security")
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(
                policy_from_hosts(["search.example"]), resolver=public_resolver
            ),
            database=database,
            permits=NavigationPermitStore(),
            search_base_url="https://search.example/find?q=",
        )

        first = await dispatcher.dispatch_url_action(command)
        duplicate = await dispatcher.dispatch_url_action(command)

        assert first.success is True
        assert duplicate.success is True
        assert len(browser.calls) == 1
        assert browser.calls[0][1] == command.action
        assert len(browser.navigation_permits) == 1
        permit = browser.navigation_permits[0]
        assert permit is not None
        assert permit.action_id == command.action_id
        assert permit.purpose.value == "navigation"
        assert (
            permit.canonical_url == "https://search.example/find?q=AXIS+browser+agent+%26+security"
        )

        expected_url = "https://search.example/find?q=AXIS+browser+agent+%26+security"
        row = await (
            await database.connection.execute(
                "SELECT sanitized_url, raw_url_hash, purpose, allowed "
                "FROM firewall_decisions WHERE action_id = ?",
                (str(command.action_id),),
            )
        ).fetchone()
        assert row is not None
        assert row["sanitized_url"] == "https://search.example/find"
        assert row["raw_url_hash"] == hashlib.sha256(expected_url.encode()).hexdigest()
        assert row["purpose"] == "navigation"
        assert row["allowed"] == 1
        stored = await database.get_action(command.action_id)
        assert stored is not None
        assert stored.state == "succeeded"
    await browser.stop()


@pytest.mark.asyncio
async def test_blocked_search_is_persisted_without_browser_execution(tmp_path) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_search_command(database, "blocked search")
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy_from_hosts([]), resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
            search_base_url="https://search.example/find?q=",
        )

        result = await dispatcher.dispatch_url_action(command)

        assert result.success is False
        assert result.error_code == "DENY_NO_ALLOW_MATCH"
        assert browser.calls == []
        expected_url = "https://search.example/find?q=blocked+search"
        row = await (
            await database.connection.execute(
                "SELECT sanitized_url, raw_url_hash, purpose, allowed "
                "FROM firewall_decisions WHERE action_id = ?",
                (str(command.action_id),),
            )
        ).fetchone()
        assert row is not None
        assert row["sanitized_url"] == "https://search.example/find"
        assert row["raw_url_hash"] == hashlib.sha256(expected_url.encode()).hexdigest()
        assert row["purpose"] == "navigation"
        assert row["allowed"] == 0
    await browser.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "go_back"},
        {"type": "click_element", "index": 0},
        {"type": "input_text", "index": 0, "text": "redacted value"},
        {"type": "switch_tab", "tabId": "mock-page-1"},
        {"type": "close_tab", "tabId": "mock-page-1"},
        {"type": "scroll_to_percent", "yPercent": 50},
        {"type": "scroll_to_top"},
        {"type": "scroll_to_bottom"},
        {"type": "scroll_to_text", "text": "heading"},
        {"type": "send_keys", "keys": "PageDown"},
        {"type": "get_dropdown_options", "index": 0},
        {"type": "select_dropdown_option", "index": 0, "text": "Option"},
        {"type": "wait", "seconds": 0},
    ],
)
async def test_every_non_url_production_action_uses_dispatcher_and_audit(
    tmp_path, payload: dict[str, object]
) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_production_command(database, payload)
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy_from_hosts([]), resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
        )

        result = await dispatcher.dispatch(command)

        stored = await database.get_action(command.action_id)
        if command.action.type in APPROVAL_REQUIRED_ACTION_TYPES:
            assert result.success is False
            assert result.error_code == "APPROVAL_REQUIRED"
            assert browser.calls == []
            assert stored is None
        else:
            assert result.success is True
            assert len(browser.calls) == 1
            assert stored is not None
            assert stored.state == "succeeded"
        decision_count = await (
            await database.connection.execute("SELECT COUNT(*) FROM firewall_decisions")
        ).fetchone()
        assert decision_count[0] == 0
    await browser.stop()


async def test_stale_observation_fails_before_persistence_and_browser_execution(tmp_path) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_production_command(database, {"type": "wait", "seconds": 0})
        command = command.model_copy(update={"expected_observation_id": "f" * 32})
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy_from_hosts([]), resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
        )

        result = await dispatcher.dispatch(command)

        assert result.success is False
        assert result.error_code == "STALE_OBSERVATION"
        assert browser.calls == []
        assert await database.get_action(command.action_id) is None
        audit_count = await (
            await database.connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE event_type='action.precondition_rejected'"
            )
        ).fetchone()
        assert audit_count[0] == 1
    await browser.stop()


async def test_mismatched_task_session_step_context_fails_before_execution(tmp_path) -> None:
    browser = MockBrowserAdapter()
    await browser.start()
    async with AxisDatabase(tmp_path / "axis.db") as database:
        command = await build_production_command(database, {"type": "wait", "seconds": 0})
        command = command.model_copy(update={"task_id": uuid4()})
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=FirewallService(policy_from_hosts([]), resolver=public_resolver),
            database=database,
            permits=NavigationPermitStore(),
        )

        result = await dispatcher.dispatch(command)

        assert result.success is False
        assert result.error_code == "INVALID_EXECUTION_CONTEXT"
        assert browser.calls == []
        assert await database.get_action(command.action_id) is None
    await browser.stop()
