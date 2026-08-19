"""Real-browser integration tests for the DirectPlaywrightAdapter contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from axis_agent.contracts import ExecutionPlan, PlanStep
from axis_agent.contracts.actions import (
    ClickElementAction,
    CloseTabAction,
    GetDropdownOptionsAction,
    GoBackAction,
    InputTextAction,
    ScrollToBottomAction,
    ScrollToTextAction,
    ScrollToTopAction,
    SelectDropdownOptionAction,
    SendKeysAction,
    SwitchTabAction,
    WaitAction,
)
from axis_agent.mcp import AxisMCPService, MCPProfile
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import ActionDispatcher, ApprovalCoordinator

pytestmark = pytest.mark.playwright


class _AllowControlledClick:
    async def approve(self, request: Any, action: Any) -> bool:
        del request, action
        return True


def _element_index(observation: Any, *, element_id: str) -> int:
    """Resolve fixture controls through stable semantic DOM metadata."""

    expected: dict[str, tuple[str, str]] = {
        "name": ("Name", "input"),
        "password": ("Password", "input"),
        "copy": ("Copy value", "button"),
        "choice": ("Choice", "select"),
        "next": ("Next page", "a"),
    }
    name, tag = expected[element_id]
    matches = [
        item
        for item in observation.interactive_elements
        if (item.get("name") == name or item.get("text") == name) and item.get("tag") == tag
    ]
    assert len(matches) == 1, observation.interactive_elements
    return int(matches[0]["index"])


@pytest.mark.asyncio
async def test_observe_and_execute_form_actions(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    async with playwright_adapter_factory() as driver:
        navigated = await driver.navigate(controlled_http_site.url("/actions"))
        assert navigated.success

        observation = await driver.adapter.observe()
        assert observation.title == "AXIS action fixture"
        assert observation.url == controlled_http_site.url("/actions")
        assert "Controlled AXIS action page" in observation.visible_text
        assert observation.page_id

        password_index = _element_index(observation, element_id="password")
        credential_entry = await driver.execute(
            InputTextAction(index=password_index, text="must-not-be-entered")
        )
        assert credential_entry.success is False
        assert credential_entry.error_code == "BROWSER_POLICY_DENIED"

        name_index = _element_index(observation, element_id="name")
        copy_index = _element_index(observation, element_id="copy")

        typed = await driver.execute(InputTextAction(index=name_index, text="AXIS Phase 2"))
        clicked = await driver.execute(ClickElementAction(index=copy_index))
        # A click may mutate or navigate the document, so AXIS deliberately
        # invalidates observation-scoped element indices after every click.
        refreshed = await driver.adapter.observe()
        choice_index = _element_index(refreshed, element_id="choice")
        options = await driver.execute(GetDropdownOptionsAction(index=choice_index))
        selected = await driver.execute(SelectDropdownOptionAction(index=choice_index, text="Beta"))

        assert typed.success
        assert clicked.success
        assert options.success
        assert selected.success
        assert "Beta" in str(options.data)
        assert "AXIS Phase 2" in (await driver.adapter.observe()).visible_text


@pytest.mark.asyncio
async def test_keyboard_scroll_wait_and_history_actions(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    async with playwright_adapter_factory() as driver:
        await driver.navigate(controlled_http_site.url("/actions"))
        observation = await driver.adapter.observe()
        name_index = _element_index(observation, element_id="name")
        next_index = _element_index(observation, element_id="next")

        assert (await driver.execute(ClickElementAction(index=name_index))).success
        assert (await driver.execute(SendKeysAction(keys="End"))).success
        assert (await driver.execute(ScrollToTextAction(text="AXIS scroll target"))).success
        assert (await driver.execute(ScrollToTopAction())).success
        assert (await driver.execute(ScrollToBottomAction())).success
        assert (await driver.execute(WaitAction(seconds=0))).success
        # The earlier click deliberately invalidated observation-scoped DOM
        # markers. Re-observe before resolving a later element reference.
        refreshed = await driver.adapter.observe()
        next_index = _element_index(refreshed, element_id="next")
        assert (await driver.execute(ClickElementAction(index=next_index))).success
        assert (await driver.adapter.observe()).title == "AXIS next page"
        assert (await driver.execute(GoBackAction())).success
        assert (await driver.adapter.observe()).title == "AXIS action fixture"


@pytest.mark.asyncio
async def test_open_switch_and_close_tab(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    async with playwright_adapter_factory() as driver:
        await driver.navigate(controlled_http_site.url("/actions"))
        opened = await driver.open_tab(controlled_http_site.url("/popup"))
        assert opened.success

        observation = await driver.adapter.observe()
        assert len(observation.tabs) == 2
        popup = next(tab for tab in observation.tabs if tab.title == "AXIS popup")
        original = next(tab for tab in observation.tabs if tab.title == "AXIS action fixture")
        popup_tab_id = str(opened.data["pageId"])
        assert popup_tab_id == popup.page_id
        assert (await driver.execute(SwitchTabAction(tab_id=popup_tab_id))).success
        assert (await driver.adapter.observe()).page_id == popup.page_id
        assert (await driver.execute(CloseTabAction(tab_id=popup_tab_id))).success
        final = await driver.adapter.observe()
        assert len(final.tabs) == 1
        assert final.page_id == original.page_id


@pytest.mark.asyncio
async def test_mcp_dispatcher_uses_one_real_playwright_observation_lease(
    playwright_adapter_factory: Callable[..., Any],
    controlled_http_site: Any,
    tmp_path: Path,
) -> None:
    async with playwright_adapter_factory() as driver:
        assert (await driver.navigate(controlled_http_site.url("/actions"))).success
        async with AxisDatabase(tmp_path / "axis-mcp.db") as database:
            task_id = uuid4()
            session_id = UUID(
                await database.create_session(
                    task_id=task_id,
                    task_summary="Controlled MCP Playwright lease test",
                    config_hash="integration-test",
                )
            )
            step_id = UUID(await database.create_step(session_id=session_id, step_number=0))
            plan = ExecutionPlan(
                session_id=session_id,
                task_id=task_id,
                objective="Exercise one controlled Playwright observation lease",
                completion_criteria=("Controlled wait and click succeed",),
                steps=(
                    PlanStep(
                        key="lease",
                        order=1,
                        objective="Use one observation per action",
                        success_criteria=("Next page opens",),
                        allowed_action_types=("wait", "click_element"),
                    ),
                ),
            )
            await database.record_plan_metadata(plan)
            service = AxisMCPService(
                profile=MCPProfile.NAVIGATOR,
                browser=driver.adapter,
                dispatcher=ActionDispatcher(
                    browser=driver.adapter,
                    firewall=driver.firewall,
                    database=database,
                    permits=driver.permits,
                ),
                approval_gate=ApprovalCoordinator(
                    database=database,
                    plan_id=plan.plan_id,
                    step_key="lease",
                    handler=_AllowControlledClick(),
                ),
                allowed_action_types=("wait", "click_element"),
                session_id=session_id,
                task_id=task_id,
                step_id=step_id,
            )

            first = await service.browser_observe()
            first_id = first["observation"]["observationId"]  # type: ignore[index]
            waited = await service.browser_execute({"type": "wait", "seconds": 0})
            assert waited["result"]["success"] is True  # type: ignore[index]
            assert waited["command"]["expectedObservationId"] == first_id  # type: ignore[index]

            second = await service.browser_observe()
            elements = second["observation"]["interactiveElements"]  # type: ignore[index]
            next_element = next(
                item
                for item in elements  # type: ignore[union-attr]
                if item.get("text") == "Next page" and item.get("tag") == "a"
            )
            clicked = await service.browser_execute(
                {"type": "click_element", "index": next_element["index"]}
            )
            assert clicked["result"]["success"] is True, clicked  # type: ignore[index]
            assert (await driver.adapter.observe()).title == "AXIS next page"
