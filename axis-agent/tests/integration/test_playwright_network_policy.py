"""Defense-in-depth tests for browser-initiated URL firewall enforcement."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from axis_agent.contracts.actions import ClickElementAction
from axis_agent.firewall import UrlPurpose

pytestmark = pytest.mark.playwright


def _evaluated(firewall: Any, path: str, purpose: UrlPurpose, *, allowed: bool) -> bool:
    return any(
        path in item.url and item.purpose is purpose and item.allowed is allowed
        for item in firewall.evaluations
    )


def _index_named(observation: Any, name: str) -> int:
    matches = [
        item
        for item in observation.interactive_elements
        if item.get("name") == name or item.get("text") == name
    ]
    assert len(matches) == 1, observation.interactive_elements
    return int(matches[0]["index"])


async def _wait_for_policy_evaluations(firewall: Any, count: int) -> None:
    """Wait for asynchronous page fetches without relying on network-idle."""

    for _ in range(100):
        if len(firewall.evaluations) >= count:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"expected at least {count} firewall evaluations")


@pytest.mark.asyncio
async def test_redirect_is_evaluated_and_blocked_before_target_request(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    controlled_http_site.clear_requests()
    async with playwright_adapter_factory(blocked_paths={"/redirect-target"}) as driver:
        result = await driver.navigate(controlled_http_site.url("/redirect"))

        assert result.success is False
        assert _evaluated(driver.firewall, "/redirect-target", UrlPurpose.REDIRECT, allowed=False)
        assert "/redirect-target" not in controlled_http_site.paths()


@pytest.mark.asyncio
async def test_iframe_subresource_and_fetch_are_checked_by_purpose(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    controlled_http_site.clear_requests()
    blocked_paths = {"/frame-content", "/blocked-subresource.js", "/blocked-api"}
    async with playwright_adapter_factory(blocked_paths=blocked_paths) as driver:
        assert (await driver.navigate(controlled_http_site.url("/network"))).success
        await _wait_for_policy_evaluations(driver.firewall, 8)
        paths = controlled_http_site.paths()

        assert "/subresource.js" in paths
        assert "/api/data" in paths
        assert "/frame-content" not in paths
        assert "/blocked-subresource.js" not in paths
        assert "/blocked-api" not in paths
        assert _evaluated(driver.firewall, "/frame-content", UrlPurpose.IFRAME, allowed=False)
        assert _evaluated(
            driver.firewall, "/blocked-subresource.js", UrlPurpose.SUBRESOURCE, allowed=False
        )
        assert _evaluated(driver.firewall, "/blocked-api", UrlPurpose.API, allowed=False)


@pytest.mark.asyncio
async def test_popup_is_blocked_before_popup_document_request(
    playwright_adapter_factory: Callable[..., Any], controlled_http_site: Any
) -> None:
    controlled_http_site.clear_requests()
    async with playwright_adapter_factory(blocked_paths={"/popup"}) as driver:
        await driver.navigate(controlled_http_site.url("/actions"))
        observation = await driver.adapter.observe()
        result = await driver.execute(
            ClickElementAction(index=_index_named(observation, "Open popup"))
        )

        assert result.success is False
        assert _evaluated(driver.firewall, "/popup", UrlPurpose.POPUP, allowed=False)
        assert "/popup" not in controlled_http_site.paths()
        assert len((await driver.adapter.observe()).tabs) == 1


@pytest.mark.asyncio
async def test_download_is_blocked_before_response_body_and_file_write(
    playwright_adapter_factory: Callable[..., Any],
    controlled_http_site: Any,
    empty_download_directory: Any,
) -> None:
    controlled_http_site.clear_requests()
    async with playwright_adapter_factory(blocked_paths={"/download"}) as driver:
        await driver.navigate(controlled_http_site.url("/actions"))
        observation = await driver.adapter.observe()
        result = await driver.execute(
            ClickElementAction(index=_index_named(observation, "Download fixture"))
        )

        assert result.success is False
        assert _evaluated(driver.firewall, "/download", UrlPurpose.DOWNLOAD, allowed=False)
        assert "/download" not in controlled_http_site.paths()
        assert list(empty_download_directory.iterdir()) == []
