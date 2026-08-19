from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlsplit
from uuid import UUID

from axis_agent.browser.base import BrowserObservation
from axis_agent.contracts.actions import Action, ActionResult
from axis_agent.firewall import NavigationPermit


class MockBrowserAdapter:
    """Deterministic adapter for standalone and contract testing."""

    def __init__(self, observation: BrowserObservation | None = None) -> None:
        self.started = False
        self.calls: list[tuple[str, Action]] = []
        self.navigation_permits: list[NavigationPermit | None] = []
        self.observation = observation or BrowserObservation(
            observation_id="0" * 32,
            page_id="mock-page-1",
            url="about:blank",
            title="Mock browser",
        )

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def observe(self) -> BrowserObservation:
        if not self.started:
            raise RuntimeError("mock browser is not started")
        return self.observation.model_copy(deep=True)

    async def execute(
        self,
        action: Action,
        *,
        action_id: str,
        navigation_permit: NavigationPermit | None = None,
        expected_observation_id: str | None = None,
        expected_page_id: str | None = None,
        expected_origin: str | None = None,
    ) -> ActionResult:
        if not self.started:
            raise RuntimeError("mock browser is not started")
        if (
            expected_observation_id is not None
            and expected_observation_id != self.observation.observation_id
        ):
            raise RuntimeError("stale browser observation")
        if expected_page_id is not None and expected_page_id != self.observation.page_id:
            raise RuntimeError("stale browser page")
        if expected_origin is not None:
            parsed = urlsplit(self.observation.url)
            hostname = parsed.hostname
            default_port = 443 if parsed.scheme == "https" else 80
            suffix = "" if parsed.port in {None, default_port} else f":{parsed.port}"
            actual_origin = None
            if parsed.scheme in {"http", "https"} and hostname is not None:
                actual_origin = f"{parsed.scheme}://{hostname.lower()}{suffix}"
            if expected_origin != actual_origin:
                raise RuntimeError("stale browser origin")
        if action.type in {"go_to_url", "open_tab", "search_google"} and navigation_permit is None:
            raise PermissionError("navigation action requires an AXIS navigation permit")
        self.calls.append((action_id, deepcopy(action)))
        self.navigation_permits.append(navigation_permit)
        return ActionResult(action_id=UUID(action_id), success=True, message=f"mock:{action.type}")
