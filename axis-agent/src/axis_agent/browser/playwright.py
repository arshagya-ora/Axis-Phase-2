"""Direct Playwright browser adapter guarded by the AXIS URL firewall."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Coroutine
from typing import Any, cast
from urllib.parse import quote_plus, urljoin, urlsplit
from uuid import UUID, uuid4

from playwright.async_api import (
    APIResponse,
    Browser,
    BrowserContext,
    Download,
    Frame,
    Locator,
    Page,
    Playwright,
    Request,
    Route,
    ViewportSize,
    WebSocketRoute,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)

from axis_agent.browser.base import (
    BrowserAdapter,
    BrowserNetworkDecision,
    BrowserObservation,
    BrowserTab,
)
from axis_agent.browser.dom import (
    DOMLimits,
    DOMObservation,
    collect_interactive_elements,
    require_locator_for_index,
    sanitize_page_text,
)
from axis_agent.config import PlaywrightSettings
from axis_agent.contracts.actions import (
    Action,
    ActionResult,
    CacheContentAction,
    ClickElementAction,
    CloseTabAction,
    DoneAction,
    GetDropdownOptionsAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    NextPageAction,
    OpenTabAction,
    PreviousPageAction,
    ScrollToBottomAction,
    ScrollToPercentAction,
    ScrollToTextAction,
    ScrollToTopAction,
    SearchGoogleAction,
    SelectDropdownOptionAction,
    SendKeysAction,
    SwitchTabAction,
    WaitAction,
)
from axis_agent.firewall import (
    FirewallDecision,
    FirewallService,
    NavigationPermit,
    NavigationPermitStore,
    UrlPurpose,
)

_MAX_NETWORK_EVENTS = 1_000
_API_TYPES = frozenset({"fetch", "xhr", "eventsource"})
_SENSITIVE_AUTOCOMPLETE_TOKENS = frozenset(
    {
        "current-password",
        "new-password",
        "one-time-code",
        "cc-number",
        "cc-csc",
        "cc-exp",
        "cc-exp-month",
        "cc-exp-year",
    }
)
_SENSITIVE_FIELD_NAME = re.compile(
    r"password|passwd|passcode|one[-_]?time|otp|secret|api[-_]?key|auth[-_]?token|"
    r"access[-_]?token|card[-_]?number|cc[-_]?(?:number|csc|cvv)|security[-_]?code",
    re.IGNORECASE,
)


class BrowserNotStartedError(RuntimeError):
    """Raised when an action is attempted before browser startup."""


class BrowserSecurityError(PermissionError):
    """Raised when a requested action is not backed by a valid AXIS permit."""


class BrowserPolicyDeniedError(PermissionError):
    """Raised when a browser-initiated action is rejected by URL policy."""


class DirectPlaywrightAdapter(BrowserAdapter):
    """AXIS-owned Playwright browser with fail-closed request interception."""

    def __init__(
        self,
        settings: PlaywrightSettings,
        firewall: FirewallService,
        permit_store: NavigationPermitStore,
    ) -> None:
        self._settings = settings
        self._firewall = firewall
        self._permit_store = permit_store
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._active_page: Page | None = None
        self._last_dom: DOMObservation | None = None
        self._cache: list[str] = []
        self._network_decisions: list[BrowserNetworkDecision] = []
        self._action_lock = asyncio.Lock()
        self._route_lock = asyncio.Lock()
        self._page_ids: dict[int, str] = {}
        self._registered_pages: set[int] = set()
        self._pending_popup_urls: set[str] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def network_decisions(self) -> tuple[BrowserNetworkDecision, ...]:
        return tuple(self._network_decisions)

    @property
    def cached_content(self) -> tuple[str, ...]:
        return tuple(self._cache)

    async def start(self) -> None:
        if self._context is not None:
            return
        self._settings.downloads_path.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        browser_type = getattr(self._playwright, self._settings.browser.value)
        viewport = ViewportSize(
            width=self._settings.viewport_width,
            height=self._settings.viewport_height,
        )
        try:
            if self._settings.user_data_dir is not None:
                self._settings.user_data_dir.mkdir(parents=True, exist_ok=True)
                self._context = await browser_type.launch_persistent_context(
                    str(self._settings.user_data_dir),
                    headless=self._settings.headless,
                    accept_downloads=False,
                    downloads_path=str(self._settings.downloads_path),
                    service_workers="block",
                    bypass_csp=False,
                    ignore_https_errors=False,
                    viewport=viewport,
                    locale=self._settings.locale,
                    timezone_id=self._settings.timezone_id,
                    slow_mo=self._settings.slow_mo_ms,
                )
            else:
                self._browser = await browser_type.launch(
                    headless=self._settings.headless,
                    downloads_path=str(self._settings.downloads_path),
                    slow_mo=self._settings.slow_mo_ms,
                )
                self._context = await self._browser.new_context(
                    accept_downloads=False,
                    service_workers="block",
                    bypass_csp=False,
                    ignore_https_errors=False,
                    viewport=viewport,
                    locale=self._settings.locale,
                    timezone_id=self._settings.timezone_id,
                )
            self._context.set_default_timeout(self._settings.default_timeout_ms)
            self._context.set_default_navigation_timeout(self._settings.navigation_timeout_ms)
            await self._context.route("**/*", self._handle_route)
            await self._context.route_web_socket("**/*", self._handle_websocket)
            self._context.on("page", self._on_page)
            self._context.on("download", self._on_download)
            for page in self._context.pages:
                self._register_page(page)
            if not self._context.pages:
                self._active_page = await self._context.new_page()
                self._register_page(self._active_page)
            else:
                self._active_page = self._context.pages[-1]
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        context = self._context
        browser = self._browser
        playwright = self._playwright
        self._context = None
        self._browser = None
        self._playwright = None
        self._active_page = None
        self._last_dom = None
        tasks = tuple(self._background_tasks)
        self._background_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if context is not None:
            with contextlib.suppress(Exception):
                await context.close()
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        if playwright is not None:
            with contextlib.suppress(Exception):
                await playwright.stop()
        self._page_ids.clear()
        self._registered_pages.clear()
        self._pending_popup_urls.clear()

    async def observe(self) -> BrowserObservation:
        async with self._action_lock:
            return await self._observe_locked()

    async def _observe_locked(self) -> BrowserObservation:
        page = self._require_page()
        self._last_dom = await collect_interactive_elements(page, limits=DOMLimits())
        try:
            visible_text = await page.locator("body").inner_text(
                timeout=self._settings.default_timeout_ms
            )
        except PlaywrightError:
            visible_text = ""
        tabs = [
            BrowserTab(
                page_id=self._page_id(tab),
                url=self._safe_observation_url(tab.url),
                title=await self._safe_title(tab),
                active=tab is page,
            )
            for tab in self._require_context().pages
            if not tab.is_closed()
        ]
        return BrowserObservation(
            observation_id=self._last_dom.observation_id,
            page_id=self._page_id(page),
            url=self._safe_observation_url(page.url),
            title=await self._safe_title(page),
            visible_text=self._redact_visible_text(visible_text),
            interactive_elements=self._last_dom.as_browser_metadata(),
            tabs=tabs,
        )

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
        async with self._action_lock:
            self._validate_execution_preconditions(
                expected_observation_id=expected_observation_id,
                expected_page_id=expected_page_id,
                expected_origin=expected_origin,
            )
            return await self._execute_locked(
                action,
                action_id=action_id,
                navigation_permit=navigation_permit,
            )

    async def _execute_locked(
        self,
        action: Action,
        *,
        action_id: str,
        navigation_permit: NavigationPermit | None,
    ) -> ActionResult:
        try:
            parsed_action_id = UUID(action_id)
        except ValueError as exc:
            raise BrowserSecurityError("action_id must be a UUID") from exc
        page = self._require_page()
        try:
            data = await self._execute_action(
                page,
                action,
                action_id=parsed_action_id,
                navigation_permit=navigation_permit,
            )
            return ActionResult(
                action_id=parsed_action_id,
                success=True,
                message=f"Executed {action.type}",
                data=data,
            )
        except BrowserSecurityError:
            raise
        except BrowserPolicyDeniedError as exc:
            return ActionResult(
                action_id=parsed_action_id,
                success=False,
                message=str(exc),
                error_code="BROWSER_POLICY_DENIED",
                retryable=False,
            )
        except Exception as exc:
            return ActionResult(
                action_id=parsed_action_id,
                success=False,
                message=f"Browser action failed: {action.type}",
                data={"exceptionType": type(exc).__name__},
                error_code="PLAYWRIGHT_ACTION_FAILED",
                retryable=False,
            )

    async def _execute_action(
        self,
        page: Page,
        action: Action,
        *,
        action_id: UUID,
        navigation_permit: NavigationPermit | None,
    ) -> dict[str, object]:
        if isinstance(action, DoneAction):
            return {"done": True, "success": action.success}
        if isinstance(action, GoToUrlAction):
            await self._navigate(
                page, action.url, action_id, navigation_permit, UrlPurpose.NAVIGATION
            )
            return {"url": self._safe_observation_url(page.url)}
        if isinstance(action, SearchGoogleAction):
            url = f"{self._settings.search_base_url}{quote_plus(action.query)}"
            await self._navigate(page, url, action_id, navigation_permit, UrlPurpose.NAVIGATION)
            return {"url": self._safe_observation_url(page.url)}
        if isinstance(action, OpenTabAction):
            context = self._require_context()
            new_page = await context.new_page()
            self._register_page(new_page)
            self._active_page = new_page
            try:
                await self._navigate(
                    new_page, action.url, action_id, navigation_permit, UrlPurpose.POPUP
                )
            except Exception:
                await new_page.close()
                self._active_page = page
                raise
            return {"pageId": self._page_id(new_page)}
        if isinstance(action, GoBackAction):
            decision_cursor = len(self._network_decisions)
            await page.go_back(wait_until="domcontentloaded")
            self._raise_for_new_denial(decision_cursor)
            self._last_dom = None
            return {"url": self._safe_observation_url(page.url)}
        if isinstance(action, ClickElementAction):
            locator = await self._require_element(page, action.index)
            await self._click_with_policy(page, locator)
            self._last_dom = None
            return {"index": action.index}
        if isinstance(action, InputTextAction):
            locator = await self._require_element(page, action.index)
            await self._require_non_sensitive_text_target(locator)
            await locator.fill(action.text)
            return {"index": action.index}
        if isinstance(action, SendKeysAction):
            await page.keyboard.press(action.keys)
            return {"keys": action.keys}
        if isinstance(action, SwitchTabAction):
            selected = self._page_by_id(action.tab_id)
            self._active_page = selected
            await selected.bring_to_front()
            self._last_dom = None
            return {"pageId": self._page_id(selected)}
        if isinstance(action, CloseTabAction):
            selected = self._page_by_id(action.tab_id)
            await selected.close()
            pages = [
                candidate
                for candidate in self._require_context().pages
                if not candidate.is_closed()
            ]
            if not pages:
                self._active_page = await self._require_context().new_page()
                self._register_page(self._active_page)
            else:
                self._active_page = pages[-1]
            self._last_dom = None
            return {"closedPageId": action.tab_id}
        if isinstance(action, CacheContentAction):
            remaining = self._settings.max_cached_content_chars - sum(map(len, self._cache))
            if remaining <= 0:
                raise ValueError("content cache limit reached")
            content = sanitize_page_text(action.content, maximum=remaining)
            self._cache.append(content)
            return {"cachedCharacters": len(content)}
        if isinstance(action, ScrollToPercentAction):
            if action.index is None:
                await page.evaluate(
                    "percent => window.scrollTo({top: "
                    "(document.documentElement.scrollHeight - window.innerHeight) "
                    "* percent / 100, behavior: 'instant'})",
                    action.y_percent,
                )
            else:
                locator = await self._require_element(page, action.index)
                await locator.evaluate(
                    "(element, percent) => { element.scrollTop = "
                    "(element.scrollHeight - element.clientHeight) * percent / 100; }",
                    action.y_percent,
                )
            return {"percent": action.y_percent}
        if isinstance(action, ScrollToTopAction):
            await self._scroll_edge(page, action.index, to_bottom=False)
            return {}
        if isinstance(action, ScrollToBottomAction):
            await self._scroll_edge(page, action.index, to_bottom=True)
            return {}
        if isinstance(action, PreviousPageAction):
            await self._scroll_page(page, action.index, direction=-1)
            return {}
        if isinstance(action, NextPageAction):
            await self._scroll_page(page, action.index, direction=1)
            return {}
        if isinstance(action, ScrollToTextAction):
            locator = page.get_by_text(action.text, exact=False).nth(action.nth - 1)
            await locator.scroll_into_view_if_needed()
            return {"occurrence": action.nth}
        if isinstance(action, GetDropdownOptionsAction):
            locator = await self._require_element(page, action.index)
            options = await locator.locator("option").all_text_contents()
            safe_options = [sanitize_page_text(option, maximum=240) for option in options[:200]]
            return {"options": safe_options}
        if isinstance(action, SelectDropdownOptionAction):
            locator = await self._require_element(page, action.index)
            await locator.select_option(label=action.text)
            # Do not return underlying option values; sites sometimes encode
            # identifiers or tokens separately from the visible label.
            return {"selected": True}
        if isinstance(action, WaitAction):
            await asyncio.sleep(action.seconds)
            return {"seconds": action.seconds}
        raise ValueError(f"unsupported action: {action.type}")

    async def _navigate(
        self,
        page: Page,
        url: str,
        action_id: UUID,
        permit: NavigationPermit | None,
        purpose: UrlPurpose,
    ) -> None:
        decision = await self._firewall.evaluate(url, purpose=purpose)
        if not decision.allowed or decision.canonical_url is None:
            raise BrowserSecurityError(f"navigation denied: {decision.reason.value}")
        if permit is None or not self._permit_store.consume(
            permit,
            action_id=action_id,
            canonical_url=decision.canonical_url,
            purpose=purpose,
            policy_hash=decision.policy_hash,
        ):
            raise BrowserSecurityError("navigation permit is missing, stale, or mismatched")
        if purpose is UrlPurpose.POPUP:
            self._pending_popup_urls.add(decision.canonical_url)
        decision_cursor = len(self._network_decisions)
        try:
            await page.goto(decision.canonical_url, wait_until="domcontentloaded")
            self._raise_for_new_denial(decision_cursor)
        finally:
            self._pending_popup_urls.discard(decision.canonical_url)
            self._last_dom = None

    async def _handle_route(self, route: Route, request: Request) -> None:
        """Validate every browser request before it reaches the network."""

        async with self._route_lock:
            try:
                purpose = self._classify_request(request)
                decision = await self._firewall.evaluate(request.url, purpose=purpose)
                self._record_network_decision(request, purpose, decision)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Request classification and policy failures are security
                # failures. No network request is allowed on ambiguity.
                with contextlib.suppress(PlaywrightError):
                    await route.abort("blockedbyclient")
                return

            if not decision.allowed:
                await route.abort("blockedbyclient")
                return
            response: APIResponse | None = None
            try:
                # Playwright does not re-run a continued route handler for all
                # redirect hops. Fetch exactly one response without following
                # redirects, validate its Location, and only then expose the
                # response to Chromium. Each fulfilled redirect becomes a new
                # routed browser request and is evaluated again.
                response = await route.fetch(
                    max_redirects=0,
                    max_retries=0,
                    timeout=self._settings.navigation_timeout_ms,
                )
                location = response.headers.get("location")
                if 300 <= response.status < 400 and location:
                    redirect_url = urljoin(request.url, location)
                    redirect_decision = await self._firewall.evaluate(
                        redirect_url,
                        purpose=UrlPurpose.REDIRECT,
                    )
                    self._record_url_decision(
                        redirect_url,
                        UrlPurpose.REDIRECT,
                        redirect_decision,
                        resource_type=request.resource_type,
                    )
                    if not redirect_decision.allowed:
                        await route.abort("blockedbyclient")
                        return
                content_disposition = response.headers.get("content-disposition", "").lower()
                if "attachment" in content_disposition:
                    download_decision = await self._firewall.evaluate(
                        request.url,
                        purpose=UrlPurpose.DOWNLOAD,
                    )
                    self._record_url_decision(
                        request.url,
                        UrlPurpose.DOWNLOAD,
                        download_decision,
                        resource_type="download",
                    )
                    # AXIS has no user-approved download capability, even
                    # when the destination itself is policy-allowed.
                    await route.abort("blockedbyclient")
                    return
                await route.fulfill(response=response)
            except asyncio.CancelledError:
                raise
            except Exception:
                with contextlib.suppress(PlaywrightError):
                    await route.abort("blockedbyclient")
            finally:
                if response is not None:
                    with contextlib.suppress(PlaywrightError):
                        await response.dispose()

    def _classify_request(self, request: Request) -> UrlPurpose:
        if request.redirected_from is not None:
            return UrlPurpose.REDIRECT
        if request.is_navigation_request():
            if request.url in self._pending_popup_urls:
                return UrlPurpose.POPUP
            try:
                frame = request.frame
            except PlaywrightError:
                # Playwright exposes a popup's first document request before
                # its Frame/Page object exists.
                return UrlPurpose.POPUP
            if frame.parent_frame is not None:
                return UrlPurpose.IFRAME
            return UrlPurpose.NAVIGATION
        if request.resource_type in _API_TYPES:
            return UrlPurpose.API
        return UrlPurpose.SUBRESOURCE

    def _record_network_decision(
        self, request: Request, purpose: UrlPurpose, decision: FirewallDecision
    ) -> None:
        self._record_url_decision(
            request.url,
            purpose,
            decision,
            resource_type=request.resource_type,
        )

    def _record_url_decision(
        self,
        url: str,
        purpose: UrlPurpose,
        decision: FirewallDecision,
        *,
        resource_type: str,
    ) -> None:
        self._network_decisions.append(
            BrowserNetworkDecision(
                url=self._safe_observation_url(url),
                purpose=purpose.value,
                allowed=decision.allowed,
                reason_code=decision.reason.value,
                resource_type=resource_type,
            )
        )
        if len(self._network_decisions) > _MAX_NETWORK_EVENTS:
            del self._network_decisions[: len(self._network_decisions) - _MAX_NETWORK_EVENTS]

    async def _require_element(self, page: Page, index: int) -> Locator:
        if self._last_dom is None:
            raise ValueError("collect a fresh browser observation before using an element index")
        locator = await require_locator_for_index(page, self._last_dom, index)
        return cast(Locator, locator)

    @staticmethod
    async def _require_non_sensitive_text_target(locator: Locator) -> None:
        """Prevent model-authored text from entering credential or payment fields."""

        input_type = (await locator.get_attribute("type") or "").strip().lower()
        autocomplete = (await locator.get_attribute("autocomplete") or "").strip().lower()
        name = (await locator.get_attribute("name") or "").strip()
        element_id = (await locator.get_attribute("id") or "").strip()
        autocomplete_tokens = frozenset(autocomplete.split())
        if (
            input_type in {"hidden", "password"}
            or autocomplete_tokens & _SENSITIVE_AUTOCOMPLETE_TOKENS
            or _SENSITIVE_FIELD_NAME.search(name) is not None
            or _SENSITIVE_FIELD_NAME.search(element_id) is not None
        ):
            raise BrowserPolicyDeniedError(
                "Credential and payment field entry is not supported by AXIS"
            )

    async def _click_with_policy(self, page: Page, locator: Locator) -> None:
        """Preflight known link/download targets and surface routed denials."""

        await self._preflight_click_target(page, locator)
        existing_pages = {id(candidate) for candidate in self._require_context().pages}
        decision_cursor = len(self._network_decisions)
        try:
            await locator.click(force=False)
            # Event handlers can complete immediately after locator.click.
            # Yield twice so routed popup requests publish their decision.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self._raise_for_new_denial(decision_cursor)
        except Exception:
            await self._close_pages_created_after(existing_pages)
            self._active_page = page
            raise

    async def _preflight_click_target(self, page: Page, locator: Locator) -> None:
        raw_href = await locator.get_attribute("href")
        if raw_href is None:
            return
        target_url = urljoin(page.url, raw_href)
        try:
            parsed = urlsplit(target_url)
        except ValueError as exc:
            raise BrowserPolicyDeniedError("Click target is not a valid URL") from exc
        if parsed.scheme.lower() not in {"http", "https"} or parsed.hostname is None:
            raise BrowserPolicyDeniedError("Click target uses a blocked URL scheme")

        download = await locator.get_attribute("download")
        target = (await locator.get_attribute("target") or "").lower()
        if download is not None:
            purpose = UrlPurpose.DOWNLOAD
        elif target == "_blank":
            purpose = UrlPurpose.POPUP
        else:
            purpose = UrlPurpose.NAVIGATION

        decision = await self._firewall.evaluate(target_url, purpose=purpose)
        self._record_url_decision(target_url, purpose, decision, resource_type="document")
        if not decision.allowed:
            raise BrowserPolicyDeniedError(
                f"Click target denied by AXIS firewall: {decision.reason.value}"
            )
        if purpose is UrlPurpose.DOWNLOAD:
            raise BrowserPolicyDeniedError(
                "Downloads are disabled until an approved download capability is enabled"
            )

    def _raise_for_new_denial(self, cursor: int) -> None:
        action_fatal_purposes = {
            UrlPurpose.NAVIGATION.value,
            UrlPurpose.REDIRECT.value,
            UrlPurpose.POPUP.value,
            UrlPurpose.DOWNLOAD.value,
        }
        denial = next(
            (
                decision
                for decision in self._network_decisions[cursor:]
                if not decision.allowed and decision.purpose in action_fatal_purposes
            ),
            None,
        )
        if denial is not None:
            raise BrowserPolicyDeniedError(
                f"Browser request denied by AXIS firewall: {denial.reason_code}"
            )

    async def _close_pages_created_after(self, existing_page_ids: set[int]) -> None:
        for candidate in tuple(self._require_context().pages):
            if id(candidate) not in existing_page_ids and not candidate.is_closed():
                with contextlib.suppress(PlaywrightError):
                    await candidate.close(run_before_unload=False)

    async def _scroll_edge(self, page: Page, index: int | None, *, to_bottom: bool) -> None:
        if index is None:
            if to_bottom:
                await page.evaluate(
                    "() => window.scrollTo({top: document.documentElement.scrollHeight})"
                )
            else:
                await page.evaluate("() => window.scrollTo({top: 0})")
            return
        locator = await self._require_element(page, index)
        if to_bottom:
            await locator.evaluate("element => { element.scrollTop = element.scrollHeight; }")
        else:
            await locator.evaluate("element => { element.scrollTop = 0; }")

    async def _scroll_page(self, page: Page, index: int | None, *, direction: int) -> None:
        if index is None:
            await page.evaluate("dy => window.scrollBy(0, dy * window.innerHeight)", direction)
            return
        locator = await self._require_element(page, index)
        await locator.evaluate(
            "(element, direction) => { element.scrollTop += direction * element.clientHeight; }",
            direction,
        )

    def _on_page(self, page: Page) -> None:
        self._register_page(page)
        self._active_page = page

    def _on_download(self, download: Download) -> None:
        self._schedule(download.cancel())

    async def _handle_websocket(self, websocket: WebSocketRoute) -> None:
        """Block all WebSockets; the current policy has no WebSocket capability."""

        try:
            decision = await self._firewall.evaluate(
                websocket.url,
                purpose=UrlPurpose.API,
            )
            self._record_url_decision(
                websocket.url,
                UrlPurpose.API,
                decision,
                resource_type="websocket",
            )
        finally:
            # Routed WebSockets do not connect unless connect_to_server is
            # called. Closing makes the fail-closed behavior explicit.
            with contextlib.suppress(PlaywrightError):
                await websocket.close(code=1008, reason="Blocked by AXIS policy")

    def _schedule(self, operation: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(operation)
        self._background_tasks.add(task)
        task.add_done_callback(self._finish_background_task)

    def _finish_background_task(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.result()

    def _register_page(self, page: Page) -> None:
        page_key = id(page)
        if page_key not in self._page_ids:
            self._page_ids[page_key] = str(uuid4())
        if page_key in self._registered_pages:
            return
        self._registered_pages.add(page_key)
        page.on("framenavigated", self._on_frame_navigated)

    def _on_frame_navigated(self, frame: Frame) -> None:
        if frame == frame.page.main_frame:
            self._last_dom = None

    def _page_id(self, page: Page) -> str:
        value = self._page_ids.get(id(page))
        if value is None:
            self._register_page(page)
            value = self._page_ids[id(page)]
        return value

    def _page_by_id(self, page_id: str) -> Page:
        for page in self._require_context().pages:
            if self._page_id(page) == page_id:
                return page
        raise LookupError("browser page no longer exists")

    def _require_page(self) -> Page:
        if self._active_page is None or self._active_page.is_closed():
            raise BrowserNotStartedError("Playwright browser is not started")
        return self._active_page

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise BrowserNotStartedError("Playwright browser is not started")
        return self._context

    @staticmethod
    async def _safe_title(page: Page) -> str:
        with contextlib.suppress(Exception):
            return sanitize_page_text(await page.title(), maximum=512)
        return ""

    @staticmethod
    def _safe_observation_url(url: str) -> str:
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
                return url if url == "about:blank" else "[blocked-url]"
            host = parsed.hostname.lower()
            if ":" in host:
                host = f"[{host}]"
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{parsed.scheme.lower()}://{host}{port}{parsed.path or '/'}"
        except ValueError:
            return "[invalid-url]"

    def _validate_execution_preconditions(
        self,
        *,
        expected_observation_id: str | None,
        expected_page_id: str | None,
        expected_origin: str | None,
    ) -> None:
        """Recheck the trusted snapshot under the same lock used for execution."""

        page = self._require_page()
        if expected_observation_id is not None and (
            self._last_dom is None or self._last_dom.observation_id != expected_observation_id
        ):
            raise BrowserSecurityError("browser observation is stale")
        if expected_page_id is not None and self._page_id(page) != expected_page_id:
            raise BrowserSecurityError("browser page is stale")
        if expected_origin is not None and self._origin(page.url) != expected_origin:
            raise BrowserSecurityError("browser origin is stale")

    @staticmethod
    def _origin(url: str) -> str | None:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https"} or hostname is None:
            return None
        default_port = 443 if parsed.scheme == "https" else 80
        suffix = "" if port in {None, default_port} else f":{port}"
        return f"{parsed.scheme}://{hostname.lower()}{suffix}"

    def _redact_visible_text(self, value: str) -> str:
        return sanitize_page_text(
            value,
            maximum=self._settings.max_observation_text_chars,
        )
