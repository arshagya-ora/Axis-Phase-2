"""Shared pytest configuration and controlled browser-integration fixtures.

The real AXIS firewall intentionally rejects loopback destinations.  The
``controlled_firewall_factory`` fixture below is therefore a *test-only*
subclass which grants one exact, randomly allocated test-server origin.  It
must never be imported by production code.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qs, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from axis_agent.contracts.actions import Action, ActionResult, GoToUrlAction, OpenTabAction
from axis_agent.firewall import (
    FirewallDecision,
    FirewallPolicy,
    FirewallService,
    NavigationPermitStore,
    ReasonCode,
    UrlPurpose,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("axis-playwright")
    group.addoption(
        "--run-playwright",
        action="store_true",
        default=False,
        help="run opt-in AXIS tests that launch a real Playwright browser",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "playwright: launches a real browser against the controlled local test site",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = config.getoption("--run-playwright") or os.getenv("AXIS_RUN_PLAYWRIGHT") == "1"
    if enabled:
        return
    skip = pytest.mark.skip(reason="use --run-playwright (or AXIS_RUN_PLAYWRIGHT=1) to run")
    for item in items:
        if "playwright" in item.keywords:
            item.add_marker(skip)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    query: dict[str, list[str]]


@dataclass(slots=True)
class ControlledTestSite:
    host: str
    port: int
    _server: ThreadingHTTPServer
    _thread: threading.Thread
    requests: list[RecordedRequest] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url(self, path: str = "/") -> str:
        normalized = path if path.startswith("/") else f"/{path}"
        return f"{self.origin}{normalized}"

    def record(self, request: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def paths(self) -> list[str]:
        with self._lock:
            return [request.path for request in self.requests]

    def clear_requests(self) -> None:
        with self._lock:
            self.requests.clear()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _AxisTestRequestHandler(BaseHTTPRequestHandler):
    server_version = "AXISControlledTestSite/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlsplit(self.path)
        site = vars(self.server)["axis_site"]
        site.record(
            RecordedRequest(
                method="GET",
                path=parsed.path,
                query=parse_qs(parsed.query, keep_blank_values=True),
            )
        )

        routes: dict[str, Callable[[], None]] = {
            "/": self._actions_page,
            "/actions": self._actions_page,
            "/popup": self._popup_page,
            "/redirect": self._redirect,
            "/redirect-target": self._redirect_target,
            "/iframe": self._iframe_page,
            "/network": self._network_page,
            "/frame-content": self._frame_content,
            "/subresource.js": self._subresource,
            "/blocked-subresource.js": self._blocked_subresource,
            "/api/data": self._api_data,
            "/blocked-api": self._blocked_api,
            "/download": self._download,
            "/next": self._next_page,
        }
        handler = routes.get(parsed.path)
        if handler is None:
            self._send_text(HTTPStatus.NOT_FOUND, "not found")
            return
        handler()

    def log_message(self, _format: str, *_args: Any) -> None:
        # Integration output stays deterministic unless a test fails.
        return

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, body.encode(), content_type="text/html; charset=utf-8")

    def _send_text(self, status: HTTPStatus, body: str) -> None:
        self._send(status, body.encode(), content_type="text/plain; charset=utf-8")

    def _actions_page(self) -> None:
        self._send_html(
            """<!doctype html>
<html><head><title>AXIS action fixture</title></head>
<body>
  <h1>Controlled AXIS action page</h1>
  <label>Name <input id="name" aria-label="Name" /></label>
  <label>Password <input id="password" type="password" aria-label="Password"
    autocomplete="current-password" /></label>
  <button id="copy" onclick="document.querySelector('#output').textContent =
    document.querySelector('#name').value">Copy value</button>
  <select id="choice" aria-label="Choice">
    <option value="alpha">Alpha</option><option value="beta">Beta</option>
  </select>
  <a id="next" href="/next">Next page</a>
  <button id="popup" onclick="window.open('/popup', '_blank')">Open popup</button>
  <a id="download" href="/download" download>Download fixture</a>
  <p id="output" aria-live="polite"></p>
  <div style="height:1600px"></div><p id="scroll-target">AXIS scroll target</p>
</body></html>"""
        )

    def _popup_page(self) -> None:
        self._send_html(
            "<!doctype html><html><head><title>AXIS popup</title></head>"
            "<body><h1>Controlled popup</h1></body></html>"
        )

    def _redirect(self) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", "/redirect-target")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _redirect_target(self) -> None:
        self._send_html(
            "<!doctype html><html><head><title>AXIS redirect target</title></head>"
            "<body><h1>Redirect completed</h1></body></html>"
        )

    def _iframe_page(self) -> None:
        self._send_html(
            "<!doctype html><html><head><title>AXIS iframe host</title></head>"
            "<body><h1>Iframe host</h1><iframe id='fixture-frame' src='/frame-content'></iframe>"
            "</body></html>"
        )

    def _frame_content(self) -> None:
        self._send_html(
            "<!doctype html><html><body><button id='frame-button'>Frame action</button></body>"
            "</html>"
        )

    def _network_page(self) -> None:
        self._send_html(
            """<!doctype html>
<html><head><title>AXIS network fixture</title>
  <script src="/subresource.js"></script>
  <script src="/blocked-subresource.js"></script>
</head><body>
  <h1>Network policy fixture</h1>
  <iframe id="fixture-frame" src="/frame-content"></iframe>
  <script>
    fetch('/api/data').then(r => r.json()).then(v => {
      document.body.dataset.apiStatus = v.status;
    });
    fetch('/blocked-api').then(r => r.json()).then(() => {
      document.body.dataset.blockedApiReached = 'true';
    }).catch(() => {
      document.body.dataset.blockedApiReached = 'false';
    });
  </script>
</body></html>"""
        )

    def _subresource(self) -> None:
        body = b"window.AXIS_SUBRESOURCE_LOADED = true;"
        self._send(HTTPStatus.OK, body, content_type="application/javascript")

    def _blocked_subresource(self) -> None:
        body = b"window.AXIS_BLOCKED_SUBRESOURCE_RAN = true;"
        self._send(HTTPStatus.OK, body, content_type="application/javascript")

    def _api_data(self) -> None:
        self._send(HTTPStatus.OK, b'{"status":"ok"}', content_type="application/json")

    def _blocked_api(self) -> None:
        self._send(HTTPStatus.OK, b'{"secret":"must-not-arrive"}', content_type="application/json")

    def _download(self) -> None:
        self._send(
            HTTPStatus.OK,
            b"controlled AXIS download\n",
            content_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="axis-fixture.txt"'},
        )

    def _next_page(self) -> None:
        self._send_html(
            "<!doctype html><html><head><title>AXIS next page</title></head>"
            "<body><h1>Next page</h1></body></html>"
        )


@pytest.fixture(scope="session")
def controlled_http_site() -> Iterator[ControlledTestSite]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AxisTestRequestHandler)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, name="axis-test-http", daemon=True)
    site = ControlledTestSite(str(host), int(port), server, thread)
    vars(server)["axis_site"] = site
    thread.start()
    try:
        yield site
    finally:
        site.stop()


@dataclass(frozen=True, slots=True)
class RecordedFirewallEvaluation:
    url: str
    purpose: UrlPurpose
    allowed: bool


class ControlledLoopbackFirewall(FirewallService):
    """Test-only policy service allowing exactly one controlled local origin."""

    def __init__(
        self,
        origin: str,
        *,
        blocked_paths: set[str] | None = None,
        blocked_purposes: set[UrlPurpose] | None = None,
    ) -> None:
        super().__init__(FirewallPolicy())
        parsed = urlsplit(origin)
        self._origin = (parsed.scheme, parsed.hostname, parsed.port)
        self._blocked_paths = blocked_paths or set()
        self._blocked_purposes = blocked_purposes or set()
        self.evaluations: list[RecordedFirewallEvaluation] = []

    async def evaluate(
        self,
        url: str,
        *,
        purpose: UrlPurpose = UrlPurpose.NAVIGATION,
        internal: bool = False,
    ) -> FirewallDecision:
        parsed = urlsplit(url)
        same_origin = (parsed.scheme, parsed.hostname, parsed.port) == self._origin
        allowed = (
            same_origin
            and parsed.path not in self._blocked_paths
            and purpose not in self._blocked_purposes
        )
        canonical = None
        resolved_ips: tuple[str, ...] = ()
        if same_origin:
            canonical = urlunsplit(
                SplitResult(parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
            )
            resolved_ips = ("127.0.0.1",)
        reason = ReasonCode.ALLOW_RULE_MATCH if allowed else ReasonCode.DENY_RULE_MATCH
        self.evaluations.append(RecordedFirewallEvaluation(url, purpose, allowed))
        return FirewallDecision(
            allowed=allowed,
            reason=reason,
            purpose=purpose,
            original_url_sha256=hashlib.sha256(
                url.encode("utf-8", errors="surrogatepass")
            ).hexdigest(),
            canonical_url=canonical,
            resolved_ips=resolved_ips,
            policy_hash=self.policy.policy_hash,
        )


@pytest.fixture
def controlled_firewall_factory(
    controlled_http_site: ControlledTestSite,
) -> Callable[..., ControlledLoopbackFirewall]:
    def factory(
        *,
        blocked_paths: set[str] | None = None,
        blocked_purposes: set[UrlPurpose] | None = None,
    ) -> ControlledLoopbackFirewall:
        return ControlledLoopbackFirewall(
            controlled_http_site.origin,
            blocked_paths=blocked_paths,
            blocked_purposes=blocked_purposes,
        )

    return factory


@pytest.fixture
def empty_download_directory(tmp_path: Path) -> Path:
    path = tmp_path / "downloads"
    path.mkdir()
    return path


@dataclass(slots=True)
class RunningPlaywrightAdapter:
    """Small integration driver around the public BrowserAdapter contract."""

    adapter: Any
    firewall: ControlledLoopbackFirewall
    permits: NavigationPermitStore

    async def execute(self, action: Action) -> ActionResult:
        action_id = uuid4()
        return await self.adapter.execute(action, action_id=str(action_id))

    async def navigate(self, url: str) -> ActionResult:
        action_id = uuid4()
        decision = await self.firewall.evaluate(url, purpose=UrlPurpose.NAVIGATION)
        if not decision.allowed:
            raise AssertionError(f"test navigation was denied: {decision.reason}")
        permit = self.permits.issue(decision, action_id=action_id)
        return await self.adapter.execute(
            GoToUrlAction(url=url),
            action_id=str(action_id),
            navigation_permit=permit,
        )

    async def open_tab(self, url: str) -> ActionResult:
        action_id = uuid4()
        decision = await self.firewall.evaluate(url, purpose=UrlPurpose.POPUP)
        if not decision.allowed:
            raise AssertionError(f"test tab navigation was denied: {decision.reason}")
        permit = self.permits.issue(decision, action_id=action_id)
        return await self.adapter.execute(
            OpenTabAction(url=url),
            action_id=str(action_id),
            navigation_permit=permit,
        )


@pytest.fixture
def playwright_adapter_factory(
    controlled_firewall_factory: Callable[..., ControlledLoopbackFirewall],
    empty_download_directory: Path,
) -> Callable[..., Any]:
    """Build a real adapter only when an opted-in test requests it.

    Importing Playwright inside the fixture lets the default offline test suite
    run without installing the optional ``browser`` dependency.
    """

    @asynccontextmanager
    async def factory(
        *,
        blocked_paths: set[str] | None = None,
        blocked_purposes: set[UrlPurpose] | None = None,
    ) -> AsyncIterator[RunningPlaywrightAdapter]:
        pytest.importorskip("playwright.async_api")
        adapter_module = pytest.importorskip("axis_agent.browser.playwright")
        adapter_type = adapter_module.DirectPlaywrightAdapter
        settings_type = adapter_module.PlaywrightSettings

        firewall = controlled_firewall_factory(
            blocked_paths=blocked_paths,
            blocked_purposes=blocked_purposes,
        )
        permits = NavigationPermitStore()
        try:
            settings = settings_type(
                headless=True,
                downloads_path=empty_download_directory,
            )
        except TypeError:
            # Some implementations keep downloads on the adapter rather than
            # settings.  Headless is the only required integration-test knob.
            settings = settings_type(headless=True)
        adapter = adapter_type(settings=settings, firewall=firewall, permit_store=permits)
        await adapter.start()
        try:
            yield RunningPlaywrightAdapter(adapter=adapter, firewall=firewall, permits=permits)
        finally:
            await adapter.stop()

    return factory
