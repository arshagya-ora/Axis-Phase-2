"""Security-focused unit tests for the direct Playwright adapter.

These tests use small protocol fakes and never launch a browser or contact the
network. Real-browser behavior lives in the opt-in integration suite.
"""

from __future__ import annotations

import subprocess
import sys
from uuid import UUID

import pytest
from pydantic import ValidationError

from axis_agent.browser.dom import sanitize_page_text
from axis_agent.browser.playwright import DirectPlaywrightAdapter
from axis_agent.config import PlaywrightSettings
from axis_agent.contracts.actions import SendKeysAction
from axis_agent.firewall import (
    FirewallService,
    NavigationPermitStore,
    UrlPurpose,
    policy_from_hosts,
)


class _FakePage:
    def __init__(self) -> None:
        self.handlers: list[tuple[str, object]] = []

    def on(self, event: str, handler: object) -> None:
        self.handlers.append((event, handler))


class _RedirectRequestWithBrokenFrame:
    url = "https://allowed.example/redirect-target"
    resource_type = "document"
    redirected_from = object()

    def is_navigation_request(self) -> bool:
        return True

    @property
    def frame(self) -> object:
        raise AssertionError("redirect classification must not inspect the frame")


class _NavigationRequestWithBrokenFrame:
    url = "https://allowed.example/"
    resource_type = "document"
    redirected_from = None

    def is_navigation_request(self) -> bool:
        return True

    @property
    def frame(self) -> object:
        raise RuntimeError("Playwright reports that the navigation has no frame")


class _FakeRoute:
    def __init__(self) -> None:
        self.abort_reason: str | None = None
        self.continued = False

    async def abort(self, error_code: str = "failed") -> None:
        self.abort_reason = error_code

    async def continue_(self) -> None:
        self.continued = True


class _FirewallMustNotRun:
    async def evaluate(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("firewall must not run after request classification fails")


def _adapter(**settings_overrides: object) -> DirectPlaywrightAdapter:
    return DirectPlaywrightAdapter(
        PlaywrightSettings(_env_file=None, **settings_overrides),  # type: ignore[arg-type]
        FirewallService(policy_from_hosts([])),
        NavigationPermitStore(),
    )


def test_redirect_is_classified_before_playwright_frame_lookup() -> None:
    adapter = _adapter()

    purpose = adapter._classify_request(_RedirectRequestWithBrokenFrame())  # type: ignore[arg-type]

    assert purpose is UrlPurpose.REDIRECT


@pytest.mark.asyncio
async def test_route_aborts_when_navigation_frame_classification_raises() -> None:
    adapter = DirectPlaywrightAdapter(
        PlaywrightSettings(_env_file=None),
        _FirewallMustNotRun(),  # type: ignore[arg-type]
        NavigationPermitStore(),
    )
    route = _FakeRoute()

    await adapter._handle_route(  # type: ignore[arg-type]
        route,
        _NavigationRequestWithBrokenFrame(),  # type: ignore[arg-type]
    )

    assert route.abort_reason == "blockedbyclient"
    assert route.continued is False


def test_page_ids_are_stable_unique_uuid_strings() -> None:
    adapter = _adapter()
    first_page = _FakePage()
    second_page = _FakePage()

    first_id = adapter._page_id(first_page)  # type: ignore[arg-type]
    repeated_id = adapter._page_id(first_page)  # type: ignore[arg-type]
    second_id = adapter._page_id(second_page)  # type: ignore[arg-type]

    assert isinstance(first_id, str)
    assert UUID(first_id).version == 4
    assert repeated_id == first_id
    assert second_id != first_id


def test_observation_text_is_bounded_and_redacts_common_secret_shapes() -> None:
    adapter = _adapter(max_observation_text_chars=1_000)
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz"
    api_key = "api_key=top-secret-value"
    jwt = "eyJheader.payload.signature"
    openai_key = "sk-" + "abcdefghijklmnop"
    payload = f"  customer   text {bearer} {api_key} {jwt} {openai_key} " + ("x" * 2_000)

    redacted = adapter._redact_visible_text(payload)

    assert len(redacted) <= 1_000
    assert "customer text" in redacted
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "top-secret-value" not in redacted
    assert jwt not in redacted
    assert openai_key not in redacted
    assert "[REDACTED]" in redacted


def test_public_page_text_sanitizer_handles_non_text_without_stringifying_it() -> None:
    assert sanitize_page_text(None, maximum=100) == ""
    assert sanitize_page_text({"password": "do-not-stringify"}, maximum=100) == ""
    with pytest.raises(ValueError, match="maximum"):
        sanitize_page_text("text", maximum=0)


def test_observation_url_removes_credentials_query_and_fragment() -> None:
    safe = DirectPlaywrightAdapter._safe_observation_url(
        "https://user:password@example.com/private/path?token=secret#account"
    )

    assert safe == "https://example.com/private/path"
    assert "password" not in safe
    assert "secret" not in safe
    assert DirectPlaywrightAdapter._safe_observation_url("file:///etc/passwd") == "[blocked-url]"


@pytest.mark.parametrize(
    "keys",
    [
        "Alt+F4",
        "alt+f4",
        "Control+W",
        "Control+Shift+W",
        "Meta+Q",
        "Meta+W",
        "Control++A",
        "UnknownKey",
        "Enter\nControl+W",
        "A" * 129,
    ],
)
def test_send_keys_contract_rejects_browser_or_os_management_shortcuts(keys: str) -> None:
    with pytest.raises(ValidationError):
        SendKeysAction(keys=keys)


@pytest.mark.parametrize("keys", ["Enter", "End", "Escape", "Control+A", "Shift+Tab"])
def test_send_keys_contract_allows_bounded_page_interaction_keys(keys: str) -> None:
    assert SendKeysAction(keys=keys).keys == keys


def test_core_browser_package_import_does_not_require_playwright_extra() -> None:
    script = r"""
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "playwright" or name.startswith("playwright."):
        raise ModuleNotFoundError("playwright intentionally unavailable")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import axis_agent.browser as browser
assert browser.MockBrowserAdapter is not None
"""

    result = subprocess.run(  # noqa: S603 - sys.executable is the current trusted interpreter
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
