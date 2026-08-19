"""Validation tests for Playwright's production browser configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from axis_agent.config import PlaywrightBrowser, PlaywrightSettings


def test_defaults_are_headless_bounded_and_downloads_are_separate() -> None:
    settings = PlaywrightSettings(_env_file=None)

    assert settings.browser is PlaywrightBrowser.CHROMIUM
    assert settings.headless is True
    assert settings.user_data_dir is None
    assert settings.downloads_path == Path(".axis-data/downloads")
    assert settings.default_timeout_ms == 15_000
    assert settings.navigation_timeout_ms == 30_000
    assert settings.max_observation_text_chars == 20_000
    assert settings.max_cached_content_chars == 100_000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("browser", "chrome"),
        ("default_timeout_ms", 999),
        ("default_timeout_ms", 120_001),
        ("navigation_timeout_ms", 999),
        ("navigation_timeout_ms", 120_001),
        ("viewport_width", 319),
        ("viewport_width", 7_681),
        ("viewport_height", 239),
        ("viewport_height", 4_321),
        ("slow_mo_ms", -1),
        ("slow_mo_ms", 5_001),
        ("max_observation_text_chars", 999),
        ("max_observation_text_chars", 100_001),
        ("max_cached_content_chars", 999),
        ("max_cached_content_chars", 1_000_001),
    ],
)
def test_resource_and_timeout_limits_fail_closed(name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        PlaywrightSettings(_env_file=None, **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    [
        "http://search.example/?q=",
        "https://search.example/search",
        "https://user:password@search.example/?q=",
        "https://search.example/?q=#fragment",
        "https:///missing-host?q=",
    ],
)
def test_search_base_url_must_be_a_clean_absolute_https_query_prefix(url: str) -> None:
    with pytest.raises(ValidationError, match="search_base_url"):
        PlaywrightSettings(_env_file=None, search_base_url=url)


def test_valid_custom_search_base_url_is_retained() -> None:
    url = "https://search.example/search?query="
    settings = PlaywrightSettings(_env_file=None, search_base_url=url)

    assert settings.search_base_url == url


def test_browser_storage_and_download_paths_must_differ() -> None:
    with pytest.raises(ValidationError, match="must be different"):
        PlaywrightSettings(
            _env_file=None,
            user_data_dir=Path(".axis-data/browser"),
            downloads_path=Path(".axis-data/browser"),
        )


def test_only_dedicated_playwright_environment_prefix_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AXIS_BROWSER", "webkit")
    monkeypatch.setenv("AXIS_PLAYWRIGHT_BROWSER", "firefox")
    monkeypatch.setenv("AXIS_PLAYWRIGHT_HEADLESS", "false")
    monkeypatch.setenv("AXIS_PLAYWRIGHT_DEFAULT_TIMEOUT_MS", "2345")

    settings = PlaywrightSettings(_env_file=None)

    assert settings.browser is PlaywrightBrowser.FIREFOX
    assert settings.headless is False
    assert settings.default_timeout_ms == 2_345


def test_unknown_constructor_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        PlaywrightSettings(_env_file=None, browser_channel="chrome")  # type: ignore[call-arg]
