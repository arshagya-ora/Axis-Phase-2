from __future__ import annotations

from collections.abc import Mapping

import pytest

from axis_agent.browser.dom import (
    DOM_COLLECTOR_SCRIPT,
    DOMCollectionError,
    DOMLimits,
    DOMObservation,
    InteractiveElement,
    StaleDOMObservationError,
    collect_interactive_elements,
    locator_for_index,
    require_locator_for_index,
)


class FakeLocator:
    def __init__(self, count: int = 1) -> None:
        self.result_count = count

    async def count(self) -> int:
        return self.result_count


class FakePage:
    def __init__(self, output: object, *, locator_count: int = 1) -> None:
        self.output = output
        self.locator_count = locator_count
        self.evaluations: list[tuple[str, object]] = []
        self.selectors: list[str] = []

    async def evaluate(self, expression: str, arg: object) -> object:
        self.evaluations.append((expression, arg))
        return self.output

    def locator(self, selector: str) -> FakeLocator:
        self.selectors.append(selector)
        return FakeLocator(self.locator_count)


def raw_element(index: int, **overrides: object) -> dict[str, object]:
    element: dict[str, object] = {
        "index": index,
        "tag": "button",
        "role": "button",
        "name": "Continue",
        "text": "Continue",
        "placeholder": "",
        "input_type": "",
        "href": "",
    }
    element.update(overrides)
    return element


@pytest.mark.asyncio
async def test_collects_allowlisted_metadata_with_contiguous_indices() -> None:
    page = FakePage(
        [
            raw_element(0),
            raw_element(
                1,
                tag="a",
                role="link",
                name="Documentation",
                text="Read docs",
                href="https://user:password@example.com/path?token=secret#private",
                value="must-never-cross-boundary",
                outerHTML="<a>must-never-cross-boundary</a>",
            ),
        ]
    )

    observation = await collect_interactive_elements(page)

    assert [element.index for element in observation.elements] == [0, 1]
    assert observation.elements[1].href == "https://example.com/path"
    metadata = observation.as_browser_metadata()[1]
    assert set(metadata) == {
        "index",
        "tag",
        "role",
        "name",
        "text",
        "placeholder",
        "input_type",
        "href",
    }
    assert "value" not in metadata
    assert "outerHTML" not in metadata

    expression, arguments = page.evaluations[0]
    assert expression == DOM_COLLECTOR_SCRIPT
    assert isinstance(arguments, Mapping)
    assert arguments["observationId"] == observation.observation_id
    assert arguments["maxElements"] == 200


@pytest.mark.asyncio
async def test_collection_caps_and_redacts_untrusted_page_output() -> None:
    limits = DOMLimits(max_elements=2, max_text_length=20, max_attribute_length=24)
    page = FakePage(
        [
            raw_element(
                0,
                name="Bearer abcdefghijklmnop",
                text="   a very long button description that must be capped   ",
                placeholder="api_key=super-secret-material",
            ),
            raw_element(1, tag="invalid tag!", href="javascript:alert(1)"),
        ]
    )

    observation = await collect_interactive_elements(page, limits=limits)

    first, second = observation.elements
    assert "abcdefghijklmnop" not in first.name
    assert first.name == "[REDACTED]"
    assert len(first.text) == limits.max_text_length
    assert "super-secret-material" not in first.placeholder
    assert second.tag == "unknown"
    assert second.href == ""


@pytest.mark.asyncio
async def test_password_element_never_exposes_returned_text_or_unknown_fields() -> None:
    page = FakePage(
        [
            raw_element(
                0,
                tag="input",
                role="textbox",
                input_type="password",
                text="actual-password",
                value="actual-password",
                name="Account password",
                placeholder="Enter password",
            )
        ]
    )

    observation = await collect_interactive_elements(page)

    password = observation.elements[0]
    assert password.input_type == "password"
    assert password.text == ""
    assert "value" not in password.as_metadata()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        None,
        {},
        ["not-an-object"],
        [raw_element(1)],
        [raw_element(True)],
    ],
)
async def test_malformed_collector_output_fails_closed(output: object) -> None:
    with pytest.raises(DOMCollectionError):
        await collect_interactive_elements(FakePage(output))


@pytest.mark.asyncio
async def test_output_cannot_exceed_configured_element_count() -> None:
    page = FakePage([raw_element(0), raw_element(1)])

    with pytest.raises(DOMCollectionError, match="exceeded"):
        await collect_interactive_elements(page, limits=DOMLimits(max_elements=1))


def test_locator_is_built_only_from_validated_observation_and_integer_index() -> None:
    observation = DOMObservation(
        observation_id="0123456789abcdef0123456789abcdef",
        elements=(InteractiveElement(index=0, tag="button"),),
    )
    page = FakePage([])

    locator = locator_for_index(page, observation, 0)

    assert isinstance(locator, FakeLocator)
    assert page.selectors == [
        '[data-axis-agent-observation="0123456789abcdef0123456789abcdef"]'
        '[data-axis-agent-index="0"]'
    ]
    for invalid in (-1, 1, True, "0"):
        with pytest.raises(IndexError):
            locator_for_index(page, observation, invalid)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_checked_locator_rejects_stale_or_ambiguous_marker() -> None:
    observation = DOMObservation(
        observation_id="0123456789abcdef0123456789abcdef",
        elements=(InteractiveElement(index=0, tag="button"),),
    )

    for locator_count in (0, 2):
        with pytest.raises(StaleDOMObservationError):
            await require_locator_for_index(
                FakePage([], locator_count=locator_count), observation, 0
            )


def test_observation_rejects_forged_identifier_and_non_contiguous_indices() -> None:
    with pytest.raises(ValueError, match="observation_id"):
        DOMObservation(observation_id='bad"] *', elements=())
    with pytest.raises(ValueError, match="contiguous"):
        DOMObservation(
            observation_id="0123456789abcdef0123456789abcdef",
            elements=(InteractiveElement(index=1, tag="button"),),
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_elements": 0},
        {"max_elements": True},
        {"max_elements": 1_001},
        {"max_text_length": 1_001},
        {"max_attribute_length": 2_049},
    ],
)
def test_limits_are_strict_and_bounded(limits: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DOMLimits(**limits)  # type: ignore[arg-type]


def test_in_page_script_has_privacy_and_visibility_guards() -> None:
    assert ".value" not in DOM_COLLECTOR_SCRIPT
    assert "outerHTML" not in DOM_COLLECTOR_SCRIPT
    assert "innerHTML" not in DOM_COLLECTOR_SCRIPT
    assert "getBoundingClientRect" in DOM_COLLECTOR_SCRIPT
    assert "data-axis-agent-index" in DOM_COLLECTOR_SCRIPT
    assert 'parsed.search = ""' in DOM_COLLECTOR_SCRIPT
    assert 'parsed.password = ""' in DOM_COLLECTOR_SCRIPT
