"""Privacy-bounded DOM observations for the Playwright browser adapter.

The collector intentionally exposes a small allowlist of metadata. It never reads
form control values, serializes HTML, or accepts selectors from an agent/model.
Element references are valid only for the observation that created them.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

INDEX_ATTRIBUTE = "data-axis-agent-index"
OBSERVATION_ATTRIBUTE = "data-axis-agent-observation"
_OBSERVATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SECRET_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
)


class DOMCollectionError(RuntimeError):
    """Raised when the in-page collector returns a malformed observation."""


class StaleDOMObservationError(LookupError):
    """Raised when an observation-local element marker is no longer present."""


class ElementLocator(Protocol):
    """The Playwright Locator surface required to resolve an observation index."""

    async def count(self) -> int: ...


class DOMPage(Protocol):
    """The small Playwright Page surface used by this module."""

    async def evaluate(self, expression: str, arg: object) -> object: ...

    def locator(self, selector: str) -> ElementLocator: ...


@dataclass(frozen=True, slots=True)
class DOMLimits:
    """Hard bounds applied both in the page and again in the trusted process."""

    max_elements: int = 200
    max_text_length: int = 240
    max_attribute_length: int = 512

    def __post_init__(self) -> None:
        _require_bounded_integer("max_elements", self.max_elements, maximum=1_000)
        _require_bounded_integer("max_text_length", self.max_text_length, maximum=1_000)
        _require_bounded_integer("max_attribute_length", self.max_attribute_length, maximum=2_048)


@dataclass(frozen=True, slots=True)
class InteractiveElement:
    """Allowlisted metadata for one visible, interactive DOM element."""

    index: int
    tag: str
    role: str = ""
    name: str = ""
    text: str = ""
    placeholder: str = ""
    input_type: str = ""
    href: str = ""

    def as_metadata(self) -> dict[str, object]:
        """Return the shape consumed by ``BrowserObservation``."""

        return {
            "index": self.index,
            "tag": self.tag,
            "role": self.role,
            "name": self.name,
            "text": self.text,
            "placeholder": self.placeholder,
            "input_type": self.input_type,
            "href": self.href,
        }


@dataclass(frozen=True, slots=True)
class DOMObservation:
    """An immutable set of element references scoped to one DOM observation."""

    observation_id: str
    elements: tuple[InteractiveElement, ...]

    def __post_init__(self) -> None:
        if _OBSERVATION_ID_PATTERN.fullmatch(self.observation_id) is None:
            raise ValueError("observation_id must be 32 lowercase hexadecimal characters")
        if any(element.index != expected for expected, element in enumerate(self.elements)):
            raise ValueError("element indices must be contiguous and start at zero")

    def element_at(self, index: int) -> InteractiveElement:
        """Return a referenced element after strict, non-coercing index validation."""

        if type(index) is not int or index < 0 or index >= len(self.elements):
            raise IndexError("element index is outside this DOM observation")
        return self.elements[index]

    def as_browser_metadata(self) -> list[dict[str, object]]:
        """Convert this observation for ``BrowserObservation.interactive_elements``."""

        return [element.as_metadata() for element in self.elements]


# This JavaScript is a constant owned by AXIS. No selector or executable source is
# interpolated into it. The only argument is a bounded data object.
DOM_COLLECTOR_SCRIPT = r"""
(args) => {
  "use strict";

  const indexAttribute = "data-axis-agent-index";
  const observationAttribute = "data-axis-agent-observation";
  const stateKey = Symbol.for("axis.agent.dom-markers.v1");
  const priorMarkers = window[stateKey];

  if (Array.isArray(priorMarkers)) {
    for (const prior of priorMarkers) {
      const node = prior.node;
      if (!(node instanceof Element) || !node.isConnected) continue;
      if (node.getAttribute(indexAttribute) === prior.assignedIndex) {
        if (prior.originalIndex === null) node.removeAttribute(indexAttribute);
        else node.setAttribute(indexAttribute, prior.originalIndex);
      }
      if (node.getAttribute(observationAttribute) === prior.assignedObservation) {
        if (prior.originalObservation === null) node.removeAttribute(observationAttribute);
        else node.setAttribute(observationAttribute, prior.originalObservation);
      }
    }
  }

  const redact = (input, limit) => {
    if (typeof input !== "string") return "";
    let output = input.replace(/\s+/g, " ").trim();
    output = output.replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b/gi, "[REDACTED]");
    output = output.replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]");
    output = output.replace(/\bAKIA[0-9A-Z]{16}\b/g, "[REDACTED]");
    output = output.replace(
      /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,
      "[REDACTED]",
    );
    output = output.replace(
      /\b(api[_ -]?key|password|secret|token)\s*[:=]\s*[^\s,;]+/gi,
      "$1=[REDACTED]",
    );
    return output.slice(0, limit);
  };

  const isVisible = (element) => {
    if (!(element instanceof HTMLElement) || !element.isConnected) return false;
    if (element.closest('[aria-hidden="true"], [inert]')) return false;
    if (element.matches(':disabled, [aria-disabled="true"]')) return false;
    const style = getComputedStyle(element);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      Number(style.opacity) === 0
    ) return false;
    const rectangle = element.getBoundingClientRect();
    return rectangle.width > 0 && rectangle.height > 0 && element.getClientRects().length > 0;
  };

  const accessibleName = (element) => {
    const direct = element.getAttribute("aria-label");
    if (direct) return direct;
    const references = (element.getAttribute("aria-labelledby") || "")
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 8);
    if (references.length) {
      return references
        .map((identifier) => document.getElementById(identifier)?.textContent || "")
        .join(" ");
    }
    return element.getAttribute("title") || "";
  };

  const inferredRole = (element) => {
    const explicit = element.getAttribute("role");
    if (explicit) return explicit;
    const tag = element.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "summary") return "button";
    if (tag !== "input") return "";
    const kind = (element.getAttribute("type") || "text").toLowerCase();
    if (kind === "checkbox") return "checkbox";
    if (kind === "radio") return "radio";
    if (kind === "button" || kind === "submit" || kind === "reset") return "button";
    return "textbox";
  };

  const safeHref = (element) => {
    const rawHref = element.getAttribute("href");
    if (!rawHref) return "";
    try {
      const parsed = new URL(rawHref, document.baseURI);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
      parsed.username = "";
      parsed.password = "";
      parsed.search = "";
      parsed.hash = "";
      return redact(parsed.toString(), args.maxAttributeLength);
    } catch {
      return "";
    }
  };

  const selector = [
    "a[href]",
    "button",
    'input:not([type="hidden"])',
    "select",
    "textarea",
    "summary",
    '[contenteditable="true"]',
    '[tabindex]:not([tabindex="-1"])',
    '[role="button"]',
    '[role="link"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="switch"]',
    '[role="tab"]',
    '[role="textbox"]',
    '[role="combobox"]',
    '[role="menuitem"]',
  ].join(",");

  const output = [];
  const markers = [];
  const candidates = document.querySelectorAll(selector);

  for (const element of candidates) {
    if (output.length >= args.maxElements) break;
    if (!isVisible(element)) continue;

    const index = output.length;
    const assignedIndex = String(index);
    const originalIndex = element.getAttribute(indexAttribute);
    const originalObservation = element.getAttribute(observationAttribute);
    element.setAttribute(indexAttribute, assignedIndex);
    element.setAttribute(observationAttribute, args.observationId);
    markers.push({
      node: element,
      originalIndex,
      originalObservation,
      assignedIndex,
      assignedObservation: args.observationId,
    });

    const tag = element.tagName.toLowerCase();
    const inputType = tag === "input"
      ? (element.getAttribute("type") || "text").toLowerCase()
      : "";
    const canExposeText =
      tag !== "input" &&
      tag !== "textarea" &&
      tag !== "select" &&
      !element.isContentEditable &&
      inputType !== "password";

    output.push({
      index,
      tag,
      role: redact(inferredRole(element), args.maxAttributeLength),
      name: redact(accessibleName(element), args.maxAttributeLength),
      text: canExposeText ? redact(element.innerText, args.maxTextLength) : "",
      placeholder: redact(
        element.getAttribute("placeholder") || "",
        args.maxAttributeLength,
      ),
      input_type: redact(inputType, 32),
      href: safeHref(element),
    });
  }

  window[stateKey] = markers;
  return output;
}
"""


async def collect_interactive_elements(
    page: DOMPage,
    *,
    limits: DOMLimits | None = None,
) -> DOMObservation:
    """Collect a bounded, privacy-safe observation from a Playwright page."""

    active_limits = limits or DOMLimits()
    observation_id = secrets.token_hex(16)
    arguments: dict[str, object] = {
        "observationId": observation_id,
        "maxElements": active_limits.max_elements,
        "maxTextLength": active_limits.max_text_length,
        "maxAttributeLength": active_limits.max_attribute_length,
    }
    raw_output = await page.evaluate(DOM_COLLECTOR_SCRIPT, arguments)
    if not isinstance(raw_output, list):
        raise DOMCollectionError("DOM collector did not return an element list")
    if len(raw_output) > active_limits.max_elements:
        raise DOMCollectionError("DOM collector exceeded the configured element limit")

    elements: list[InteractiveElement] = []
    for expected_index, raw_element in enumerate(raw_output):
        if not isinstance(raw_element, Mapping):
            raise DOMCollectionError("DOM collector returned a non-object element")
        returned_index = raw_element.get("index")
        if type(returned_index) is not int or returned_index != expected_index:
            raise DOMCollectionError("DOM collector returned invalid element indices")
        elements.append(_parse_element(expected_index, raw_element, active_limits))

    return DOMObservation(observation_id=observation_id, elements=tuple(elements))


def locator_for_index(
    page: DOMPage,
    observation: DOMObservation,
    index: int,
) -> ElementLocator:
    """Build a locator from an observation-local integer, never a caller selector."""

    observation.element_at(index)
    selector = (
        f'[{OBSERVATION_ATTRIBUTE}="{observation.observation_id}"][{INDEX_ATTRIBUTE}="{index}"]'
    )
    return page.locator(selector)


async def require_locator_for_index(
    page: DOMPage,
    observation: DOMObservation,
    index: int,
) -> ElementLocator:
    """Resolve an index and fail if its observation marker is stale or ambiguous."""

    locator = locator_for_index(page, observation, index)
    if await locator.count() != 1:
        raise StaleDOMObservationError(
            "element marker is missing or ambiguous; collect a fresh DOM observation"
        )
    return locator


def _parse_element(
    index: int,
    raw: Mapping[object, object],
    limits: DOMLimits,
) -> InteractiveElement:
    tag = _safe_text(raw.get("tag"), 32).lower()
    if _TAG_PATTERN.fullmatch(tag) is None:
        tag = "unknown"
    input_type = _safe_text(raw.get("input_type"), 32).lower() if tag == "input" else ""
    text = "" if input_type == "password" else _safe_text(raw.get("text"), limits.max_text_length)
    return InteractiveElement(
        index=index,
        tag=tag,
        role=_safe_text(raw.get("role"), limits.max_attribute_length),
        name=_safe_text(raw.get("name"), limits.max_attribute_length),
        text=text,
        placeholder=_safe_text(raw.get("placeholder"), limits.max_attribute_length),
        input_type=input_type,
        href=_safe_href(raw.get("href"), limits.max_attribute_length),
    )


def _safe_text(raw: object, maximum: int) -> str:
    return sanitize_page_text(raw, maximum=maximum)


def sanitize_page_text(raw: object, *, maximum: int) -> str:
    """Normalize, redact, and bound untrusted text collected from a page.

    The same trusted-process boundary is used for DOM metadata, visible page
    text, and agent-requested cache entries. Non-string objects are ignored so
    their repr cannot accidentally serialize secrets into a model prompt.
    """

    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
    if not isinstance(raw, str):
        return ""
    output = _WHITESPACE_PATTERN.sub(" ", raw).strip()
    for pattern in _SECRET_PATTERNS:
        output = pattern.sub("[REDACTED]", output)
    return output[:maximum]


def _safe_href(raw: object, maximum: int) -> str:
    if not isinstance(raw, str):
        return ""
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https"} or parsed.hostname is None:
            return ""
        port = parsed.port
    except ValueError:
        return ""

    hostname = parsed.hostname
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = parsed.path or "/"
    return _safe_text(
        urlunsplit((parsed.scheme.lower(), netloc, path, "", "")),
        maximum,
    )


def _require_bounded_integer(name: str, value: int, *, maximum: int) -> None:
    if type(value) is not int or value <= 0 or value > maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
