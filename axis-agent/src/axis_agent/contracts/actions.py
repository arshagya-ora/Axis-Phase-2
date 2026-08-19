"""Typed, discriminated AXIS browser action vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import Field, TypeAdapter, field_validator

from axis_agent.contracts.base import ContractModel


class IntentAction(ContractModel):
    intent: str = ""


class DoneAction(ContractModel):
    type: Literal["done"] = "done"
    text: str
    success: bool


class SearchGoogleAction(IntentAction):
    type: Literal["search_google"] = "search_google"
    query: str = Field(min_length=1)


class GoToUrlAction(IntentAction):
    type: Literal["go_to_url"] = "go_to_url"
    url: str = Field(min_length=1)


class GoBackAction(IntentAction):
    type: Literal["go_back"] = "go_back"


class ClickElementAction(IntentAction):
    type: Literal["click_element"] = "click_element"
    index: int = Field(ge=0)
    xpath: str | None = None


class InputTextAction(IntentAction):
    type: Literal["input_text"] = "input_text"
    index: int = Field(ge=0)
    text: str
    xpath: str | None = None


class SwitchTabAction(IntentAction):
    type: Literal["switch_tab"] = "switch_tab"
    tab_id: str = Field(min_length=1, max_length=128)


class OpenTabAction(IntentAction):
    type: Literal["open_tab"] = "open_tab"
    url: str = Field(min_length=1)


class CloseTabAction(IntentAction):
    type: Literal["close_tab"] = "close_tab"
    tab_id: str = Field(min_length=1, max_length=128)


class CacheContentAction(IntentAction):
    type: Literal["cache_content"] = "cache_content"
    content: str = ""


class ScrollToPercentAction(IntentAction):
    type: Literal["scroll_to_percent"] = "scroll_to_percent"
    y_percent: int = Field(alias="yPercent", ge=0, le=100)
    index: int | None = Field(default=None, ge=0)


class ScrollToTopAction(IntentAction):
    type: Literal["scroll_to_top"] = "scroll_to_top"
    index: int | None = Field(default=None, ge=0)


class ScrollToBottomAction(IntentAction):
    type: Literal["scroll_to_bottom"] = "scroll_to_bottom"
    index: int | None = Field(default=None, ge=0)


class PreviousPageAction(IntentAction):
    type: Literal["previous_page"] = "previous_page"
    index: int | None = Field(default=None, ge=0)


class NextPageAction(IntentAction):
    type: Literal["next_page"] = "next_page"
    index: int | None = Field(default=None, ge=0)


class ScrollToTextAction(IntentAction):
    type: Literal["scroll_to_text"] = "scroll_to_text"
    text: str = Field(min_length=1)
    nth: int = Field(default=1, ge=1)


class SendKeysAction(IntentAction):
    type: Literal["send_keys"] = "send_keys"
    keys: str = Field(min_length=1, max_length=128)

    @field_validator("keys")
    @classmethod
    def validate_safe_key_chord(cls, value: str) -> str:
        """Allow only page-scoped keyboard commands with reviewed semantics."""

        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("keys must be one bounded Playwright key chord")
        parts = value.split("+")
        if any(not part for part in parts):
            raise ValueError("keys contains an empty chord component")

        safe_single_keys = {
            "Backspace",
            "Delete",
            "End",
            "Enter",
            "Escape",
            "Home",
            "Insert",
            "PageDown",
            "PageUp",
            "Space",
            "Tab",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "ArrowUp",
        }
        if len(parts) == 1 and parts[0] in safe_single_keys:
            return value
        if parts == ["Control", "A"] or (
            len(parts) == 2 and parts[0] == "Shift" and parts[1] in safe_single_keys
        ):
            return value
        raise ValueError("keys chord is not allowed by AXIS browser policy")


class GetDropdownOptionsAction(IntentAction):
    type: Literal["get_dropdown_options"] = "get_dropdown_options"
    index: int = Field(ge=0)


class SelectDropdownOptionAction(IntentAction):
    type: Literal["select_dropdown_option"] = "select_dropdown_option"
    index: int = Field(ge=0)
    text: str


class WaitAction(IntentAction):
    type: Literal["wait"] = "wait"
    seconds: int = Field(default=3, ge=0, le=120)


Action = Annotated[
    DoneAction
    | SearchGoogleAction
    | GoToUrlAction
    | GoBackAction
    | ClickElementAction
    | InputTextAction
    | SwitchTabAction
    | OpenTabAction
    | CloseTabAction
    | CacheContentAction
    | ScrollToPercentAction
    | ScrollToTopAction
    | ScrollToBottomAction
    | PreviousPageAction
    | NextPageAction
    | ScrollToTextAction
    | SendKeysAction
    | GetDropdownOptionsAction
    | SelectDropdownOptionAction
    | WaitAction,
    Field(discriminator="type"),
]

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

# Protocol 3.0 continues to parse the complete legacy action vocabulary above.
# New model-driven execution uses this deliberately smaller set.  In particular,
# completion is a Navigator outcome (not a browser action), content caching is
# not model-controlled, and provider-specific search/scroll aliases are omitted.
ProductionAction = Annotated[
    GoToUrlAction
    | GoBackAction
    | ClickElementAction
    | InputTextAction
    | SwitchTabAction
    | OpenTabAction
    | CloseTabAction
    | ScrollToPercentAction
    | ScrollToTopAction
    | ScrollToBottomAction
    | ScrollToTextAction
    | SendKeysAction
    | GetDropdownOptionsAction
    | SelectDropdownOptionAction
    | WaitAction,
    Field(discriminator="type"),
]

ProductionActionType = Literal[
    "go_to_url",
    "go_back",
    "click_element",
    "input_text",
    "switch_tab",
    "open_tab",
    "close_tab",
    "scroll_to_percent",
    "scroll_to_top",
    "scroll_to_bottom",
    "scroll_to_text",
    "send_keys",
    "get_dropdown_options",
    "select_dropdown_option",
    "wait",
]

PRODUCTION_ACTION_TYPES: Final[frozenset[ProductionActionType]] = frozenset(
    {
        "go_to_url",
        "go_back",
        "click_element",
        "input_text",
        "switch_tab",
        "open_tab",
        "close_tab",
        "scroll_to_percent",
        "scroll_to_top",
        "scroll_to_bottom",
        "scroll_to_text",
        "send_keys",
        "get_dropdown_options",
        "select_dropdown_option",
        "wait",
    }
)
APPROVAL_REQUIRED_ACTION_TYPES: Final[frozenset[ProductionActionType]] = frozenset(
    {"click_element", "input_text", "select_dropdown_option", "send_keys"}
)
PRODUCTION_ACTION_ADAPTER: TypeAdapter[ProductionAction] = TypeAdapter(ProductionAction)


def parse_action(value: object) -> Action:
    return ACTION_ADAPTER.validate_python(value, strict=True)


def parse_production_action(value: object) -> ProductionAction:
    """Parse one action from the canonical production surface.

    ``intent`` and ``xpath`` remain accepted only as protocol-3 compatibility
    fields.  They are discarded before validation, so neither hidden reasoning
    nor a caller-controlled selector crosses the production execution boundary.
    Every other unknown field is still rejected by the strict action model.
    """

    if isinstance(value, ContractModel):
        payload: object = value.model_dump(mode="python", by_alias=False)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = value

    if isinstance(payload, dict):
        payload.pop("intent", None)
        payload.pop("xpath", None)
    return PRODUCTION_ACTION_ADAPTER.validate_python(payload, strict=True)


def production_action_payload(value: object) -> dict[str, object]:
    """Return the stable action payload persisted and sent across MCP."""

    action = parse_production_action(value)
    return action.model_dump(
        mode="json",
        by_alias=True,
        exclude={"intent", "xpath"},
        exclude_none=True,
    )


class ActionResult(ContractModel):
    action_id: UUID
    success: bool
    message: str = ""
    data: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False


class ActionCommand(ContractModel):
    action_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    task_id: UUID
    step_id: UUID
    ordinal: int = Field(ge=0)
    expected_observation_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
    )
    expected_page_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_origin: str | None = Field(default=None, max_length=2_048)
    action: Action

    @field_validator("expected_origin")
    @classmethod
    def require_clean_http_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("expected_origin must be a clean HTTP(S) origin") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("expected_origin must be a clean HTTP(S) origin")
        default_port = 443 if parsed.scheme == "https" else 80
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        suffix = "" if port in {None, default_port} else f":{port}"
        return f"{parsed.scheme}://{host}{suffix}"
