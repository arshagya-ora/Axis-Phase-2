from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from axis_agent.contracts.actions import Action, ActionResult
from axis_agent.contracts.base import ContractModel
from axis_agent.firewall import NavigationPermit


class BrowserTab(ContractModel):
    page_id: str
    url: str
    title: str = ""
    active: bool = False


class BrowserObservation(ContractModel):
    observation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    page_id: str
    url: str
    title: str = ""
    visible_text: str = ""
    interactive_elements: list[dict[str, object]] = Field(default_factory=list)
    tabs: list[BrowserTab] = Field(default_factory=list)


class BrowserNetworkDecision(ContractModel):
    url: str
    purpose: str
    allowed: bool
    reason_code: str
    resource_type: str


@runtime_checkable
class BrowserAdapter(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def observe(self) -> BrowserObservation: ...

    async def execute(
        self,
        action: Action,
        *,
        action_id: str,
        navigation_permit: NavigationPermit | None = None,
        expected_observation_id: str | None = None,
        expected_page_id: str | None = None,
        expected_origin: str | None = None,
    ) -> ActionResult: ...
