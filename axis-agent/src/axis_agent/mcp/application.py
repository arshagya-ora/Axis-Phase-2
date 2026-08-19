"""Production application wiring for the AXIS-owned MCP stdio child."""

from __future__ import annotations

import argparse
import asyncio
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

from axis_agent.browser import BrowserAdapter
from axis_agent.config import PlaywrightBrowser, PlaywrightSettings
from axis_agent.contracts.actions import PRODUCTION_ACTION_TYPES
from axis_agent.firewall import (
    FirewallPolicyLoadError,
    FirewallService,
    NavigationPermitStore,
    load_policy_file,
)
from axis_agent.mcp.client import ManagedGatewayProcessConfig
from axis_agent.mcp.service import (
    AxisMCPService,
    ManagedStepApprovalFactory,
    MCPGatewayConfigurationError,
    MCPProfile,
)
from axis_agent.mcp.stdio import create_stdio_server
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import ActionDispatcher, ApprovalCoordinator
from axis_agent.runtime.approvals import TrustedSiteAutoApprovalHandler


class _PlaywrightAdapterFactory(Protocol):
    def __call__(
        self,
        settings: PlaywrightSettings,
        firewall: FirewallService,
        permit_store: NavigationPermitStore,
    ) -> BrowserAdapter: ...


class _AsyncStdioServer(Protocol):
    async def run_stdio_async(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GatewayRuntime:
    """Resources that must live on the MCP server's event loop."""

    service: AxisMCPService
    database: AxisDatabase


def build_gateway_runtime(config: ManagedGatewayProcessConfig) -> GatewayRuntime:
    """Construct, but do not start, one fail-closed gateway runtime."""

    try:
        policy = load_policy_file(config.firewall_policy_path)
    except FirewallPolicyLoadError as exc:
        raise MCPGatewayConfigurationError("gateway firewall policy is invalid") from exc

    firewall = FirewallService(policy)
    permit_store = NavigationPermitStore()
    browser_settings = PlaywrightSettings(
        _env_file=None,
        browser=PlaywrightBrowser.CHROMIUM,
        headless=True,
        user_data_dir=None,
        downloads_path=config.downloads_path,
        default_timeout_ms=15_000,
        navigation_timeout_ms=30_000,
        viewport_width=1440,
        viewport_height=900,
        locale="en-US",
        timezone_id="UTC",
        slow_mo_ms=0,
        max_observation_text_chars=20_000,
        max_cached_content_chars=100_000,
        search_base_url=ActionDispatcher.DEFAULT_SEARCH_BASE_URL,
    )
    browser = _load_playwright_adapter_factory()(browser_settings, firewall, permit_store)
    database = AxisDatabase(config.database_path)
    dispatcher = ActionDispatcher(
        browser=browser,
        firewall=firewall,
        database=database,
        permits=permit_store,
        search_base_url=browser_settings.search_base_url,
        action_timeout_seconds=30,
    )
    approval_gate = None
    approval_factory: ManagedStepApprovalFactory | None = None
    if config.profile is MCPProfile.NAVIGATOR:
        if config.plan_id is None or config.step_key is None:
            raise MCPGatewayConfigurationError("Navigator approval context is missing")
        auto_approval_handler = TrustedSiteAutoApprovalHandler(firewall)

        def create_step_approval(
            *,
            plan_id: UUID,
            step_key: str,
        ) -> ApprovalCoordinator:
            return ApprovalCoordinator(
                database=database,
                plan_id=plan_id,
                step_key=step_key,
                handler=auto_approval_handler,
            )

        approval_factory = create_step_approval

        approval_gate = ApprovalCoordinator(
            database=database,
            plan_id=config.plan_id,
            step_key=config.step_key,
            handler=auto_approval_handler,
        )
    service = AxisMCPService(
        profile=config.profile,
        browser=browser,
        dispatcher=dispatcher,
        approval_gate=approval_gate,
        allowed_action_types=(
            config.allowed_action_types if config.profile is MCPProfile.NAVIGATOR else None
        ),
        session_id=config.session_id,
        task_id=config.task_id,
        step_id=config.step_id,
        managed_step_approval_factory=approval_factory,
    )
    return GatewayRuntime(service=service, database=database)


async def serve_gateway_runtime(
    runtime: GatewayRuntime,
    *,
    server_factory: Callable[[AxisMCPService], _AsyncStdioServer] | None = None,
) -> None:
    """Serve one runtime and release browser/database resources on every exit."""

    factory = _default_server_factory if server_factory is None else server_factory
    try:
        await runtime.database.connect()
        await runtime.database.initialize()
        server = factory(runtime.service)
        await server.run_stdio_async()
    finally:
        try:
            await runtime.service.stop()
        finally:
            await runtime.database.close()


async def run_gateway_application(config: ManagedGatewayProcessConfig) -> None:
    """Build and serve the configured gateway in one asynchronous lifecycle."""

    await serve_gateway_runtime(build_gateway_runtime(config))


def parse_gateway_arguments(
    arguments: Sequence[str] | None = None,
) -> ManagedGatewayProcessConfig:
    """Parse the fixed arguments emitted by :func:`build_stdio_launch_params`."""

    parser = argparse.ArgumentParser(prog="python -m axis_agent.mcp", allow_abbrev=False)
    parser.add_argument(
        "--profile", required=True, choices=tuple(item.value for item in MCPProfile)
    )
    parser.add_argument("--database-path", required=True, type=Path)
    parser.add_argument("--firewall-policy-path", required=True, type=Path)
    parser.add_argument("--downloads-path", required=True, type=Path)
    parser.add_argument("--session-id", required=True, type=UUID)
    parser.add_argument("--task-id", required=True, type=UUID)
    parser.add_argument("--step-id", required=True, type=UUID)
    parser.add_argument("--plan-id", type=UUID)
    parser.add_argument("--step-key")
    parser.add_argument(
        "--allowed-action-type",
        action="append",
        choices=sorted(PRODUCTION_ACTION_TYPES),
        default=[],
    )
    namespace = parser.parse_args(arguments)
    try:
        return ManagedGatewayProcessConfig(
            profile=MCPProfile(namespace.profile),
            database_path=namespace.database_path,
            firewall_policy_path=namespace.firewall_policy_path,
            downloads_path=namespace.downloads_path,
            session_id=namespace.session_id,
            task_id=namespace.task_id,
            step_id=namespace.step_id,
            allowed_action_types=tuple(namespace.allowed_action_type),
            plan_id=namespace.plan_id,
            step_key=namespace.step_key,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the managed MCP gateway without writing non-protocol data to stdout."""

    config = parse_gateway_arguments(arguments)
    asyncio.run(run_gateway_application(config))
    return 0


def _load_playwright_adapter_factory() -> _PlaywrightAdapterFactory:
    try:
        module = importlib.import_module("axis_agent.browser.playwright")
    except ImportError as exc:
        raise MCPGatewayConfigurationError(
            "managed MCP gateway requires the optional Playwright browser runtime"
        ) from exc
    factory = getattr(module, "DirectPlaywrightAdapter", None)
    if factory is None:
        raise MCPGatewayConfigurationError(
            "installed browser runtime does not provide DirectPlaywrightAdapter"
        )
    return cast(_PlaywrightAdapterFactory, factory)


def _default_server_factory(service: AxisMCPService) -> _AsyncStdioServer:
    server = create_stdio_server(service)
    if not hasattr(server, "run_stdio_async"):
        raise MCPGatewayConfigurationError("installed MCP SDK lacks async stdio support")
    return cast(_AsyncStdioServer, server)
