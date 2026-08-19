"""Standalone entry point for the AXIS Planner/Navigator workflow."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from axis_agent.app import AxisApplication
from axis_agent.browser import MockBrowserAdapter
from axis_agent.config import AxisSettings, ModelMode, load_settings
from axis_agent.contracts import TaskRequest, WorkflowStatus
from axis_agent.devtools import MockPlannerNavigatorRuntime
from axis_agent.firewall import (
    FirewallPolicy,
    FirewallPolicyLoadError,
    FirewallService,
    NavigationPermitStore,
    load_policy_file,
)
from axis_agent.mcp import AxisMCPService, MCPProfile
from axis_agent.openai_client import OpenAIClientRuntime
from axis_agent.persistence import AxisDatabase
from axis_agent.runtime import (
    ActionDispatcher,
    MCPServiceBrowserTools,
    TwoAgentWorkflow,
    WorkflowLimits,
)

app = typer.Typer(no_args_is_help=True, help="AXIS two-agent browser automation")
agent_app = typer.Typer(no_args_is_help=True, help="Run the fixed Planner/Navigator workflow")
app.add_typer(agent_app, name="agent")


class AgentRunMode(StrEnum):
    LIVE = "live"
    MOCK = "mock"


def _emit(value: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            typer.echo(f"{key}: {item}")
        return
    typer.echo(value)


def _settings_or_exit() -> AxisSettings:
    try:
        return load_settings()
    except ValidationError as exc:
        typer.echo(f"configuration invalid: {exc}", err=True)
        raise typer.Exit(1) from exc


def _load_policy(path: Path | None, settings: AxisSettings) -> FirewallPolicy:
    target = path or settings.firewall_policy_path
    if target is None:
        return FirewallPolicy()
    try:
        return load_policy_file(target)
    except FirewallPolicyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


async def _run_mock_workflow(
    settings: AxisSettings,
    task: TaskRequest,
) -> dict[str, object]:
    policy = _load_policy(None, settings)
    permits = NavigationPermitStore()
    firewall = FirewallService(policy)
    browser = MockBrowserAdapter()
    async with AxisDatabase(settings.database_path) as database:
        await database.store_firewall_policy(
            policy_hash=policy.policy_hash,
            rules=[rule.model_dump(mode="json") for rule in policy.rules],
        )
        dispatcher = ActionDispatcher(
            browser=browser,
            firewall=firewall,
            database=database,
            permits=permits,
            action_timeout_seconds=settings.action_timeout_seconds,
        )
        service = AxisMCPService(
            profile=MCPProfile.NAVIGATOR,
            browser=browser,
            dispatcher=dispatcher,
        )
        tools = MCPServiceBrowserTools(service=service, database=database)
        workflow = TwoAgentWorkflow(
            runner=MockPlannerNavigatorRuntime(),
            database=database,
            browser_tools=tools,
            config_hash=settings.config_hash,
            provider_name="mock",
            planner_model="mock-planner",
            navigator_model="mock-navigator",
            limits=WorkflowLimits(
                max_steps=settings.max_steps,
                max_actions_per_step=settings.max_actions_per_step,
                max_failures=settings.max_failures,
                model_timeout_seconds=settings.model_timeout_seconds,
                workflow_timeout_seconds=settings.workflow_timeout_seconds,
            ),
        )
        try:
            result = await workflow.run(task)
        finally:
            await service.stop()
    return {
        "mode": "mock",
        "externalCalls": 0,
        "result": result.model_dump(mode="json", by_alias=True),
    }


async def _run_live_workflow(
    settings: AxisSettings,
    *,
    prompt: str,
    start_url: str | None,
) -> dict[str, object]:
    if settings.model_mode is not ModelMode.OCI_OPENAI:
        raise ValueError("live mode requires AXIS_MODEL_MODE=oci_openai")
    async with OpenAIClientRuntime(settings) as client:
        application = AxisApplication(settings=settings, client=client)
        result = await application.run(prompt, start_url=start_url)
    return {
        "mode": "live",
        "result": result.model_dump(mode="json", by_alias=True),
    }


@agent_app.command("run")
def agent_run(
    task_input: Annotated[
        str,
        typer.Option(
            "--input",
            help="Task text, or '-' to read it from standard input without shell history",
        ),
    ] = "-",
    mode: Annotated[
        AgentRunMode,
        typer.Option("--mode", help="Explicit live or offline-mock execution mode"),
    ] = AgentRunMode.LIVE,
    start_url: Annotated[
        str | None,
        typer.Option("--start-url", help="Optional initial firewall-approved URL"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Run the complete Planner/Navigator workflow without implicit fallbacks."""

    settings = _settings_or_exit()
    prompt = typer.get_text_stream("stdin").read() if task_input == "-" else task_input
    try:
        if mode is AgentRunMode.MOCK:
            task = TaskRequest(prompt=prompt.strip())
            payload = asyncio.run(_run_mock_workflow(settings, task))
        else:
            payload = asyncio.run(
                _run_live_workflow(
                    settings,
                    prompt=prompt.strip(),
                    start_url=start_url,
                )
            )
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        typer.echo(f"agent run failed: {type(exc).__name__}", err=True)
        raise typer.Exit(1) from exc
    _emit(payload, json_output)
    result = payload["result"]
    if not isinstance(result, dict) or result.get("status") != WorkflowStatus.SUCCEEDED.value:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
