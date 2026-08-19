"""Simple production composition for the two AXIS direct-SDK agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from axis_agent.config import AxisSettings, PlaywrightSettings
from axis_agent.contracts import (
    PRODUCTION_ACTION_TYPES,
    ExecutionPlan,
    NavigatorInput,
    NavigatorOutcome,
    PlannerInput,
    TaskRequest,
    WorkflowResult,
)
from axis_agent.firewall import (
    FirewallPolicy,
    FirewallService,
    RuleEffect,
    UrlPurpose,
    load_policy_file,
)
from axis_agent.mcp.direct_client import TaskScopedPlaywrightMCP
from axis_agent.navigator import Navigator
from axis_agent.openai_client import StructuredOutputClient
from axis_agent.persistence import AxisDatabase
from axis_agent.planner import Planner
from axis_agent.runtime.workflow import TwoAgentWorkflow, WorkflowLimits


class AxisApplicationError(RuntimeError):
    """A stable local composition/configuration failure."""


@dataclass(slots=True)
class _DirectTwoAgentRuntime:
    planner: Planner
    navigator: Navigator

    async def run_planner(
        self,
        planner_input: PlannerInput,
        *,
        run_config: object | None = None,
    ) -> ExecutionPlan:
        del run_config
        return await self.planner.plan_input(planner_input)

    async def run_navigator(
        self,
        navigator_input: NavigatorInput,
        *,
        run_config: object | None = None,
    ) -> NavigatorOutcome:
        del run_config
        return await self.navigator.run_navigator(navigator_input)


class AxisApplication:
    """Run Planner then Navigator with one isolated MCP browser per task."""

    def __init__(
        self,
        *,
        settings: AxisSettings,
        client: StructuredOutputClient,
        playwright_settings: PlaywrightSettings | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._playwright_settings = playwright_settings or PlaywrightSettings()

    async def run(
        self,
        task_text: str,
        start_url: str | None = None,
    ) -> WorkflowResult:
        task_prompt = task_text.strip()
        if not task_prompt:
            raise ValueError("task_text must not be empty")
        policy_path, policy = self._production_policy()
        approved_start_host: str | None = None
        if start_url is not None:
            decision = await FirewallService(policy).evaluate(
                start_url.strip(),
                purpose=UrlPurpose.NAVIGATION,
            )
            if not decision.allowed or decision.canonical_url is None:
                raise AxisApplicationError("initial URL is blocked by the firewall")
            safe_start_url = self._safe_model_url(decision.canonical_url)
            approved_start_host = urlsplit(safe_start_url).hostname
            task_prompt = f"{task_prompt}\nInitial URL: {safe_start_url}"
        task = TaskRequest(prompt=task_prompt)

        database_path = self._settings.database_path.resolve()
        downloads_path = self._playwright_settings.downloads_path.resolve()
        authorized_domains = {
            rule.host for rule in policy.rules if rule.enabled and rule.effect is RuleEffect.ALLOW
        }
        if approved_start_host is not None:
            authorized_domains.add(approved_start_host.lower().rstrip("."))
        domains = tuple(sorted(authorized_domains))
        if not domains:
            raise AxisApplicationError("live execution requires at least one firewall allow rule")

        planner = Planner(
            client=self._client,
            model=self._settings.planner_model,
            authorized_domains=domains,
            authorized_action_types=PRODUCTION_ACTION_TYPES,
            max_steps=self._settings.max_steps,
        )
        mcp_browser = TaskScopedPlaywrightMCP(
            database_path=database_path,
            firewall_policy_path=policy_path,
            downloads_path=downloads_path,
            timeout_seconds=min(300, max(130, self._settings.model_timeout_seconds)),
        )
        navigator = Navigator(
            client=self._client,
            model=self._settings.navigator_model,
            browser=mcp_browser,
        )
        runtime = _DirectTwoAgentRuntime(planner=planner, navigator=navigator)

        async with AxisDatabase(database_path) as database:
            await database.store_firewall_policy(
                policy_hash=policy.policy_hash,
                rules=[rule.model_dump(mode="json") for rule in policy.rules],
            )
            workflow = TwoAgentWorkflow(
                runner=runtime,
                database=database,
                browser_tools=navigator,
                config_hash=self._settings.config_hash,
                provider_name="oci-openai-direct",
                planner_model=self._settings.planner_model,
                navigator_model=self._settings.navigator_model,
                limits=WorkflowLimits(
                    max_steps=self._settings.max_steps,
                    max_actions_per_step=self._settings.max_actions_per_step,
                    max_failures=self._settings.max_failures,
                    model_timeout_seconds=self._settings.model_timeout_seconds,
                    workflow_timeout_seconds=self._settings.workflow_timeout_seconds,
                ),
            )
            try:
                return await workflow.run(task)
            finally:
                await navigator.close()

    def _production_policy(self) -> tuple[Path, FirewallPolicy]:
        configured = self._settings.firewall_policy_path
        if configured is None:
            raise AxisApplicationError("live execution requires AXIS_FIREWALL_POLICY_PATH")
        try:
            policy_path = configured.resolve(strict=True)
        except OSError:
            raise AxisApplicationError("firewall policy file is unavailable") from None
        if not policy_path.is_file():
            raise AxisApplicationError("firewall policy path must identify a file")
        try:
            policy = load_policy_file(policy_path)
        except Exception:
            raise AxisApplicationError("firewall policy is invalid") from None
        return policy_path, policy

    @staticmethod
    def _safe_model_url(url: str) -> str:
        parsed = urlsplit(url)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


__all__ = ["AxisApplication", "AxisApplicationError"]
