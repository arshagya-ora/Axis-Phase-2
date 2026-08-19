from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from axis_agent.app import AxisApplication, AxisApplicationError
from axis_agent.config import AxisSettings

ModelT = TypeVar("ModelT", bound=BaseModel)


class NeverCalledStructuredClient:
    """Injected client proving policy failures happen before any model call."""

    def __init__(self) -> None:
        self.calls = 0

    async def parse_structured(
        self,
        *,
        model: str,
        input: str,
        output_type: type[ModelT],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelT:
        del model, input, output_type, instructions, max_output_tokens
        self.calls += 1
        raise AssertionError("policy rejection must happen before a model call")


def live_settings(tmp_path: Path, *, policy_path: Path | None) -> AxisSettings:
    return AxisSettings(
        _env_file=None,
        model_mode="oci_openai",
        oci_project_ocid="ocid1.generativeaiproject.oc1.iad.exampleproject",
        oci_genai_api_key="synthetic-application-test-key",
        firewall_policy_path=policy_path,
        database_path=tmp_path / "axis.db",
    )


def forbid_mcp_construction(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise AssertionError("policy rejection must happen before MCP construction")


@pytest.mark.asyncio
async def test_application_rejects_missing_firewall_before_model_or_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = NeverCalledStructuredClient()
    settings = live_settings(tmp_path, policy_path=None)
    monkeypatch.setattr("axis_agent.app.TaskScopedPlaywrightMCP", forbid_mcp_construction)

    application = AxisApplication(settings=settings, client=client)
    with pytest.raises(AxisApplicationError, match="AXIS_FIREWALL_POLICY_PATH"):
        await application.run("Inspect the approved site")

    assert client.calls == 0
    assert not settings.database_path.exists()


@pytest.mark.asyncio
async def test_application_rejects_policy_with_no_allow_rules_before_model_or_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "empty-policy.json"
    policy_path.write_text('{"rules": []}', encoding="utf-8")
    client = NeverCalledStructuredClient()
    settings = live_settings(tmp_path, policy_path=policy_path)
    monkeypatch.setattr("axis_agent.app.TaskScopedPlaywrightMCP", forbid_mcp_construction)

    application = AxisApplication(settings=settings, client=client)
    with pytest.raises(AxisApplicationError, match="at least one firewall allow rule"):
        await application.run("Inspect the approved site")

    assert client.calls == 0
    assert not settings.database_path.exists()
