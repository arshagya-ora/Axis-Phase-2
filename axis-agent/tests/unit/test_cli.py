from __future__ import annotations

from typer.testing import CliRunner

import axis_agent.cli as cli_module
from axis_agent.cli import app
from axis_agent.config import AxisSettings
from axis_agent.contracts import TaskRequest

runner = CliRunner()


def test_agent_run_executes_complete_mock_workflow_without_external_calls(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "workflow.db"
    monkeypatch.setenv("AXIS_DATABASE_PATH", str(database_path))

    result = runner.invoke(
        app,
        ["agent", "run", "--mode", "mock", "--input", "Verify AXIS standalone", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert '"status": "succeeded"' in result.output
    assert '"externalCalls": 0' in result.output
    assert "Verify AXIS standalone" not in result.output
    assert database_path.exists()


def test_agent_run_accepts_mock_stdin(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIS_DATABASE_PATH", str(tmp_path / "mock.db"))
    stdin_result = runner.invoke(
        app,
        ["agent", "run", "--mode", "mock", "--json"],
        input="stdin task\n",
    )
    assert stdin_result.exit_code == 0, stdin_result.output


def test_agent_live_failure_never_falls_back_to_mock(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = AxisSettings(
        _env_file=None,
        model_mode="oci_openai",
        oci_project_ocid="ocid1.generativeaiproject.oc1.iad.exampleproject",
        oci_genai_api_key="synthetic-cli-live-test-key",
    )
    live_prompts: list[str] = []
    mock_tasks: list[TaskRequest] = []

    async def fail_live(
        received_settings: AxisSettings,
        *,
        prompt: str,
        start_url: str | None,
    ) -> dict[str, object]:
        assert received_settings is settings
        assert start_url is None
        live_prompts.append(prompt)
        raise RuntimeError("sensitive synthetic provider detail")

    async def forbidden_mock(
        received_settings: AxisSettings,
        task: TaskRequest,
    ) -> dict[str, object]:
        assert received_settings is settings
        mock_tasks.append(task)
        raise AssertionError("live mode must never fall back to mock")

    monkeypatch.setattr(cli_module, "load_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "_run_live_workflow", fail_live)
    monkeypatch.setattr(cli_module, "_run_mock_workflow", forbidden_mock)

    refused = runner.invoke(
        app,
        ["agent", "run", "--mode", "live", "--input", "do not send"],
    )

    assert refused.exit_code != 0
    assert "agent run failed: RuntimeError" in refused.output
    assert "sensitive synthetic provider detail" not in refused.output
    assert "do not send" not in refused.output
    assert live_prompts == ["do not send"]
    assert mock_tasks == []
