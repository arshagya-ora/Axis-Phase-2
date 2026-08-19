from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

from axis_agent.firewall import policy_from_hosts
from axis_agent.mcp import MCPProfile
from axis_agent.mcp.client import (
    ManagedGatewayProcessConfig,
    build_stdio_launch_params,
    tool_names_for_profile,
)


def _config(tmp_path: Path, profile: MCPProfile) -> ManagedGatewayProcessConfig:
    policy_path = tmp_path / "firewall.json"
    policy_path.write_text(
        policy_from_hosts(["example.com"]).model_dump_json(),
        encoding="utf-8",
    )
    navigator = profile is MCPProfile.NAVIGATOR
    return ManagedGatewayProcessConfig(
        profile=profile,
        database_path=tmp_path / "axis.db",
        firewall_policy_path=policy_path,
        downloads_path=tmp_path / "downloads",
        session_id=uuid4(),
        task_id=uuid4(),
        step_id=uuid4(),
        allowed_action_types=("wait",) if navigator else (),
        plan_id=uuid4() if navigator else None,
        step_key="test_step" if navigator else None,
    )


def test_direct_navigator_client_has_fixed_tool_and_launch_boundary(tmp_path: Path) -> None:
    config = _config(tmp_path, MCPProfile.NAVIGATOR)
    params = build_stdio_launch_params(
        config,
        source_environment={
            "PATH": "safe-path",
            "TEMP": "safe-temp",
            "AXIS_OCI_GENAI_API_KEY": "must-not-cross",
            "AXIS_OCI_PROJECT_OCID": "must-not-cross",
            "AXIS_NAVIGATOR_MODEL": "must-not-cross",
        },
    )

    assert tool_names_for_profile(MCPProfile.NAVIGATOR) == (
        "axis_bind_step",
        "browser_observe",
        "browser_execute",
    )
    assert params["command"] == str(Path(sys.executable).resolve())
    assert params["args"][0:3] == ["-I", "-m", "axis_agent.mcp"]
    assert params["env"] == {"PATH": "safe-path", "TEMP": "safe-temp"}
    assert not any("shell" in argument.lower() for argument in params["args"])


def test_process_config_rejects_relative_or_missing_security_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="database_path must be absolute"):
        ManagedGatewayProcessConfig(
            profile=MCPProfile.PLANNER,
            database_path=Path("axis.db"),
            firewall_policy_path=tmp_path / "missing.json",
            downloads_path=tmp_path / "downloads",
            session_id=uuid4(),
            task_id=uuid4(),
            step_id=uuid4(),
        )

    with pytest.raises(ValueError, match="firewall_policy_path cannot be resolved"):
        ManagedGatewayProcessConfig(
            profile=MCPProfile.PLANNER,
            database_path=tmp_path / "axis.db",
            firewall_policy_path=tmp_path / "missing.json",
            downloads_path=tmp_path / "downloads",
            session_id=uuid4(),
            task_id=uuid4(),
            step_id=uuid4(),
        )


def test_planner_has_no_runtime_mcp_client_surface() -> None:
    assert "browser_execute" not in tool_names_for_profile(MCPProfile.PLANNER)
