"""Fixed launch configuration for the local AXIS MCP gateway.

The model-facing process never receives a command, module name, tool list, or
child environment from model input.  Those values are closed over here so the
MCP subprocess remains an AXIS-owned security boundary.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict
from uuid import UUID

from axis_agent.contracts.actions import PRODUCTION_ACTION_TYPES, ProductionActionType
from axis_agent.mcp.environment import build_child_environment
from axis_agent.mcp.service import MCPGatewayConfigurationError, MCPProfile

_MODULE_NAME: Final = "axis_agent.mcp"
_PLANNER_TOOL_NAMES: Final[tuple[str, ...]] = ()
_NAVIGATOR_TOOL_NAMES: Final[tuple[str, ...]] = (
    "axis_bind_step",
    "browser_observe",
    "browser_execute",
)


class StdioLaunchParams(TypedDict):
    """Fixed stdio parameters consumed by the direct MCP Python client."""

    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str
    encoding: str
    encoding_error_handler: Literal["strict"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedGatewayProcessConfig:
    """Trusted filesystem inputs for one isolated MCP gateway child."""

    profile: MCPProfile
    database_path: Path
    firewall_policy_path: Path
    downloads_path: Path
    session_id: UUID
    task_id: UUID
    step_id: UUID
    allowed_action_types: tuple[ProductionActionType, ...] = ()
    plan_id: UUID | None = None
    step_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, MCPProfile):
            raise ValueError("profile must be an MCPProfile")
        for field_name in ("session_id", "task_id", "step_id"):
            if not isinstance(getattr(self, field_name), UUID):
                raise ValueError(f"{field_name} must be a UUID")
        if len(set(self.allowed_action_types)) != len(self.allowed_action_types):
            raise ValueError("allowed_action_types must be unique")
        if not set(self.allowed_action_types) <= PRODUCTION_ACTION_TYPES:
            raise ValueError("allowed_action_types contains an unknown production action")
        if self.profile is MCPProfile.NAVIGATOR:
            if not self.allowed_action_types:
                raise ValueError("Navigator gateway requires allowed_action_types")
            if not isinstance(self.plan_id, UUID):
                raise ValueError("Navigator gateway requires a plan_id UUID")
            if (
                self.step_key is None
                or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.step_key) is None
            ):
                raise ValueError("Navigator gateway requires a valid step_key")
        elif self.allowed_action_types or self.plan_id is not None or self.step_key is not None:
            raise ValueError("Planner gateway cannot receive Navigator execution context")
        database_path = _resolve_absolute_path(self.database_path, "database_path")
        firewall_policy_path = _resolve_absolute_path(
            self.firewall_policy_path,
            "firewall_policy_path",
            must_exist=True,
        )
        downloads_path = _resolve_absolute_path(self.downloads_path, "downloads_path")

        if not firewall_policy_path.is_file():
            raise ValueError("firewall_policy_path must identify a file")
        if database_path.exists() and not database_path.is_file():
            raise ValueError("database_path must identify a file")
        if downloads_path.exists() and not downloads_path.is_dir():
            raise ValueError("downloads_path must identify a directory")
        if database_path == firewall_policy_path:
            raise ValueError("database and firewall policy paths must be different")
        if database_path == downloads_path:
            raise ValueError("database and downloads paths must be different")

        object.__setattr__(self, "database_path", database_path)
        object.__setattr__(self, "firewall_policy_path", firewall_policy_path)
        object.__setattr__(self, "downloads_path", downloads_path)


def tool_names_for_profile(profile: MCPProfile) -> tuple[str, ...]:
    """Return the immutable client-side allowlist for an AXIS MCP profile."""

    if profile is MCPProfile.PLANNER:
        return _PLANNER_TOOL_NAMES
    return _NAVIGATOR_TOOL_NAMES


def build_stdio_launch_params(
    config: ManagedGatewayProcessConfig,
    *,
    source_environment: Mapping[str, str] | None = None,
) -> StdioLaunchParams:
    """Build a non-extensible stdio launch specification for the gateway."""

    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file() or not executable.is_absolute():
        raise MCPGatewayConfigurationError("Python executable is unavailable")

    # Isolated mode ignores PYTHON* variables and the current directory for
    # imports.  Every remaining argument has a fixed position and meaning.
    arguments = [
        "-I",
        "-m",
        _MODULE_NAME,
        f"--profile={config.profile.value}",
        f"--database-path={config.database_path}",
        f"--firewall-policy-path={config.firewall_policy_path}",
        f"--downloads-path={config.downloads_path}",
        f"--session-id={config.session_id}",
        f"--task-id={config.task_id}",
        f"--step-id={config.step_id}",
    ]
    if config.profile is MCPProfile.NAVIGATOR:
        arguments.extend(
            [
                f"--plan-id={config.plan_id}",
                f"--step-key={config.step_key}",
                *(
                    f"--allowed-action-type={action_type}"
                    for action_type in config.allowed_action_types
                ),
            ]
        )
    return StdioLaunchParams(
        command=str(executable),
        args=arguments,
        env=build_child_environment(source_environment),
        # A trusted installation directory avoids making an attacker-writable
        # data directory the native process working directory.
        cwd=str(executable.parent),
        encoding="utf-8",
        encoding_error_handler="strict",
    )


def _resolve_absolute_path(path: Path, name: str, *, must_exist: bool = False) -> Path:
    raw = str(path)
    if any(character in raw for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{name} contains an invalid character")
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        return path.resolve(strict=must_exist)
    except OSError as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
