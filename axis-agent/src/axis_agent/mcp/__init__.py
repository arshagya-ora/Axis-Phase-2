"""Narrow MCP gateway used only by the AXIS Navigator.

The package deliberately exposes AXIS-owned tools rather than a raw browser MCP
server.  Importing it never imports the optional MCP SDK.
"""

from axis_agent.mcp.client import (
    ManagedGatewayProcessConfig,
    build_stdio_launch_params,
    tool_names_for_profile,
)
from axis_agent.mcp.direct_client import DirectMCPClientError, TaskScopedPlaywrightMCP
from axis_agent.mcp.environment import build_child_environment
from axis_agent.mcp.service import (
    ActionDispatcher,
    AxisMCPService,
    MCPGatewayConfigurationError,
    MCPGatewayError,
    MCPGatewayExecutionError,
    MCPInvalidArgumentsError,
    MCPProfile,
    MCPToolNotAllowedError,
)

__all__ = [
    "ActionDispatcher",
    "AxisMCPService",
    "MCPGatewayConfigurationError",
    "MCPGatewayError",
    "MCPGatewayExecutionError",
    "MCPInvalidArgumentsError",
    "MCPProfile",
    "MCPToolNotAllowedError",
    "ManagedGatewayProcessConfig",
    "DirectMCPClientError",
    "build_stdio_launch_params",
    "build_child_environment",
    "TaskScopedPlaywrightMCP",
    "tool_names_for_profile",
]
