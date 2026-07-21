"""MCP（Model Context Protocol）客户端集成包。

本包负责将外部 MCP 服务器提供的工具能力接入 TwinSpark 工具体系，
使 Agent 能够透明地调用来自不同 MCP 服务器的远程工具。

核心组件：
- MCPClient: 单个 MCP 服务器的异步连接客户端，管理连接生命周期
- MCPTool: 适配器，将 MCP 远程工具映射为 TwinSpark Tool 接口
- MCPManager: 多服务器生命周期管理器，负责批量连接、工具发现与注册

异常体系：
- MCPConnectionError: MCP 服务器连接相关错误（连接失败、超时等）
- MCPToolError: MCP 工具调用相关错误（执行失败、返回异常等）
"""

from __future__ import annotations

from twinspark.tools.mcp.client import MCPClient
from twinspark.tools.mcp.adapter import MCPTool, create_mcp_tools
from twinspark.tools.mcp.manager import MCPManager


class MCPConnectionError(Exception):
    """MCP 服务器连接错误。

    当与 MCP 服务器建立连接失败、连接超时或连接意外断开时抛出。
    包含服务器名称和原始错误信息，便于上层诊断。
    """

    def __init__(self, server_name: str, message: str, cause: Exception | None = None):
        self.server_name = server_name
        self.cause = cause
        super().__init__(f"[{server_name}] 连接错误: {message}")


class MCPToolError(Exception):
    """MCP 工具调用错误。

    当调用 MCP 服务器上的工具执行失败时抛出。
    包含工具名称、服务器名称和错误详情。
    """

    def __init__(self, tool_name: str, server_name: str, message: str, cause: Exception | None = None):
        self.tool_name = tool_name
        self.server_name = server_name
        self.cause = cause
        super().__init__(f"[{server_name}:{tool_name}] 调用错误: {message}")


__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPManager",
    "MCPConnectionError",
    "MCPToolError",
    "create_mcp_tools",
]
