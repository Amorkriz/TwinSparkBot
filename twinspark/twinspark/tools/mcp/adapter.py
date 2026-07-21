"""MCP 工具适配器 —— 将 MCP 远程工具映射为 TwinSpark Tool 接口。

本模块实现了 MCPTool 适配器类和批量创建工厂函数。
MCPTool 继承自 TwinSpark 的 Tool 基类，使得 MCP 远程工具
可以像本地工具一样被 ToolRegistry 管理和被 Agent 调用。

工具命名规则：使用 "服务器名:工具名" 格式，避免多服务器间的名称冲突。
"""

from __future__ import annotations

import logging
from typing import Any

from twinspark.tools.base import Tool

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """将 MCP 远程工具适配为 TwinSpark Tool 接口。

    每个 MCPTool 实例代表一个来自外部 MCP 服务器的工具，
    通过持有的 MCPClient 引用将调用委托给远程服务器执行。

    工具名称格式为 "server_name:tool_name"，确保在多服务器
    环境下不会产生命名冲突。

    Attributes:
        is_async: 始终为 True，因为 MCP 调用必然是异步的。
        name: 格式为 "服务器名:工具名"。
        description: 来自 MCP 服务器的工具描述。
        parameters: 来自 MCP 服务器的工具参数 JSON Schema。
    """

    # MCP 调用必然是异步的
    is_async: bool = True

    def __init__(self, server_name: str, tool_schema: dict[str, Any], client: "MCPClient") -> None:
        """初始化 MCP 工具适配器。

        Args:
            server_name: MCP 服务器标识名称，作为工具名称前缀。
            tool_schema: MCP 服务器返回的工具描述字典，包含：
                - name: 工具原始名称
                - description: 工具描述
                - inputSchema: 参数的 JSON Schema
            client: 关联的 MCPClient 实例，用于实际的远程调用。
        """
        # 组合命名：服务器名:工具名
        self.name: str = f"{server_name}:{tool_schema['name']}"
        self.description: str = tool_schema.get("description", "")
        self.parameters: dict[str, Any] = tool_schema.get(
            "inputSchema", {"type": "object", "properties": {}}
        )
        # 持有客户端引用，用于委托调用
        self._client = client
        # 保存远程工具原始名称（不含服务器前缀），用于实际调用
        self._remote_name: str = tool_schema["name"]
        self._server_name: str = server_name

    async def run(self, **kwargs: Any) -> str:
        """委托给 MCP 服务器执行工具调用。

        将调用通过 MCPClient 转发到远程 MCP 服务器，
        并返回序列化后的结果字符串。

        Args:
            **kwargs: 符合该工具 parameters schema 的调用参数。

        Returns:
            工具执行结果的字符串表示。

        Note:
            调用失败时不会向外抛出异常，而是返回包含错误信息的字符串，
            让 Agent 能够感知到失败并做出相应决策。
        """
        try:
            result = await self._client.call_tool(self._remote_name, kwargs)
            logger.debug("工具 [%s] 调用成功", self.name)
            return result
        except Exception as exc:
            error_msg = f"工具调用失败 [{self.name}]: {exc}"
            logger.error(error_msg)
            # 返回错误信息而非抛出异常，让 Agent 能感知失败
            return f"[错误] {error_msg}"

    def __repr__(self) -> str:
        connected = self._client.is_connected if self._client else False
        return (
            f"MCPTool(name={self.name!r}, "
            f"remote={self._remote_name!r}, "
            f"connected={connected})"
        )


def create_mcp_tools(
    server_name: str,
    tool_schemas: list[dict[str, Any]],
    client: "MCPClient",
) -> list[MCPTool]:
    """批量创建 MCPTool 实例的工厂函数。

    从 MCP 服务器返回的工具描述列表中，为每个工具创建一个对应的
    MCPTool 适配器实例。

    Args:
        server_name: MCP 服务器标识名称。
        tool_schemas: 工具描述字典列表，每个元素应包含：
            - name (str): 工具名称
            - description (str, optional): 工具描述
            - inputSchema (dict, optional): 参数 JSON Schema
        client: 关联的 MCPClient 实例。

    Returns:
        MCPTool 实例列表，与输入的 tool_schemas 顺序一致。
        如果某个工具 schema 格式异常，会跳过该工具并记录警告日志。
    """
    tools: list[MCPTool] = []

    for schema in tool_schemas:
        # 校验必要字段
        if not isinstance(schema, dict) or "name" not in schema:
            logger.warning(
                "MCP 服务器 [%s] 返回了无效的工具 schema（缺少 name 字段），已跳过: %s",
                server_name, schema,
            )
            continue

        try:
            tool = MCPTool(server_name=server_name, tool_schema=schema, client=client)
            tools.append(tool)
        except Exception as exc:
            logger.warning(
                "MCP 服务器 [%s] 创建工具 [%s] 失败，已跳过: %s",
                server_name, schema.get("name", "unknown"), exc,
            )

    logger.info(
        "MCP 服务器 [%s] 成功创建 %d/%d 个工具适配器",
        server_name, len(tools), len(tool_schemas),
    )
    return tools


# 类型提示用的前向引用（避免循环导入）
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinspark.tools.mcp.client import MCPClient
