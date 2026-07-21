"""MCPTool 适配器和 create_mcp_tools 工厂函数的单元测试。

验证 MCP 远程工具到 TwinSpark Tool 接口的适配逻辑，包括：
- 命名规则（服务器名:工具名 格式）
- OpenAI function-calling schema 的正确转换
- run() 方法对 MCPClient 的委托调用
- 错误处理和工厂函数的批量创建逻辑
"""

from __future__ import annotations

import os

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from twinspark.tools.base import Tool
from twinspark.tools.mcp.adapter import MCPTool, create_mcp_tools
from twinspark.tools.mcp import MCPToolError


# --------------------------------------------------------------------------- #
# 辅助设施
# --------------------------------------------------------------------------- #
def _make_mock_client(*, is_connected: bool = True) -> MagicMock:
    """创建一个模拟的 MCPClient 实例。"""
    client = MagicMock()
    client.call_tool = AsyncMock(return_value="调用成功")
    type(client).is_connected = PropertyMock(return_value=is_connected)
    client.name = "mock-server"
    return client


def _sample_tool_schema(
    name: str = "search",
    description: str = "搜索文档",
    input_schema: dict | None = None,
) -> dict[str, Any]:
    """构建一个标准的工具 schema 字典。"""
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema
        or {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量"},
            },
            "required": ["query"],
        },
    }


# --------------------------------------------------------------------------- #
# MCPTool 基本属性测试
# --------------------------------------------------------------------------- #
class TestMCPToolAttributes:
    """MCPTool 属性和 schema 转换测试。"""

    def test_mcp_tool_name_format(self):
        """验证命名格式为 "服务器名:工具名"。"""
        client = _make_mock_client()
        schema = _sample_tool_schema(name="read_file")

        tool = MCPTool(server_name="filesystem", tool_schema=schema, client=client)

        assert tool.name == "filesystem:read_file"

    def test_mcp_tool_name_preserves_server_prefix(self):
        """不同服务器下同名工具应产生不同的工具名称。"""
        client = _make_mock_client()
        schema = _sample_tool_schema(name="search")

        tool_a = MCPTool(server_name="server-a", tool_schema=schema, client=client)
        tool_b = MCPTool(server_name="server-b", tool_schema=schema, client=client)

        assert tool_a.name != tool_b.name
        assert tool_a.name == "server-a:search"
        assert tool_b.name == "server-b:search"

    def test_mcp_tool_schema_conversion(self):
        """验证 to_openai_schema() 输出符合 OpenAI function-calling 格式。"""
        client = _make_mock_client()
        schema = _sample_tool_schema(
            name="web_search",
            description="在网上搜索信息",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

        tool = MCPTool(server_name="web", tool_schema=schema, client=client)
        openai_schema = tool.to_openai_schema()

        # 顶层结构
        assert openai_schema["type"] == "function"
        assert "function" in openai_schema

        func = openai_schema["function"]
        # 函数名应为组合名称
        assert func["name"] == "web:web_search"
        # 描述来自 schema
        assert func["description"] == "在网上搜索信息"
        # 参数 schema 原样传递
        assert func["parameters"]["type"] == "object"
        assert "query" in func["parameters"]["properties"]

    def test_mcp_tool_is_async(self):
        """验证 MCPTool 的 is_async 属性始终为 True。"""
        client = _make_mock_client()
        schema = _sample_tool_schema()

        tool = MCPTool(server_name="s", tool_schema=schema, client=client)

        assert tool.is_async is True

    def test_mcp_tool_inherits_from_base_tool(self):
        """验证 MCPTool 是 Tool 基类的子类。"""
        client = _make_mock_client()
        schema = _sample_tool_schema()

        tool = MCPTool(server_name="s", tool_schema=schema, client=client)

        assert isinstance(tool, Tool)

    def test_mcp_tool_default_input_schema(self):
        """当 tool_schema 缺少 inputSchema 时，使用空参数默认值。"""
        client = _make_mock_client()
        schema = {"name": "no_params", "description": "无参数工具"}

        tool = MCPTool(server_name="s", tool_schema=schema, client=client)

        assert tool.parameters == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------- #
# MCPTool.run() 委托和错误处理测试
# --------------------------------------------------------------------------- #
class TestMCPToolRun:
    """MCPTool run() 方法测试。"""

    async def test_mcp_tool_run_delegates_to_client(self):
        """run() 方法正确委托给 MCPClient.call_tool()，传递原始工具名和参数。"""
        client = _make_mock_client()
        client.call_tool.return_value = "查询结果: 3 条记录"

        schema = _sample_tool_schema(name="query_db")
        tool = MCPTool(server_name="database", tool_schema=schema, client=client)

        result = await tool.run(query="SELECT *", limit=10)

        # 验证 call_tool 被调用，使用原始工具名（不含服务器前缀）
        client.call_tool.assert_awaited_once_with(
            "query_db", {"query": "SELECT *", "limit": 10}
        )
        assert result == "查询结果: 3 条记录"

    async def test_mcp_tool_run_handles_error(self):
        """工具执行出错时，返回包含错误信息的字符串而非抛出异常。"""
        client = _make_mock_client()
        client.call_tool.side_effect = MCPToolError(
            tool_name="broken_tool",
            server_name="s",
            message="远程执行失败",
        )

        schema = _sample_tool_schema(name="broken_tool")
        tool = MCPTool(server_name="s", tool_schema=schema, client=client)

        result = await tool.run(param="value")

        # 不应抛出异常，而是返回错误描述字符串
        assert "[错误]" in result
        assert "调用失败" in result or "调用错误" in result

    async def test_mcp_tool_run_handles_generic_exception(self):
        """工具执行抛出通用异常时，同样返回错误字符串。"""
        client = _make_mock_client()
        client.call_tool.side_effect = RuntimeError("网络超时")

        schema = _sample_tool_schema(name="timeout_tool")
        tool = MCPTool(server_name="net", tool_schema=schema, client=client)

        result = await tool.run()

        assert "[错误]" in result
        assert "网络超时" in result


# --------------------------------------------------------------------------- #
# create_mcp_tools 工厂函数测试
# --------------------------------------------------------------------------- #
class TestCreateMCPTools:
    """create_mcp_tools 工厂函数测试。"""

    def test_create_mcp_tools_factory(self):
        """批量创建工具：根据 schema 列表创建对应数量的 MCPTool 实例。"""
        client = _make_mock_client()
        schemas = [
            _sample_tool_schema(name="tool_a", description="工具A"),
            _sample_tool_schema(name="tool_b", description="工具B"),
            _sample_tool_schema(name="tool_c", description="工具C"),
        ]

        tools = create_mcp_tools("my-server", schemas, client)

        assert len(tools) == 3
        assert all(isinstance(t, MCPTool) for t in tools)
        assert tools[0].name == "my-server:tool_a"
        assert tools[1].name == "my-server:tool_b"
        assert tools[2].name == "my-server:tool_c"

    def test_create_mcp_tools_skips_invalid_schema(self):
        """工厂函数跳过缺少 name 字段的无效 schema，不影响其他工具创建。"""
        client = _make_mock_client()
        schemas = [
            _sample_tool_schema(name="valid_tool"),
            {"description": "无 name 字段的无效 schema"},  # 无效
            _sample_tool_schema(name="another_valid"),
        ]

        tools = create_mcp_tools("srv", schemas, client)

        # 仅有效的两个被创建
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "srv:valid_tool" in names
        assert "srv:another_valid" in names

    def test_create_mcp_tools_empty_list(self):
        """空 schema 列表返回空工具列表。"""
        client = _make_mock_client()

        tools = create_mcp_tools("empty-srv", [], client)

        assert tools == []

    def test_create_mcp_tools_non_dict_schema_skipped(self):
        """非字典类型的 schema 项被跳过。"""
        client = _make_mock_client()
        schemas = [
            "not a dict",  # type: ignore
            None,  # type: ignore
            _sample_tool_schema(name="ok"),
        ]

        tools = create_mcp_tools("srv", schemas, client)

        assert len(tools) == 1
        assert tools[0].name == "srv:ok"
