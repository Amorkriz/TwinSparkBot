"""MCPClient 单元测试。

使用 unittest.mock.AsyncMock 模拟 MCP SDK 的 stdio_client、sse_client 和 ClientSession，
验证 MCPClient 的连接管理、工具列表获取、工具调用以及重试机制等核心功能。
所有测试完全自包含，不依赖任何外部服务。
"""

from __future__ import annotations

import os

# 确保导入 config 相关模块不会因缺少环境变量而失败
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinspark.tools.mcp import MCPConnectionError, MCPToolError
from twinspark.tools.mcp.client import MCPClient, _MAX_RETRIES


# --------------------------------------------------------------------------- #
# 辅助 Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeToolInfo:
    """模拟 MCP SDK 返回的工具信息对象。"""

    name: str
    description: str
    inputSchema: dict


@dataclass
class FakeTextBlock:
    """模拟 MCP SDK 返回的文本内容块。"""

    text: str


@dataclass
class FakeListToolsResult:
    """模拟 session.list_tools() 返回值。"""

    tools: list


@dataclass
class FakeCallToolResult:
    """模拟 session.call_tool() 返回值。"""

    content: list


def _make_fake_session() -> AsyncMock:
    """创建模拟的 ClientSession，预设 initialize/list_tools/call_tool 方法。"""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=FakeListToolsResult(tools=[]))
    session.call_tool = AsyncMock(
        return_value=FakeCallToolResult(content=[FakeTextBlock(text="ok")])
    )
    return session


@asynccontextmanager
async def _fake_stdio_transport(*args, **kwargs):
    """模拟 stdio_client 上下文管理器，返回假的 (read, write) 对。"""
    yield (MagicMock(), MagicMock())


@asynccontextmanager
async def _fake_sse_transport(*args, **kwargs):
    """模拟 sse_client 上下文管理器，返回假的 (read, write) 对。"""
    yield (MagicMock(), MagicMock())


# --------------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------------- #
class TestMCPClientConnect:
    """MCPClient 连接相关测试。"""

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_connect_stdio_success(self, mock_stdio, mock_session_cls):
        """stdio 传输方式成功连接：验证 stdio_client 和 ClientSession 被正确调用。"""
        # 设置 mock
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(
            name="test-server",
            transport="stdio",
            command="python",
            args=["-m", "mcp_server"],
        )

        await client.connect()

        # 验证已连接
        assert client.is_connected is True
        # 验证 initialize 被调用（MCP 握手）
        fake_session.initialize.assert_awaited_once()

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.sse_client")
    async def test_connect_sse_success(self, mock_sse, mock_session_cls):
        """SSE 传输方式成功连接：验证 sse_client 被使用且传入了正确的 URL。"""
        mock_sse.return_value = _fake_sse_transport()
        fake_session = _make_fake_session()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(
            name="sse-server",
            transport="sse",
            url="http://localhost:8080/sse",
        )

        await client.connect()

        assert client.is_connected is True
        # 验证 sse_client 被调用时传入了正确的 URL
        mock_sse.assert_called_once_with("http://localhost:8080/sse")

    def test_connect_invalid_transport(self):
        """无效传输类型：构造时应直接抛出 ValueError。"""
        with pytest.raises(ValueError, match="不支持的传输方式"):
            MCPClient(name="bad", transport="websocket")

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_disconnect(self, mock_stdio, mock_session_cls):
        """正常断开连接：disconnect 后 is_connected 变为 False。"""
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(name="dc-test", transport="stdio", command="echo")
        await client.connect()
        assert client.is_connected is True

        await client.disconnect()
        assert client.is_connected is False

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_disconnect_idempotent(self, mock_stdio, mock_session_cls):
        """多次 disconnect 是安全的（幂等操作）。"""
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()
        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(name="dc-test2", transport="stdio", command="echo")
        await client.connect()

        # 多次 disconnect 不应抛出异常
        await client.disconnect()
        await client.disconnect()
        assert client.is_connected is False

    @patch("twinspark.tools.mcp.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_connect_retry_on_failure(self, mock_stdio, mock_session_cls, mock_sleep):
        """连接失败后重试：验证指数退避重试机制，最终所有重试耗尽后抛出 MCPConnectionError。"""
        # 让 stdio_client 始终抛出异常
        @asynccontextmanager
        async def _failing_transport(*args, **kwargs):
            raise ConnectionError("模拟连接失败")
            yield  # noqa: unreachable - needed for async generator syntax

        mock_stdio.return_value = _failing_transport()
        # 每次重试都返回一个新的 failing transport
        mock_stdio.side_effect = lambda *a, **kw: _failing_transport()

        client = MCPClient(name="retry-test", transport="stdio", command="fail")

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect()

        # 验证错误信息包含重试次数
        assert f"{_MAX_RETRIES}" in str(exc_info.value)
        # 验证 sleep 被调用了 (MAX_RETRIES - 1) 次（最后一次失败不需要 sleep）
        assert mock_sleep.await_count == _MAX_RETRIES - 1


class TestMCPClientTools:
    """MCPClient 工具列表和调用相关测试。"""

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_list_tools(self, mock_stdio, mock_session_cls):
        """获取工具列表：验证返回格式为包含 name/description/inputSchema 的字典列表。"""
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()

        # 模拟返回两个工具
        fake_session.list_tools.return_value = FakeListToolsResult(
            tools=[
                FakeToolInfo(
                    name="read_file",
                    description="读取文件内容",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
                FakeToolInfo(
                    name="write_file",
                    description="写入文件",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                ),
            ]
        )

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(name="tool-test", transport="stdio", command="echo")
        await client.connect()

        tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"
        assert tools[0]["description"] == "读取文件内容"
        assert "path" in tools[0]["inputSchema"]["properties"]
        assert tools[1]["name"] == "write_file"

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_call_tool_success(self, mock_stdio, mock_session_cls):
        """成功调用工具：验证参数被正确传递，结果被正确序列化返回。"""
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()

        # 模拟 call_tool 返回带文本内容的结果
        fake_session.call_tool.return_value = FakeCallToolResult(
            content=[FakeTextBlock(text="文件内容: hello world")]
        )

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(name="call-test", transport="stdio", command="echo")
        await client.connect()

        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})

        assert result == "文件内容: hello world"
        fake_session.call_tool.assert_awaited_once_with(
            "read_file", {"path": "/tmp/test.txt"}
        )

    async def test_call_tool_not_connected(self):
        """未连接时调用工具：应抛出 MCPConnectionError。"""
        client = MCPClient(name="not-connected", transport="stdio", command="echo")

        with pytest.raises(MCPConnectionError, match="未连接"):
            await client.call_tool("any_tool", {})

    async def test_list_tools_not_connected(self):
        """未连接时获取工具列表：应抛出 MCPConnectionError。"""
        client = MCPClient(name="not-connected", transport="stdio", command="echo")

        with pytest.raises(MCPConnectionError, match="未连接"):
            await client.list_tools()

    @patch("twinspark.tools.mcp.client.ClientSession")
    @patch("twinspark.tools.mcp.client.stdio_client")
    async def test_call_tool_multi_content_blocks(self, mock_stdio, mock_session_cls):
        """工具返回多个内容块时，结果按换行拼接。"""
        mock_stdio.return_value = _fake_stdio_transport()
        fake_session = _make_fake_session()
        fake_session.call_tool.return_value = FakeCallToolResult(
            content=[
                FakeTextBlock(text="第一行"),
                FakeTextBlock(text="第二行"),
            ]
        )

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=fake_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = MCPClient(name="multi-block", transport="stdio", command="echo")
        await client.connect()

        result = await client.call_tool("some_tool", {})
        assert result == "第一行\n第二行"
