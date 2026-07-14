"""Tests for the tools base class and registry (Task 4).

These tests define throwaway ``Tool`` subclasses locally — no concrete tools
live in the ``twinspark.tools`` package yet.
"""

from __future__ import annotations

import pytest

from twinspark.tools import Tool, ToolRegistry, registry as default_registry
from twinspark.tools.base import Tool as BaseTool


class EchoTool(Tool):
    """A trivial synchronous tool used only for testing."""

    name = "echo"
    description = "Echo the provided message back to the caller."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Text to echo."},
        },
        "required": ["message"],
    }
    is_async = False

    def run(self, **kwargs):
        return kwargs.get("message", "")


class AsyncPingTool(Tool):
    """A trivial asynchronous tool used only for testing."""

    name = "ping"
    description = "Return 'pong'."
    is_async = True

    async def run(self, **kwargs):
        return "pong"


@pytest.fixture
def reg() -> ToolRegistry:
    """A fresh, isolated registry for each test."""
    return ToolRegistry()


def test_default_registry_starts_empty():
    # The global singleton must expose no tools before any registration.
    assert default_registry.get_openai_schemas() == []


def test_tool_defaults_and_abstract():
    # Tool is abstract: cannot be instantiated directly.
    with pytest.raises(TypeError):
        BaseTool()

    # Default parameters schema on a subclass that doesn't override it.
    class NoArgTool(Tool):
        name = "noarg"
        description = "no args"

        def run(self, **kwargs):
            return None

    tool = NoArgTool()
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.is_async is False


def test_to_openai_schema_structure():
    schema = EchoTool().to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"].startswith("Echo")
    assert schema["function"]["parameters"]["type"] == "object"
    assert "message" in schema["function"]["parameters"]["properties"]


def test_register_instance_and_get(reg: ToolRegistry):
    tool = EchoTool()
    returned = reg.register(tool)
    assert returned is tool  # register returns the same object

    fetched = reg.get("echo")
    assert fetched is tool
    assert reg.get("missing") is None
    assert "echo" in reg
    assert len(reg) == 1


def test_register_class_as_decorator(reg: ToolRegistry):
    @reg.register
    class DecoratedTool(Tool):
        name = "decorated"
        description = "registered via decorator"

        def run(self, **kwargs):
            return "ok"

    # Decorator returns the class unchanged.
    assert isinstance(DecoratedTool, type)
    instance = reg.get("decorated")
    assert isinstance(instance, DecoratedTool)


def test_list_tools_and_schemas(reg: ToolRegistry):
    echo = EchoTool()
    ping = AsyncPingTool()
    reg.register(echo)
    reg.register(ping)

    tools = reg.list_tools()
    assert tools == [echo, ping]  # registration order preserved

    schemas = reg.get_openai_schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"echo", "ping"}
    assert all(s["type"] == "function" for s in schemas)


def test_duplicate_registration_raises(reg: ToolRegistry):
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.register(EchoTool())


def test_empty_name_raises(reg: ToolRegistry):
    class Nameless(Tool):
        name = ""
        description = "no name"

        def run(self, **kwargs):
            return None

    with pytest.raises(ValueError):
        reg.register(Nameless())


def test_register_invalid_type_raises(reg: ToolRegistry):
    with pytest.raises(TypeError):
        reg.register(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        reg.register(dict)  # a class, but not a Tool subclass


def test_clear(reg: ToolRegistry):
    reg.register(EchoTool())
    assert len(reg) == 1
    reg.clear()
    assert len(reg) == 0
    assert reg.get_openai_schemas() == []


@pytest.mark.asyncio
async def test_async_tool_run():
    result = await AsyncPingTool().run()
    assert result == "pong"
