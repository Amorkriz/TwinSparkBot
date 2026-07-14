"""Minimal in-memory registry for TwinSpark tools.

The registry maps a tool ``name`` to a single :class:`~twinspark.tools.base.Tool`
instance and exposes helpers the agent core needs — most importantly
:meth:`ToolRegistry.get_openai_schemas`, which returns the function-calling
schemas for every registered tool.

A module-level singleton :data:`registry` is provided for global use so tool
modules can simply do::

    from twinspark.tools.registry import registry

    @registry.register
    class MyTool(Tool):
        ...

or register an instance directly::

    registry.register(MyTool())

No concrete tools are registered here; that happens in later tasks.
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Type, Union

from twinspark.tools.base import Tool


class ToolRegistry:
    """A minimal name -> :class:`Tool` instance registry."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Union[Tool, Type[Tool]]) -> Union[Tool, Type[Tool]]:
        """Register a tool by instance or class.

        Supports two usage styles:

        * Direct call with an instance: ``registry.register(MyTool())``.
        * Decorator on a ``Tool`` subclass: ``@registry.register`` — the class
          is instantiated with no arguments and the *class* is returned
          unchanged so the decorated name still refers to the class.

        Args:
            tool: A :class:`Tool` instance, or a :class:`Tool` subclass to be
                instantiated with no arguments.

        Returns:
            The same object that was passed in (instance or class), enabling
            decorator usage.

        Raises:
            TypeError: If *tool* is neither a ``Tool`` instance nor subclass.
            ValueError: If the tool's ``name`` is empty or already registered.
        """
        if inspect.isclass(tool):
            if not issubclass(tool, Tool):
                raise TypeError(f"{tool!r} is not a Tool subclass")
            instance = tool()
            self._add(instance)
            return tool

        if not isinstance(tool, Tool):
            raise TypeError(f"{tool!r} is not a Tool instance or subclass")

        self._add(tool)
        return tool

    def _add(self, tool: Tool) -> None:
        """Insert *tool* into the registry, validating its name."""
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has an empty tool name")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool registered under *name*, or ``None`` if absent."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return all registered tool instances (registration order)."""
        return list(self._tools.values())

    def get_openai_schemas(self) -> List[Dict]:
        """Return the OpenAI function-calling schema for every tool.

        The agent core (Task 6) passes this list as the ``tools`` argument to
        the chat-completions call. When no tools are registered this returns an
        empty list.
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def clear(self) -> None:
        """Remove all registered tools (primarily useful for tests)."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


#: Module-level default registry singleton for global use.
registry = ToolRegistry()
