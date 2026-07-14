"""Abstract base class for TwinSpark tools.

A *tool* is a single capability the agent can invoke through the LLM
function-calling interface (e.g. reading a file, running a shell command,
searching the web).  Concrete tools subclass :class:`Tool`, declare their
``name`` / ``description`` / ``parameters`` (a JSON Schema describing the
call arguments) and implement :meth:`run`.

This module deliberately contains *no* concrete tools — it only provides the
contract that later tasks build on.  The registry (see
``twinspark.tools.registry``) collects :class:`Tool` instances and exposes
their OpenAI function-calling schemas to the agent core.
"""

from __future__ import annotations

import abc
from typing import Any, Dict


class Tool(abc.ABC):
    """Abstract base class every TwinSpark tool must extend.

    Subclasses are expected to set the class attributes below (or override
    them as instance attributes) and implement :meth:`run`.

    Attributes:
        name: Unique identifier used by the LLM to reference the tool.
        description: Human/LLM-readable summary of what the tool does. Used
            verbatim in the function-calling schema.
        parameters: JSON Schema (draft-07 style ``object``) describing the
            accepted keyword arguments. Defaults to an empty parameter object.
        is_async: Whether :meth:`run` is a coroutine. The agent core inspects
            this flag to decide between ``await tool.run(...)`` and a plain
            ``tool.run(...)`` call.
    """

    #: Unique tool name exposed to the model. Subclasses MUST override.
    name: str = ""

    #: Short description of the tool's behaviour, surfaced to the model.
    description: str = ""

    #: JSON Schema for the tool's arguments. Defaults to a no-argument object.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    #: Set to ``True`` when :meth:`run` is defined as ``async def``.
    is_async: bool = False

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool.

        Concrete tools implement the actual behaviour here. The method may be
        synchronous or, when :attr:`is_async` is ``True``, defined as
        ``async def run(self, **kwargs)``.

        Args:
            **kwargs: Arguments matching the tool's :attr:`parameters` schema.

        Returns:
            The tool result. The type is tool-specific; the agent core is
            responsible for serialising it back to the model.

        Raises:
            NotImplementedError: If a subclass fails to implement this method.
        """
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool.

        Returns:
            A dict of the shape::

                {
                    "type": "function",
                    "function": {
                        "name": <self.name>,
                        "description": <self.description>,
                        "parameters": <self.parameters>,
                    },
                }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, is_async={self.is_async})"
