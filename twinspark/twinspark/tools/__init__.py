"""Tools subsystem: base :class:`Tool` contract and the :class:`ToolRegistry`.

Concrete tools are added in later tasks. This package currently exposes only
the abstractions and the module-level default ``registry`` singleton.
"""

from twinspark.tools.base import Tool
from twinspark.tools.registry import ToolRegistry, registry

__all__ = ["Tool", "ToolRegistry", "registry"]
