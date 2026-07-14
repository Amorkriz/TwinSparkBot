"""Passive skill subsystem.

Skills are Markdown documents (``SKILL.md``) that are *loaded and injected*,
never executed. :class:`~twinspark.skills.loader.SkillLoader` discovers and
parses them; :class:`~twinspark.skills.retriever.SkillRetriever` selects the
most relevant ones for a query and renders them for system-prompt injection.
"""

from twinspark.skills.loader import Skill, SkillLoader
from twinspark.skills.retriever import SkillRetriever

__all__ = ["Skill", "SkillLoader", "SkillRetriever"]
"""Skills subsystem (placeholder; implemented in later tasks)."""
