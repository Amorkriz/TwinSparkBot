"""Passive skill loading for TwinSpark.

A *skill* is a Markdown document (``SKILL.md``) that captures a reusable
workflow, procedure, or best-practice. Unlike an executable tool, a skill is
**passive**: it is never run. Instead its body is retrieved by relevance and
injected into the system prompt (see :mod:`twinspark.skills.retriever`) so the
model can follow the guidance in-context.

Layout on disk::

    <skills_dir>/<category>/<name>/SKILL.md

Each ``SKILL.md`` begins with a YAML frontmatter block delimited by ``---``
lines, followed by a free-form Markdown body::

    ---
    name: skill-name
    description: One sentence on what this does and when to use it.
    tags: [tag1, tag2]
    ---

    # Human Title
    Detailed steps / procedure / best-practices...

The frontmatter format follows the authoring standard used by hermes-agent's
``agent/learn_prompt.py``; only ``name``, ``description`` and ``tags`` are
consumed here.

Skill source directory
----------------------
By default skills are loaded from ``cfg.skills_dir`` (``~/.twinspark/skills/``).
The repository also ships a small set of *example* skills under
``twinspark/skills/examples/`` purely as reference material — they are **not**
loaded automatically. To try one, copy its ``<category>/<name>/`` directory
into your ``skills_dir``::

    cp -r twinspark/skills/examples/general/example-skill \
          ~/.twinspark/skills/general/

Tests (and other callers) may point :class:`SkillLoader` at any directory by
passing ``skills_dir`` explicitly, which also avoids importing the runtime
config (and therefore the ``DASHSCOPE_API_KEY`` requirement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

__all__ = ["Skill", "SkillLoader"]

# Name of the manifest file inside every ``<category>/<name>/`` directory.
SKILL_FILENAME = "SKILL.md"


@dataclass
class Skill:
    """A single loaded skill.

    Attributes:
        name: The skill's identifier (from frontmatter ``name``; falls back to
            the containing directory name).
        description: One-sentence summary of what the skill does / when to use.
        tags: List of lowercase keyword tags used for retrieval matching.
        body: The Markdown body (everything after the frontmatter block).
        category: The category directory the skill lives under (e.g. ``general``).
        path: Absolute path to the ``SKILL.md`` file it was parsed from.
    """

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    body: str = ""
    category: str = ""
    path: Optional[Path] = None


class SkillLoader:
    """Discovers and parses ``SKILL.md`` files under a skills directory.

    Args:
        skills_dir: Directory to scan. When ``None`` (the default) the value is
            resolved lazily from :func:`twinspark.config.get_config`
            (``cfg.skills_dir``) on first use — importing this module never
            triggers config loading, so tests can construct a loader with an
            explicit path without needing ``DASHSCOPE_API_KEY``.
    """

    def __init__(self, skills_dir: Optional[str | Path] = None) -> None:
        # Store the raw value; resolve lazily so that merely constructing the
        # loader (or importing the module) never requires the runtime config.
        self._skills_dir_arg: Optional[Path] = (
            Path(skills_dir).expanduser() if skills_dir is not None else None
        )

    @property
    def skills_dir(self) -> Path:
        """Return the resolved skills directory, consulting config if needed."""
        if self._skills_dir_arg is not None:
            return self._skills_dir_arg
        # Import lazily so tests that pass an explicit dir don't require the
        # DASHSCOPE_API_KEY that get_config() demands.
        from twinspark.config import get_config

        return get_config().skills_dir

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_all(self) -> list[Skill]:
        """Recursively load every valid skill under the skills directory.

        Scans ``<skills_dir>/<category>/<name>/SKILL.md``. A missing or empty
        directory yields an empty list. Files whose frontmatter cannot be
        parsed are skipped with a logged warning rather than raising.

        Returns:
            A list of :class:`Skill`, sorted by ``(category, name)`` for stable
            output.
        """
        base = self.skills_dir
        if not base.exists() or not base.is_dir():
            return []

        skills: list[Skill] = []
        for manifest in sorted(base.rglob(SKILL_FILENAME)):
            skill = self._parse_skill_md(manifest)
            if skill is not None:
                skills.append(skill)

        skills.sort(key=lambda s: (s.category, s.name))
        return skills

    def _parse_skill_md(self, path: Path) -> Optional[Skill]:
        """Parse a single ``SKILL.md`` into a :class:`Skill`.

        Separates the YAML frontmatter (between the first pair of ``---``
        lines) from the Markdown body and parses the frontmatter with
        :func:`yaml.safe_load`. Missing fields fall back to sensible defaults;
        the ``name`` defaults to the containing directory name. Returns
        ``None`` (and logs a warning) when the file cannot be read or the YAML
        is malformed, so a bad skill never crashes ``load_all``.

        The ``category`` is derived from the path relative to ``skills_dir``:
        the first path component is treated as the category (``""`` when the
        skill sits directly under the skills root).
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Skipping skill %s: cannot read file (%s)", path, exc)
            return None

        frontmatter_text, body = self._split_frontmatter(text)

        meta: dict = {}
        if frontmatter_text is not None:
            try:
                parsed = yaml.safe_load(frontmatter_text)
            except yaml.YAMLError as exc:
                logger.warning("Skipping skill %s: invalid YAML frontmatter (%s)", path, exc)
                return None
            if parsed is None:
                meta = {}
            elif isinstance(parsed, dict):
                meta = parsed
            else:
                logger.warning(
                    "Skipping skill %s: frontmatter is not a mapping (%r)", path, type(parsed)
                )
                return None

        category = self._derive_category(path)
        default_name = path.parent.name or "unnamed-skill"

        name = str(meta.get("name") or default_name).strip()
        description = str(meta.get("description") or "").strip()
        tags = self._normalize_tags(meta.get("tags"))

        return Skill(
            name=name,
            description=description,
            tags=tags,
            body=body.strip(),
            category=category,
            path=path,
        )

    # ------------------------------------------------------------------ #
    # Summaries
    # ------------------------------------------------------------------ #
    def list_skills(self) -> list[dict]:
        """Return lightweight skill summaries for the CLI ``/skills`` and API.

        Returns:
            A list of dicts ``{name, description, category, tags}`` — one per
            loaded skill, in the same order as :meth:`load_all`.
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "tags": list(s.tags),
            }
            for s in self.load_all()
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_frontmatter(text: str) -> tuple[Optional[str], str]:
        """Split ``text`` into (frontmatter, body).

        The frontmatter is the block between the first two ``---`` fences when
        the document *starts* with ``---``. When there is no frontmatter the
        first element is ``None`` and the body is the whole document.
        """
        stripped = text.lstrip("\ufeff")  # tolerate a leading BOM
        lines = stripped.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, text

        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter = "\n".join(lines[1:idx])
                body = "\n".join(lines[idx + 1 :])
                return frontmatter, body

        # Opening fence but no closing fence: treat everything as frontmatter.
        return "\n".join(lines[1:]), ""

    def _derive_category(self, path: Path) -> str:
        """Derive the category (first path component under ``skills_dir``)."""
        try:
            rel = path.relative_to(self.skills_dir)
        except ValueError:
            # Path not under skills_dir (shouldn't happen via load_all): fall
            # back to the grandparent directory name.
            return path.parent.parent.name if path.parent.parent else ""
        parts = rel.parts
        # parts == (category, name, "SKILL.md"); a skill directly under the
        # root would be (name, "SKILL.md") and thus have no category.
        return parts[0] if len(parts) >= 3 else ""

    @staticmethod
    def _normalize_tags(raw: object) -> list[str]:
        """Coerce a frontmatter ``tags`` value into a clean ``list[str]``.

        Accepts a YAML list (``[a, b]``) or a comma-separated string
        (``"a, b"``); anything else yields an empty list. Tags are stripped,
        lowercased and de-duplicated while preserving order.
        """
        items: list[str]
        if raw is None:
            items = []
        elif isinstance(raw, str):
            items = raw.split(",")
        elif isinstance(raw, (list, tuple)):
            items = [str(x) for x in raw]
        else:
            items = []

        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            tag = item.strip().lower()
            if tag and tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result
