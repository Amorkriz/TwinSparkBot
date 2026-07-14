"""Query-driven retrieval of passive skills for prompt injection.

:class:`SkillRetriever` scores loaded :class:`~twinspark.skills.loader.Skill`
objects against a user query using a lightweight keyword-overlap heuristic (no
embeddings), selects the most relevant few within a character budget, and
renders them into a block of text that the agent core (Task 6) injects into the
system prompt.

The scoring is intentionally simple and dependency-free: both the query and
each skill's searchable text (name + description + tags) are tokenised into
lowercase word tokens, and a skill's score is the sum of query-token
frequencies it contains, with a small weight boost for matches in the name and
tags (which are stronger relevance signals than prose).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Sequence, Union

from twinspark.skills.loader import Skill, SkillLoader

__all__ = ["SkillRetriever"]

# Unicode-aware word tokeniser (mirrors the memory store's tokenizer).
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

# Relative weights: matches in name/tags are stronger signals than description.
_NAME_WEIGHT = 3.0
_TAG_WEIGHT = 2.0
_DESC_WEIGHT = 1.0

_DEFAULT_TOP_N = 3
_DEFAULT_CHAR_BUDGET = 4000

_INJECTION_HEADER = "## 可用技能参考"
_TRUNCATION_NOTICE = "\n\n> [技能内容因长度限制被截断]"


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase word tokens."""
    return _TOKEN_RE.findall(text.lower())


class SkillRetriever:
    """Rank passive skills against a query and render them for injection.

    Args:
        source: Either a :class:`~twinspark.skills.loader.SkillLoader` (whose
            :meth:`~twinspark.skills.loader.SkillLoader.load_all` is called to
            obtain the skill pool) or a ready-made sequence of
            :class:`~twinspark.skills.loader.Skill` objects (handy for tests).
    """

    def __init__(self, source: Union[SkillLoader, Sequence[Skill]]) -> None:
        if isinstance(source, SkillLoader):
            self._loader: Optional[SkillLoader] = source
            self._skills: Optional[list[Skill]] = None
        else:
            self._loader = None
            self._skills = list(source)

    def _all_skills(self) -> list[Skill]:
        """Return the skill pool, loading it from the loader on demand."""
        if self._skills is not None:
            return self._skills
        assert self._loader is not None  # one of the two is always set
        return self._loader.load_all()

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_n: int = _DEFAULT_TOP_N,
        char_budget: int = _DEFAULT_CHAR_BUDGET,
    ) -> list[Skill]:
        """Return the skills most relevant to ``query``.

        Skills are scored by keyword overlap between the query and each skill's
        name + description + tags, then sorted by score (descending). At most
        ``top_n`` skills are returned, and skills are only appended while the
        running total of their body lengths stays within ``char_budget`` — once
        a skill's body would exceed the remaining budget, selection stops.

        Args:
            query: The user query / current task description. An empty or
                whitespace-only query returns an empty list.
            top_n: Maximum number of skills to return.
            char_budget: Maximum cumulative size (in characters) of the
                returned skills' bodies.

        Returns:
            The selected skills, highest score first. Skills with a zero score
            (no keyword overlap) are excluded.
        """
        if not query or not query.strip():
            return []
        if top_n <= 0 or char_budget <= 0:
            return []

        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, int, Skill]] = []
        for idx, skill in enumerate(self._all_skills()):
            score = self._score(query_tokens, skill)
            if score > 0:
                # idx as tie-breaker keeps ordering stable and deterministic.
                scored.append((score, idx, skill))

        # Highest score first; ties resolved by original (idx) order.
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[Skill] = []
        used = 0
        for _score, _idx, skill in scored:
            if len(selected) >= top_n:
                break
            body_len = len(skill.body)
            if used + body_len > char_budget:
                # Stop appending once the budget would be exceeded.
                break
            selected.append(skill)
            used += body_len
        return selected

    def _score(self, query_tokens: Counter, skill: Skill) -> float:
        """Weighted keyword-overlap score of ``skill`` against the query."""
        name_tokens = Counter(_tokenize(skill.name))
        tag_tokens = Counter(_tokenize(" ".join(skill.tags)))
        desc_tokens = Counter(_tokenize(skill.description))

        score = 0.0
        for token, q_count in query_tokens.items():
            if token in name_tokens:
                score += _NAME_WEIGHT * q_count * name_tokens[token]
            if token in tag_tokens:
                score += _TAG_WEIGHT * q_count * tag_tokens[token]
            if token in desc_tokens:
                score += _DESC_WEIGHT * q_count * desc_tokens[token]
        return score

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def build_injection_text(
        self,
        skills: Iterable[Skill],
        char_budget: int = _DEFAULT_CHAR_BUDGET,
    ) -> str:
        """Render ``skills`` into a system-prompt injection block.

        Produces a section like::

            ## 可用技能参考

            ### skill-name
            <description>

            <body>

            ---

            ### other-skill
            ...

        The whole block is capped at ``char_budget`` characters; if rendering
        would exceed it, the text is truncated at the budget and a clear notice
        (:data:`_TRUNCATION_NOTICE`) is appended.

        Args:
            skills: The skills to render (typically the output of
                :meth:`retrieve`).
            char_budget: Maximum size of the returned string.

        Returns:
            The injection block, or an empty string when ``skills`` is empty.
        """
        skills = list(skills)
        if not skills:
            return ""

        parts: list[str] = [_INJECTION_HEADER, ""]
        for i, skill in enumerate(skills):
            if i > 0:
                parts.append("---")
                parts.append("")
            parts.append(f"### {skill.name}")
            if skill.description:
                parts.append(skill.description)
            if skill.body:
                parts.append("")
                parts.append(skill.body)
            parts.append("")

        text = "\n".join(parts).rstrip() + "\n"

        if len(text) > char_budget:
            # Reserve room for the notice so the final string still fits.
            keep = max(0, char_budget - len(_TRUNCATION_NOTICE))
            text = text[:keep].rstrip() + _TRUNCATION_NOTICE
        return text
