"""Tests for the passive skill subsystem (loader + retriever)."""

from __future__ import annotations

from pathlib import Path

import pytest

from twinspark.skills.loader import Skill, SkillLoader
from twinspark.skills.retriever import SkillRetriever


# --------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------- #
def _write_skill(base: Path, category: str, name: str, content: str) -> Path:
    """Write a ``SKILL.md`` at ``base/category/name/SKILL.md`` and return it."""
    d = base / category / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


GOOD_SKILL = """\
---
name: summarize-text
description: 把长文本压缩成要点摘要。
tags: [summary, text, nlp]
---

# 总结长文本

将长文档提炼为若干条要点。

## 步骤
1. 通读全文,识别主题。
2. 抽取关键句。
3. 用要点列表输出。
"""

GIT_SKILL = """\
---
name: git-commit
description: 编写规范的 git commit message。
tags: [git, commit]
---

# Git 提交信息

遵循约定式提交格式。
"""

MISSING_FIELDS_SKILL = """\
---
tags: [misc]
---

# 无名技能

frontmatter 缺少 name 和 description。
"""

BAD_YAML_SKILL = """\
---
name: broken
description: 坏的 YAML
tags: [a, b
  bad: : indent
---

# 坏技能

这个文件的 frontmatter 无法被解析。
"""

NO_FRONTMATTER_SKILL = """\
# 纯正文

这个文件没有 frontmatter。
"""


@pytest.fixture()
def skills_dir(tmp_path) -> Path:
    """A skills directory populated with several SKILL.md variants."""
    base = tmp_path / "skills"
    _write_skill(base, "general", "summarize-text", GOOD_SKILL)
    _write_skill(base, "dev", "git-commit", GIT_SKILL)
    _write_skill(base, "misc", "no-name", MISSING_FIELDS_SKILL)
    _write_skill(base, "broken", "bad-yaml", BAD_YAML_SKILL)
    return base


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #
def test_load_all_parses_valid_skills(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir=skills_dir)
    skills = loader.load_all()

    names = {s.name for s in skills}
    # The good skills load; the bad-YAML one is skipped.
    assert "summarize-text" in names
    assert "git-commit" in names
    assert "broken" not in names

    by_name = {s.name: s for s in skills}
    good = by_name["summarize-text"]
    assert good.category == "general"
    assert good.description == "把长文本压缩成要点摘要。"
    assert good.tags == ["summary", "text", "nlp"]
    assert "总结长文本" in good.body
    assert good.path is not None and good.path.name == "SKILL.md"


def test_bad_yaml_skipped_not_crashing(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir=skills_dir)
    # Should not raise even though one SKILL.md has malformed YAML.
    skills = loader.load_all()
    assert all(s.name != "broken" for s in skills)
    # 3 valid skills remain (good, git, missing-fields).
    assert len(skills) == 3


def test_missing_fields_get_defaults(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir=skills_dir)
    by_name = {s.name: s for s in loader.load_all()}
    # name defaults to the containing directory name.
    assert "no-name" in by_name
    skill = by_name["no-name"]
    assert skill.description == ""
    assert skill.tags == ["misc"]
    assert skill.category == "misc"


def test_no_frontmatter_uses_dir_name(tmp_path) -> None:
    base = tmp_path / "skills"
    _write_skill(base, "general", "plain", NO_FRONTMATTER_SKILL)
    loader = SkillLoader(skills_dir=base)
    skills = loader.load_all()
    assert len(skills) == 1
    assert skills[0].name == "plain"
    assert skills[0].description == ""
    assert "纯正文" in skills[0].body


def test_list_skills_summary(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir=skills_dir)
    summary = loader.list_skills()
    assert isinstance(summary, list)
    assert all(set(item.keys()) == {"name", "description", "category", "tags"} for item in summary)
    entry = next(i for i in summary if i["name"] == "summarize-text")
    assert entry["category"] == "general"
    assert entry["tags"] == ["summary", "text", "nlp"]


def test_empty_dir_returns_empty(tmp_path) -> None:
    empty = tmp_path / "skills"
    empty.mkdir()
    loader = SkillLoader(skills_dir=empty)
    assert loader.load_all() == []
    assert loader.list_skills() == []


def test_missing_dir_returns_empty(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    loader = SkillLoader(skills_dir=missing)
    assert loader.load_all() == []


# --------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------- #
def _sample_skills() -> list[Skill]:
    return [
        Skill(
            name="summarize-text",
            description="把长文本压缩成要点摘要。",
            tags=["summary", "text"],
            body="A" * 100,
            category="general",
        ),
        Skill(
            name="git-commit",
            description="编写规范的 git commit message。",
            tags=["git", "commit"],
            body="B" * 100,
            category="dev",
        ),
        Skill(
            name="python-debug",
            description="调试 python 程序。",
            tags=["python", "debug"],
            body="C" * 100,
            category="dev",
        ),
    ]


def test_retrieve_hits_by_query() -> None:
    retriever = SkillRetriever(_sample_skills())
    hits = retriever.retrieve("怎样写好 git commit", top_n=3)
    assert hits
    assert hits[0].name == "git-commit"


def test_retrieve_orders_by_score() -> None:
    skills = [
        Skill(name="git-basic", description="git 基础", tags=["git"], body="x"),
        Skill(name="git-commit", description="git commit 提交", tags=["git", "commit"], body="y"),
    ]
    retriever = SkillRetriever(skills)
    # Query mentions both git and commit; git-commit should outrank git-basic.
    hits = retriever.retrieve("git commit", top_n=2)
    assert [s.name for s in hits] == ["git-commit", "git-basic"]


def test_retrieve_via_loader(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir=skills_dir)
    retriever = SkillRetriever(loader)
    hits = retriever.retrieve("summary text", top_n=3)
    assert hits and hits[0].name == "summarize-text"


def test_empty_query_returns_empty() -> None:
    retriever = SkillRetriever(_sample_skills())
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_no_match_returns_empty() -> None:
    retriever = SkillRetriever(_sample_skills())
    assert retriever.retrieve("完全无关的查询词汇xyzzy") == []


def test_char_budget_limits_selection() -> None:
    skills = [
        Skill(name="git-a", description="git commit a", tags=["git"], body="A" * 100),
        Skill(name="git-b", description="git commit b", tags=["git"], body="B" * 100),
        Skill(name="git-c", description="git commit c", tags=["git"], body="C" * 100),
    ]
    retriever = SkillRetriever(skills)
    # Budget only fits two 100-char bodies.
    hits = retriever.retrieve("git commit", top_n=5, char_budget=250)
    assert len(hits) == 2


def test_top_n_limits_selection() -> None:
    retriever = SkillRetriever(_sample_skills())
    hits = retriever.retrieve("git python text", top_n=1)
    assert len(hits) == 1


# --------------------------------------------------------------------- #
# Injection text
# --------------------------------------------------------------------- #
def test_build_injection_text_contains_skills() -> None:
    retriever = SkillRetriever(_sample_skills())
    hits = retriever.retrieve("git commit", top_n=1)
    text = retriever.build_injection_text(hits)
    assert "## 可用技能参考" in text
    assert "### git-commit" in text
    assert "编写规范的 git commit message。" in text


def test_build_injection_text_empty() -> None:
    retriever = SkillRetriever(_sample_skills())
    assert retriever.build_injection_text([]) == ""


def test_build_injection_text_truncates() -> None:
    big = Skill(name="big", description="d", tags=["t"], body="Z" * 5000)
    retriever = SkillRetriever([big])
    text = retriever.build_injection_text([big], char_budget=200)
    assert len(text) <= 200
    assert "截断" in text
