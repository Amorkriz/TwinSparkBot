---
name: git-commit-message
description: 写清晰规范的 git commit message,遵循约定式提交。
tags: [git, commit, workflow, changelog]
---

# 规范的 Git Commit Message

这个技能给出编写清晰、可检索、可用于生成 changelog 的 git 提交信息的
标准做法。它只规定“写什么、怎么写”,不代替你执行任何 git 命令。

## 何时使用
- 准备提交代码前,需要一条描述本次改动的 commit message。
- 想让提交历史遵循约定式提交(Conventional Commits)以便自动生成日志。

## 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type**:改动类型,常用取值:
  - `feat` 新功能
  - `fix` 修复缺陷
  - `docs` 仅文档
  - `refactor` 重构(既不修 bug 也不加功能)
  - `test` 增删测试
  - `chore` 构建/依赖/杂项
- **scope**(可选):影响范围,如模块或文件名。
- **subject**:一句话祈使句,首字母小写,结尾不加句号,建议 ≤ 50 字符。

## 正文与脚注
- **body**:说明“为什么”这么改,而不是简单复述“改了什么”。每行建议 ≤ 72 字符。
- **footer**:关联 issue 或标注破坏性变更,例如:
  - `Closes #123`
  - `BREAKING CHANGE: 配置项 foo 已被移除,请改用 bar。`

## 示例

```
feat(skills): 新增被动技能检索与注入

按查询对技能做关键词打分,取 top-N 并在 char_budget 内
拼成系统提示片段。技能只做参考注入,不执行。

Closes #42
```

## 注意事项
- 一次提交只做一件事;跨越多个关注点时拆成多个提交。
- subject 用祈使句(“add”而非“added/adds”)。
- 避免无信息量的信息,如 “update” / “fix bug” / “wip”。
