# retool_small_llm · 知识库索引

本目录存放本项目（ReTool-style Agentic RL for Mathematical Reasoning with Small LLMs）的知识文档，按**文档功能**分三类归档，不再平铺。

## 目录结构

| 目录 | 用途 | 现有文件 |
|---|---|---|
| `architecture/` | 课题架构与研究设计 | `project_architecture.md`（从 0 到 1 行动指南）、`phase0_environment_setup.md`（verl+SuperPOD 环境打通 Runbook）、`superpod_workspace.md`（**集群目录/环境/路径速查表**） |
| `literature/` | 论文阅读与笔记 | `reading_list.md`（阅读清单）、`study_notes.md`（ReTool 共读笔记） |
| `presentation/` | 各次汇报材料 | `presentation_1_plan.md`（第一次汇报方案定稿）、`presentation_1_slides.md`（第一次汇报逐页稿） |

> 论文原文（PDF / txt）在项目根目录的 `references/`，不在此处。

## 归档约定（后续新增文档时遵守）

- **新论文笔记 / 阅读清单** → 放 `literature/`。
- **新的课题设计 / 实验设计文档** → 放 `architecture/`。
- **每次汇报的材料**（方案、逐页稿、演练稿）→ 放 `presentation/`，按汇报编号命名：第二次汇报用 `presentation_2_*`，依此类推。
- 跨目录引用一律写**相对路径**（如 `../architecture/project_architecture.md`），保证移动后链接仍有效。

## 快速导航

- 想回看「为什么做这个课题 / 三阶段管线 / 实验矩阵」→ `architecture/project_architecture.md`
- **上集群操作前必看**「东西放在哪 / 环境什么版本 / 路径怎么写」→ `architecture/superpod_workspace.md`
- 想回看「论文读到了哪、结论是什么」→ `literature/study_notes.md` + `literature/reading_list.md`
- 想准备第一次汇报 → `presentation/presentation_1_plan.md`（方案）+ `presentation/presentation_1_slides.md`（逐页稿）
