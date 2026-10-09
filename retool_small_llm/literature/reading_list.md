# 论文阅读清单（ReTool Small LLM 课题）

> 阅读顺序按"投入产出比"排列。每篇标注**精读/选读**与**怎么读**。
> 状态图例：⬜ 未读 / 🔄 阅读中 / ✅ 读完

## Tier 0 — 现在就读（精读）

### 1. ReTool：课题原型
- **arXiv**: [2504.11536](https://arxiv.org/abs/2504.11536) ｜ [PDF 下载](https://arxiv.org/pdf/2504.11536)
- **必读章节**：Abstract、Intro、Method（ReTool-SFT 冷启动 + Tool-augmented RL）、实验设置与结果表
- **略读**：附录训练细节
- **读完应能回答**：三阶段分别做什么？奖励怎么判？大模型上比纯文本 RL 强多少（AIME 数字）？
- 状态：✅ 已共读完成（2026-09-06，全文收官总结见 study_notes.md 讨论7）

### 2. DeepSeek-R1 技术报告：GRPO 与规则奖励的出处
- **arXiv**: [2501.12948](https://arxiv.org/abs/2501.12948) ｜ [PDF 下载](https://arxiv.org/pdf/2501.12948)
- **必读章节**：Intro、**R1-Zero 部分**（纯 RL 训练方式）、数学/代码 RL 设置
- **略读**：蒸馏（distillation）、与各模型的对比细节
- **读完应能回答**：GRPO 和 PPO 差在哪？为什么数学题能用规则奖励？
- 状态：⬜

## Tier 1 — 读完 ReTool 后（精读核心部分）

### 3. DAPO：首选 RL 算法
- **arXiv**: [2503.14476](https://arxiv.org/abs/2503.14476) ｜ [PDF 下载](https://arxiv.org/pdf/2503.14476)
- **必读**：Abstract + **4 个关键技术点**（Clip-Higher / Dynamic Sampling / Token-Level PG Loss / Overlong Reward Shaping）+ 实验表
- **读完应能回答**：DAPO 修了 GRPO 的哪 4 个坑？动态采样为什么对小模型重要？
- 状态：⬜

### 4. PPO：RL 理论基础（只读核心）
- **arXiv**: [1707.06347](https://arxiv.org/abs/1707.06347) ｜ [PDF 下载](https://arxiv.org/pdf/1707.06347)
- **必读**：Sec 3 "PPO with Clipped Objective"（clipped surrogate objective）、GAE 概念
- **读完应能回答**："限制更新幅度防崩"是什么意思？为什么 GRPO 砍掉价值网络？
- 状态：⬜

## Tier 1.5 — 直接竞品（务必精读，写 Related Work 绕不开）

### 4b. ToRL：我的直接竞品（小模型 + RL + 工具）
- **arXiv**: [2503.23383](https://arxiv.org/abs/2503.23383) ｜ [PDF 下载](https://arxiv.org/pdf/2503.23383)
- **背景**：TIGER-AI-Lab 出品。ReTool 论文评价它"在 Qwen2.5-Math 1.5B/7B 上用 RL 学工具使用，效果 suboptimal"——而我课题做的就是这件事（换 Qwen3 + DAPO/GRPO/PPO 对比）
- **必读重点**：① 它 suboptimal 的具体原因（数据？算法？设置？）——避免我踩同样的坑；② 它的方法与 ReTool 的差异
- **代码**：`TIGER-AI-Lab/verl-tool`（ToRL 是原生 recipe，可作对照实现）
- **读完应能回答**：我凭什么能做得比 ToRL 好？（动机句）
- 状态：⬜

## Tier 2 — 背景与参考（选读关键结论）

### 5. SimpleRL-Zoo：小模型 RL 的最佳参考
- **arXiv**: [2503.18892](https://arxiv.org/abs/2503.18892) ｜ [PDF 下载](https://arxiv.org/pdf/2503.18892)
- **只读两个关键发现**：① 训练数据难度必须匹配模型能力；② 冷启动 SFT 会限制 RL 探索（与 ReTool 的做法形成对照——是潜在的讨论/研究点）
- 状态：⬜

### 6. verl（HybridFlow）：训练框架设计
- **arXiv**: [2409.19256](https://arxiv.org/abs/2409.19256) ｜ [PDF 下载](https://arxiv.org/pdf/2409.19256)
- **只了解设计思想**：actor 训练与 rollout 解耦、多轮 agentic 支持。不深究系统细节
- 状态：⬜

### 7. ToRA：工具集成推理经典基线（背景）
- **arXiv**: [2309.17452](https://arxiv.org/abs/2309.17452) ｜ [PDF 下载](https://arxiv.org/pdf/2309.17452)
- **只读**：Intro + 方法图。理解"监督式工具集成（SFT）"与"RL 式工具策略"的区别
- 状态：⬜

## 非论文必读

### 8. ReTool recipe Notion 实操文档
- **链接**: [verl reTool recipe](https://app.notion.com/p/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
- **用途**：理论→实践的桥梁。等论文看完、开始搭环境时再细读
- 状态：⬜

---

## 阅读心法（新手适用）

1. **别按小说顺序啃**——每篇先看 Abstract + 图 + 表格，再回到方法。
2. **遇到公式不要卡**——先划过去，把"这段在解决什么问题"记下来，我们讨论时再补数学。
3. **每篇读完写 3 句话**：① 它在做什么 ② 关键做法 ③ 对我课题意味着什么。
4. **带着问题来讨论**——读到任何不懂的段落，直接贴过来，我逐段给你讲。
