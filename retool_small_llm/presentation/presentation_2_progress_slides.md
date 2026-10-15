# Second Presentation — Progress Report: Initial Results（中英对照逐页讲稿）

> **本文件是什么**：Meeting 3「Progress Report, initial results expected」（10–15 分钟）的完整讲稿。每页给出：**幻灯片文字（中英）**、**配图说明**、**口头逐字稿（中英对照）**。
> **用法**：先读 §1（时间分配）→ §2（逐页）→ §3（预期提问）→ §4（演练提示）。讲稿全程中英对照，可只念一种语言。
> **上传文件名**：`<CourseCode>-2 Progress <Your-Full-Name> ReTool-style Agentic RL for Mathematical`（课程代码 CSIT6910 / MSBD6910 尚未确认，上传前问清）。
> **关联数据**：本讲稿所有数字来自冻结协议评测（AIME 2024/2025 × 12 采样，job 686226）与 Stage 1 验收报告（DAPO 271 步训练，job 675200）。

---

## 0. 使用说明（仅此一节为中文）

- 本次汇报是**进度报告**：主线是「我已完成什么 → 首批结果是什么 → 接下来预期什么」，背景与方法只做极简回顾（评审已经听过 Proposal）。
- 每个数字都能溯源：结果表 → job 686226；训练曲线 → job 675200（见 §6 参考文献）。
- 时长目标 **约 11 分钟**，落在 10–15 分钟区间。§4 给了压缩/超时的兜底。
- 只讲你（**成员 A**）的切片：Qwen3-1.7B + GRPO vs DAPO。不展开 B/C 的实验。

---

## 1. 时间分配（目标 ≈ 11 分钟）

| 页 | 标题 | 时长 |
|---|---|---|
| 1 | 标题页 | 0:15 |
| 2 | 回顾：目标与承诺 | 0:40 |
| 3 | 已完成：管线端到端跑通 | 0:50 |
| 4 | 冻结评测协议 | 0:40 |
| 5 | **首批结果总表** | 1:20 |
| 6 | 发现一：RL 增益成立且显著 | 1:00 |
| 7 | 发现二：增益由工具驱动 | 0:50 |
| 8 | 发现三：SFT 负迁移 | 0:50 |
| 9 | 假设检验状态 | 1:00 |
| 10 | 训练曲线与稳定性证据 | 0:50 |
| 11 | 下一步与预期 | 1:00 |
| 12 | 总结 | 0:30 |

**叙事主线**：*我承诺了什么（2）→ 我做到了什么（3–4）→ 首批结果是什么（5）→ 三个发现（6–8）→ 假设现在站不站得住（9–10）→ 接下来预期什么（11–12）。*

---

## 2. 逐页讲稿

---

### Slide 1 — 标题页（Title）

**幻灯片文字（中文）**

> # ReTool 式小模型 agentic RL：进展报告
> **Progress Report — Initial Results**
> [姓名] · [课程代码] · 2026-10-20/21
> *我的切片：Qwen3-1.7B · GRPO vs DAPO 受控对比*

**幻灯片文字（English）**

> # ReTool-style Agentic RL for Mathematical Reasoning with Small LLMs
> **Progress Report — Initial Results**
> [Your Full Name] · [Course Code] · Oct 20–21, 2026
> *My slice: Qwen3-1.7B · a controlled GRPO-vs-DAPO comparison*

**配图**：标题版式；右侧放一个小图——「LLM ⇄ 沙箱」循环箭头（ReTool 核心循环）。无数据。

**口头逐字稿**

> 中文：各位好。我是 [姓名]。上次汇报我提出了一个受控对比——在最小的可行模型 Qwen3-1.7B 上复现 ReTool 管线，只改 RL 更新规则，对比 GRPO 与 DAPO。今天报告进度与首批结果。
>
> EN: Good [morning/afternoon]. I'm [name]. Last time I proposed a controlled comparison — reproducing the ReTool pipeline on the smallest viable model, Qwen3-1.7B, varying only the RL update rule between GRPO and DAPO. Today I report the progress and the first results.

---

### Slide 2 — 回顾：目标与承诺（Recap）

**幻灯片文字（中文）**

> **一句话目标**：在 Qwen3-1.7B 上端到端复现 ReTool，**冻结**基座/数据/沙箱/奖励/评测，**只改 RL 更新**（GRPO vs DAPO，等步预算）。
>
> **我承诺的首批交付**：① 冻结协议基线表（tool + text-only）；② SFT 检查点；③ 一组 RL（DAPO）结果；④ GRPO 对照。

**幻灯片文字（English）**

> **Objective in one line**: reproduce ReTool end-to-end on Qwen3-1.7B, **freezing** base/data/sandbox/reward/eval and **varying only the RL update** (GRPO vs DAPO, equal step budget).
>
> **Promised at this milestone**: ① frozen-protocol baseline table (tool + text-only); ② SFT checkpoint; ③ one RL (DAPO) result; ④ the GRPO comparison.

**配图**：极简——把 Proposal 里的「只动 RL update」那张图缩小复现，其余阶段打锁。

**口头逐字稿**

> 中文：目标不变，一句带过。我承诺这次汇报交付四样：冻结协议的基线表、SFT 检查点、一组 RL 结果、以及 GRPO 对照。前三样今天都有，第四样是「预期下一步」——稍后讲。
>
> EN: Objective unchanged — one sentence: freeze everything, change only the RL update. I promised four things at this milestone: the frozen-protocol baseline, the SFT checkpoint, one RL result, and the GRPO comparison. The first three are here today; the fourth is my expected next step.

---

### Slide 3 — 已完成：管线端到端跑通（Done）

**幻灯片文字（中文）**

> **管线三段全部打通（Qwen3-1.7B）**
>
> | 阶段 | 状态 | 产物 |
> |---|---|---|
> | 环境 + 沙箱 | ✅ | verl 0.6.1 + vllm 0.10.0 + 本地代码沙箱 |
> | 冷启动 SFT | ✅ | 2 epoch，`global_step_124` |
> | RL（DAPO） | ✅ | **271 步 · 9h08m · 8×H800**，reward 由负转正 |
>
> 全程 **8×H800 纯 FSDP**，无 OOM。

**幻灯片文字（English）**

> **All three stages run end-to-end (Qwen3-1.7B)**
>
> | Stage | Status | Output |
> |---|---|---|
> | Env + sandbox | ✅ | verl 0.6.1 + vllm 0.10.0 + local code sandbox |
> | Cold-start SFT | ✅ | 2 epochs, `global_step_124` |
> | RL (DAPO) | ✅ | **271 steps · 9h08m · 8×H800**, reward turns positive |
>
> All on **8×H800, pure FSDP, no OOM**.

**配图**：三格横向流程图，全部打绿勾；右侧小注「271 steps / 9h08m / no OOM」。

**口头逐字稿**

> 中文：进度上，三段管线全部跑通：环境与沙箱、两轮 SFT、以及一组完整的 DAPO 训练——271 步、9 小时、8 张 H800 纯 FSDP、无 OOM。这意味着我现在手上有三个可评测的检查点：base、SFT、RL。
>
> EN: On progress, all three stages are done: the environment and sandbox, a two-epoch SFT, and one full DAPO run — 271 steps, 9 hours, on eight H800s with pure FSDP, no OOM. That gives me three checkpoints to evaluate: base, SFT, and RL.

---

### Slide 4 — 冻结评测协议（Frozen protocol）

**幻灯片文字（中文）**

> **一套评测脚本跑遍所有检查点**
>
> - 评测集：**AIME 2024 + 2025**（各 30 题），**每题 12 次采样**
> - 两种模式：**tool-enabled** 与 **text-only**，每个检查点都跑
> - 奖励：**只看最终答案**（`\boxed{}` 对 +1 / 错 −1），**不为代码可执行性给分**
> - 解码固定：temperature=1.0 · top-p=0.6 · thinking mode 关闭
> - **AIME 永不进训练集**

**幻灯片文字（English）**

> **One eval script across every checkpoint**
>
> - Eval: **AIME 2024 + 2025** (30 each) · **12 samples per problem**
> - Two modes: **tool-enabled** and **text-only**, run at every checkpoint
> - Reward: **final answer only** (`\boxed{}` correct → +1 / wrong → −1), **no reward for executable code**
> - Fixed decoding: temperature=1.0 · top-p=0.6 · thinking mode OFF
> - **AIME never appears in training**

**配图**：检查清单卡片；高亮「不为可执行性给分」「AIME 不进训练集」。

**口头逐字稿**

> 中文：可比性靠这条冻结协议：AIME 2024 加 2025，每题 12 次采样，工具和纯文本两种模式都跑，奖励只看最终 boxed 答案对错，解码参数固定，AIME 不进训练集。下面所有数字都出自这一套脚本。
>
> EN: Comparability rests on this frozen protocol: AIME 2024 and 2025, twelve samples per problem, both tool and text-only modes, reward from the final boxed answer only, fixed decoding, and AIME never in training. Every number that follows comes from this one script.

---

### Slide 5 — 首批结果总表（Headline results）⭐

**幻灯片文字（中文）**

> **AIME 正确率（12 采样协议）——Average@12 / Pass@12**
>
> | 模型 | 模式 | AIME24 Avg | AIME24 Pass | AIME25 Avg | AIME25 Pass |
> |---|---|---|---|---|---|
> | Base | tool | 22.8 | 44.4 | 22.5 | 32.1 |
> | Base | text | 16.4 | 34.7 | 16.9 | 25.2 |
> | SFT | tool | 11.7 | 35.9 | 14.4 | 31.6 |
> | SFT | text | 5.0 | 14.0 | 8.9 | 22.6 |
> | **RL (DAPO)** | **tool** | **41.1** | **70.5** | **35.0** | **55.3** |
> | RL (DAPO) | text | 10.8 | 26.3 | 8.1 | 20.5 |
>
> Avg@12 = 平均每次采样答对率；Pass@12 = 12 次里至少一次答对。

**幻灯片文字（English）**

> **AIME accuracy (12-sample protocol) — Average@12 / Pass@12**
>
> | Model | Mode | AIME24 Avg | AIME24 Pass | AIME25 Avg | AIME25 Pass |
> |---|---|---|---|---|---|
> | Base | tool | 22.8 | 44.4 | 22.5 | 32.1 |
> | Base | text | 16.4 | 34.7 | 16.9 | 25.2 |
> | SFT | tool | 11.7 | 35.9 | 14.4 | 31.6 |
> | SFT | text | 5.0 | 14.0 | 8.9 | 22.6 |
> | **RL (DAPO)** | **tool** | **41.1** | **70.5** | **35.0** | **55.3** |
> | RL (DAPO) | text | 10.8 | 26.3 | 8.1 | 20.5 |
>
> Avg@12 = per-sample solve rate; Pass@12 = ≥1 correct in 12 draws.

**配图**：这张表就是主图。用加粗/高亮把 RL-tool 那一行标出来；下面加一行小注解释 Avg 与 Pass。

**口头逐字稿**

> 中文：这是今天的主表。三个检查点、两种模式、两套 AIME、12 次采样。先看最关键的一行——RL 加工具：AIME2024 的 Pass@12 到了 **70.5%**，AIME2025 是 **55.3%**，而 base 加工具只有 44.4% 和 32.1%。两个指标一起看：Avg@12 衡量「稳不稳」，Pass@12 衡量「会不会」。下面三页我把这张表拆成三个发现。
>
> EN: This is the headline table — three checkpoints, two modes, two AIME sets, twelve samples. Look first at the most important row, RL with tool: Pass@12 reaches **70.5%** on AIME 2024 and **55.3%** on 2025, against 44.4% and 32.1% for the base with tool. Read the two metrics together — Avg@12 is reliability, Pass@12 is capability. The next three slides turn this table into three findings.

---

### Slide 6 — 发现一：RL 增益成立且显著（Finding 1）

**幻灯片文字（中文）**

> **base → RL（tool 模式）**
>
> | | AIME24 Avg@12 | AIME24 Pass@12 | AIME25 Avg@12 | AIME25 Pass@12 |
> |---|---|---|---|---|
> | Base | 22.8 | 44.4 | 22.5 | 32.1 |
> | RL | **41.1** | **70.5** | **35.0** | **55.3** |
> | Δ base→RL | **+18.3** | **+26.1** | **+12.5** | **+23.2** |
>
> **「SFT→RL 增益在最小模型上是否成立」→ 成立（两个口径都为正）。** 容量门槛判断看干净的 **base→RL**：1.7B 的 **+18.3** 小于 32B 的 **+40.3** → **支持容量门槛假说**。Proposal 原口径 **SFT→RL = +29.4** 反而大于 32B 的 +26.1，但被 **SFT 负迁移虚高**（1.7B SFT 11.7 < base 22.8；32B 是 +14），不可直接比较。

**幻灯片文字（English）**

> **base → RL (tool mode)**
>
> | | AIME24 Avg@12 | AIME24 Pass@12 | AIME25 Avg@12 | AIME25 Pass@12 |
> |---|---|---|---|---|
> | Base | 22.8 | 44.4 | 22.5 | 32.1 |
> | RL | **41.1** | **70.5** | **35.0** | **55.3** |
> | Δ base→RL | **+18.3** | **+26.1** | **+12.5** | **+23.2** |
>
> **Does the SFT→RL gain survive the smallest model? Yes — positive under both metrics.** The capacity test uses the clean **base→RL**: 1.7B's **+18.3** is smaller than 32B's **+40.3** → **consistent with a capacity threshold**. The proposal's original **SFT→RL = +29.4** is actually larger than 32B's +26.1, but it is **inflated by SFT negative transfer** (1.7B SFT 11.7 < base 22.8; 32B's is +14), so the two are not directly comparable.

**配图**：两组柱状图（base vs RL，AIME24/25 各一组），标注 Δ。

**口头逐字稿**

> 中文：发现一，RL 增益成立。工具模式下，base 到 RL 的 Avg@12 从 22.8 涨到 41.1，Pass@12 从 44.4 涨到 70.5——显著且真实。所以核心研究问题「SFT→RL 增益是否成立」答案是「成立」，两个口径都为正。但容量门槛的判断要用干净的 base→RL 口径：1.7B 是 +18.3，32B 是 +40.3，增益保留但幅度变小。这里要诚实交代一个口径细节——Proposal 里 H2 写的是「SFT→RL 增益小于 +26」，按字面算我的 SFT→RL 是 +29.4，反而更大。但那是因为我的 SFT 是负迁移（11.7 低于 base 22.8），把基线压低了，而 32B 的 SFT 是正向 +14，所以这个差被虚高，干净的容量对照应该看 base→RL。
>
> EN: Finding one — the RL gain is real. In tool mode, Avg@12 rises from 22.8 to 41.1 and Pass@12 from 44.4 to 70.5 — large and genuine. So the answer to my core question, does the SFT→RL gain hold, is yes — positive under both metrics. But the capacity test uses the clean base→RL metric: 1.7B is +18.3 against 32B's +40.3 — the gain survives, but smaller. I should be honest about one metric nuance: the proposal's H2 was written as "SFT→RL gain below +26." Read literally, my SFT→RL is +29.4, actually larger. But that's because my SFT is negative transfer (11.7, below base 22.8), which depresses the baseline, whereas 32B's SFT is a positive +14 — so that gap is inflated, and the clean capacity comparison is base→RL.

---

### Slide 7 — 发现二：增益由工具驱动（Finding 2）

**幻灯片文字（中文）**

> **RL 学到的不是「更会算」，而是「更会用工具」**
>
> | | tool（Avg@12） | text（Avg@12） |
> |---|---|---|
> | Base | 22.8 | 16.4 |
> | RL | **41.1** | **10.8** ← 反而更低 |
>
> **平均交互轮次**：base **2.0** → SFT **4.8** → RL **8.6**（工具模式）
>
> 纯文本 RL 反而退化到 base 以下 —— 增益全部来自**策略性工具调用**，而非闭卷推理变强。

**幻灯片文字（English）**

> **RL learned *when to use the tool* — not how to calculate better**
>
> | | tool (Avg@12) | text (Avg@12) |
> |---|---|---|
> | Base | 22.8 | 16.4 |
> | RL | **41.1** | **10.8** ← actually lower |
>
> **Average turns**: base **2.0** → SFT **4.8** → RL **8.6** (tool mode)
>
> Text-only RL drops *below* base — the entire gain comes from **strategic tool use**, not stronger closed-book reasoning.

**配图**：左「tool vs text」对比柱；右「轮次 2.0→4.8→8.6」上升折线，箭头标注「用满工具」。

**口头逐字稿**

> 中文：发现二更关键——这个增益是从哪来的。看 RL 的纯文本那一行：AIME24 只有 10.8，比 base 的 16.4 还低。也就是说，RL 之后模型闭卷解题能力没有变强，反而略降；它变强的地方只在工具模式。再看交互轮次：base 几乎不调工具（2.0 轮），SFT 学会格式（4.8），RL 用满工具（8.6，接近 8 轮上限）。结论很干净：RL 学的是「何时、如何调用工具」这一策略，而不是把数学本身算得更准。
>
> EN: Finding two is the sharper one — where does the gain come from. Look at RL in text-only mode: only 10.8 on AIME 2024, *below* base's 16.4. So after RL the model is not better at closed-book math — if anything slightly worse — and it is better only in tool mode. Now the turns: base barely calls the tool (2.0), SFT learns the format (4.8), RL uses it heavily (8.6, near the 8-turn cap). The clean conclusion: RL learns *when and how to call the tool* — a strategy — not how to do the math better.

---

### Slide 8 — 发现三：SFT 负迁移（Finding 3）

**幻灯片文字（中文）**

> **SFT 教了格式，却伤了准确率**
>
> | | base | SFT | Δ |
> |---|---|---|---|
> | tool（Avg@12） | 22.8 | **11.7** | **−11.1** |
> | text（Avg@12） | 16.4 | **5.0** | **−11.4** |
>
> **机制**：SFT 灌入的 `Answer:` 格式与 `\boxed{}` 判分冲突 → 模型被带偏。RL 正是建立在这个格式能力之上，才实现反超。
>
> → 这是一个诚实呈现的 **negative result**。

**幻灯片文字（English）**

> **SFT taught the format — but hurt accuracy**
>
> | | base | SFT | Δ |
> |---|---|---|---|
> | tool (Avg@12) | 22.8 | **11.7** | **−11.1** |
> | text (Avg@12) | 16.4 | **5.0** | **−11.4** |
>
> **Mechanism**: SFT's `Answer:` format conflicts with `\boxed{}` scoring → the model is pulled off-track. RL then builds on this format capability to overtake.
>
> → an honestly reported **negative result**.

**配图**：base vs SFT 的对比柱，下降用红色箭头；旁边小字「格式习得 ✓ / 准确率 ✗」。

**口头逐字稿**

> 中文：发现三是一个诚实的负面结果——SFT 负迁移。base 工具模式 22.8，SFT 掉到 11.7，纯文本 16.4 掉到 5.0，两边都掉了大约 11 个点。原因我们定位到了：SFT 数据里的 `Answer:` 格式和判分用的 `\boxed{}` 冲突，把模型带偏了。但要注意——SFT 教会了模型「怎么调工具」（轮次 2→4.8），正是这个格式能力，让 RL 有基础去反超。所以 SFT 的价值是格式习得，代价是准确率；两者在 RL 阶段被重新组织。这是 Guideline 要求的「清晰报告失败情形」，我如实呈现。
>
> EN: Finding three is an honest negative result — SFT negative transfer. Base in tool mode is 22.8; SFT drops to 11.7, and text mode from 16.4 to 5.0 — roughly 11 points down in both. We traced the cause: the `Answer:` format in the SFT data conflicts with the `\boxed{}` scorer, pulling the model off-track. But note — SFT did teach the model *how* to call the tool (turns 2→4.8), and it is exactly this format capability that lets RL overtake. So SFT's value is format acquisition at a cost to accuracy, and RL reorganizes the two. This is the guideline's "report failures clearly," and I report it as is.

---

### Slide 9 — 假设检验状态（Hypothesis status）

**幻灯片文字（中文）**

> | 假设 | 状态 | 判定 |
> |---|---|---|
> | H1 SFT 提升格式、对准确率提升有限 | ⚠️ 部分证伪 | 格式 ✓（轮次↑），但准确率**反降** |
> | H2 RL 增益 < 32B（容量门槛） | ✅ 支持（base→RL 口径） | base→RL +18.3 < 32B 的 +40.3；SFT→RL +29.4 被负迁移虚高 |
> | H3 DAPO 比 GRPO 更稳定 | ⏳ 待测 | GRPO（E3）尚未跑 |
> | H4 长度先升后降 | 🔶 部分证据 | 轮次/长度持续上升，未见拐点 |
>
> *H2 口径注：Proposal 原写「SFT→RL < +26」，但 1.7B 的 SFT 负迁移（11.7 < base 22.8）使该口径虚高为 +29.4；容量判定改用干净的 base→RL（+18.3 < 32B +40.3）。*
>
> *任何假设被证伪都不等于失败——H1 的证伪本身就有价值。*

**幻灯片文字（English）**

> | Hypothesis | Status | Verdict |
> |---|---|---|
> | H1 SFT raises format, little accuracy gain | ⚠️ partly falsified | format ✓ (turns↑), but accuracy **dropped** |
> | H2 RL gain < 32B (capacity threshold) | ✅ supported (base→RL) | base→RL +18.3 < 32B's +40.3; SFT→RL +29.4 inflated by negative transfer |
> | H3 DAPO more stable than GRPO | ⏳ pending | GRPO (E3) not yet run |
> | H4 length rises then falls | 🔶 partial | turns/length still rising, no peak yet |
>
> *H2 metric note: the proposal wrote "SFT→RL < +26," but 1.7B's negative-transfer SFT (11.7 < base 22.8) inflates that metric to +29.4; the capacity verdict uses the clean base→RL (+18.3 < 32B's +40.3).*
>
> *A falsified hypothesis is not a failure — H1's falsification is itself informative.*

**配图**：四行状态卡，用颜色区分 ✅/⚠️/⏳/🔶。

**口头逐字稿**

> 中文：把结果映射回我上次提的四个假设。H1 部分证伪：SFT 确实提升了格式（轮次上升），但准确率不是「有限提升」而是「下降」。H2 支持，但要说明口径：Proposal 写的是 SFT→RL +26，我的 SFT 是负迁移、这个口径被虚高；改用干净的 base→RL 口径，+18.3 确实小于 32B 的 +40.3，容量门槛成立。H3 待测：GRPO 还没跑，这是下一步。H4 只有部分证据：轮次和长度还在上升，还没看到拐点。再次强调——H1 被证伪不是失败，它本身就说明「小模型的瓶颈不在格式习得」，是有价值的信息。
>
> EN: Mapping the results back to my four hypotheses. H1 is partly falsified: SFT did raise the format (turns up), but accuracy didn't rise a little — it dropped. H2 is supported, with a metric caveat: the proposal wrote SFT→RL +26, but my negative-transfer SFT inflates that metric; on the clean base→RL metric, +18.3 is indeed smaller than 32B's +40.3, consistent with a capacity threshold. H3 is pending: GRPO isn't run yet — that's next. H4 has partial evidence: turns and length are still rising, no peak yet. And again — H1's falsification is not a failure; it says the small model's bottleneck is not format acquisition, which is itself informative.

---

### Slide 10 — 训练曲线与稳定性证据（Training curve）

**幻灯片文字（中文）**

> **DAPO 271 步 · reward 由负转正（critic/score/mean）**
>
> | 步数 | 0 | 50 | 100 | 150 | 200 | 250 | 271 |
> |---|---|---|---|---|---|---|---|
> | score | −0.643 | −0.106 | −0.099 | −0.052 | **+0.217** | +0.151 | **+0.230** |
>
> 同时：平均轮次 4.47 → 10.05；响应长度 2292 → 4821 token。
>
> → 早期即跨零、稳定为正，无熵坍塌迹象（H3 的早期信号）。

**幻灯片文字（English）**

> **DAPO 271 steps · reward crosses zero (critic/score/mean)**
>
> | Step | 0 | 50 | 100 | 150 | 200 | 250 | 271 |
> |---|---|---|---|---|---|---|---|
> | score | −0.643 | −0.106 | −0.099 | −0.052 | **+0.217** | +0.151 | **+0.230** |
>
> Also: mean turns 4.47 → 10.05; response length 2292 → 4821 tokens.
>
> → crosses zero early, stays positive, no entropy collapse (early signal for H3).

**配图**：reward 折线（标注零点穿越）；副图放轮次/长度上升曲线。

**口头逐字稿**

> 中文：训练曲线给了稳定性证据。reward 从 −0.643 一路升到 +0.230，跨过零点并稳定为正——模型确实学会了通过工具拿到正确奖励。轮次和响应长度同步上升，说明它学到的是更长、更有效的工具链，而不是敷衍的短答。全程没有熵坍塌。这给 H3 一个早期正向信号，但要下结论还得等 GRPO。
>
> EN: The training curve is my stability evidence. Reward climbs from −0.643 to +0.230, crosses zero and stays positive — the model genuinely learns to earn reward through the tool. Turns and response length rise together, so it is learning longer, more effective tool chains, not lazy short answers. No entropy collapse throughout. That's an early positive signal for H3, but I need GRPO before concluding.

---

### Slide 11 — 下一步与预期（Next steps）

**幻灯片文字（中文）**

> **已交付**：基线表 ✅ · SFT ✅ · DAPO ✅
> **预期下一步**：
> 1. **GRPO 对照（E3）**——`clip_ratio_high=0.0`，同 271 步预算 → 检验 H3
> 2. **稳定性指标**——熵、退化组比例、回答长度随步数曲线
> 3. **工具行为四指标**——调用率、沙箱成功率、调用时机、自我纠错
> 4. 报告定稿与可复现包（终期 11/25–26，报告 11/30）

**幻灯片文字（English）**

> **Delivered**: baseline table ✅ · SFT ✅ · DAPO ✅
> **Expected next**:
> 1. **GRPO comparison (E3)** — `clip_ratio_high=0.0`, same 271-step budget → tests H3
> 2. **Stability metrics** — entropy, degenerate-group ratio, length-vs-step curves
> 3. **Tool-behavior indicators** — call rate, sandbox success, call timing, self-correction
> 4. Report finalization + reproducibility package (final 11/25–26, report 11/30)

**配图**：已完成打勾清单 + 下一步编号列表；把 GRPO 标成高亮。

**口头逐字稿**

> 中文：最后是预期。已交付三项，接下来按优先级：第一，GRPO 对照，只改 clip 上界、同样的 271 步预算，直接检验 DAPO 是否更稳；第二，稳定性指标曲线；第三，工具行为的四项指标；最后是报告定稿。其中 GRPO 是回答我研究问题里「GRPO vs DAPO」那一半的关键，也是这次汇报唯一还没落地的一项。
>
> EN: Finally, what's expected. Three items are delivered; next, in order: first, the GRPO comparison — change only the clip upper bound, same 271-step budget — to test H3 directly; second, the stability-metric curves; third, the four tool-behavior indicators; last, the report. GRPO is the one promised item still outstanding, and it is the key to the "GRPO vs DAPO" half of my research question.

---

### Slide 12 — 总结（Takeaway）

**幻灯片文字（中文）**

> **一句话结论**
> *ReTool 式 agentic RL 在 1.7B 上成立：AIME2024 Pass@12 从 44.4% → **70.5%**，且增益全部来自策略性工具调用；幅度小于 32B，指向容量门槛。GRPO 对照是下一关键步骤。*

**幻灯片文字（English）**

> **One-sentence takeaway**
> *ReTool-style agentic RL holds at 1.7B: AIME2024 Pass@12 from 44.4% → **70.5%**, driven entirely by strategic tool use; the gain is smaller than at 32B, pointing to a capacity threshold. The GRPO comparison is the next key step.*

**配图**：大号高亮结论框；右上一行小字「谢谢 · 欢迎提问」。

**口头逐字稿**

> 中文：总结一句话：ReTool 式 agentic RL 在最小模型上成立——AIME2024 的 Pass@12 从 44.4% 到 70.5%，增益全部来自策略性工具调用，幅度比 32B 小，指向容量门槛；GRPO 对照是下一步关键。谢谢，欢迎提问。
>
> EN: To close in one sentence: ReTool-style agentic RL holds at the smallest model — AIME2024 Pass@12 goes from 44.4% to 70.5%, driven entirely by strategic tool use, smaller than at 32B and pointing to a capacity threshold; the GRPO comparison is the next key step. Thank you, and I'm happy to take questions.

---

## 3. 预期提问与应答（中英对照）

**Q1 — 「SFT 为什么反而退步了？」**
> 中文：SFT 数据里的 `Answer:` 格式与判分要求的 `\boxed{}` 冲突，把模型带偏了。我们已用「温和 2-epoch + 剥模板」修复到格式不冲突，但准确率下降本身仍在——这说明 SFT 的价值在格式习得，不在准确率，准确率要留给 RL 去补。
> EN: The `Answer:` format in the SFT data conflicts with the `\boxed{}` scorer, pulling the model off-track. We mitigated the format conflict with a gentle 2-epoch SFT plus template stripping, but the accuracy drop itself remains — SFT's value is format acquisition, not accuracy, and accuracy is RL's job.

**Q2 — 「RL 的增益会不会只是工具的功劳，模型本身没变强？」**
> 中文：正是如此，而且这是结论而非缺陷。RL 纯文本模式（10.8）低于 base（16.4），说明闭卷能力没变强；增益全部来自工具模式。ReTool 的核心主张就是「RL 学的是何时用工具」，我的结果在 1.7B 上复现了这一点。
> EN: Exactly, and that's the conclusion, not a flaw. RL in text-only mode (10.8) is below base (16.4), so closed-book ability didn't improve; the entire gain is in tool mode. ReTool's core claim is that RL learns *when* to use the tool — my result reproduces that at 1.7B.

**Q3 — 「GRPO 对比在哪？为什么还没跑？」**
> 中文：GRPO（E3）是唯一尚未落地的一项，排在下一步第一位。它只改 `clip_ratio_high` 一个变量、同样的 271 步预算，会在 W6–7 完成。本次汇报已交付前三项（基线/SFT/DAPO）。
> EN: GRPO (E3) is the one item still outstanding, and it's first on my next-steps list. It changes only `clip_ratio_high` at the same 271-step budget, planned for W6–7. This report has already delivered the first three (baseline / SFT / DAPO).

**Q4 — 「为什么 Pass@12 比 Avg@12 高这么多？该信哪个？」**
> 中文：两个口径，互补。Avg@12 是「平均每次采样答对率」，衡量可靠性；Pass@12 是「12 次里至少一次答对」，衡量模型有没有能力解出。竞赛场景常报 Pass@N，因为它反映上限；Avg@N 反映稳定复现。两个都报更诚实。
> EN: Two complementary metrics. Avg@12 is the per-sample solve rate, a measure of reliability; Pass@12 is "≥1 correct in 12 draws," a measure of capability. Competition settings often report Pass@N for the ceiling, and Avg@N for consistency — reporting both is the honest choice.

**Q5 — 「12 次采样会不会有噪声？结论稳吗？」**
> 中文：有约 ±1% 的采样噪声（同配置两次独立跑，base-text 差 0.8%），但关键差距都是十几到二十几个点，远超噪声量级，方向性结论稳健。
> EN: There is roughly ±1% sampling noise (two independent runs of the same base-text config differ by 0.8%), but the key gaps are ten to twenty points — far above noise — so the directional conclusions are robust.

**Q6 — 「和 32B 的 26.7→40.9→67.0 怎么比？」**
> 中文：我的是 22.8→11.7→41.1（Avg@12），两处不同：我的 SFT 是负迁移（32B 是 +14），所以不能直接套「SFT→RL」的 +26；但「base→RL」的增益 +18.3 确实小于 32B 的 +40.3，支持容量门槛。我会在报告里明确这个口径差异。
> EN: Mine is 22.8→11.7→41.1 (Avg@12), with two differences: my SFT is negative transfer (32B's is +14), so I can't directly reuse the "+26 over SFT"; but the base→RL gain of +18.3 is genuinely smaller than 32B's +40.3, supporting a capacity threshold. I'll state this metric nuance explicitly in the report.

**Q7 — 「训练/评测数据有没有重叠？」**
> 中文：无重叠。训练用 ReTool-SFT 和 DAPO-Math-17k，评测只用 AIME 2024/2025，AIME 从不进训练集；三人会做一次交集检查确认。
> EN: No overlap. Training uses ReTool-SFT and DAPO-Math-17k; evaluation uses only AIME 2024/2025, which never enter training; the three of us will do a one-time intersection check.

---

## 4. 演练提示与兜底

**通用**
- **用结果开场（Slide 5 是钩子）**。评审前两分钟就定调：70.5% 这个数字别藏，先在 Slide 3–4 快速铺垫协议，Slide 5 立刻抛主表。
- **只讲自己的切片**。全程「我」，不展开 B/C 的实验。
- **三处放慢**：主表（Slide 5）、发现二「工具驱动」（Slide 7）、一句结论（Slide 12）。这三处承载论点。
- **数字只说关键几位**。70.5、41.1、18.3、40.3 这几个记住即可；被问细节再指表。
- **计时演练**：完整跑一遍 + 再跑一遍确认能准时落在结论句上。

**若被压到 10 分钟**（砍约 1 分钟）
1. Slide 2 只念一句目标，跳过承诺清单。
2. Slide 4 只念「12 采样 + 只看最终答案」两条。
3. Slide 9 只讲 H1 证伪 + H2 支持，H3/H4 一句话带过。

**若超时**：先压缩 Slide 3（已完成）——管线三段直接并成一句话「三段都跑通了，手里有三个检查点」。

**开场/结尾句（背诵）**
- 开场：*「上次我提出在 1.7B 上只改 RL 规则、对比 GRPO 与 DAPO；今天报告首批结果。」*
- 结尾：*「……增益全部来自策略性工具调用，GRPO 对照是下一步关键。谢谢。」*

---

## 5. 参考文献（幻灯片脚注 / 自查）

- **ReTool** — Feng et al., *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*, arXiv:2504.11536。
- **DAPO** — Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476。
- **GRPO** — Shao et al., *DeepSeekMath*, arXiv:2402.03300。
- **ToRL** — *Scaling Tool-Integrated RL*, arXiv:2503.23383。
- **本报告数据来源**：
  - 冻结协议评测（12 采样）→ SLURM job **686226**（`eval.slurm`，AIME 2024/2025，6 组全部 COMPLETED 退出码 0）。
  - DAPO 训练曲线 → SLURM job **675200**（271 步，9h08m，8×H800）。
  - 配套脚本 → `code/retool-small-llm/eval_qwen3_1.7b.sh`、`eval.slurm`、`merge_rl_hf.slurm`。
