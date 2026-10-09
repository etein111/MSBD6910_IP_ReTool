# First Presentation — Project Objectives (English slide-by-slide)

> **What this file is:** the complete content for your **Meeting 2 "Project Objectives"** presentation (10–15 min, in English). For every slide it gives: the **on-slide text** (paste-ready), a **visual spec** (what chart/table/diagram to draw), the **source/citation** for each claim, and **speaker notes** (a spoken script + delivery cues).
>
> **How to use:** read §1 (timing) → §2 (slides) → §3 (anticipated Q&A) → §4 (delivery tips). Everything on the slides and in the speaker notes is already in English; only this header and §0 are in Chinese for your convenience.

---

## 0. 使用说明（中文，仅此一节）

- 汇报是 **英文** 的：下方所有 slide 文字、speaker notes、Q&A 都是英文，直接可用。
- 上传文件名格式（会议要求）：`<CourseCode>-1 Proposal <Your-Full-Name> ReTool-style Agentic RL for Mathematical`（projectName 取前五个词）。**课程代码 CSIT6910 / MSBD6910 尚未确认，上传前问清。**
- 时长控制：目标 **约 11 分钟**（落在 10–15 分钟区间内，偏保守）。§4 给了「压缩到 10 分钟」的兜底方案。
- 本汇报只讲你（**成员 A**）自己的切片（Qwen3-1.7B + GRPO vs DAPO），不展开其他成员的实验。
- 每个 slide 的「Citation」列是我从你已读材料里核过的出处，汇报时不必逐页念，但被问到"这个数字哪来的"要能指得出来。

---

## 1. Timing plan (target ≈ 11 min)

| Slide | Title | Time |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | What ReTool does | 1:00 |
| 3 | The headline result (32B) | 1:10 |
| 4 | Three gaps = my entry point | 1:00 |
| 5 | Objective | 1:00 |
| 6 | Method & frozen protocol | 1:10 |
| 7 | Experiment matrix | 0:45 |
| 8 | Hypotheses | 1:00 |
| 9 | Deliverables & feasibility | 0:45 |
| 10 | Timeline & decision gates | 1:10 |
| 11 | Risks & contributions | 1:00 |
| 12 | This week & takeaway | 0:30 |

**Narrative arc:** *Here is a proven result (2–3) → here is what is missing from it (4) → here is exactly what I will do (5–7) → here is what I predict and how I know it is feasible (8–10) → here is why it matters and what is next (11–12).*

---

## 2. Slides

---

### Slide 1 — Title

**On-slide text**

> # ReTool-style Agentic RL for Mathematical Reasoning with Small LLMs
> **Project Objectives & Detailed Plan — First Presentation**
> [Your Full Name] · [Course Code] · [Date]
> *My slice: Qwen3-1.7B · a controlled GRPO-vs-DAPO comparison*

**Visual** — title layout; a small hero graphic on the right: two boxes **"LLM ⇄ sandbox"** with an arrow loop (the core ReTool loop). No data yet.

**Citations** — none (title slide).

**Speaker notes**

> "Good [morning/afternoon]. I'm [name], and my project is *ReTool-style agentic RL for mathematical reasoning with small LLMs*. Today I'll walk through the objectives and a detailed plan. My personal slice is on the smallest model, Qwen3-1.7B, where I run a controlled comparison of two reinforcement-learning algorithms, GRPO and DAPO. I'll spend most of the time on the plan, not the background."

---

### Slide 2 — What ReTool does

**On-slide text**

> **ReTool: teaching a model *when* to call a code tool**
>
> - **Problem:** text-only chain-of-thought fails at exact arithmetic — one wrong step propagates to a wrong final answer.
> - **Idea:** interleave natural-language reasoning with **Python code run in a sandbox**; the model sees the output (or the error) and continues.
> - **Training = two stages:**
>   - **SFT** (cold-start): learn the *format* — "how to call the tool".
>   - **RL** (tool-augmented): learn the *strategy* — "when to call it".
> - **Reward is outcome-only:** correct final `\boxed{}` answer → +1, otherwise −1. *Tool calls are never rewarded by themselves.*

**Visual** — a two-panel before/after diagram: (left) a text-only chain with a red X on a miscalculated step; (right) "reason → `code` → sandbox returns 7 → reason continues". Label the right panel "in-loop feedback".

**Citations** — ReTool (arXiv:2504.11536), §Abstract & §2.

**Speaker notes**

> "First, sixty seconds of background. ReTool tackles a specific failure of text-only reasoning: the model does arithmetic inside its own weights, so a single miscalculation cascades. ReTool instead lets the model call a Python interpreter inside a sandbox and read the real result back. The key training design is a *two-stage* recipe — SFT first to learn the call format, then RL to learn the strategy — and crucially the reward only checks whether the final boxed answer is correct. [POINT AT LAST BULLET] The tool call itself is never rewarded, which is what prevents the model from just spamming the tool."

---

### Slide 3 — The headline result (why this matters)

**On-slide text**

> **At 32B, "tool × RL" is multiplicative — not additive** (AIME 2024)
>
> | Stage | AIME 2024 | Training steps |
> |---|---|---|
> | Base model (untrained) | 26.7 | — |
> | + cold-start SFT | 40.9 | — |
> | + tool-augmented RL | **67.0** | **400** |
> | text-only RL (control, no tool) | 40.0 | > 1000 |
>
> **Takeaway:** SFT alone ≈ RL alone ≈ 40; combined = 67. Converges **~2.5× faster** and answers are **~40% shorter**.

**Visual** — a **bar chart** of the four numbers (26.7 / 40.9 / 67.0 / 40.0), with an arrow between "SFT 40.9" and "RL 67.0" labeled *"the RL jump"*, and a dashed bracket showing "tool ≈ RL ≈ 40, tool × RL = 67".

**Citations** — ReTool Table 1 (§3); AIME 2024 = `HuggingFaceH4/aime_2024`.

**Speaker notes**

> "The reason anyone cares is this table. On AIME 2024, the untrained 32B model scores 26.7. Adding SFT gets 40.9. Adding RL on top gets 67 — a 27-point jump. The clean control is the bottom row: text-only RL, no tool, reaches only 40.0 even after more than a thousand steps. [SLOW DOWN] So the tool alone is worth about as much as RL alone, but the combination is multiplicative. That's the story I'm testing at a much smaller scale."

---

### Slide 4 — Three gaps = my entry point

**On-slide text**

> **ReTool's boundaries leave three open questions**
>
> | Gap | Open question |
> |---|---|
> | **Scale** — only 32B backbones; the one small-model attempt (ToRL, 1.5B/7B) was called "suboptimal" | Does the gain require model capacity? |
> | **Algorithm** — only PPO, which needs a value network | How do critic-free GRPO / DAPO compare in agentic RL? |
> | **Efficiency** — the −40% length & faster convergence were shown only at 32B | Do small models get the same efficiency win? |

**Visual** — a three-column "Gap → Question" diagram, each row as a card. Put a red circle on "Scale" (my primary framing).

**Citations** — ReTool (§3, single scale + PPO only); ToRL (arXiv:2503.23383); GRPO (arXiv:2402.03300); DAPO (arXiv:2503.14476).

**Speaker notes**

> "So why is this a project and not a re-run? ReTool leaves three gaps. First, *scale*: everything was done at 32B, and the only prior small-model attempt — a method called ToRL — was judged 'suboptimal', which is an unresolved question, not a settled answer. Second, *algorithm*: ReTool used only PPO, which needs a value network; on a limited GPU budget, the critic-free algorithms GRPO and DAPO are the realistic choice, and nobody has compared them in this agentic setting. Third, *efficiency*: the shorter-answer, faster-convergence result is also only shown at 32B. [PAUSE] My project sits in the first two gaps."

---

### Slide 5 — Objective

**On-slide text**

> **Objective: a controlled algorithm comparison on the smallest model**
>
> - Reproduce the ReTool pipeline **end-to-end on Qwen3-1.7B**.
> - **Freeze** base model, data, sandbox, reward, and evaluation protocol.
> - **Vary only the RL update** — GRPO vs DAPO — under an **equal step budget**.
>
> *"ReTool was validated only at 32B and only with PPO. This project reproduces the pipeline at the smallest viable model and provides a controlled stability comparison of critic-free algorithms in small-model agentic RL."*

**Visual** — a diagram of the pipeline with a padlock over every stage and a single highlighted, swap-able box labeled "RL update: GRPO ⇄ DAPO" (the only moving part).

**Citations** — objective restated from my proposal (based on Guideline §5 "change only the RL update"); PPO value-network cost — Schulman et al. (arXiv:1707.06347).

**Speaker notes**

> "Here is the objective in one sentence: I reproduce the full ReTool pipeline on the smallest practical model, and I freeze everything — model, data, sandbox, reward, evaluation — and change exactly one thing, the RL update rule, GRPO versus DAPO, at an identical number of training steps. [POINT] That single-variable design is what lets me make a *causal* claim about algorithm stability, rather than a story about different setups."

---

### Slide 6 — Method & frozen protocol

**On-slide text**

> **Three-stage pipeline (all on Qwen3-1.7B)**
>
> `Baseline (untrained, tool & text-only)` → `SFT (ReTool-SFT → learn format)` → `RL (DAPO-Math-17k, DAPO/GRPO → learn strategy)`
>
> **Frozen protocol — one eval script throughout**
> - Eval: **AIME 2024 & 2025** · 12 samples/problem · last `\boxed{}` · **both tool & text-only** modes.
> - Reward: **final-answer only** (no executability reward); **AIME never in training**.
> - Equal step budget **N (200–300)** · thinking mode **OFF** · model pinned by HF commit.

**Visual** — a horizontal flow diagram for the pipeline (three boxes + arrows), with a compact checklist card below for the protocol. Highlight "no executability reward" and "AIME never in training".

**Citations** — ReTool §2 (two-stage recipe); Guideline §3 (fixed protocol, 12 samples, last boxed answer, thinking off); datasets `JoeYing/ReTool-SFT`, `BytedTsinghua-SIA/DAPO-Math-17k`, `HuggingFaceH4/aime_2024`, `yentinglin/aime_2025`.

**Speaker notes**

> "Concretely, a three-stage pipeline on the 1.7B: a baseline, cold-start SFT to learn the tool format, then RL to learn the strategy — with the same evaluation script after every stage. The comparability rests on a frozen protocol. Evaluation is AIME 2024 and 2025, twelve samples per problem, last boxed answer, in both tool and text-only modes. The reward is outcome-only — I explicitly do *not* reward code that merely runs, which prevents reward hacking — and AIME never appears in training. [POINT AT 'thinking mode OFF'] One practical note: thinking mode is off, because thinking mode rarely writes code in this setting."

---

### Slide 7 — Experiment matrix

**On-slide text**

> **Experiment matrix (my slice)**
>
> | ID | Experiment | Base | Update | Priority |
> |---|---|---|---|---|
> | E0a | Baseline — tool mode | 1.7B | — | core |
> | E0b | Baseline — text-only | 1.7B | — | core |
> | E1 | Cold-start SFT | 1.7B | SFT | core |
> | E2 | RL — **DAPO** | from E1 | DAPO | core |
> | E3 | RL — **GRPO** | from E1 | GRPO | core |
> | E5 | PPO (reference) | 1.7B | PPO | optional |
>
> E5 is optional: PPO needs a **value network** (much higher memory). E0–E3 are never cut.

**Visual** — the table with E0a–E3 shaded "core (never cut)" and E5 grayed "optional".

**Citations** — my proposal (experiment matrix); PPO critic memory — Schulman et al. (arXiv:1707.06347).

**Speaker notes**

> "The experiment matrix is small and deliberate. E0 is the baseline in both modes. E1 is the SFT checkpoint. Then E2 and E3 are the heart of the project: from the *same* SFT checkpoint, I train once with DAPO and once with GRPO at the same step budget. I list PPO as optional because it needs a value network — significantly more memory — which is risky on a small-model, long-context setup. The core is E0 through E3, and those never get cut."

---

### Slide 8 — Hypotheses

**On-slide text**

> **Four falsifiable hypotheses**
>
> - **H1** — an untrained 1.7B cannot reliably produce valid tool calls; SFT raises **format rate / sandbox success** far more than **accuracy**.
> - **H2** — RL improves accuracy beyond SFT, but the gain is **smaller than the +26 pts at 32B**.
> - **H3** — at equal step budget, **DAPO is more stable** than GRPO (fewer degenerate groups / entropy collapse).
> - **H4** — response length is **non-monotonic**: first up (more tool turns), then down.
>
> *Any hypothesis that gets falsified is still a valid result — not a failure.*

**Visual** — four cards, each with a tiny "if true → / if falsified →" footnote under it.

**Citations** — my proposal (§B.1 hypotheses); entropy-collapse framing — DAPO (arXiv:2503.14476).

**Speaker notes**

> "I've written four hypotheses, each falsifiable. H1: SFT mainly fixes the format, not the accuracy. H2: RL still helps at 1.7B, but less than the 27 points at 32B — this directly tests whether capacity is a threshold. H3 is the core contribution: at equal budget, DAPO should be more stable than GRPO. H4 is a small correction to the 'tools make answers shorter' story — I expect length to rise first, then fall. [EMPHASIZE LAST LINE] And I want to be explicit: a falsified hypothesis is a valid result, not a failed project — either outcome has scientific value."

---

### Slide 9 — Deliverables & feasibility

**On-slide text**

> **Deliverables**
> benchmark table (base / +SFT / +RL(DAPO) / +RL(GRPO) × AIME 24/25 × tool & text-only) · stability curves (reward, entropy, degenerate groups, length) · tool-behavior analysis · two qualitative traces · reproducibility package
>
> **Feasibility**
> Median **78 human-hours** vs **88-h budget** (11 wks × 8 h). Biggest items: environment & recipe (12–20 h) and report (16–22 h); RL (DAPO + GRPO) ≈ 14–23 h. Median fits with ~10 h margin.

**Visual** — a compact deliverables checklist (left) + a horizontal bar showing **62 / 78 / 95** against a vertical line at **88** (right).

**Citations** — my proposal (§B.1 deliverables; §D.1 work breakdown).

**Speaker notes**

> "The deliverables go beyond a final score — the benchmark table across stages, stability curves, a tool-behavior analysis, two annotated traces, and a reproducibility package. On feasibility: a median of 78 human-hours against an 88-hour budget, so it fits with a small margin. The two big items are the environment stage and the report; the RL runs themselves are bounded because I train to a fixed step budget, not to convergence."

---

### Slide 10 — Timeline & decision gates

**On-slide text**

> **11-week plan with three decision gates**
>
> | Week | Work | Gate |
> |---|---|---|
> | W1 (9/16–22) | accounts, code reading, frozen protocol, smoke eval | this presentation |
> | W2 (9/23–29) | environment + training smoke run | **G1** recipe end-to-end |
> | W3 (9/30–10/6) | E0 baseline; start writing | **G2** eval cost on estimate |
> | W4 (10/7–13) | E1 SFT | — |
> | W5 (10/14–20) | E2 DAPO, early curves | **G3** RL curve rising · progress 10/20 |
> | W6–9 | E3 GRPO → algorithm comparison → tool analysis | main result |
> | W10–11 | report + reproducibility → final presentation | final 11/25 · report 11/30 |

**Visual** — a Gantt-style table; color the three Gate rows as red diamonds.

**Citations** — my proposal (§E schedule; §D.3 gates); Guideline meeting dates.

**Speaker notes**

> "The schedule runs backward from three gates, each an early-warning check. Gate 1, end of week 2: the recipe must run end-to-end — data loads, sandbox executes, rewards aren't all identical. Gate 2 checks the evaluation cost against my estimate. Gate 3 at week 5: the DAPO reward curve must be rising with normal tool calls. Each gate has a pre-agreed fallback, so if anything slips I cut scope early instead of discovering it in week 10."

---

### Slide 11 — Risks & contributions

**On-slide text**

> **Top risks → mitigations**
> - **Time overrun** → cut expansion first, never cut baseline/DAPO; debug on 1.7B only.
> - **Training instability (entropy collapse)** → DAPO clip-higher / dynamic sampling + rollback checkpoints.
> - **Reward hacking** → outcome-only reward + monitor tool-call distribution.
> - *Safety valve:* course allows extension to Jan 2027.
>
> **Expected contributions**
> 1. First *controlled* GRPO-vs-DAPO comparison in **small-model agentic RL**.
> 2. Decompose "capacity" into observable indicators (format rate, sandbox success, call timing, self-correction).
> 3. Test reproducibility under **constrained compute**; evidence on the ToRL "suboptimal" claim.

**Visual** — two-column layout: risks (left) and contributions (right).

**Citations** — my proposal (§8 risks; §C.2 significance); DAPO (arXiv:2503.14476).

**Speaker notes**

> "The three risks that matter. Time overrun is the top one — the rule is cut the extras, never the baseline or the DAPO run. Training instability, entropy collapse, is a known small-model failure mode; DAPO's clip-higher plus rollback checkpoints are the defense. Reward hacking is contained by the outcome-only reward and monitoring. [MOVE TO CONTRIBUTIONS] On the other side, the contributions: the first controlled algorithm comparison in small-model agentic RL, a decomposition of 'capacity' into observable indicators, and a practical test of reproducibility under constrained compute — plus direct evidence on whether ToRL's 'small models are suboptimal' verdict still holds."

---

### Slide 12 — This week & takeaway

**On-slide text**

> **This week (by 9/21)**
>
> - SuperPOD account + read login / Slurm / quota / storage docs
> - Read `05-retool` README + six modules; write the 200-word "question → trajectory" note
> - Set up environment; smoke-evaluate 1–2 problems (tool & text-only)
> - Rehearse; upload slides before the meeting
>
> **One-sentence takeaway:**
> *ReTool's multiplicative gain is proven only at 32B and only with PPO — I will test whether it survives the smallest model, and which critic-free algorithm drives it most stably.*

**Visual** — a checklist (top) + a large highlighted takeaway box (bottom).

**Citations** — Guideline §3 (four starting steps).

**Speaker notes**

> "Finally, the immediate plan: this week I finish reading the training repository, run a one-or-two-problem smoke evaluation in both modes so I can show real numbers next time, and rehearse. [SLOW, LAST LINE] To close with the takeaway: ReTool's multiplicative gain is proven only at 32B and only with PPO — my project tests whether it survives the smallest model, and which critic-free algorithm drives it most stably. Thank you; happy to take questions."

---

## 3. Anticipated Q&A (English)

> These are the questions most likely to come up, with ready answers. (Derived from your proposal §G.)

**Q1 — "Three of you on the same project — how do you avoid duplicating each other?"**
> One backbone each: 1.7B, 2B, and 4B, each with a *different* question — algorithm comparison, tool-vs-text, and SFT-necessity. We share a frozen evaluation protocol, data, reward, and sandbox, so our results merge into one table, but each person codes, trains, and writes independently. This matches the guideline's requirement that no two people present the same model and algorithm.

**Q2 — "Won't your progress be blocked by each other?"**
> No. We publish the interface contract once in week 1; it freezes *values*, not *adoption time*, so we adopt it asynchronously. Each of us builds our own environment and draws conclusions that don't cite the others' data. The only real coupling is GPU: if the quota is shared, our jobs queue, but while queuing nobody is waiting on anyone else's *results*.

**Q3 — "Why 1.7B and not 4B?"**
> The guideline's principle is to pick the smallest model you can actually finish. 1.7B makes pipeline debugging cheapest, and it's the most informative point for the "capacity threshold" question. The 4B is handled by a teammate as a separate slice.

**Q4 — "Why compare GRPO and DAPO, and why not PPO?"**
> ReTool used only PPO, which needs a value network and is memory-heavy. Under a limited GPU budget, GRPO and DAPO are the realistic critic-free choices, and their relative stability in an agentic setting — with multi-turn interaction and sparse rewards — is genuinely unknown. PPO is optional for reference only.

**Q5 — "Why run both algorithms for the *same* number of steps?"**
> It solves three things at once: it keeps the training cost inside quota, it makes the comparison stricter (identical compute), and it keeps our three-person scale ladder comparable. Training-to-convergence is not required to answer the stability question.

**Q6 — "What if the model never learns to call the tool?"**
> That is a valid result and I will report it clearly, as the guideline asks. My E0a baseline answers part of this *before* any training, and the cold-start SFT stage is the main defense against it.

**Q7 — "How do you prevent reward hacking?"**
> The reward is outcome-only — a correct final boxed answer, nothing for executability — which is ReTool's design choice. On top of that I audit the tool-call count distribution and watch the gap between format rate and accuracy.

**Q8 — "How do you guarantee a fair evaluation?"**
> A single evaluation script everywhere; 12 samples per problem at every checkpoint; identical decoding parameters for the base and trained models; and every random seed recorded.

**Q9 — "Is there train/eval overlap?"**
> No. Training uses ReTool-SFT and DAPO-Math-17k; evaluation uses only AIME 2024 and 2025, which never enter training. The three of us will do a one-time intersection check to confirm.

**Q10 — "Is the workload realistic?"**
> Median estimate is 78 human-hours against an 88-hour budget (11 weeks × 8 h). The main risk is the environment stage, which is why Gate 1 sits at the end of week 2 to catch a slow start early.

---

## 4. Delivery tips & fallback

**General delivery**
- **Lead with the result (Slides 2–3), then the gap (4).** The audience will decide in the first two minutes whether this is a real project; the 26.7 → 40.9 → 67.0 table is your strongest hook — don't bury it.
- **Keep the story on *your* slice.** Every slide speaks as "I" — this presentation covers only your Qwen3-1.7B / GRPO-vs-DAPO slice.
- **Slow down at three places:** the multiplicative-gain table (Slide 3), the frozen-protocol list (Slide 6), and the one-sentence takeaway (Slide 12). These carry the argument.
- **Rehearse with a timer once at full length**, then once again to confirm you can land the takeaway sentence in time.
- **Cite only when asked.** Don't read arXiv numbers aloud on every slide; be *able* to name them (ReTool 2504.11536, DAPO 2503.14476, GRPO 2402.03300, ToRL 2503.23383, PPO 1707.06347).

**If you are capped at 10 minutes** (cut ≈ 1 min):
1. Slide 5 — read only the three objective bullets; skip the positioning quote.
2. Slide 9 — say only the feasibility number ("78 hours vs 88") and skip reading the deliverables list.
3. Slide 11 — keep only the contributions; drop the risk bullets.

**If you run long**, the first slide to shrink is Slide 4 (three gaps) — the scale gap alone is enough to motivate the project; you can mention the algorithm and efficiency gaps in one breath.

**Opening / closing lines to memorize**
- Open: *"ReTool's multiplicative gain — 27 points on AIME — is proven only at 32B and only with PPO."*
- Close: *"…which critic-free algorithm drives it most stably. Thank you."*

---

## 5. Reference list (for the slide footnotes / your own recall)

- **ReTool** — Feng et al., *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*, arXiv:2504.11536.
- **DAPO** — Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476.
- **GRPO** — Shao et al., *DeepSeekMath*, arXiv:2402.03300 (also DeepSeek-R1, arXiv:2501.12948).
- **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- **ToRL** — *Scaling Tool-Integrated RL*, arXiv:2503.23383.
- **SimpleRL-Zoo** — arXiv:2503.18892.
- **Training code** — `github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool`.
- **Official recipe** — `github.com/verl-project/verl-recipe/tree/main/retool`.
- **Models** — `Qwen/Qwen3-1.7B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`.
- **Datasets** — SFT `JoeYing/ReTool-SFT`; RL `BytedTsinghua-SIA/DAPO-Math-17k`; eval `HuggingFaceH4/aime_2024`, `yentinglin/aime_2025`.
- **Course material** — *IP NZ7 Student Guideline* (SuperPOD usage, starting steps, exploration directions, presentation rules).
