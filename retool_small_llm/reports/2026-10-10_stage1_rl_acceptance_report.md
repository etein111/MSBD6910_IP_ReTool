# ReTool Agentic RL 阶段性验收报告（Stage 1）

> **日期**：2026-10-10
> **阶段范围**：环境搭建 → SFT 冷启动 → 多卡 RL 训练（DAPO/GRPO）跑通并验证有效
> **一句话结论**：✅ ReTool 式 agentic RL 在 **Qwen3-1.7B** 上完整跑通（8×H800、271 步、9h08m），**AIME 最佳采样正确率近翻倍**（AIME2024 `best@4` 29.3%→50.6%，AIME2025 26.3%→48.3%）。

---

## 1. 项目目标

在 **小参数模型（Qwen3-1.7B）** 上复现 **ReTool** 式的 agentic RL：让模型在数学推理任务中**主动调用工具**（本地 Python 代码执行沙箱 `LocalCodeTool`），通过多轮工具交互 + RL（GRPO/DAPO）学会「边想边算」的解题策略。

- 技术栈：`verl 0.6.1`（FSDP）+ `vllm 0.10.0`（异步多轮 rollout）+ `hermes` 工具调用格式。
- 平台：HKUST SuperPOD（`normal` 分区，账户 `mscbdtspod2026`，8×H800 单节点）。
- 实验承诺：**只动 RL update 规则**（GRPO vs DAPO），其余组件固定，便于对照。

---

## 2. 本阶段完成的工作

### 2.1 环境与数据
| 项 | 内容 |
|---|---|
| 环境 | conda env `retool`，verl 0.6.1 + vllm 0.10.0 + 本地工具沙箱 `LocalCodeTool`（subprocess，`num_workers=8`） |
| 训练数据 | `DAPO-Math-17k`，去重后 **17,398 行**（自研 `make_dedup_dataset.py`，按 prompt 文本去重，179 万行 → 17,398 行） |
| 评测数据 | AIME 2025 + AIME 2024（各 30 题），`best@4` / `mean@4` / `score@4` 三指标 |
| 工具定义 | `local_tool_config.yaml`：单工具 `code_interpreter`，`default_timeout=30`，`max_output_chars=4096` |

### 2.2 SFT 冷启动（含负迁移修复）
- **问题**：初版 SFT（6 epoch）产生**负迁移** —— RL 起点 `critic/score/mean` 掉到 **−0.881**（base 模型是 −0.622），且模型被 SFT 灌入的 `Answer:` 格式带偏，与 answer_format 要求的 `\boxed{}` 冲突，RL 永远到不了 `score/max=1.0`。
- **修复（双管齐下）**：
  1. **温和 SFT**：epoch 6→2、lr 显式 1e-5，job 671667（24min，val/loss 0.453），ckpt `global_step_124`。
  2. **格式冲突修复**：自研 `recipe/retool/retool_dataset_utils.py` 的 `map_fn2` 先剥掉 DAPO 外层模板（prefix/suffix 里的 `Answer: $Answer`）再追加 `\boxed{}`，消除与 answer_format 的冲突。
- **验证**：merge（job 671846 转 HF）+ RL 冒烟（job 671847，6 步）—— `critic/score/mean:-0.6156`（回升到 base 水平）、**`score/max:1.0`**、`aborted_ratio:0.0`。**0% 负迁移已解除**。

### 2.3 多卡 RL 训练（本阶段核心）
- **拓扑验证**：先踩了 SP/TP 的坑（照抄 7B recipe 的 `train_sp=4/infer_tp=4` → job 672236 报 `all_to_all` NCCL OOM），改 **纯 FSDP + 8×vllm(TP=1)** 后 8 卡冒烟 job 672270 通过（6m24s，`max_memory_reserved_gb` 53 无 OOM）。
- **正式训练 job 675200**（本阶段最终产物）：`run_qwen3_1.7b_dapo.sh` + `dapo.slurm`，**COMPLETED，退出码 0，9h08m，271/271 步**。

### 2.4 最终训练配置（job 675200）
| 类别 | 关键项 | 值 |
|---|---|---|
| 模型 | 冷启动 | `qwen3-1.7b-instruct-sft-2ep/global_step_124-hf`（HF safetensors） |
| 算法 | adv_estimator / clip | `grpo` / `clip_ratio_low=0.2, clip_ratio_high=0.28`（DAPO；GRPO 对照改 0.0） |
| 算法 | KL | 全关（`use_kl_in_reward=False, kl_coef=0`） |
| 优化 | lr / batch | `actor_lr=1e-6`，`train_batch_size=64`，`ppo_mini_batch_size=16` |
| 采样 | n / max_turns | `n_resp_per_prompt=4`，`max_turns=8`（多轮工具循环） |
| 序列 | prompt / response | `max_prompt_length=2048`，`max_response_length=8192` |
| 拓扑 | 并行 | 8×H800，纯 FSDP（`train_sp=1`）+ 8×vllm(TP=1)，`offload=False` |
| rollout | 模式 | vllm async + `multi_turn.format=hermes`，`gpu_memory_utilization=0.7` |
| 调度 | 保存/评测 | `save_freq=20`，`test_freq=50`，`val_before_train=False`，`total_epochs=1` |

---

## 3. 实现的结果

### 3.1 训练曲线（reward 由负转正）
`critic/score/mean`（多轮工具调用链上的总奖励，初始即负：因为冷启动模型不会正确调用工具）：

| 步数 | 0 | 50 | 100 | 150 | 200 | 250 | 271（终） |
|---|---|---|---|---|---|---|---|
| critic/score/mean | −0.643 | −0.106 | −0.099 | −0.052 | **+0.217** | +0.151 | **+0.230** |

- **reward 从 −0.643 一路升到 +0.230，跨过零点并稳定为正** —— 模型确实学会了通过工具调用拿到正确奖励。
- 同时 **num_turns 4.47 → 10.05**、**response_length 2292 → 4821 token** —— 模型学会了**更长的、更有效的工具调用链**（而非一开始的「短、错」）。

### 3.2 AIME 评测（正确率近翻倍）
| 指标 | AIME 2024 起点 → 终点 | AIME 2025 起点 → 终点 |
|---|---|---|
| `acc/best@4`（最佳采样正确率） | **29.3% → 50.6%**（峰值 61.3%@step250） | **26.3% → 48.3%** |
| `acc/mean@4`（平均正确率） | 18.3% → 37.5% | 16.7% → 35.8% |
| `score/mean@4`（部分分奖励） | −0.598 → −0.075 | −0.625 → −0.098 |

- **`best@4` 是最关键指标**（4 次采样里至少一次答对），代表模型「有能力」解出题目 —— AIME 2024 从 29.3% 升到 50.6%，**接近翻倍**。
- `mean@4` 同步提升（18.3%→37.5% / 16.7%→35.8%），说明不是偶然采样，是**稳定性的真实提升**。

### 3.3 产物
- checkpoint：`qwen3-1.7b-dapo/global_step_{20,40,…,260,271}`（每 20 步 + 最终步），`latest_checkpointed_iteration=271`，支持断点续训。
- TensorBoard 日志：`retool/qwen3-1.7b-dapo/`（160+ 指标 tag）。

---

## 4. 关键技术决策与踩坑（本阶段沉淀，已同步进 `architecture/superpod_workspace.md`）

| # | 坑 | 根因 | 解法 |
|---|---|---|---|
| 1 | SFT 负迁移 | 6ep SFT 过拟合 + `Answer:` 格式与 `\boxed{}` 冲突 | 2ep 温和 SFT + `map_fn2` 剥模板（见 §2.2） |
| 2 | 多卡 `all_to_all` NCCL OOM | 照抄 7B 的 `train_sp=4/infer_tp=4`，1.7B 单卡放得下，SP/TP 只添乱 | `train_sp=1/infer_tp=1/offload=False`（纯 FSDP） |
| 3 | rollout 极慢（首步 1h+） | SFT 冷启动模型生成**错误工具名**（`code_editor`/`code_executor`，配置叫 `code_interpreter`）+ 坏 JSON，`max_turns=16` 时响应不收敛顶到 8192 token | `max_turns=8` + `n=4`（冒烟实测 ~3.3 轮即收敛、41s/步） |
| 4 | 账户 GPU 分钟预算 | `GrpTRESMins=gres/gpu=9600`（≈160 GPU·h），8 卡 `--time` 上限 ≈17h 且随历史消耗递减 | 提交前 `sacct` 累加已用，`--time=15h` 留余量 |

**核心洞察**：**小模型（1.7B）+ agentic RL 的最大瓶颈不是训练吞吐，而是多轮工具 rollout 的采样速度**（模型生成不稳定 → 工具调用错乱 → 响应拖到上限）。因此 `n=4 / max_turns=8`（而非官方 7B 的 `n=16 / max_turns=16`）是「在有限 GPU 预算内先跑出有效奖励曲线」的关键取舍。

---

## 5. 结论与下一步

### 结论
1. ✅ **ReTool 式 agentic RL 在 1.7B 上完整跑通且有效**：reward 由负转正，AIME `best@4` 近翻倍 —— 证明「小模型 + 工具调用 + RL」这条路线在该课题上可行。
2. ✅ **多卡（8×H800 纯 FSDP）跑通**，无 OOM，9h08m 完成一整轮 epoch。
3. ✅ 本阶段产出了**可复现、可对照**的正式训练脚本（`run_qwen3_1.7b_dapo.sh`），实验变量已隔离为「只动 `clip_ratio_high`」。

### 下一步（待办）
- [ ] **GRPO 对照实验**：`clip_ratio_high=0.0` + `experiment_name=qwen3-1.7b-grpo`，验证 DAPO 的高 clip 上界在 agentic 场景是否带来增益。
- [ ] **AIME 2024 固定评测脚本**（官方 eval 目前只覆盖 AIME2025，需补充 2024）。
- [ ] **查证 DAPO-Math-17k 规模**：名字叫 17k，但原始 repo 拉到 179 万行，需确认是否 repo 更新（本阶段按去重后 17,398 行训练）。
- [ ] 若预算允许，可尝试 `n=8` 或 `max_turns=12` 探索采样效率上界（**仅在 GRPO 对照做完、预算有余量时**）。

---

## 附：相关文件索引
| 文件 | 说明 |
|---|---|
| `code/retool-small-llm/run_qwen3_1.7b_dapo.sh` | RL 正式训练 launch（本报告 §2.4 配置） |
| `code/retool-small-llm/dapo.slurm` | RL 正式训练 SLURM 包装（job 675200 所用） |
| `code/retool-small-llm/dapo_smoke_8gpu.slurm` | 8 卡多卡冒烟 SLURM 包装 |
| `retool_small_llm/architecture/superpod_workspace.md` | 工作区全貌 + 13 条踩坑 + 更新日志（时刻对齐文档） |
| `retool_small_llm/architecture/phase0_environment_setup.md` | Phase 0 环境搭建记录 |
| `code/retool-small-llm/recipe/retool/retool_dataset_utils.py` | 自研格式冲突修复（`map_fn2` 剥 DAPO 模板） |
