# Phase 0 环境打通 Runbook（verl + SuperPOD）

> 版本：v0.1（2026-09-27）
> 用途：把「verl 最小管线在 SuperPOD 上跑通（Qwen3-1.7B）」这一 Gate 1 目标拆成可执行步骤。
> 配套：`project_architecture.md`（课题总架构）、`superpod_workspace.md`（**集群目录/环境/路径速查表 —— 实操时看那份**）。
> 本文档记录「怎么从零装起来」；装好之后的日常对照看 `superpod_workspace.md`。

---

## 0. 当前状态（2026-09-27）

> ⚠️ 本表 2026-09-27 更新。**日常路径/环境速查请看 `superpod_workspace.md`**，本表只记进度。

| 项 | 状态 |
|---|---|
| 官方 verl-recipe 已 clone 并读完 `retool/` 全部关键文件 | ✅ 完成 |
| verl 版本锁定 | ✅ `verl==0.6.1`（tag `v0.6.1`） |
| 算法口径确认 | ✅ 官方 recipe 即「GRPO + DAPO clip-higher」；GRPO vs DAPO 是 `clip_ratio_high` 一个开关 |
| SuperPOD 登录 / Slurm 打通 | ✅ 登录 `slogin-01`，账号 `mscbdtspod2026`，分区 `normal`，已跑通首个作业 |
| conda 环境 + 软件栈 | ✅ `retool` 环境：torch 2.7.1+cu126 / vllm 0.10.0 / verl 0.6.1 / flash-attn 2.8.3.post1 / xformers 0.0.31，GPU 节点验证通过（**2026-09-27 升过 vllm 到 0.10.0**，见 `superpod_workspace.md` 坑 #8） |
| 数据下载 + 目录重排 | ✅ 模型 Qwen3-1.7B、DAPO-Math-17k、AIME 2024/2025，已软链重排为 `dataset_clean/` |
| 沙箱方案 | ✅ 定为**自研本地 subprocess 沙箱**（`LocalCodeTool`），已在 login node 跑通预检 19/19 |
| Qwen3-1.7B chat template 前置检查 | ✅ 通过：含 `<tool_call>`，`eos_token='<|im_end|>'`。**⚠️ 2026-10-08 更正：Qwen3 命名与 Qwen2.5 相反 —— `Qwen/Qwen3-1.7B` 就是 Instruct 版**（含 tool_call 模板），`Qwen3-1.7B-Base` 才是 base，**不存在 `-Instruct` 后缀** |
| 代码上传通道 | ✅ 打通：登录主机名 `superpod.ust.hk`（**不是** `slogin-01.cm.cluster`），`scp` 免密。命令见 `superpod_workspace.md` §4 |
| 版本漂移修复 | ✅ 降级 transformers 4.57.6 / datasets 3.6.0（5.x 与 vllm 0.8.5 不兼容）；**2026-09-27 二次修复**：vllm 0.8.5→0.9.2（`get_tcp_uri`/`CoreEngineProcManager` 缺失）→ **0.10.0**（`AsyncLLM.wait_for_requests_to_drain` 缺失），torch→2.7.1+cu126、flash-attn→2.8.3.post1、新增 xformers 0.0.31、numpy→1.26.4（verl 要 `<2.0`）、transformers→4.53.2（aimv2 冲突） |
| 数据 schema + 字段映射 | ✅ `data_check2.py` 11/11：AIME 列名 `Problem/Answer`、`problem/answer` 齐全，`map_fn`/`map_fn2` 真实首行跑通 |
| 冒烟测试（Gate 1） | ✅ **通过**：job 615359（vllm 0.8.5 `get_tcp_uri`）→ 615381（vllm 0.9.2 `wait_for_requests_to_drain`）→ 615461（`KeyError: num_turns`）→ 补丁后 **job 615480 COMPLETED（10:54）**。四条验收全过（数据 64/60、`LocalCodeTool`+多轮、reward max=1/min=−1、`global_step_5` 落盘） |
| **SFT 冷启动（Gate 2）** | ✅ **完成**（2026-10-08）：`sft_preprocess.py` 把 `JoeYing/ReTool-SFT`（2000 行）转多轮 tool 消息 → `data/dataset_clean/JoeYing/ReTool-SFT/train.parquet`；FSDP SFT 单卡 6 epoch，**job 663756 COMPLETED（~80min）**，最终 ckpt `checkpoints/qwen3-1.7b-instruct-sft/global_step_372/`。连踩 4 坑后跑通：fp32 OOM → `model_dtype=bf16` → `micro_batch=2`（logits 显存与模型大小无关）→ `truncation=left`（6 样本 >8192 token） |

---

## 1. 第一步（人工，只能你来做）：SuperPOD 登录

**这是当前唯一卡点。** 我无法替你登录，需你本人操作。关键规则（摘自 Guideline）：

- 从**校园网或 HKUST VPN** 连接。
- 用 **HKUST 用户名登录（不是邮箱）**，密码登录走 **Duo MFA**。
- **login node 没有 GPU**，所有 GPU 作业走 **Slurm** 提交。
- 学生账号通常只有**有限分区 + GPU 时配额**，超额会杀掉正在跑的任务。
- home 目录小，**别把大 checkpoint 堆在 home**（放 `/scratch` 或作业目录）。

官方文档（先读完再动手）：

| 主题 | 链接 |
|---|---|
| SuperPOD 总览 | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod |
| 登录 | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/usage-tips/login |
| 提交第一个 Slurm 作业 | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job |
| Slurm 调度 | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/slurm |
| 教学课账号 | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/account-for-teaching-course |
| FAQ | https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/faq |

**验收**：能 `ssh <username>@<login-node>` 登录，`sinfo` / `squeue` 能看到分区和队列；确认自己的 GPU 时配额和可用分区名。

---

## 2. 装 verl 0.6.1 及依赖

在 SuperPOD 上（建议用 conda/mamba 建独立环境）：

```bash
conda create -n retool python=3.11 -y && conda activate retool
pip install verl==0.6.1          # recipe 锁定的版本
```

或者用 recipe 自带的安装脚本（会读 `REQUIRED_VERL.txt`，保证版本一致）：

```bash
cd verl-recipe && ./install_verl.sh --recipe retool --method pip --yes
```

**依赖要点**（H800 上的常见坑，`project_architecture.md` §Phase0 已列）：

- **FlashAttention**：H800 上从源码编译易失败，优先用**预编译 wheel**（匹配 CUDA 版本的）。
- **vLLM**：注意与 torch/CUDA 版本匹配；recipe 里 `export VLLM_USE_V1=1`。
- **Ray**：verl 的分布式调度依赖，装完确认 `ray start --head` 能起。

**验收**：`python -c "import verl; print(verl.__version__)"` 打印 `0.6.1`，`python -c "import vllm"` 不报错。

---

## 3. 沙箱（SandboxFusion）—— 已知最难的一步

官方 recipe 用的是 **SandboxFusion**：一个独立的代码执行服务，`sandbox_fusion_tool_config.yaml` 里配的是 `http://localhost:8080/run_code`。它**不是** pip 装上就行，需要单独部署一个 daemon（C++ 实现，通常要 Docker/容器）。SuperPOD 上能不能跑 Docker 需要先确认。

三个选项，按推荐顺序：

| 方案 | 说明 | 适合 |
|---|---|---|
| **A. SandboxFusion**（官方默认） | 与 verl recipe 原生匹配，`verl.tools.sandbox_fusion_tools` 直接支持 | 正式实验（4B/多节点、并发大） |
| **B. 移植 05-retool 的本地 subprocess 沙箱** | 把 `agentic-rl-lab/05-retool/sandbox.py` 的 `LocalPythonSandbox` 逻辑写成一个 verl `BaseTool`，纯 Python、无需 Docker | **Gate 1 冒烟测试最快**，1.7B 小并发够用 |
| **C. Daytona 云沙箱** | recipe 自带 `daytona_sandbox_tool.py`，走云端 | 不想管沙箱部署，但要额外服务/账号 |

**建议**：Gate 1 先走 **B**（本地 subprocess 沙箱，零部署、纯 Python，05-retool 已验证逻辑），把「数据→采样→沙箱执行→reward→更新」整条链路先跑通；等上 4B 正式实验、并发大时再评估要不要 A（SandboxFusion）。

> 这一步正是「把 05-retool 学到的 sandbox.py / protocol.py 逻辑移植进 verl」的落点——它们框架无关，逻辑直接复用。

---

## 4. 数据准备

| 数据 | 用途 | HuggingFace ID | 格式 |
|---|---|---|---|
| RL 训练集 | RL | `BytedTsinghua-SIA/DAPO-Math-17k` | verl 需要的 parquet（`prompt` + `reward_model.ground_truth`） |
| SFT 训练集 | 冷启动 | `JoeYing/ReTool-SFT`（多轮版 `swordfaith/ReTool-SFT-multi-turn`） | 由 `retool_sft_preprocess.py` 转成多轮 tool 消息 |
| 评测 AIME 2024 | val | recipe 用 `Maxwell-Jia/AIME_2024`（Guideline 另列 `HuggingFaceH4/aime_2024`） | 与 recipe 一致用 Maxwell-Jia |
| 评测 AIME 2025 | val | `yentinglin/aime_2025` | 同 recipe |

- RL 数据 `map_fn2`（`retool_dataset_utils.py`）会往 prompt 里加 `\boxed{}` 答题格式要求；AIME 用 `map_fn`。
- **别把 AIME 训练集混进 RL 训练**（Guideline 明确要求）。

---

## 5. 适配 Qwen3-1.7B（相对 7B 脚本 `run_qwen2_7b_dapo.sh` 的改动）

把 `run_qwen2_7b_dapo.sh` 复制一份改成 1.7B，参数如下：

| 参数 | 7B 脚本原值 | 1.7B 建议值 | 理由 |
|---|---|---|---|
| `model.path` | Qwen2.5-7B SFT ckpt | SFT 后 ckpt `checkpoints/qwen3-1.7b-instruct-sft/global_step_372`（对齐官方：RL 从 SFT 冷启动 ckpt 出发） | 你的模型 |
| `infer_tp` (vllm) | 4 | 1 | 1.7B 单卡即可 |
| `train_sp` | 4 | 1 | 同上 |
| `train_batch_size` | 64 | 32 | 冒烟先用小 |
| `ppo_mini_batch_size` | 16 | 8 | |
| `n_resp_per_prompt`（组大小 G） | 16 | 8 | 架构文档 §4.2 G=8 |
| `max_response_length` | 16384 | 8192 | 1.7B 回答短 |
| `fsdp_config.*_offload` | True | False | 1.7B 全参 8×80G 放得下 |
| `actor_lr` | 1e-6 | 1e-6 | 「freeze everything」不动机 |

**两个必须验证的点**（上 SuperPOD 后对着装好的 verl 0.6.1 确认）：

1. **`rollout.multi_turn.format=hermes` 对 Qwen3 是否兼容**。Qwen2.5/Qwen3 的原生工具格式是 `<tool_call>`…`<tool_response>`，与 hermes 接近但不完全等同；verl 0.6.1 的 multi-turn format 注册表里若另有 `qwen` 格式，优先用 `qwen`。
2. **Qwen3-1.7B 是 Instruct 模型（更正 2026-10-08）**：Qwen3 命名反转，`Qwen3-1.7B` 即 Instruct（含 tool_call 模板），`Qwen3-1.7B-Base` 才是 base。虽然模板存在，但**预训练 Instruct 未经工具调用 SFT 对齐，不会主动稳定地调工具、也不会稳定输出 `\boxed{}`**（这正是 H1）。所以 1.7B 的「端到端复现」**必须 SFT 先行**（冷启动对齐工具调用格式），否则冒烟测试里工具调用率和 boxed 格式率都会是 0。Guideline Step 1 也强调：ReTool 复现关 thinking mode。**（2026-10-08 已按用户决策：用 Instruct 对齐官方，base 作为可选的额外消融）**

### GRPO vs DAPO —— 你的核心实验开关

官方 `run_qwen2_7b_dapo.sh` 本身就是「GRPO + DAPO clip-higher」：

```
actor.clip_ratio_low=0.2    → 下限 clip 到 1-0.2 = 0.8
actor.clip_ratio_high=0.28  → 上限 clip 到 1+0.28 = 1.28
```

与 05-retool 的 `clip_low_threshold=0.8 / clip_high_threshold=1.28` **完全一致**。所以：

| 实验 | 改动 | 对应矩阵 |
|---|---|---|
| **DAPO** | `clip_ratio_high=0.28`（recipe 默认） | E2 主实验 |
| **GRPO** | `clip_ratio_high=0.0`（无上限 clip） | E3 对比 |
| PPO | `adv_estimator=gae`（需 critic/价值网络，重） | E4 可选 |

其余全部冻结（模型、数据、沙箱、评测、step 预算），只动这一个 knob——这正是汇报里承诺的「freeze everything, vary only RL update」。

---

## 6. Smoke test 验收（Guideline Step 4）

短跑（几十 step 量级）能证明四件事，就说明链路通了：

1. 数据能加载；
2. 沙箱真的执行了代码；
3. **reward 不全相同**（不是全 0 或全常数）；
4. 短训练能跑完并产出 checkpoint。

**先做 1–2 题的评测冒烟**，再上 30 题全量评测，别在坏配置上烧配额。跑完立刻 `squeue` / 配额面板查 GPU 时消耗。

---

## 7. 下一步行动

**Phase 0 已全部落地（2026-09-27）：**

- [x] **登录 SuperPOD**（§1）—— `superpod.ust.hk`，账号 `mscbdtspod2026`
- [x] 确认配额 / 可用分区（`normal`）/ 存储位置（checkpoint 放 `/scratch`，home 小）
- [x] 建 conda 环境装 verl 0.6.1 + vLLM 0.10.0 + FlashAttn（§2）
- [x] 确定沙箱方案 B：本地 subprocess 沙箱（`LocalCodeTool`，预检 19/19）
- [x] 下数据（§4）—— Qwen3-1.7B、DAPO-Math-17k、AIME 2024/2025，软链重排 `dataset_clean/`
- [x] 写 1.7B 版 launch 脚本（`smoke.slurm`）+ 验证 Qwen3 format（hermes + `<tool_call>` 模板通过）
- [x] 跑 5-step smoke test，确认四件事（§6）—— **Gate 1 ✅ job 615480**

**真正的下一步（Phase 1 及之后）：**

- [x] SFT 冷启动：`sft_preprocess.py`（`JoeYing/ReTool-SFT` → 多轮 tool 消息）+ FSDP SFT 训练 —— **✅ 完成（2026-10-08，job 663756，ckpt `global_step_372`）**
- [ ] RL 训练（GRPO/DAPO）：写 `run_qwen3_1.7b_dapo.sh`，`actor_rollout_ref.model.path` 指向 SFT ckpt `checkpoints/qwen3-1.7b-instruct-sft/global_step_372`（对齐官方 dapo 脚本的衔接方式）
- [ ] AIME 2024 评测加入固定评测脚本（官方 eval 只覆盖 AIME25）
- [ ] 查证 DAPO-Math-17k 为何 179 万行（名字叫 17k）
