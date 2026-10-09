# SuperPOD 工作区地图（集群目录 · 环境基线 · 路径速查）

> 版本：v1.0（2026-09-27）
> 用途：**集群上什么放在哪里、环境是什么版本**的唯一对照表。每次上集群操作前先看这份，
> 新增目录/文件后**立即回来更新**，避免几周后「东西放哪了、当时装的什么版本」全忘光。
> 配套：`phase0_environment_setup.md`（怎么从零装起来）、`project_architecture.md`（课题设计）。

---

## 1. 集群环境基线（已装好，勿重复折腾）

| 项 | 值 |
|---|---|
| 集群 | HKUST SuperPOD |
| **SSH 登录主机名（从外面连用这个）** | `superpod.ust.hk`（→ `143.89.184.2`） |
| 登录节点（集群内部名） | `slogin-01` —— 仅节点间互称，**从校外解析不了**，别拿它做 ssh/scp 目标 |
| **提交账号（必须带）** | `--account=mscbdtspod2026` |
| 计算分区 | `normal`（`dgx-*` 节点，每节点 8× **NVIDIA H800 80GB**） |
| 用户组 | `cyeah spprojects spodcpu spodnormal mscbdtspod2026` |
| conda base | `/cm/shared/apps/Anaconda3/2023.09-0`（只读，**不要往这里装东西**） |
| conda 环境 | `retool`（Python 3.11）→ `/home/cyeah/.conda/envs/retool` |
| 存储 | home 200G（NFS，放环境）· `/scratch` 518T（放数据/checkpoint/输出） |
| GPU 驱动 | 580.173.02 → CUDA 13.0（向下兼容，用预编译 wheel 不编译） |

**已装软件版本（改动前先确认，别随手升级）**

| 包 | 版本 | 备注 |
|---|---|---|
| torch | `2.7.1+cu126` | **已升**（原 2.6.0+cu124 → 2.7.0+cu128 → 2.7.1+cu126）：vllm 0.10.0 锁 `torch==2.7.1`（plain，默认 cu126），PEP 440 里 `+cu128` 不等于 `2.7.1` |
| vllm | `0.10.0` | **已升**（原 0.8.5.post1 → 0.9.2 → 0.10.0）：verl 0.6.1 异步 rollout 需 `AsyncLLM.wait_for_requests_to_drain`（0.10.0 才加）；0.10.0 是 CUDA 无关 abi3 wheel |
| verl | `0.6.1` | 官方 recipe 锁定版本（tag `v0.6.1`） |
| flash-attn | `2.8.3.post1` | **已升**（原 2.7.4.post1）：`+cu12torch2.7` wheel，ABI 兼容 torch 2.7.1 |
| xformers | `0.0.31` | **新增**：vllm 0.10.0 的硬依赖（abi3 wheel，不用编译），是 0.10.0 的默认 attention 后端 |
| transformers | `4.53.2` | **锁精确版**：见下方「vllm↔transformers 版本耦合」 |
| datasets | `3.6.0` | 已降级（原 5.0.1 的 `load_dataset(裸路径)` 行为变了），保持不变 |
| tokenizers | `0.21.4` | 随 transformers 4.53 自动降级（原 0.22.2） |
| numpy | `1.26.4` | **已降回**：verl 0.6.1 要求 `numpy<2.0.0`，但 vllm 0.10.0 不锁 numpy、会拉到 2.2.6 |

> **版本漂移教训（2026-09-27）**：verl 0.6.1 的 `requirements.txt` 对 `transformers`/`datasets`
> **不锁上限**，pip 装了 5.x 大版本。但 vllm 0.8.5 只兼容 transformers 4.x。修法：
> `pip install "transformers>=4.51.1,<5.0.0" "datasets>=3.0,<4.0"`（tokenizers 会自动跟回 0.22.x）。
>
> **opentelemetry 良性冲突**：pip 会警告 `vllm 需 opentelemetry<1.27，实际 1.45`。
> vllm 只在显式开启 tracing 时用 otel，冒烟不碰；`import vllm` 实测正常，**先不追**。
>
> **vllm↔transformers 版本耦合（2026-09-27 升级后的新约束）**：vllm 0.9.x/0.10.0 在 import 时会
> `AutoConfig.register("aimv2", AIMv2Config)`，而 `aimv2` 从 **transformers 4.54.0** 起进了
> 官方，会撞 `ValueError: 'aimv2' is already used`。所以 transformers 必须 `<4.54.0`。最终锁
> **4.53.2**：同时满足 vllm 0.10.0 的 `>=4.53.2` 下限、又避开 aimv2。**为什么选 0.10.0 不选 0.10.1**：
> 0.10.1 要求 `transformers>=4.55.0`（自带 aimv2），会逼 transformers 再升、重引冲突；0.10.0 只要
> `>=4.53.2`，4.53.2 正好够用。verl 0.6.1 锁 numpy<2.0.0，vllm 0.10.0 锁 torch==2.7.1 ——
> 三者在「升 vllm」上是同一根绳，动一个要联动其余（torch/transformers/numpy/xformers 一起 pin）。

**踩过的坑（别再踩）**

1. **`sbatch` 必须带 `--account=mscbdtspod2026`**，否则提交被拒。
2. **login node 没有 GPU** —— `torch.cuda.is_available()` 在 login node 返回 False 是正常的，别当成环境坏了。
3. **Slurm 脚本里 `conda activate` 会失败** —— Ubuntu `.bashrc` 对非交互 shell 提前 `return`。必须写成：
   `source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh` 再 `conda activate retool`。
4. **必须 `export PYTHONNOUSERSITE=1`** —— 否则 `~/.local/lib/python3.11/site-packages` 会混进环境
   （retool 也是 py3.11，路径同名），造成 torch/nvidia 库分裂、版本错乱的诡异报错。已写入 `env.sh`。
5. **别往 conda base 里装东西**（共享只读），pip 会静默退到 `~/.local`，制造上面第 4 条的污染。
6. **从外面 ssh/scp 用 `superpod.ust.hk`，别用 `slogin-01.cm.cluster`**。`.cm.cluster` 是集群
   内部节点间互称的域名，校外 DNS（含校园 VPN）解析不了，会得到 `NXDOMAIN` /
   `Could not resolve hostname`，**看起来像网络不通，其实只是名字用错了**。
   （2026-09-27 踩过：据此误判「Mac 连不上集群、只能 heredoc 粘贴」，浪费了一轮。）
7. **非交互 shell 里 `sbatch`/`squeue` 是教学占位符**。`/usr/bin/sbatch` 其实指向
   `tutorial-slurm`（打印「load slurm module」帮助）。交互式登录 shell 靠 `.bashrc:118`
   的 `module load slurm` 换到真版。所以从 Mac 用 ssh 跑 sbatch 时，要先
   `source /etc/profile.d/modules.sh; module load slurm`，否则 sbatch 会打开一个 pager 卡住。
8. **verl 0.6.1 的多轮工具循环是 async-only，vllm 必须升到 0.10.0**。这是一场三级跳：
   - vllm 0.8.5.post1 → 报 `ImportError: cannot import name 'get_tcp_uri' from 'vllm.utils'`，
     且缺 `vllm.v1.engine.utils.CoreEngineProcManager`（0.9.x 的 V1 引擎才有）。
   - vllm 0.9.2 → 上面的解决了，但 `AgentLoopManager` 初始化时又报
     `AttributeError: 'AsyncLLM' object has no attribute 'wait_for_requests_to_drain'`
     （该方法 **vllm 0.10.0 才加**，见 0.9.3/0.10.0 的 async_llm.py 对比）。
   - vllm 0.10.0 → 全绿。根因：`AgentLoopManager`（驱动沙箱工具循环）只在
     `rollout.mode="async"` 时创建（`ray_trainer.py:769-773`），sync 模式没有工具循环，
     所以不能退回 sync 规避。**教训：装前先读 verl 代码实际调的 vllm API 反查版本，别先装 0.8.5 再一个个跳。**
   另：**别用 `pip download --no-binary :none:` 检查 wheel 里有没有某模块**——它是对空包集合禁二进制
   （空操作），会下到 wheel 然后 `tar tzf *.tar.gz` 找不到东西，误判「没有这模块」。
   正确：`pip download` 后 `unzip -l *.whl | grep models/aimv2`。
9. **verl-recipe/retool 与 verl 0.6.1 的 `num_turns` 契约断裂**。recipe 的 `retool.py` `compute_score`
   读 `extra_info["num_turns"]`，但 verl 0.6.1 把轮数存成 `AgentLoopOutput.num_turns` →
   non_tensor_batch 的 `__num_turns__`，没放进 `extra_info`（`extra_fields` 里只有 `turn_scores`/`tool_rewards`），
   报 `KeyError: 'num_turns'`。**修法**：跑 `patch_verl_num_turns.py` 补丁到 installed 的
   `verl/experimental/agent_loop/tool_agent_loop.py`，构造 `AgentLoopOutput` 时把 `num_turns`
   一并塞进 `extra_fields`（幂等，可重复跑）。**重装 verl 后要重打**（patch 脚本在 `code/work/`）。

---

## 2. 集群目录树（实际路径）

根目录：**`/scratch/mscbdtspod2026/cyeah/retool/`**

```
/scratch/mscbdtspod2026/cyeah/retool/
├── env.sh                      # 环境变量（每个作业都 source 它）
│
├── code/                       # 代码
│   ├── verl-recipe/            # 官方 recipe 仓库（git clone，只读参考，别改）
│   └── work/                   # ★ 运行目录 —— 所有训练命令都从这里跑
│       ├── recipe/retool/      # 从 verl-recipe/retool 复制而来，可自由改
│       │   ├── retool.py               # 官方：自定义数据集类 + reward 函数
│       │   ├── retool_dataset_utils.py # 官方：字段映射 map_fn/map_fn2
│       │   ├── retool_sft_preprocess.py# 官方：SFT 数据预处理
│       │   ├── sft_preprocess.py       # ★ 自研：官方 SFT 预处理的本地沙箱版（tool_schema 改用 local_tool_config，与 RL 一致）
│       │   ├── sandbox_fusion_tool_config.yaml # 官方沙箱配置（需独立部署，未用）
│       │   ├── local_code_tool.py      # ★ 自研：本地 subprocess 沙箱 Tool
│       │   └── local_tool_config.yaml  # ★ 自研：指向本地沙箱的 tool 配置
│       ├── run_qwen3_1.7b_smoke.sh     # ★ 冒烟测试 launch 脚本
│       ├── run_qwen3_1.7b_sft.sh       # ★ SFT 冷启动 launch 脚本（bf16+micro_batch=2+truncation=left）
│       ├── smoke.slurm                 # ★ Slurm 作业（冒烟，调 run_qwen3_1.7b_smoke.sh）
│       ├── sft.slurm                   # ★ Slurm 作业（SFT，调 run_qwen3_1.7b_sft.sh）
│       ├── preflight_check.py          # ★ 冒烟前预检：Tool 执行行为（login node 可跑）
│       ├── data_check2.py              # ★ 冒烟前预检：数据 schema + map_fn/map_fn2
│       ├── import_check.py             # ★ 冒烟前预检：vllm/verl/数据集/Tool import
│       └── make_smoke_subset.py        # ★ 切 64 行 DAPO 子集（pyarrow 直切）
│
├── data/                       # 数据（软链重排后才是 recipe 能用的形态）
│   ├── model/Qwen3-1.7B/       # 基座模型（~3.8G，HF 原始下载）
│   ├── dataset/                # HF 原始下载（层级是 HF 仓库原样，不能直接用）
│   │   ├── BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet
│   │   ├── Maxwell-Jia/AIME_2024/aime_2024_problems.parquet
│   │   └── yentinglin/aime_2025/{data,part1,part2}/*.parquet
│   └── dataset_clean/          # ★ 软链重排 —— launch 脚本用这个
│       ├── BytedTsinghua-SIA/DAPO-Math-17k/        (179万行，全量)
│       ├── BytedTsinghua-SIA/DAPO-Math-17k-smoke/  (64行，冒烟用，pyarrow 直切的真子集)
│       ├── Maxwell-Jia/AIME_2024/            (30题)
│       ├── yentinglin/aime_2025/             (30题，来自 data/ 那份)
│       └── JoeYing/ReTool-SFT/               (2000行，SFT 冷启动，sft_preprocess.py 产出 train.parquet)
│
├── checkpoints/                # 训练产出的 checkpoint（按 experiment_name 分子目录）
│   └── qwen3-1.7b-instruct-sft/  # ★ SFT 冷启动产物：global_step_{63..372}，最终 372（6 epoch × 62 step）
├── outputs/                    # Slurm 日志（.out）与临时产物
└── hf/                         # HF_HOME 缓存（模型/数据集下载缓存）
```

**为什么要 `dataset_clean/`**：`recipe/retool/retool.py` 用「路径末两段」判断数据集类型
（`data_source = "/".join(path.split("/")[-2:])`，命中 `Maxwell-Jia/AIME_2024` /
`yentinglin/aime_2025` 才走 AIME 字段解析）。所以传进去的路径**必须以 repo 名结尾**。
`dataset_clean/` 用软链把每个 parquet 摆到正确层级，**不用改官方代码**就能工作。

**为什么要 `DAPO-Math-17k-smoke/` 子集**：`retool.py` 的 `CustomRLHFDataset._read_files_and_tokenize`
**重写时没调用父类**，所以 `data.train_max_samples` 采样键对它**无效**——不切子集就会每次冒烟
都先对全量 179 万行跑一遍 `map_fn2`，白白烧时间。`make_smoke_subset.py` 用 pyarrow 直切前 64 行，
冒烟脚本指向子集目录。**正式跑记得把 launch 脚本的 `dapo_math_17k` 改回全量目录。**

---

## 3. 路径速查（写脚本时直接抄）

`env.sh` 内容（已创建）：

```bash
export PYTHONNOUSERSITE=1
export PROJECT_ROOT=/scratch/mscbdtspod2026/cyeah/retool
export HF_HOME=$PROJECT_ROOT/hf
export DATA_ROOT=$PROJECT_ROOT/data
export CKPT_ROOT=$PROJECT_ROOT/checkpoints
```

**所有 Slurm 脚本统一头部**：

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh
```

**launch 脚本里用的数据路径**：

```bash
# 冒烟用 64 行子集；正式跑改回 DAPO-Math-17k 全量目录
dapo_math_17k=$DATA_ROOT/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-smoke
aime_2024=$DATA_ROOT/dataset_clean/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset_clean/yentinglin/aime_2025
model_path=$DATA_ROOT/model/Qwen3-1.7B
tool_config_path=recipe/retool/local_tool_config.yaml   # 相对 work/ 目录
```

---

## 4. 本地 ↔ 集群 对照（改动两边都要同步）

本地项目里的自研代码，与集群上的落地位置：

| 本地（Mac，项目仓库） | 集群（SuperPOD） |
|---|---|
| `code/retool-small-llm/recipe/retool/local_code_tool.py` | `.../retool/code/work/recipe/retool/local_code_tool.py` |
| `code/retool-small-llm/recipe/retool/local_tool_config.yaml` | `.../retool/code/work/recipe/retool/local_tool_config.yaml` |
| `code/retool-small-llm/run_qwen3_1.7b_smoke.sh` | `.../retool/code/work/run_qwen3_1.7b_smoke.sh` |
| `code/retool-small-llm/smoke.slurm` | `.../retool/code/work/smoke.slurm` |
| `code/retool-small-llm/preflight_check.py` | `.../retool/code/work/preflight_check.py` |
| `code/retool-small-llm/data_check2.py` | `.../retool/code/work/data_check2.py` |
| `code/retool-small-llm/import_check.py` | `.../retool/code/work/import_check.py` |
| `code/retool-small-llm/make_smoke_subset.py` | `.../retool/code/work/make_smoke_subset.py` |
| `code/verl-recipe/retool/*`（官方参考，只读） | `.../retool/code/verl-recipe/retool/*` + 复制到 `work/recipe/retool/` |

> **约定**：改自研代码一律**先改本地**，再用 `scp` 推上集群，保证本地仓库是唯一真相源。
> 直接在集群上改会导致两边漂移。

**把本地代码推上集群（在 Mac 上跑，不是在集群上跑）**

前置（只需做一次）：把 Mac 的公钥装到集群

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub cyeah@superpod.ust.hk
```

之后用 `scp` 推文件（走密钥认证，无需密码）：

```bash
# 本地基准目录：/Users/chuan/Documents/HKUST/Academic/Courses/IP_6910A
LOCAL=/Users/chuan/Documents/HKUST/Academic/Courses/IP_6910A/code/retool-small-llm
REMOTE=cyeah@superpod.ust.hk:/scratch/mscbdtspod2026/cyeah/retool/code/work

scp $LOCAL/recipe/retool/local_code_tool.py     $REMOTE/recipe/retool/
scp $LOCAL/recipe/retool/local_tool_config.yaml $REMOTE/recipe/retool/
scp $LOCAL/run_qwen3_1.7b_smoke.sh              $REMOTE/
scp $LOCAL/smoke.slurm                          $REMOTE/
```

> 只推**自研**文件。`recipe/retool/` 里官方那几个（`retool.py`、`retool_dataset_utils.py`、
> `retool_sft_preprocess.py`）是当初从 `code/verl-recipe/retool/` 复制上去的，别覆盖。

**提交冒烟作业 + 监控（在集群 login node 上）**

```bash
# 交互式登录 shell 里 sbatch 已是真版；若从脚本/ssh 里跑，先 source module：
source /etc/profile.d/modules.sh; module load slurm

cd /scratch/mscbdtspod2026/cyeah/retool/code/work
sbatch smoke.slurm          # 提交，记下 job id
squeue -u $USER             # 看状态：PD=排队 / R=运行
tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_smoke_<jobid>.out   # 看日志
```

> 只推**自研**文件。`recipe/retool/` 里官方那几个（`retool.py`、`retool_dataset_utils.py`、
> `retool_sft_preprocess.py`）是当初从 `code/verl-recipe/retool/` 复制上去的，别覆盖。

---

## 5. 关键设计决策（记录「为什么这么定」）

| 决策 | 选择 | 理由 |
|---|---|---|
| 训练框架 | **verl 0.6.1 + SuperPOD**（非 PyTRIO 云） | Guideline 指向 SuperPOD；免费、符合已汇报方案 |
| 沙箱 | **本地 subprocess 沙箱**（自研 `LocalCodeTool`），非官方 SandboxFusion | SandboxFusion 要独立部署 HTTP 服务（通常需 Docker），计算节点未必允许；冒烟阶段零部署优先 |
| 模型 | **Qwen/Qwen3-1.7B**（= Instruct 版） | Guideline 建议「选最小的能跑完的」；1.7B 单卡 80G 舒适。**⚠️ Qwen3 命名反转：`Qwen3-1.7B` 是 Instruct（含 tool_call 模板），`-Base` 才是 base，无 `-Instruct` 后缀**（2026-10-08 踩过：HF 上 `Qwen3-1.7B-Instruct` 是 404 不存在） |
| RL 算法 | GRPO + DAPO clip-higher（`clip_ratio_high=0.28`） | 官方 recipe 即此配置；**GRPO 对比只需把它改成 `0.0`** |
| 实验变量 | 只动 RL update，其余全冻结 | 对应汇报中承诺的「freeze everything, vary only RL update」 |

---

## 6. 待办 / 尚未落地

- [x] 确认登录主机名 = `superpod.ust.hk`（见 §1；`slogin-01.cm.cluster` 校外不可解析）
- [x] 把自研代码推上集群（`scp`，命令见 §4；前置：`ssh-copy-id` 装公钥）
- [x] 前置检查：Qwen3-1.7B 的 chat template —— **已确认含 `<tool_call>`**，
      `eos_token='<|im_end|>'`，模板长 4168。**无需换 Instruct，无需自定义模板**。
      （但仍需 SFT 冷启动：模板存在 ≠ 基座模型会主动调工具，这正是 H1）
- [x] 冒烟前预检三件套（login node 可跑，不占 GPU）：Tool 执行 19/19、数据 schema+映射 11/11、import 全绿
- [x] 跑通 5 步冒烟测试（Gate 1）—— job 615359（`get_tcp_uri`）→ 615381（`wait_for_requests_to_drain`）→ 615461（`KeyError: num_turns`）→ 打 num_turns 补丁后 **job 615480 COMPLETED（10:54）**。四条验收全过：① `dataset len` 64/60；② `LocalCodeTool` 初始化 + `num_turns` 2–6 多轮工具调用；③ `critic/score` max=1.0/min=−1.0（reward 不全同，mean −0.62→−0.24 有信号）；④ `global_step_5/actor/{model,optim,extra_state}*.pt` + `huggingface/` 落盘
- [x] SFT 阶段代码（`sft_preprocess.py` + FSDP SFT 训练）—— **✅ 完成（2026-10-08）**：job 663756 COMPLETED，ckpt `checkpoints/qwen3-1.7b-instruct-sft/global_step_372`
- [ ] RL 训练（GRPO/DAPO）：`run_qwen3_1.7b_dapo.sh`，`actor_rollout_ref.model.path` → SFT ckpt `global_step_372`（对齐官方 dapo 脚本）
- [ ] AIME 2024 评测加入固定评测脚本（官方 eval 只覆盖 AIME25）
- [ ] 确认 DAPO-Math-17k 为何有 179 万行（名字叫 17k，需查证是否 repo 更新）

---

## 7. 更新记录

| 日期 | 改动 |
|---|---|
| 2026-09-27 | 初版：环境装完（torch/vllm/verl/flash-attn）、数据下载+软链重排、自研沙箱与冒烟脚本写好 |
| 2026-09-27 | 修正登录主机名：`superpod.ust.hk`（原误记为 `slogin-01.cm.cluster`，校外 NXDOMAIN）；§4 补 `ssh-copy-id` + `scp` 实际命令；§6 勾掉 chat template 前置检查 |
| 2026-09-27 | 打通 scp 推代码（免密）；修 `local_code_tool.py` 的 auto-print bug（AST 精确补 print，替代官方朴素实现，避免把 `x=1` 误包成 `print(x=1)`）；**降级 transformers 4.57.6 / datasets 3.6.0**（5.x 与 vllm 0.8.5 不兼容）；切 64 行 DAPO 子集；建 `smoke.slurm` 并提交 **job 615359** |
| 2026-09-27 | **升级 vllm 0.8.5→0.9.2** 修异步 rollout 报错（`get_tcp_uri`/`CoreEngineProcManager` 缺失，见坑 #8）；配套升 torch 2.7.0+cu128、flash-attn 2.8.3.post1，降 numpy 1.26.4、transformers 4.53.2；重提 smoke **job 615381** |
| 2026-09-27 | **再升级 vllm 0.9.2→0.10.0** 修 `AsyncLLM.wait_for_requests_to_drain` 缺失（该法 0.10.0 才加）；torch 2.7.0+cu128→2.7.1+cu126、新增 xformers 0.0.31、numpy 重锁 1.26.4；import gate 全绿；重提 smoke **job 615461** |
| 2026-09-27 | **打 num_turns 补丁**：verl 0.6.1 没把轮数放 `extra_info`，recipe `compute_score` 读 `extra_info["num_turns"]` 报 `KeyError`；`patch_verl_num_turns.py` 把 `num_turns` 注入 `tool_agent_loop.py` 的 `extra_fields`；重提 smoke **job 615480** |
| 2026-09-27 | **Gate 1 通过 ✅** job 615480 `COMPLETED`（10:54，MaxRSS 62GB，~2300 tok/s，`aborted_ratio:0.0`）。四条验收全绿：数据加载（`dataset len` 64/60）、沙箱执行（`LocalCodeTool` 初始化 + `num_turns` 2–6 + `tool_calls` 计时非零）、reward 不全同（每步 `score/max=1.0,min=−1.0`）、checkpoint 落盘（`global_step_5/actor/*.pt` + `huggingface/`） |
| 2026-10-08 | **Qwen3 命名反转发现**：`Qwen/Qwen3-1.7B` 是 Instruct 版（`tokenizer_config.json` 含 tool_call `chat_template`），`-Base` 才是 base，**不存在 `-Instruct` 后缀**（HF 上 `Qwen3-1.7B-Instruct` 是 404）。据此把 SFT `MODEL_PATH` 定为 `Qwen3-1.7B`，对齐官方「用 Instruct 冷启动」 |
| 2026-10-08 | **SFT 冷启动完成 ✅（Gate 2）**：`sft_preprocess.py` 产 `train.parquet`（2000 行多轮 tool 消息）；FSDP SFT 单卡 6 epoch，**job 663756 COMPLETED（~80min，13s/it×372 step）**，ckpt `qwen3-1.7b-instruct-sft/global_step_{63,126,189,252,315,372}`。连踩 4 坑：① fp32 OOM（job 663732，verl FSDP SFT 默认 `model_dtype=fp32`）→ ② `model.fsdp_config.model_dtype=bf16` 后仍 OOM（job 663739，logits 显存 = micro_batch×seq×vocab 与模型大小无关，官方 7B 靠 8 卡+ulysses SP=4 摊 1/8，单卡扛不住 micro_batch=4）→ ③ `micro_batch_size_per_gpu=2` 跑通但遇超长样本（job 663747，6 样本 >8192 token，`truncation=error` 崩）→ ④ `data.truncation=left` 全绿（job 663756） |
