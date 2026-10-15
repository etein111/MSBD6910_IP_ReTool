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
10. **SFT→RL 衔接：verl 0.6.1 的 RL 从 `model.path` 根目录加载 tokenizer + config + HF 权重，但 SFT ckpt 三者都不在根目录**。两处要补（2026-10-09 踩过，job 664052/664071/668747 连环 FAILED）：
    - **tokenizer/config 在 `huggingface/` 子目录**：`fsdp_workers.py:302` `hf_tokenizer(local_path)` 与 `:320` `AutoConfig.from_pretrained(local_path)` 都读**根目录**，而 SFT 把它们存进 `huggingface/`。症状 `TypeError: vocab_file=None`（慢速 tokenizer 在根目录找不到 vocab.json）。修法：`cp ckpt/huggingface/* ckpt/` 铺到根目录。
    - **缺 HF 权重（safetensors）**：SFT 默认 `trainer.checkpoint.save_contents=["model","optimizer","extra"]` **不含 `hf_model`**，ckpt 只有 FSDP 权重 `model_world_size_1_rank_0.pt`。RL `_build_model_optimizer:377` 用 `AutoModelForCausalLM.from_pretrained(local_path)` 加载 HF 权重，报 `OSError: no file named pytorch_model.bin / model.safetensors`。修法：用 verl 自带 merger 转 HF：`python -m verl.model_merger merge --backend fsdp --local_dir <ckpt> --target_dir <ckpt>-hf`（从 `local_dir/huggingface/config.json` 读 config），RL 的 `model_path` 指向 `-hf` 目录。
11. **账户有 GPU 分钟预算，不是墙钟上限（2026-10-10 踩过）**：account `mscbdtspod2026` 有 `GrpTRESMins=gres/gpu=9600`（≈160 GPU·h 总量），QOS MaxWall=3d、MaxSubmit=2。8×GPU 的 `--time` 上限 ≈ 9600/8/60 ≈ 17h，且随历史 job 消耗递减。`sbatch --time=48:00:00`（23040 GPU·min）会报 `AssocGrpGRESMinutes / Job violates accounting/QOS policy`。**提交前先 `sacct -A mscbdtspod2026 -X` 累加已用 GPU·min，`--time ≤ (9600−已用)/8`，别按墙钟拍脑袋。**
12. **1.7B 多卡别用 ulysses SP / vllm TP（2026-10-10 踩过）**：照抄官方 7B recipe 的 `train_sp=4/infer_tp=4`，8 卡冒烟 job 672236 报 `all_to_all → ncclUnhandledCudaError → Cuda failure 'out of memory'`。根因 ulysses SP 的 `all_to_all` NCCL collective 是给 7B+/长序列摊显存的，1.7B 单卡放得下，SP/TP 只添乱。**修法**：`train_sp=1 / infer_tp=1 / offload=False`（纯 FSDP 数据并行 + 8 份 vllm(TP=1)），8 卡冒烟 672270 通过，`max_memory_reserved_gb` 53。
13. **工具名不匹配 + max_turns 过大 → rollout 极慢（2026-10-10 踩过）**：`local_tool_config.yaml` 里工具名是 `code_interpreter`，但 SFT-2ep 冷启动模型常生成错名（`code_editor`/`code_executor`/`solving_quadratic`）和坏 JSON（`Failed to decode tool call`）。`max_turns=16` 时模型不收敛、响应顶到 8192 token，job 675119 首步 rollout 1h+ 未走完。**修法**：`max_turns=8`（冒烟 672270 实测 ~3.3 轮收敛、响应均值 1778 token、41s/步）+ `n_resp_per_prompt=4`，正式 job 675200 落到 ~108s/步。
14. **RL checkpoint 合并的 --local_dir 要指向 actor/ 子目录（2026-10-15 踩过）**：verl 0.6.1 的 RL checkpoint 结构是 `global_step_271/actor/`（含 `model_world_size_8_rank_*.pt` + `huggingface/` + `fsdp_config.json`），而 SFT 那次是平铺在根目录。merge 时 RL 要 `--local_dir .../global_step_271/actor`（不是根目录），否则 merger 找不到分片。产出 `global_step_271-hf/`（model.safetensors 4.06GB），供独立评测当 model_path。

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
│       │   ├── retool_dataset_utils.py # ★ 自研修复版：字段映射 map_fn/map_fn2（map_fn2 剥 DAPO 外层模板，修格式冲突）
│       │   ├── retool_sft_preprocess.py# 官方：SFT 数据预处理
│       │   ├── sft_preprocess.py       # ★ 自研：官方 SFT 预处理的本地沙箱版（tool_schema 改用 local_tool_config，与 RL 一致）
│       │   ├── sandbox_fusion_tool_config.yaml # 官方沙箱配置（需独立部署，未用）
│       │   ├── local_code_tool.py      # ★ 自研：本地 subprocess 沙箱 Tool
│       │   └── local_tool_config.yaml  # ★ 自研：指向本地沙箱的 tool 配置
│       ├── run_qwen3_1.7b_smoke.sh     # ★ 冒烟测试 launch 脚本
│       ├── run_qwen3_1.7b_sft.sh       # ★ SFT 冷启动 launch 脚本（bf16+micro_batch=2+truncation=left）
│       ├── run_qwen3_1.7b_dapo_smoke.sh# ★ RL 冒烟 launch：model_path → SFT ckpt，验证 SFT→RL 衔接
│       ├── smoke.slurm                 # ★ Slurm 作业（冒烟，调 run_qwen3_1.7b_smoke.sh）
│       ├── sft.slurm                   # ★ Slurm 作业（SFT，调 run_qwen3_1.7b_sft.sh）
│       ├── dapo_smoke.slurm            # ★ Slurm 作业（RL 冒烟，调 run_qwen3_1.7b_dapo_smoke.sh）
│       ├── dapo_smoke_8gpu.slurm      # ★ Slurm 作业（8 卡 RL 冒烟，覆盖 n_gpus_per_node=8 + experiment_name）
│       ├── run_qwen3_1.7b_dapo.sh     # ★ RL 正式训练 launch（DAPO/GRPO，全量 dedup 数据，8 卡，n=4/max_turns=8）
│       ├── dapo.slurm                  # ★ Slurm 作业（RL 正式，调 run_qwen3_1.7b_dapo.sh，--time=15h）
│       ├── merge_sft_hf.slurm          # ★ Slurm 作业（FSDP→HF 合并，调 verl.model_merger）
│       ├── merge_rl_hf.slurm           # ★ Slurm 作业（RL final FSDP→HF 合并；--local_dir 要指向 .../global_step_271/actor）
│       ├── eval_qwen3_1.7b.sh          # ★ 冻结协议独立评测 launch（val-only：12 采样，tool/text，参数化 EVAL_MODEL_PATH/EVAL_MODE/EVAL_EXP）
│       ├── eval_smoke.slurm            # ★ Slurm 作业（评测冒烟：base+text，验证 eval 脚本端到端）
│       ├── eval.slurm                  # ★ Slurm 作业（全套 6 组评测：base/sft/rl × tool/text 循环）
│       ├── make_dedup_dataset.py       # ★ DAPO-Math-17k 去重（179万→17398 行，按 prompt 文本）
│       ├── watch_job.sh                # ★ 通用 job 监控（本地循环 + 短 SSH 抗掉线）
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
│       ├── BytedTsinghua-SIA/DAPO-Math-17k/        (179万行，全量——名字叫17k是发布bug，实际~17k独特题×~100次重复)
│       ├── BytedTsinghua-SIA/DAPO-Math-17k-smoke/  (64行，冒烟用，pyarrow 直切的真子集)
│       ├── BytedTsinghua-SIA/DAPO-Math-17k-dedup/  (★ 17398行，按 prompt[0]["content"] 去重，RL 正式训练用这个)
│       ├── Maxwell-Jia/AIME_2024/            (30题)
│       ├── yentinglin/aime_2025/             (30题，来自 data/ 那份)
│       └── JoeYing/ReTool-SFT/               (2000行，SFT 冷启动，sft_preprocess.py 产出 train.parquet)
│
├── checkpoints/                # 训练产出的 checkpoint（按 experiment_name 分子目录）
│   ├── qwen3-1.7b-instruct-sft/      # SFT 6ep（已弃用）：global_step_{63..372}
│   ├── qwen3-1.7b-instruct-sft-2ep/  # ★ SFT 2ep 冷启动（现用）：global_step_{62,124}
│   │   └── global_step_124-hf/       # ★ HF 格式（safetensors），RL 冷启动 model_path 指这
│   ├── qwen3-1.7b-dapo-smoke-8gpu/   # 8 卡 RL 冒烟（job 672270）产物
│   ├── qwen3-1.7b-dapo/              # ★ RL 正式训练（job 675200，271 步/epoch）
│   │   └── global_step_271-hf/       # ★ RL final 的 HF 合并（merge_rl_hf.slurm，job 685931），独立评测 model_path 指这
│   ├── qwen3-1.7b-eval-*/           # ★ 独立评测 tensorboard 目录（6 组+smoke，job 686226/685932）
│   └── qwen3-1.7b-dapo-cancelled-step20/  # 被取消的 675119 留的 step20 ckpt（mv 归档，防自动恢复冲突）
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
| `code/retool-small-llm/recipe/retool/retool_dataset_utils.py`（★ 自研修复版：map_fn2 剥 DAPO 模板） | `.../retool/code/work/recipe/retool/retool_dataset_utils.py` |
| `code/retool-small-llm/run_qwen3_1.7b_smoke.sh` | `.../retool/code/work/run_qwen3_1.7b_smoke.sh` |
| `code/retool-small-llm/run_qwen3_1.7b_dapo.sh` | `.../retool/code/work/run_qwen3_1.7b_dapo.sh` |
| `code/retool-small-llm/run_qwen3_1.7b_dapo_smoke.sh` | `.../retool/code/work/run_qwen3_1.7b_dapo_smoke.sh` |
| `code/retool-small-llm/dapo.slurm` | `.../retool/code/work/dapo.slurm` |
| `code/retool-small-llm/dapo_smoke_8gpu.slurm` | `.../retool/code/work/dapo_smoke_8gpu.slurm` |
| `code/retool-small-llm/smoke.slurm` | `.../retool/code/work/smoke.slurm` |
| `code/retool-small-llm/preflight_check.py` | `.../retool/code/work/preflight_check.py` |
| `code/retool-small-llm/data_check2.py` | `.../retool/code/work/data_check2.py` |
| `code/retool-small-llm/import_check.py` | `.../retool/code/work/import_check.py` |
| `code/retool-small-llm/make_smoke_subset.py` | `.../retool/code/work/make_smoke_subset.py` |
| `code/retool-small-llm/eval_qwen3_1.7b.sh` | `.../retool/code/work/eval_qwen3_1.7b.sh` |
| `code/retool-small-llm/eval_smoke.slurm` | `.../retool/code/work/eval_smoke.slurm` |
| `code/retool-small-llm/eval.slurm` | `.../retool/code/work/eval.slurm` |
| `code/retool-small-llm/merge_rl_hf.slurm` | `.../retool/code/work/merge_rl_hf.slurm` |
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

> 只推**自研**文件。`recipe/retool/` 里官方那几个（`retool.py`、`retool_sft_preprocess.py`）
> 是当初从 `code/verl-recipe/retool/` 复制上去的，别覆盖。**例外**：`retool_dataset_utils.py`
> 已被自研修复版（map_fn2 剥 DAPO 模板）覆盖，它的真源是 `code/retool-small-llm/recipe/retool/`。

**提交冒烟作业 + 监控（在集群 login node 上）**

```bash
# 交互式登录 shell 里 sbatch 已是真版；若从脚本/ssh 里跑，先 source module：
source /etc/profile.d/modules.sh; module load slurm

cd /scratch/mscbdtspod2026/cyeah/retool/code/work
sbatch smoke.slurm          # 提交，记下 job id
squeue -u $USER             # 看状态：PD=排队 / R=运行
tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_smoke_<jobid>.out   # 看日志
```

> 只推**自研**文件。`recipe/retool/` 里官方那几个（`retool.py`、`retool_sft_preprocess.py`）
> 是当初从 `code/verl-recipe/retool/` 复制上去的，别覆盖。**例外**：`retool_dataset_utils.py`
> 已被自研修复版（map_fn2 剥 DAPO 模板）覆盖，它的真源是 `code/retool-small-llm/recipe/retool/`。

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
- [x] **FSDP→HF 合并**（SFT→RL 衔接的权重侧修复）：`merge_sft_hf.slurm`（**job 668790 COMPLETED，1min**）经 `verl.model_merger --backend fsdp` 转出 `global_step_372-hf/`，含 `model.safetensors`（4.06GB）+ tokenizer/config 全在根目录；chat_template(4168)、`<tool_call>`、vocab 151669 均已验证
- [x] RL 冒烟（验证 SFT→RL 衔接全链路）—— **✅ 衔接打通（job 668819 COMPLETED）**：tokenizer + HF 权重加载 + reward 信号 + ckpt 落盘全过。**但发现 SFT 冷启动负迁移**：base 首轮 19% 正确 vs SFT 0% 正确（256 rollout 零答对，max=−0.60）
- [x] **诊断 SFT 负迁移根因**（`diag_sft_rollout.py` 跑 rollout 轨迹）：SFT 模型 4 类失败——①纯冗长推理不收敛（2048 token 空转不调工具不给答案）②理解错题→代码逻辑错（题 10 把公司数当节点数）③代码 SyntaxError + `<tool_call>` JSON 截断解析失败 ④**`\boxed{}` 格式全丢**（0/4 次输出，reward `strict_box_verify=True` 只认 boxed）。根因：**6 epoch 对 1.7B 过拟合/灾难性遗忘**，把 base 的「长推理→boxed」换成了「调工具但代码写不对」。（lr 本是 verl 默认 **1e-5**，此前误读成 1e-3 是 `fsdp_sft_trainer.py:522` 硬编码日志标签 `"train/lr(1e-3)"` 的锅——数值=真实 lr×1000，反推 step1 值 0.00027=1e-5×(1/37) 线性 warmup。）base 模型本身不调工具（num_turns 2.06），靠超长单轮推理硬算蒙对 19%
- [x] **base 同款诊断对比**（`diag_sft_rollout.py` 跑 base `/scratch/.../data/model/Qwen3-1.7B`，**job 669552，8192 预算**，因 base 自然 response_length≈6919 不能被 2048 截断）：4 题 0 对，但**失败模式与 SFT 判然不同**——base 是「长 `<think>` 推理、多数不给 boxed、超时不收敛」（题 0/5 打满 8192 token 仍无答案；题 20 5870 token 后 boxed='25' 但错；题 10 2864 token 后竟也吐 `<tool_call>print("2014")` 然后答 `Answer: $2014$`）。**关键二次发现**：base 也卡在 `\boxed{}` 格式上（4 题只有 1 题给 boxed），题 10 直接跟了 DAPO 原 prompt 的 `Answer: $Answer` 格式而非 boxed——印证 §5 记录的「DAPO prompt 与 `answer_format` 格式冲突」对 base/SFT 都是独立瓶颈，不只是 SFT 负迁移。
- [x] **格式冲突修复**：自研 `recipe/retool/retool_dataset_utils.py`（map_fn2 剥 DAPO 外层模板再追加 `\boxed{}`），scp 覆盖 work/recipe/retool/
- [x] **温和 SFT 重训**：epoch 6→2、lr 1e-5、experiment_name `-2ep`，**job 671667 COMPLETED（24min，val/loss 0.453）**，ckpt `global_step_124`
- [x] **merge + RL 冒烟验证负迁移已修复 ✅**（2026-10-09）：merge **job 671846** 转出 `-2ep/global_step_124-hf`（model.safetensors 4.06GB）；RL 冒烟 **job 671847 COMPLETED（6 步全跑完，退出码 0）**。结果：`critic/score/mean:-0.6156`（旧 SFT −0.881 → 回升到 base −0.622 水平）、**`score/max:1.0`**（至少一个样本产出与答案匹配的 `\boxed{}`，旧 SFT 格式冲突时永远到不了 1.0）、`aborted_ratio:0.0`、num_turns 6.125。日志尾部 `DataLoader worker killed by signal: Killed` 是 **atexit 关闭阶段的良性清理报错**（`Exception ignored in atexit callback`，出现在 ckpt `global_step_6` 落盘之后），非训练中崩溃；GPU `max_memory_reserved_gb:74.75` < 80GB 无显存 OOM。**结论：温和 SFT（2ep）+ 格式修复双管齐下，0% 负迁移已解除**
- [ ] RL 正式训练（GRPO/DAPO）：`run_qwen3_1.7b_dapo.sh`，`model.path` → `-2ep/global_step_124-hf`，数据用 `DAPO-Math-17k-dedup`（17,398 行去重后）——**待冒烟验证通过后**
- [x] **8 卡 RL 冒烟验证多卡加速 ✅（job 672270 COMPLETED，6m24s）**：纯 FSDP + 8×vllm(TP=1)，8-rank ckpt 落盘，reward 信号正常，无 OOM（`max_memory_reserved_gb` 53.07）
- [x] **RL 正式训练（job 675200）✅ COMPLETED（9h08m，退出码 0，271/271 步）**：`run_qwen3_1.7b_dapo.sh`（n=4 / max_turns=8 / SP=1 / TP=1 / offload=False），`model.path`→`-2ep/global_step_124-hf`，数据 `DAPO-Math-17k-dedup`（17,398 行→271 步/epoch），`--time=15h`。终态 `critic/score/mean` **-0.643→+0.230**；AIME2024 `acc/best@4` **29.3%→50.6%**、`acc/mean@4` 18.3%→37.5%；AIME2025 `acc/best@4` **26.3%→48.3%**、`acc/mean@4` 16.7%→35.8%；num_turns 4.5→10.1。ckpt `global_step_{20..260,271}` 落盘，`latest_checkpointed_iteration=271`
- [x] **冻结协议独立评测（12 采样，tool/text，base/SFT/RL 三检查点）✅（2026-10-15）**：`eval_qwen3_1.7b.sh`（val-only：`val_before_train=True` + `total_training_steps=1` + `n_resp_per_prompt_val=12`）+ `merge_rl_hf.slurm`（RL final 合并）+ `eval.slurm`（6 组循环）。冒烟 job **685932**（base+text）通过，全套 job **686226 COMPLETED（42min，6 组全 OK）**。结果（AIME 2024/2025 acc@12）：base-tool 22.8/22.5、base-text 16.4/16.9、sft-tool 11.7/14.4（负迁移）、sft-text 5.0/8.9、**rl-tool mean 41.1/35.0、best 70.5/55.3**、rl-text 10.8/8.1；num_turns（tool）2.0→4.8→8.6
- [ ] **GRPO 对照（E3）**：`clip_ratio_high=0.0` + `experiment_name=qwen3-1.7b-grpo`，同 271 步预算 → 检验 H3（DAPO vs GRPO 稳定性）
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
| 2026-10-09 | **SFT→RL 衔接连踩 2 坑并修复**：① tokenizer 加载报 `TypeError: vocab_file=None`（job 664052/664071）——SFT 把 tokenizer/config 存到 `huggingface/` 子目录，而 RL `_build_model_optimizer:302` 从 `model.path` **根目录**读 tokenizer，`cp ckpt/huggingface/* ckpt/` 修掉；② HF 权重缺失报 `OSError: no file named model.safetensors`（job 668747）——SFT 默认 `save_contents` 不含 `hf_model`，只存 FSDP 权重，而 RL `:377` 用 `AutoModelForCausalLM.from_pretrained` 读 HF 权重，`verl.model_merger --backend fsdp` 合并出 `global_step_372-hf/`（**job 668790 COMPLETED，model.safetensors 4.06GB**）。合并后 tokenizer 验证齐全，`run_qwen3_1.7b_dapo_smoke.sh` 的 `model_path` 改指 `-hf` 目录，重提 RL 冒烟 **job 668819** |
| 2026-10-09 | **RL 冒烟通过 + 发现 SFT 冷启动负迁移**：job 668819 COMPLETED，衔接全链路（tokenizer + HF 权重 + reward + ckpt）全过。但 **base 首轮 19% 正确 vs SFT 0% 正确**（256 rollout 零答对，step1 max=−0.60）。用 `diag_sft_rollout.py` 复现 verl hermes 工具循环看轨迹，定位 SFT 失败 4 类：①冗长推理不收敛（空转不调工具不给答案）②理解错题→代码逻辑错 ③代码 SyntaxError + `<tool_call>` JSON 截断 ④**`\boxed{}` 全丢**（reward `strict_box_verify=True` 只认 boxed）。根因 = **6 epoch 对 1.7B 灾难性遗忘**（lr 本是 verl 默认 1e-5，误读成 1e-3 是 `fsdp_sft_trainer.py:522` 硬编码日志标签的锅），把 base 的「长推理→boxed」换成「调工具但代码写不对」。base 本身不调工具（num_turns 2.06）靠长推理蒙对 19% |
| 2026-10-09 | **base 同款诊断对比（job 669552，8192 预算）**：base 4 题 0 对但失败模式与 SFT 不同——base「长 `<think>` 推理、多数不给 boxed、超时不收敛」（题 0/5 打满 8192 仍无答案；题 20 5870 token 后 boxed='25' 错；题 10 2864 token 后吐 `print("2014")` 再答 `Answer: $2014$`）。**二次发现**：base 也卡 `\boxed{}` 格式（4 题仅 1 题给 boxed），题 10 直接跟 DAPO 的 `Answer: $Answer` 而非 boxed——「DAPO prompt 与 answer_format 格式冲突」对 base/SFT 都是独立瓶颈，不单是 SFT 负迁移 |
| 2026-10-09 | **温和 SFT 重训 + 格式冲突修复（决定后落地）**：① 修格式——DAPO-Math-17k 题目外层模板（prefix `Solve...Answer: $Answer...\n\n` + suffix `Remember...after "Answer:".`）与 answer_format 的 `\boxed{}` 冲突，自研 `recipe/retool/retool_dataset_utils.py` 的 `map_fn2` 先剥模板再追加 boxed（AIME 走 `map_fn` 不变），scp 覆盖 work/recipe/retool/；② 温和 SFT——epoch 6→2、lr 显式 1e-5、experiment_name → `-2ep`，**job 671667 COMPLETED（24min，val/loss 0.453）**，ckpt `global_step_{62,124}`；③ 待 merge → RL 冒烟验证 0% 是否回升 |
| 2026-10-09 | **merge + RL 冒烟验证负迁移已修复 ✅**：merge **job 671846**（`-2ep/global_step_124-hf`，model.safetensors 4.06GB）；RL 冒烟 **job 671847 COMPLETED（6 步，退出码 0）**。`critic/score/mean:-0.6156`（旧 SFT −0.881 → 回升到 base −0.622 水平）、**`score/max:1.0`**（有样本产出匹配 `\boxed{}`，旧 SFT 格式冲突时永远到不了）、`aborted_ratio:0.0`、num_turns 6.125。日志尾部 `DataLoader worker killed` 是 atexit 关闭阶段良性清理报错（在 ckpt `global_step_6` 落盘之后，非训练中崩溃），GPU `max_memory_reserved_gb:74.75` 无 OOM。**0% 负迁移已解除，可进正式 RL 训练** |
| 2026-10-10 | **8 卡多卡验证 ✅（job 672270 COMPLETED，6m24s）**：纯 FSDP + 8×vllm(TP=1)，8-rank ckpt 落盘、reward 信号正常、`max_memory_reserved_gb` 53.07 无 OOM。**先踩了 SP/TP 的坑**：照抄 7B recipe 的 `train_sp=4/infer_tp=4` → job 672236 报 `all_to_all` NCCL OOM（见坑 #12），改成 SP=1/TP=1 后通过 |
| 2026-10-10 | **发现 GPU 分钟预算约束**：`GrpTRESMins=gres/gpu=9600`（≈160 GPU·h），8×GPU 的 `--time` 上限 ≈17h。`--time=48h` 被拒（`AssocGrpGRESMinutes`），见坑 #11 |
| 2026-10-10 | **正式 RL 首跑（job 674976）卡 val_before_train 被取消**：`n_resp_per_prompt_val=30` × 60 题 = 1800 响应 × 16 轮工具循环，70+ 分钟没走完。修：`val_before_train=False` + `test_freq=50` + `n_val=8` |
| 2026-10-10 | **正式 RL 二跑（job 675119）卡 rollout 被取消**：`max_turns=16` + n=8 时，SFT-2ep 模型生成错工具名（`code_editor`/`code_executor`/`solving_quadratic`，配置里叫 `code_interpreter`）+ 坏 JSON，响应不收敛顶到 8192 token，首步 rollout 1h+。日志显示实际 ~2.5min/步、到 step 20 存了 ckpt（说明非死锁，是太慢）。修：回退冒烟实证的 `max_turns=8` + `n_resp_per_prompt=4`（见坑 #13） |
| 2026-10-10 | **正式 RL 三跑（job 675200，当前）✅ 正在学习**：n=4/max_turns=8/SP=1/TP=1，`--time=15h`（预算余量内）。~108s/步；133/271 步（~50%）时 `critic/score/mean` **-0.643→+0.247**、AIME2024 `best@4` **-0.355→+0.052（过零）**、AIME2025 `mean@4` **-0.625→-0.474**、num_turns 4.5→9.3（学到更长的工具调用链）。stale ckpt/tb 已 mv 归档防自动恢复冲突 |
| 2026-10-10 | **正式 RL 三跑（job 675200）✅ COMPLETED（9h08m，退出码 0，271/271 步）**：终态 `critic/score/mean` **-0.643→+0.230**、num_turns 4.47→10.05（模型学会用 ~10 轮工具调用链）；**AIME2024 `acc/best@4` 29.3%→50.6%、`acc/mean@4` 18.3%→37.5%**；**AIME2025 `acc/best@4` 26.3%→48.3%、`acc/mean@4` 16.7%→35.8%**。ckpt `global_step_{20..260,271}` 落盘（`latest_checkpointed_iteration=271`）。**结论：ReTool agentic RL 在 1.7B 上跑通且有效——AIME 最佳采样正确率近翻倍** |
| 2026-10-15 | **独立评测脚本 + 冻结协议 12 采样评测完成 ✅**：新增 4 脚本（`eval_qwen3_1.7b.sh` val-only 评测、`eval_smoke.slurm` 冒烟、`eval.slurm` 6 组循环、`merge_rl_hf.slurm` RL final 合并）。merge job **685931**（`global_step_271/actor` → `global_step_271-hf`，model.safetensors 4.06GB，1min）；冒烟 job **685932**（base+text，10min，退出码 0）；全套 job **686226 COMPLETED（42min，6 组全 OK）**。**结果（AIME 2024/2025 acc@12）**：base-tool 22.8/22.5、base-text 16.4/16.9、sft-tool 11.7/14.4、sft-text 5.0/8.9、**rl-tool mean 41.1/35.0、best 70.5/55.3**、rl-text 10.8/8.1；num_turns（tool）2.0→4.8→8.6。**三发现**：①RL 增益成立（base→rl-tool mean +18.3、best 44.4→70.5）②增益全由工具驱动（rl-text 10.8 反低于 base-text 16.4）③SFT 负迁移（sft-tool 11.7 < base-tool 22.8，`Answer:` vs `\boxed{}` 冲突）。产出讲稿 `presentation/presentation_2_progress_slides.md` |
