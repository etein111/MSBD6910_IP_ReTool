#!/bin/bash
# ReTool RL 正式训练：DAPO/GRPO on Qwen3-1.7B（SFT 2ep 冷启动），单节点 8×H800。
#
# 与 run_qwen3_1.7b_dapo_smoke.sh 的区别：
#   1) 数据：DAPO-Math-17k-smoke(64 行) → DAPO-Math-17k-dedup(全量 17,398 行)
#   2) 规模：单卡 → 8 卡（纯 FSDP + 8 份 vllm(TP=1)；1.7B 无需 SP/TP，见下方拓扑说明）
#   3) 时长：total_training_steps=5 → total_epochs=1（≈ ceil(17398/64) ≈ 272 步）
#   4) max_turns 保持 8（冒烟实证）、n_resp_per_prompt=4、val_before_train 关、save_freq 5→20、test_freq 50
#   （n=4 而非官方 16、max_turns=8 而非 16：8 卡 × 18h 受账户 GPU 分钟预算 gres/gpu=9600 限制，
#     ——多轮工具 rollout 极慢。job 675119 实测 max_turns=16 时，SFT-2ep 模型反复生成
#     错误的工具名('code_editor'/'solving_quadratic'，配置里叫 code_interpreter)与坏 JSON，
#     响应不收敛、一路顶到 8192 token，首步 rollout 1h+ 未走完。冒烟(672270)用 max_turns=8
#     时模型 ~3.3 轮即收敛、响应均值 1778 token、41s/步。故回到 8 轮 + n=4，优先把预算花在
#     「能出奖励曲线」的训练上）
#
# 实验变量（GRPO vs DAPO，项目承诺「只动 RL update」）：
#   DAPO  → clip_ratio_high=0.28（本文件默认）
#   GRPO  → 把 clip_ratio_high 改成 0.0，并同步改 experiment_name（如 -grpo）
#
# 用法：cd /scratch/mscbdtspod2026/cyeah/retool/code/work && bash run_qwen3_1.7b_dapo.sh

set -x
export VLLM_USE_V1=1

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/mscbdtspod2026/cyeah/retool}
DATA_ROOT=$PROJECT_ROOT/data
CKPT_ROOT=$PROJECT_ROOT/checkpoints

dapo_math_17k=$DATA_ROOT/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-dedup
aime_2024=$DATA_ROOT/dataset_clean/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset_clean/yentinglin/aime_2025
model_path=$CKPT_ROOT/qwen3-1.7b-instruct-sft-2ep/global_step_124-hf

train_files="['$dapo_math_17k']"
test_files="['$aime_2025', '$aime_2024']"

tool_config_path=recipe/retool/local_tool_config.yaml

project_name=retool
experiment_name=qwen3-1.7b-dapo
default_local_dir=$CKPT_ROOT/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28   # DAPO；GRPO 对照改成 0.0

max_turns=8
max_prompt_length=2048
max_response_length=8192
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=4
n_resp_per_prompt_val=4

# ================= 8 卡拓扑（纯 FSDP，1.7B 不需要 SP/TP） =================
# ulysses SP 的 all_to_all 与 vllm TP 都是为 7B+ 大模型/长序列设计的；对 1.7B 反而
# 触发 all_to_all CUDA OOM（8 卡冒烟 job 672236 实证）。1.7B 单卡放得下，8 卡就做
# 纯 FSDP（数据并行，训练吞吐×8）+ 8 份 vllm(TP=1)（采样吞吐×8）。
infer_tp=1      # vllm tensor parallel
train_sp=1      # 不用 ulysses sequence parallel
offload=False   # 1.7B + optimizer 显存放得下，无需 offload

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=50 \
    trainer.total_epochs=1 "$@"
