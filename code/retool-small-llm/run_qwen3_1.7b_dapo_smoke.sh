#!/bin/bash
# ReTool RL smoke：验证 SFT 冷启动 ckpt 能被 RL（GRPO/DAPO）正确加载，并跑 5 步看 reward 信号。
#
# 与 run_qwen3_1.7b_smoke.sh 的唯一区别：model_path 从 base 换成 SFT 冷启动产物
#   checkpoints/qwen3-1.7b-instruct-sft/global_step_372
# 这一步是 SFT→RL 衔接的最大风险点（verl 能否从 FSDP 格式 ckpt 加载权重）。
# 预期：SFT 后模型已会调工具 + 输出 \boxed{}，初始 reward 应显著高于 base 冒烟（mean −0.62）。
#
# 用法：cd /scratch/mscbdtspod2026/cyeah/retool/code/work && bash run_qwen3_1.7b_dapo_smoke.sh

set -x

export VLLM_USE_V1=1

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/mscbdtspod2026/cyeah/retool}
DATA_ROOT=$PROJECT_ROOT/data
CKPT_ROOT=$PROJECT_ROOT/checkpoints

dapo_math_17k=$DATA_ROOT/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-smoke
aime_2024=$DATA_ROOT/dataset_clean/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset_clean/yentinglin/aime_2025
model_path=$CKPT_ROOT/qwen3-1.7b-instruct-sft/global_step_372

train_files="['$dapo_math_17k']"
test_files="['$aime_2025', '$aime_2024']"

tool_config_path=recipe/retool/local_tool_config.yaml

project_name=retool
experiment_name=qwen3-1.7b-dapo-smoke
default_local_dir=$CKPT_ROOT/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=2048
max_response_length=8192
actor_lr=1e-6

# ================= 冒烟规模（单卡） =================
train_batch_size=8
ppo_mini_batch_size=8
n_resp_per_prompt=4
n_resp_per_prompt_val=4
total_training_steps=5

infer_tp=1
train_sp=1
offload=False

actor_max_token_len_per_gpu=$(( max_prompt_length + max_response_length ))
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=-1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=1 "$@"
