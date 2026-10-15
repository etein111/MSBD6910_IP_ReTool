#!/bin/bash
# ReTool 独立评测（val-only）：在冻结协议下对任意 HF checkpoint 跑 AIME 2024/2025 评测。
#
# 与训练脚本 run_qwen3_1.7b_dapo.sh 同源（同一 vllm rollout / 沙箱 / 判分 / 数据），只改：
#   - model.path        ← EVAL_MODEL_PATH（base / SFT-2ep-hf / RL-final-hf 任一 HF 目录）
#   - multi_turn        ← EVAL_MODE=tool|text（tool=hermes 工具循环；text=单轮纯文本）
#   - val 采样数         ← n_resp_per_prompt_val=12（接口契约 12 采样）
#   - 只跑 val 不训练     ← val_before_train=True + total_training_steps=1
#
# 用法（在 work/ 目录）：
#   EVAL_MODEL_PATH=<hf-dir> EVAL_MODE=tool EVAL_EXP=qwen3-1.7b-eval-base-tool \
#       bash eval_qwen3_1.7b.sh
#
# 产出：tensorboard 目录 checkpoints/<EVAL_EXP>/，含
#   val-core/aime_2024/acc/best@12|mean@12、val-aux/.../score/mean@12、num_turns、response_length 等

set -x
export VLLM_USE_V1=1

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/mscbdtspod2026/cyeah/retool}
DATA_ROOT=$PROJECT_ROOT/data
CKPT_ROOT=$PROJECT_ROOT/checkpoints

# —— 必填参数 ——
model_path=${EVAL_MODEL_PATH:?必须设置 EVAL_MODEL_PATH（HF 权重目录）}
eval_mode=${EVAL_MODE:-tool}
experiment_name=${EVAL_EXP:?必须设置 EVAL_EXP（独立实验名，避免 checkpoint 冲突）}

dapo_smoke=$DATA_ROOT/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-smoke
aime_2024=$DATA_ROOT/dataset_clean/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset_clean/yentinglin/aime_2025

train_files="['$dapo_smoke']"          # 只当占位，不真正训练（1 步即停）
test_files="['$aime_2025', '$aime_2024']"

project_name=retool
default_local_dir=$CKPT_ROOT/$experiment_name

# ================= algorithm（与训练一致，仅评测用不到训练步）=================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
actor_lr=1e-6

max_turns=8
max_prompt_length=2048
max_response_length=8192

train_batch_size=8
ppo_mini_batch_size=8
n_resp_per_prompt=4          # 训练采样数（只跑 1 步，取 4 与冒烟一致，避免 GRPO n=1 边界）
n_resp_per_prompt_val=12     # ★ 评测采样数 = 12（接口契约）

infer_tp=1
train_sp=1
offload=False

actor_max_token_len_per_gpu=$(( max_prompt_length + max_response_length ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

# ================= 多轮工具 / 纯文本 =================
if [ "$eval_mode" = "tool" ]; then
    MULTI_TURN_ARGS=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns
        actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/retool/local_tool_config.yaml
        actor_rollout_ref.rollout.multi_turn.format=hermes
    )
else
    MULTI_TURN_ARGS=(actor_rollout_ref.rollout.multi_turn.enable=False)
fi

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
    "${MULTI_TURN_ARGS[@]}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=1000000 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 "$@"
