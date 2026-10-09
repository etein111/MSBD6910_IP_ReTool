#!/bin/bash
# ReTool SFT 冷启动：Qwen3-1.7B-Instruct 多轮工具调用，单卡 2 epoch（温和版）。
# 产出 ckpt 作为 RL 的起点（对齐官方：官方用 Instruct 做 SFT 冷启动）。
#
# 2026-10-09 由 6 epoch 改成 2 epoch：6 epoch 对 1.7B 灾难性遗忘（把 base 的
# 「长推理→boxed」换成「调工具但代码写不对」，RL 首轮 19%→0%）。lr 本就是 1e-5
# （verl 默认，之前误读成 1e-3 是 fsdp_sft_trainer.py:522 硬编码日志标签的锅），
# 此处显式写 optim.lr=1e-5 固定意图。
#
# 基线来自官方 recipe 的 run_qwen2_7b_sft.sh，改动：
#   - 模型用 Qwen/Qwen3-1.7B（Qwen3 不带后缀即 Instruct 版；官方 SFT 用 Qwen2.5-7B-Instruct）
#     注意：Qwen3 命名与 Qwen2.5 相反 —— Qwen3-1.7B = Instruct，Qwen3-1.7B-Base = base
#   - 单卡：nproc_per_node=1、ulysses_sequence_parallel_size=1、trainer.n_gpus_per_node=1
#   - model.fsdp_config.model_dtype=bf16：verl FSDP SFT 默认 fp32，单卡下 logits/激活
#     显存翻倍直接 OOM（复现：CUDA OOM 反向分配 18.54 GiB）；对齐官方 dapo 脚本用 bf16
#   - micro_batch_size_per_gpu 4 -> 2：logits 显存 = micro_batch × seq × vocab，官方 7B 用
#     8 卡 + ulysses SP=4 把 logits 切成 1/8，单卡 SP=1 全扛；降到 2 才放得下 H800 80GB
#   - data.truncation=left：数据集有 6 个样本 >8192 token（最长 14824），默认 truncation=error
#     会崩；left 截断保留末尾 answer（label），砍 prompt 开头
#   - max_length 16384 -> 8192（对齐 RL，省显存）
#   - logger 只用 console（不依赖 wandb 登录）
#   - 路径走 PROJECT_ROOT/data/dataset_clean/ 与 checkpoints/
#
# 用法：在有 `recipe/` 包的目录（code/work/）下运行
#   cd /scratch/mscbdtspod2026/cyeah/retool/code/work
#   bash run_qwen3_1.7b_sft.sh

set -x

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/mscbdtspod2026/cyeah/retool}
DATA_ROOT=$PROJECT_ROOT/data
CKPT_ROOT=$PROJECT_ROOT/checkpoints

TRAIN_DATA=$DATA_ROOT/dataset_clean/JoeYing/ReTool-SFT/train.parquet
EVAL_DATA=$TRAIN_DATA   # 与官方一致：SFT 用同一份做 val（不做真正 hold-out）
MODEL_PATH=$DATA_ROOT/model/Qwen3-1.7B

project_name=retool
experiment_name=qwen3-1.7b-instruct-sft-2ep
SAVE_PATH=$CKPT_ROOT/$experiment_name

nnodes=1
nproc_per_node=1

torchrun --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --master-addr=127.0.0.1 \
    --master-port=29500 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=8192 \
    data.truncation=left \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=1e-5 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=2 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true "$@"
