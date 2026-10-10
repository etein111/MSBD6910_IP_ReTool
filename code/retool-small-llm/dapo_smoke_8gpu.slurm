#!/bin/bash
# 8 卡多卡冒烟：验证 verl 纯 FSDP 多卡（train_sp=1 / infer_tp=1，1.7B 不需要 SP/TP）在 SuperPOD 可用，5 步即停。
# 复用 run_qwen3_1.7b_dapo_smoke.sh（64 行 smoke 数据 + total_training_steps=5，脚本内本就是 sp=1/tp=1/offload=False），
# 只覆盖 n_gpus_per_node=8 + 独立实验名，从而验证「多卡能不能跑通」而不烧正式时长。
# 教训（job 672236 踩过）：照抄官方 7B 的 train_sp=4/infer_tp=4 会触发 ulysses all_to_all CUDA OOM，1.7B 无需 SP/TP。
#
# 提交：sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/dapo_smoke_8gpu.slurm
# 日志：tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_smoke_8gpu_<jobid>.out

#SBATCH --job-name=retool_smoke_8gpu
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_smoke_8gpu_%j.out

set -e

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== node ==="; hostname
echo "=== gpu  ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== cwd  ==="; pwd

cd /scratch/mscbdtspod2026/cyeah/retool/code/work

bash run_qwen3_1.7b_dapo_smoke.sh \
    trainer.n_gpus_per_node=8 \
    trainer.experiment_name=qwen3-1.7b-dapo-smoke-8gpu \
    trainer.default_local_dir=/scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo-smoke-8gpu
