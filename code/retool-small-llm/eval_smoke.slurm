#!/bin/bash
# 评测脚本冒烟：先跑一个最小配置（base 模型 + text-only）验证 eval_qwen3_1.7b.sh 端到端可用，
# 再上全套 6 组（3 checkpoint × 2 模式）。这样能最早暴露「base HF 直接加载」与「multi_turn 关闭」
# 两个新路径的问题，不浪费预算在 6 组全跑上。
#
# 提交：sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/eval_smoke.slurm
# 日志：tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_eval_smoke_%j.out

#SBATCH --job-name=retool_eval_smoke
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_eval_smoke_%j.out

set -e

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

cd /scratch/mscbdtspod2026/cyeah/retool/code/work

EVAL_MODEL_PATH=/scratch/mscbdtspod2026/cyeah/retool/data/model/Qwen3-1.7B \
EVAL_MODE=text \
EVAL_EXP=qwen3-1.7b-eval-smoke-base-text \
    bash eval_qwen3_1.7b.sh

echo "=== 冒烟评测完成，tensorboard 目录 ==="
ls -la /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-eval-smoke-base-text/
