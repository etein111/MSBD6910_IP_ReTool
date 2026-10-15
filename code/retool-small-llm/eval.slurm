#!/bin/bash
# ReTool 全套独立评测：3 个 checkpoint × 2 模式（tool/text）在 AIME 2024/2025 上按 12 采样协议出数。
# 用循环顺序跑 6 组，单组失败不中断（记录 FAILED），最后打印汇总。
#
# 依赖：RL final 已 merge 成 global_step_271-hf（merge_rl_hf.slurm）；冒烟 eval_smoke.slurm 已通过。
#
# 提交：sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/eval.slurm
# 日志：tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_eval_%j.out

#SBATCH --job-name=retool_eval
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_eval_%j.out

set -e

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

cd /scratch/mscbdtspod2026/cyeah/retool/code/work

BASE=/scratch/mscbdtspod2026/cyeah/retool/data/model/Qwen3-1.7B
SFT=/scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft-2ep/global_step_124-hf
RL=/scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo/global_step_271-hf

# 6 组：exp|model|mode（按序跑，tool 慢、text 快）
CONFIGS=(
  "qwen3-1.7b-eval-base-tool|$BASE|tool"
  "qwen3-1.7b-eval-base-text|$BASE|text"
  "qwen3-1.7b-eval-sft-tool|$SFT|tool"
  "qwen3-1.7b-eval-sft-text|$SFT|text"
  "qwen3-1.7b-eval-rl-tool|$RL|tool"
  "qwen3-1.7b-eval-rl-text|$RL|text"
)

declare -a RESULTS

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r EXP MODEL MODE <<< "$cfg"
  echo ""
  echo "========== 评测：$EXP (mode=$MODE) =========="
  if EVAL_MODEL_PATH="$MODEL" EVAL_MODE="$MODE" EVAL_EXP="$EXP" bash eval_qwen3_1.7b.sh; then
    echo "[$EXP] OK"
    RESULTS+=("[$EXP] OK")
  else
    echo "[$EXP] FAILED (exit=$?)"
    RESULTS+=("[$EXP] FAILED")
  fi
done

echo ""
echo "========== 汇总 =========="
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "===== 各评测 tensorboard 目录 ====="
ls -d /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-eval-*/ 2>/dev/null
