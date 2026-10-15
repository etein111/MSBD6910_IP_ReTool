#!/bin/bash
# 把 RL 正式训练（job 675200）的最终 checkpoint global_step_271（8-rank FSDP 分片）合并成 HF 格式。
# 注意：RL 的 verl 0.6.1 checkpoint 结构是 global_step_271/actor/（含 model_world_size_8_rank_*.pt +
# huggingface/ + fsdp_config.json），所以 --local_dir 要指向 .../global_step_271/actor（不是根目录）。
# SFT 那次 checkpoint 在根目录（无 actor/ 子目录），故 --local_dir 指向根目录；两者布局不同。
# 产出 global_step_271-hf/（model.safetensors），供独立评测脚本当 model_path 加载。
#
# 提交（login node）：
#   sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/merge_rl_hf.slurm

#SBATCH --job-name=retool_merge_rl
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_merge_rl_%j.out

set -e

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

cd /scratch/mscbdtspod2026/cyeah/retool/code/work

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo/global_step_271/actor \
    --target_dir /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo/global_step_271-hf

echo "=== merge 完成，输出目录 ==="
ls -la /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo/global_step_271-hf/
