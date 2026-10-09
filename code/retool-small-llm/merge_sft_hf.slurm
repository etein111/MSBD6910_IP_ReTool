#!/bin/bash
# 把 SFT 的 verl FSDP checkpoint（model_world_size_1_rank_0.pt）合并成 HF 格式（safetensors）。
# 原因：verl 0.6.1 的 RL `_build_model_optimizer` 用 AutoModelForCausalLM.from_pretrained 从 model.path 根目录
# 加载 HF 权重，而 SFT 默认 save_contents=["model","optimizer","extra"] 不含 hf_model，只有 FSDP 权重。
# 用 verl 自带 model_merger 转换，产出 global_step_124-hf/，RL 的 model_path 指向它。
#（2-epoch 温和版 SFT 的最终 ckpt 是 global_step_124；6-epoch 旧版是 372。）
#
# 提交（login node）：
#   sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/merge_sft_hf.slurm

#SBATCH --job-name=retool_merge
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_merge_%j.out

set -e

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

cd /scratch/mscbdtspod2026/cyeah/retool/code/work

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft-2ep/global_step_124 \
    --target_dir /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft-2ep/global_step_124-hf

echo "=== merge 完成，输出目录 ==="
ls -la /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft-2ep/global_step_124-hf/
