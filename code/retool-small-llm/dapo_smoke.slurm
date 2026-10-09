#!/bin/bash
# ReTool RL 冒烟：验证 SFT 冷启动 ckpt（2ep 版 global_step_124）能被 RL（GRPO/DAPO）正确加载，单卡 5 步。
# 与 smoke.slurm 的唯一区别：运行 run_qwen3_1.7b_dapo_smoke.sh（model_path 指向 SFT ckpt）。
#
# 提交（在 login node）：
#   sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/dapo_smoke.slurm
# 看日志：
#   tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_dapo_smoke_<jobid>.out

#SBATCH --job-name=retool_dapo_smoke
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_dapo_smoke_%j.out

set -e

# 环境（login node 无 GPU，这里才在 GPU 节点上跑）
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== node ==="; hostname
echo "=== gpu  ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== cwd  ==="; pwd

# 必须在包含 recipe/ 包的目录下运行（class_name 走 importlib 动态导入）
cd /scratch/mscbdtspod2026/cyeah/retool/code/work

bash run_qwen3_1.7b_dapo_smoke.sh
