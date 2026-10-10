#!/bin/bash
# ReTool RL 正式训练（DAPO/GRPO）—— 单节点 8×H800，Qwen3-1.7B + SFT 2ep 冷启动。
# 运行 run_qwen3_1.7b_dapo.sh；全量 DAPO-Math-17k-dedup（17,398 条），≈272 步/epoch。
#（normal 分区：单节点 8 GPU；账户 mscbdtspod2026 有 GPU 分钟预算 gres/gpu=9600 ≈ 160 GPU·h。
#  此前失败的/取消的冒烟+正式 run 已累计消耗约 1447 GPU·min，剩 ~8153 GPU·min ≈ 17h@8GPU，
#  故 --time 取 15h（8×15×60=7200 GPU·min），留 ~950 GPU·min 余量。18h(8640) 会超预算被拒。
#  实测 n=4/max_turns=8 配置 ~1.3min/步 × 272 步 ≈ 6h，15h 有充足余量确保跑完。）
#
# 提交（在 login node）：
#   sbatch /scratch/mscbdtspod2026/cyeah/retool/code/work/dapo.slurm
# 看日志：
#   tail -f /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_dapo_<jobid>.out

#SBATCH --job-name=retool_dapo
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_dapo_%j.out

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

bash run_qwen3_1.7b_dapo.sh
