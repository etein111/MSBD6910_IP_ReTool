#!/bin/bash
# 监控 RL 冒烟 job 664052（SFT 冷启动 → RL 加载验证）。本地循环 + 每轮全新短 SSH，抗掉线。
# 冒烟约 11 分钟（base 冒烟 10:54），循环 80 轮 × 30s ≈ 40 分钟覆盖全程。
JOB=664052
OUT=/scratch/mscbdtspod2026/cyeah/retool/outputs/retool_dapo_smoke_$JOB.out
prev=""
for i in $(seq 1 80); do
  line=$(ssh -o ConnectTimeout=25 cyeah@superpod.ust.hk "source /etc/profile.d/modules.sh 2>/dev/null
module load slurm 2>/dev/null
st=\$(sacct -j $JOB --noheader -o State 2>/dev/null | head -1 | tr -d ' ')
if echo \"\$st\" | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEM|NODE_FAIL|BOOT_FAIL'; then
  echo \"TERMINAL:\$st\"
  echo '--- reward / 关键指标 ---'
  grep -oE '(critic/score|critic/rewards|actor/|perf|global_step|step).{0,120}' $OUT 2>/dev/null | tail -6
  if [ \"\$st\" = COMPLETED ]; then
    echo '--- ckpt 落盘 ---'
    ls -la /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-dapo-smoke/ 2>/dev/null | head -30
  fi
  exit 0
fi
grep -oE '(critic/score/[a-z]+|critic/rewards/[a-z]+).{0,60}' $OUT 2>/dev/null | tail -2
grep -nE 'global_step|Saved|saved|checkpoint|Traceback|Error|OOM|CUDA out of memory|assert|KeyError|No such file|RuntimeError|ValueError|Failed to load|load_checkpoint' $OUT 2>/dev/null | tail -4" 2>/dev/null)
  if [ -n "$line" ] && [ "$line" != "$prev" ]; then
    echo "$line"
    prev="$line"
  fi
  echo "$line" | grep -q 'TERMINAL' && break
  sleep 30
done
