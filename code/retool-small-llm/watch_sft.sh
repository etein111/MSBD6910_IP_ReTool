#!/bin/bash
# 监控 SFT job 663756 直到终态；每轮全新短 SSH，抗掉线；只在新内容出现时输出。
# 训练约 87 分钟（372 step × ~14s），循环 200 轮 × 45s ≈ 150 分钟覆盖全程。
JOB=663756
prev=""
for i in $(seq 1 200); do
  line=$(ssh -o ConnectTimeout=25 cyeah@superpod.ust.hk 'source /etc/profile.d/modules.sh 2>/dev/null
module load slurm 2>/dev/null
st=$(sacct -j 663756 --noheader -o State 2>/dev/null | head -1 | tr -d " ")
if echo "$st" | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEM|NODE_FAIL|BOOT_FAIL"; then
  echo "TERMINAL:$st"
  if [ "$st" = "COMPLETED" ]; then echo "CKPT:"; ls -la /scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft/ 2>/dev/null | head -30; fi
  exit 0
fi
grep -oE "Epoch [0-9]/6: *[0-9]+%\|[^]]*\| [0-9]+/62[^]]*\]" /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_sft_663756.out 2>/dev/null | tail -1
grep -nE "Saved|saved|checkpoint|global_step|Traceback|Error|OOM|CUDA out of memory|assert|KeyError|No such file|sequence_length" /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_sft_663756.out 2>/dev/null | tail -4' 2>/dev/null)
  if [ -n "$line" ] && [ "$line" != "$prev" ]; then
    echo "$line"
    prev="$line"
  fi
  echo "$line" | grep -q 'TERMINAL' && break
  sleep 45
done
