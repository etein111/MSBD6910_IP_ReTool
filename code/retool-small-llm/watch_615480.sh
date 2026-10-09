#!/bin/bash
# 轮询 smoke job 615480 直到终态；每轮一次全新短 SSH，抗长连接被断。
# 只在出现新内容时输出（去重），终态时退出。
JOB=615480
prev=""
for i in $(seq 1 40); do
  line=$(ssh -o ConnectTimeout=25 cyeah@superpod.ust.hk 'source /etc/profile.d/modules.sh 2>/dev/null
module load slurm 2>/dev/null
st=$(sacct -j 615480 --noheader -o State 2>/dev/null | head -1 | tr -d " ")
if echo "$st" | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEM|NODE_FAIL|BOOT_FAIL"; then echo "TERMINAL:$st"; exit 0; fi
grep -E "Training Progress|reward|checkpoint|saved|num_turns|Traceback|KeyError|RayActorError|CUDA out of memory|OOM|dataset|sample" /scratch/mscbdtspod2026/cyeah/retool/outputs/retool_smoke_615480.out 2>/dev/null | tail -6' 2>/dev/null)
  if [ -n "$line" ] && [ "$line" != "$prev" ]; then
    echo "$line"
    prev="$line"
  fi
  echo "$line" | grep -q 'TERMINAL' && break
  sleep 45
done
