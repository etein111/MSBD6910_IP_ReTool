#!/bin/bash
# 通用监控：本地循环 + 每轮全新短 SSH，抗掉线。用法：bash watch_job.sh <jobid> <outfile> [ckpt_dir]
# 冒烟约 11 分钟，循环 100 轮 × 25s ≈ 40 分钟覆盖全程。
JOB=${1:?jobid}
OUT=${2:?outfile}
CK=${3:-}
prev=""
for i in $(seq 1 100); do
  line=$(ssh -o ConnectTimeout=25 cyeah@superpod.ust.hk "source /etc/profile.d/modules.sh 2>/dev/null
module load slurm 2>/dev/null
st=\$(sacct -j $JOB --noheader -o State 2>/dev/null | head -1 | tr -d ' ')
if echo \"\$st\" | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEM|NODE_FAIL|BOOT_FAIL'; then
  echo \"TERMINAL:\$st\"
  echo '--- 关键指标 ---'
  grep -oE '(critic/score/[a-z]+|critic/rewards/[a-z]+|actor/entropy|perf/mfu).{0,70}' $OUT 2>/dev/null | tail -8
  if [ \"\$st\" = COMPLETED ] && [ -n \"$CK\" ]; then
    echo '--- ckpt 落盘 ---'; ls -la $CK 2>/dev/null | head -30
  fi
  exit 0
fi
grep -oE '(critic/score/[a-z]+|critic/rewards/[a-z]+).{0,60}' $OUT 2>/dev/null | tail -2
grep -nE 'global_step|Saved|checkpoint|Traceback|Error|OOM|CUDA out of memory|assert|KeyError|No such file|RuntimeError|ValueError|Failed to load|TypeError|vocab_file' $OUT 2>/dev/null | grep -viE 'set -x|model.path|actor_rollout_ref|trainer\.|algorithm\.|data\.' | tail -5" 2>/dev/null)
  if [ -n "$line" ] && [ "$line" != "$prev" ]; then
    echo "$line"
    prev="$line"
  fi
  echo "$line" | grep -q 'TERMINAL' && break
  sleep 25
done
