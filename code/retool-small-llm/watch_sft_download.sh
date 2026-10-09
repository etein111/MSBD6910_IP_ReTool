#!/bin/bash
# 监控 ModelScope Instruct 下载完成；每轮全新短 SSH，抗掉线。
for i in $(seq 1 40); do
  line=$(ssh -o ConnectTimeout=25 cyeah@superpod.ust.hk 'L=/scratch/mscbdtspod2026/cyeah/retool/outputs/download_instruct_ms.log
D=/scratch/mscbdtspod2026/cyeah/retool/data/model/Qwen3-1.7B-Instruct
if grep -q "DONE" $L 2>/dev/null; then echo "DOWNLOAD_DONE"; ls $D 2>/dev/null | tr "\n" " "; echo; exit 0; fi
if grep -qiE "error|traceback|exception|failed|FileNotFound" $L 2>/dev/null; then echo "DOWNLOAD_ERROR:"; tail -8 $L; exit 0; fi
n=$(ls $D 2>/dev/null | wc -l); echo "downloading... files=$n"' 2>/dev/null)
  [ -n "$line" ] && echo "$line"
  echo "$line" | grep -qE "DOWNLOAD_DONE|DOWNLOAD_ERROR" && break
  sleep 40
done
