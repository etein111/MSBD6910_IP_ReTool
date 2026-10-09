#!/bin/bash
# 升级 vllm 0.8.5 -> 0.9.2，修复 verl 0.6.1 异步 rollout 缺 get_tcp_uri / CoreEngineProcManager 的问题。
#
# 为什么：verl 0.6.1 的 async rollout（vllm_async_server.py）针对 vllm 0.9.x 的 V1 引擎写的，
#   而 0.8.5 缺 get_tcp_uri、vllm.v1.engine.utils 模块。同步模式（mode=sync）又没有多轮工具循环
#   （AgentLoopManager 只在 mode==async 时创建），所以必须升级 vllm。
# 版本选择：vllm 0.9.x 全系锁 torch 2.7.0；0.9.2 是 0.9 系最新、最贴合 verl 0.6.1 的 API。
#   0.9.2 的 PyPI wheel 是 cu128 编译，所以 torch 也要 cu128。
#
# 运行（login node）：
#   nohup bash upgrade_vllm.sh > /scratch/mscbdtspod2026/cyeah/retool/outputs/upgrade_vllm.log 2>&1 &
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== $(date) 开始升级 ==="
echo "--- 当前版本 ---"
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || true
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null || true

echo
echo "=== 1/3 升级 torch 2.7.0 + cu128（torchaudio/torchvision 同步）==="
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.0+cu128" "torchaudio==2.7.0+cu128" "torchvision==0.22.0+cu128"

echo
echo "=== 2/3 升级 vllm 0.9.2 ==="
pip install "vllm==0.9.2"

echo
echo "=== 3/3 重装 flash-attn 2.8.3.post1（torch 2.7.0 + cu12，abiFALSE）==="
curl -L --retry 3 -o /tmp/flash_attn_torch27.whl \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install /tmp/flash_attn_torch27.whl

echo
echo "=== $(date) 升级完成，验证版本 ==="
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python3 -c "import verl; print('verl', verl.__version__)"

echo
echo "=== 验证 get_tcp_uri 原生存在 + 异步 rollout 能 import ==="
python3 -c "from vllm.utils import get_tcp_uri; print('get_tcp_uri OK:', get_tcp_uri('10.0.0.1', 1234))"
python3 -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager OK')"

echo
echo "=== DONE ==="
