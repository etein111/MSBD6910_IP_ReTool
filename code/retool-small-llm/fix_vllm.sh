#!/bin/bash
# 收尾修复 upgrade_vllm.sh 的两个遗留问题：
#   1. flash-attn wheel 下载好了但存成了非标准文件名（下划线），pip 不认 -> 改名重装
#   2. vllm 0.9.2 把 numpy 升到 2.2.6，但 verl 0.6.1 要求 numpy<2.0.0 -> 降回 1.26.x
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

WHEEL=/tmp/flash_attn_torch27.whl
WHEEL_OK=/tmp/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=== $(date) 收尾开始 ==="

# 1) flash-attn：改名（若已下好）或重新下载，然后装
if [ ! -f "$WHEEL_OK" ]; then
    if [ -f "$WHEEL" ]; then
        echo "已下载，改名 $WHEEL -> $WHEEL_OK"
        mv "$WHEEL" "$WHEEL_OK"
    else
        echo "重新下载 flash-attn wheel"
        curl -L --retry 3 -o "$WHEEL_OK" \
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    fi
fi
echo "=== 安装 flash-attn ==="
pip install "$WHEEL_OK"

# 2) numpy 降回 1.26.x（verl 0.6.1 要 numpy<2.0.0）
echo "=== 降级 numpy <2.0.0 ==="
pip install "numpy<2.0.0"

echo "=== $(date) 收尾完成，最终验证 ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import verl; print('verl', verl.__version__)"

echo "=== 验证 async rollout 依赖面 ==="
python3 -c "from vllm.utils import get_tcp_uri, FlexibleArgumentParser; print('get_tcp_uri OK:', get_tcp_uri('10.0.0.1', 1234))"
python3 -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager OK')"
python3 -c "from vllm.v1.engine.async_llm import AsyncLLM; from vllm.v1.engine.core import EngineCoreProc; from vllm.v1.executor.abstract import Executor; print('V1 engine 类 OK')"

echo
echo "=== 完整 import verl 异步 rollout 模块（最终 gate）==="
python3 -c "import verl.workers.rollout.vllm_rollout.vllm_async_server as vas; print('✓ vllm_async_server import 成功')"

echo
echo "=== DONE ==="
