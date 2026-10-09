#!/bin/bash
# 收尾修复 4：pip 降级 transformers 时残留了 4.55.4 的 models/aimv2/（官方 4.54.1 没有 aimv2）。
# 彻底卸载 + 清残留目录 + 重装干净 4.54.1。
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "=== site-packages: $SP ==="

echo "=== 1) 卸载 transformers ==="
pip uninstall -y transformers

echo "=== 2) 清掉可能残留的 aimv2 / 整个 transformers 目录 ==="
if [ -d "$SP/transformers" ]; then
    echo "发现残留 $SP/transformers，删除"
    rm -rf "$SP/transformers"
fi

echo "=== 3) 重装干净 transformers 4.54.1 ==="
pip install "transformers==4.54.1"

echo
echo "=== 4) 确认 aimv2 已消失 ==="
if python3 -c "import transformers.models.aimv2" 2>/dev/null; then
    echo "!! aimv2 仍在，清理失败"
    exit 1
else
    echo "✓ aimv2 已不存在"
fi
python3 -c "import transformers; print('transformers', transformers.__version__)"
python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)"

echo
echo "=== 5) 异步 rollout 完整依赖面（最终 gate）==="
python3 -c "from vllm.utils import get_tcp_uri, FlexibleArgumentParser; print('get_tcp_uri OK:', get_tcp_uri('10.0.0.1', 1234))"
python3 -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager OK')"
python3 -c "from vllm.v1.engine.async_llm import AsyncLLM; from vllm.v1.engine.core import EngineCoreProc; from vllm.v1.executor.abstract import Executor; print('V1 engine 类 OK')"
python3 -c "import verl.workers.rollout.vllm_rollout.vllm_async_server as vas; print('vllm_async_server import OK')"

echo
echo "=== DONE ==="
