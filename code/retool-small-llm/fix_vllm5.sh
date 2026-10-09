#!/bin/bash
# 收尾修复 5：真正正确的 transformers 版本是 4.53.2（aimv2 在 4.54.0 才加入，4.53.x 没有）。
# 之前 4.54.1 还撞 aimv2，是因为 4.54.1 本来就带 aimv2（我之前用错 --no-binary :none: 误判了）。
# 干净重装到 4.53.2。
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== 1) 卸载 transformers ==="
pip uninstall -y transformers

echo "=== 2) 清残留目录（保险）==="
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
[ -d "$SP/transformers" ] && rm -rf "$SP/transformers" && echo "清掉 $SP/transformers" || echo "无残留目录"

echo "=== 3) 重装 transformers 4.53.2 ==="
pip install "transformers==4.53.2"

echo
echo "=== 4) 确认 aimv2 已消失 ==="
if python3 -c "import transformers.models.aimv2" 2>/dev/null; then
    echo "!! aimv2 仍在"
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
