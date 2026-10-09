#!/bin/bash
# 收尾修复 3：transformers 降到 <4.55.0（aimv2 是 4.55.0 才进的，4.54.x 没有，vllm 0.9.2 注册才不撞）
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== $(date) 降 transformers 到 >=4.51.1,<4.55.0 ==="
pip install "transformers>=4.51.1,<4.55.0"

echo
echo "=== 最终版本 ==="
python3 -c "import transformers; print('transformers', transformers.__version__)"
python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)"

echo
echo "=== 异步 rollout 完整依赖面（最终 gate）==="
python3 -c "from vllm.utils import get_tcp_uri, FlexibleArgumentParser; print('get_tcp_uri OK:', get_tcp_uri('10.0.0.1', 1234))"
python3 -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager OK')"
python3 -c "from vllm.v1.engine.async_llm import AsyncLLM; from vllm.v1.engine.core import EngineCoreProc; from vllm.v1.executor.abstract import Executor; print('V1 engine 类 OK')"
python3 -c "import verl.workers.rollout.vllm_rollout.vllm_async_server as vas; print('vllm_async_server import OK')"

echo
echo "=== DONE ==="
