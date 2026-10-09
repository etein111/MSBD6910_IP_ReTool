#!/bin/bash
# 收尾修复 6：vllm 0.9.2 -> 0.10.0。
# 根因：verl 0.6.1 异步 rollout 调 AsyncLLM.wait_for_requests_to_drain()，
#       该方法在 vllm 0.10.0 才加入（0.9.x 没有，报 AttributeError）。
# 且 vllm 0.10.0 是 CUDA 无关的 abi3 wheel，锁 torch==2.7.1（plain，默认 cu126），
# 要求 transformers>=4.53.2（当前已是 4.53.2，不用动），新增硬依赖 xformers==0.0.31。
set -euo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate retool
source /scratch/mscbdtspod2026/cyeah/retool/env.sh

echo "=== [1/4] 升 torch 2.7.0 -> 2.7.1（vllm 0.10.0 锁 torch==2.7.1，不带 +cu 后缀）==="
pip install "torch==2.7.1" "torchaudio==2.7.1" "torchvision==0.22.1"

echo
echo "=== [2/4] 升 vllm 0.9.2 -> 0.10.0（自动装 xformers==0.0.31）==="
pip install "vllm==0.10.0"

echo
echo "=== [3/4] numpy 锁回 <2.0（vllm 0.10.0 不锁 numpy，会拉到 2.x）==="
pip install "numpy<2.0.0"

echo
echo "=== [4/4] transformers 锁回 4.53.2（防 pip 顺带升到 4.54+ 引入 aimv2）==="
pip install "transformers==4.53.2"

echo
echo "=== 版本快照 ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import transformers, numpy, tokenizers, verl; print('transformers', transformers.__version__, '| numpy', numpy.__version__, '| tokenizers', tokenizers.__version__, '| verl', verl.__version__)"
python3 -c "import xformers; print('xformers', xformers.__version__)"
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null || echo "!! flash_attn import 失败"

echo
echo "=== 异步 rollout 完整 gate（含 wait_for_requests_to_drain）==="
python3 -c "from vllm.utils import get_tcp_uri; print('get_tcp_uri OK:', get_tcp_uri('10.0.0.1', 1234))"
python3 -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager OK')"
python3 -c "from vllm.v1.engine.async_llm import AsyncLLM; assert hasattr(AsyncLLM, 'wait_for_requests_to_drain'), 'wait_for_requests_to_drain 缺失!'; print('AsyncLLM.wait_for_requests_to_drain OK')"
python3 -c "import vllm; print('vllm import（含 aimv2 注册）OK')"
python3 -c "import verl.workers.rollout.vllm_rollout.vllm_async_server as vas; print('vllm_async_server import OK')"

echo
echo "=== DONE ==="
