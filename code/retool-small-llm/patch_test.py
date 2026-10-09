"""验证：verl 0.6.1 的异步 rollout 缺的 `get_tcp_uri`，是不是补一个别名就能过 import。

背景
====
verl 0.6.1 的 `vllm_async_server.py` 第 38 行
    from vllm.utils import FlexibleArgumentParser, get_tcp_uri
在 vllm 0.8.5 上 ImportError：`get_tcp_uri` 是 vllm 0.9.0 才有的名字，
但它的实现与 0.8.5 已有的 `get_distributed_init_method` 完全相同。

本脚本：
  1. 确认 0.8.5 缺 get_tcp_uri、但有 get_distributed_init_method
  2. 打别名后，尝试 import 整个 vllm_async_server 模块，看是否还有别的缺失符号
"""

import sys


def main() -> int:
    import vllm.utils as vu

    print(f"get_tcp_uri 存在？ {hasattr(vu, 'get_tcp_uri')}")
    print(f"get_distributed_init_method 存在？ {hasattr(vu, 'get_distributed_init_method')}")

    if hasattr(vu, "get_distributed_init_method") and not hasattr(vu, "get_tcp_uri"):
        vu.get_tcp_uri = vu.get_distributed_init_method
        print(f"补别名后 get_tcp_uri('10.0.0.1', 1234) = {vu.get_tcp_uri('10.0.0.1', 1234)!r}")

    print()
    print("尝试 import verl 的异步 rollout 模块……")
    try:
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vas  # noqa: F401

        print("✓ vllm_async_server import 成功")
        print(f"  是否含 vLLMReplica：{hasattr(vas, 'vLLMReplica')}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"✗ import 仍失败：{type(exc).__name__}: {exc}")
        # 打印完整 traceback 方便定位下一个缺失符号
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
