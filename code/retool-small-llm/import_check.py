"""核心导入体检 —— main_ppo 启动前最后一次 import 级验证。

单独成脚本是为了绕开 bash 嵌套引号问题（之前 inline -c 里带 ✓ 和双引号被吃掉）。
"""

import sys


def main():
    results = []
    print("=" * 60)

    # 1. vllm（带 opentelemetry 冲突时能否 import）
    try:
        import vllm  # noqa: F401

        print(f"  ✓ vllm {vllm.__version__}")
        results.append(True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ vllm import 失败: {type(exc).__name__}: {exc}")
        results.append(False)

    # 2. verl
    try:
        import verl  # noqa: F401

        print(f"  ✓ verl {verl.__version__}")
        results.append(True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ verl import 失败: {type(exc).__name__}: {exc}")
        results.append(False)

    # 3. verl 数据集模块（之前 AutoModelForVision2Seq 报错处）
    try:
        from verl.utils.dataset import RLHFDataset  # noqa: F401

        print("  ✓ verl.utils.dataset.RLHFDataset")
        results.append(True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ RLHFDataset 失败: {type(exc).__name__}: {exc}")
        results.append(False)

    # 4. 自研沙箱 Tool（主入口之前已验过 19/19，这里只确认还能 import）
    try:
        from recipe.retool.local_code_tool import LocalCodeTool  # noqa: F401

        print("  ✓ recipe.retool.local_code_tool.LocalCodeTool")
        results.append(True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ LocalCodeTool 失败: {type(exc).__name__}: {exc}")
        results.append(False)

    # 5. opentelemetry 版本 vs vllm 约束（只报告，不判死）
    try:
        from importlib.metadata import version

        for pkg in ["opentelemetry-api", "opentelemetry-sdk"]:
            try:
                print(f"  · {pkg} = {version(pkg)}")
            except Exception:  # noqa: BLE001
                print(f"  · {pkg} = <未安装>")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 读 opentelemetry 版本失败: {exc}")

    print("=" * 60)
    ok = all(results)
    print("结论：" + ("全绿，可提交冒烟作业" if ok else "有失败项，先修"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
