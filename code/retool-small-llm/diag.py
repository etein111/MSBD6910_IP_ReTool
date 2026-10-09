"""版本兼容性 + parquet 加载方式诊断。

两个待查问题：
  A. transformers 版本与 verl 0.6.1 不兼容（AutoModelForVision2Seq 导入失败）
  B. datasets.load_dataset(<parquet 文件路径>) 报找不到文件，但 os.path.exists 为真
"""

import os
import sys
from importlib.metadata import version, requires

DAPO = (
    "/scratch/mscbdtspod2026/cyeah/retool/data/dataset_clean/"
    "BytedTsinghua-SIA/DAPO-Math-17k/dapo-math-17k.parquet"
)

print("=" * 70)
print("A. 版本")
print("=" * 70)
for pkg in [
    "torch", "vllm", "verl", "transformers", "datasets", "tokenizers",
    "huggingface-hub", "accelerate", "numpy", "pyarrow",
]:
    try:
        print(f"  {pkg:20s} {version(pkg)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {pkg:20s} <未安装/查询失败: {exc}>")

print()
print("verl 0.6.1 对关键包的版本约束（来自 METADATA）：")
try:
    reqs = requires("verl") or []
    for r in reqs:
        low = r.lower()
        if any(k in low for k in ["transformers", "datasets", "vllm", "torch", "numpy"]):
            print(f"  {r}")
except Exception as exc:  # noqa: BLE001
    print(f"  <读取失败: {exc}>")

print()
print("=" * 70)
print("B. parquet 加载")
print("=" * 70)
print(f"os.path.exists      : {os.path.exists(DAPO)}")
print(f"os.path.islink      : {os.path.islink(DAPO)}")
print(f"os.path.realpath    : {os.path.realpath(DAPO)}")
print(f"realpath exists     : {os.path.exists(os.path.realpath(DAPO))}")
try:
    st = os.stat(DAPO)
    print(f"size                : {st.st_size:,} bytes")
except Exception as exc:  # noqa: BLE001
    print(f"stat 失败            : {exc}")

print()
print("--- 方式 1：官方 retool.py 的写法 datasets.load_dataset(<file>) ---")
try:
    import datasets

    ds = datasets.load_dataset(DAPO)["train"]
    print(f"  ✓ 成功，行数={len(ds):,} 列={ds.column_names}")
except Exception as exc:  # noqa: BLE001
    print(f"  ✗ {type(exc).__name__}: {str(exc)[:300]}")

print()
print("--- 方式 2：datasets.load_dataset('parquet', data_files=<file>) ---")
try:
    ds = datasets.load_dataset("parquet", data_files=DAPO)["train"]
    print(f"  ✓ 成功，行数={len(ds):,} 列={ds.column_names}")
    print(f"  首行 prompt 类型: {type(ds[0].get('prompt')).__name__}")
except Exception as exc:  # noqa: BLE001
    print(f"  ✗ {type(exc).__name__}: {str(exc)[:300]}")

print()
print("--- 方式 3：pyarrow 直读（绕过 datasets） ---")
try:
    import pyarrow.parquet as pq

    t = pq.read_table(DAPO)
    print(f"  ✓ 成功，行数={t.num_rows:,}")
    print(f"  schema: {t.schema}")
except Exception as exc:  # noqa: BLE001
    print(f"  ✗ {type(exc).__name__}: {str(exc)[:300]}")
