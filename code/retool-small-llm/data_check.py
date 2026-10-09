"""数据体检 —— 提交冒烟作业前必须过。

验三件事，每一件不通过都会让冒烟跑出「看起来在跑、其实全错」的结果：
  1. 三个 parquet 的**实际列名/schema** 是否满足 retool.py 的字段映射要求
  2. `data_source = "/".join(path.split("/")[-2:])` 在**当前目录布局下**到底推出来什么
     —— 推错了 AIME 会静默走成 DAPO 的映射，评测数字全废
  3. 上游 RLHFDataset 解析 data_files 的方式（决定上面第 2 条是文件路径还是目录路径）

用法（code/work/ 下）：python data_check.py
"""

import inspect
import os
import sys

import datasets

R = "/scratch/mscbdtspod2026/cyeah/retool"
DAPO = f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k/dapo-math-17k.parquet"
AIME24 = f"{R}/data/dataset_clean/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet"
AIME25 = (
    f"{R}/data/dataset_clean/yentinglin/aime_2025/"
    "train-00000-of-00001-243207c6c994e1bd.parquet"
)

AIME_SOURCES = ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    _results.append((ok, label))
    print(f"  {'✓' if ok else '✗'} {label}")
    if detail:
        print(f"      → {detail}")


def probe(label: str, path: str) -> None:
    """打印一个 parquet 的行数、列名、以及关键列的类型与样例。"""
    print(f"\n--- {label} ---")
    print(f"路径：{path}")
    if not os.path.exists(path):
        check(False, f"{label} 文件存在", f"找不到 {path}")
        return
    check(True, f"{label} 文件存在")
    ds = datasets.load_dataset(path)["train"]
    print(f"行数：{len(ds):,}")
    print(f"列名：{ds.column_names}")
    for col in ds.column_names:
        val = ds[0][col]
        kind = type(val).__name__
        preview = repr(val)
        if len(preview) > 160:
            preview = preview[:160] + "…"
        print(f"   · {col}: {kind} = {preview}")


def main() -> int:
    print("=" * 70)
    print("1. 上游 RLHFDataset 如何解析 data_files")
    print("=" * 70)
    try:
        from verl.utils.dataset.rl_dataset import RLHFDataset

        src = inspect.getsource(RLHFDataset._get_data_files)
        print(src)
    except Exception as exc:  # noqa: BLE001
        check(False, "读取 RLHFDataset._get_data_files", repr(exc))

    print("=" * 70)
    print("2. 三个 parquet 的真实 schema")
    print("=" * 70)
    probe("DAPO-Math-17k（RL 训练集）", DAPO)
    probe("AIME 2024（评测）", AIME24)
    probe("AIME 2025（评测）", AIME25)

    print()
    print("=" * 70)
    print("3. data_source 推导 —— 决定走 map_fn 还是 map_fn2")
    print("=" * 70)
    for label, path, expect_aime in [
        ("DAPO-Math-17k", DAPO, False),
        ("AIME 2024", AIME24, True),
        ("AIME 2025", AIME25, True),
    ]:
        derived = "/".join(path.split("/")[-2:])
        is_aime = derived in AIME_SOURCES
        ok = is_aime == expect_aime
        check(
            ok,
            f"{label}: data_source = {derived!r} → "
            f"{'map_fn(AIME)' if is_aime else 'map_fn2(DAPO)'}",
            f"期望 {'map_fn(AIME)' if expect_aime else 'map_fn2(DAPO)'}",
        )

    print()
    print("=" * 70)
    print("4. 字段映射能不能真的跑通（拿真实首行试一次）")
    print("=" * 70)
    sys.path.insert(0, os.getcwd())
    from recipe.retool.retool_dataset_utils import map_fn, map_fn2

    # DAPO 首行走 map_fn2
    dapo_ds = datasets.load_dataset(DAPO)["train"]
    src = "/".join(DAPO.split("/")[-2:])
    row = dict(dapo_ds[0])
    try:
        out = map_fn2(row) if src not in AIME_SOURCES else map_fn(row, data_source=src)
        check(True, "DAPO 首行字段映射成功", f"产出键={sorted(out.keys())}")
    except Exception as exc:  # noqa: BLE001
        check(False, "DAPO 首行字段映射失败", f"{type(exc).__name__}: {exc}")

    # AIME 两份都走 map_fn
    for label, path in [("AIME 2024", AIME24), ("AIME 2025", AIME25)]:
        ds = datasets.load_dataset(path)["train"]
        src = "/".join(path.split("/")[-2:])
        try:
            out = map_fn(dict(ds[0]), data_source=src)
            check(
                True,
                f"{label} 字段映射成功",
                f"data_source={out['data_source']!r} "
                f"ground_truth={out['reward_model']['ground_truth']!r}",
            )
        except Exception as exc:  # noqa: BLE001
            check(False, f"{label} 字段映射失败", f"{type(exc).__name__}: {exc}")

    print()
    print("=" * 70)
    passed = sum(1 for ok, _ in _results if ok)
    print(f"结果：{passed}/{len(_results)} 通过")
    failed = [label for ok, label in _results if not ok]
    for label in failed:
        print(f"  ✗ {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
