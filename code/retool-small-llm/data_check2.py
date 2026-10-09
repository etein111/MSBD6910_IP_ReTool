"""数据路径最终体检 —— 只测 main_ppo 真实会走的调用。

覆盖：
  A. 三个目录 load_dataset 成功 + data_source 路由正确（map_fn vs map_fn2）
  B. AIME 两份 parquet 的列名满足 map_fn 的字段访问（Problem/Answer, problem/answer）
  C. map_fn / map_fn2 用真实首行跑一遍，确认无异常、产出键正确
"""

import os
import sys

import datasets

sys.path.insert(0, os.getcwd())
from recipe.retool.retool_dataset_utils import map_fn, map_fn2  # noqa: E402

R = "/scratch/mscbdtspod2026/cyeah/retool"
DIRS = {
    "DAPO": f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k",
    "AIME24": f"{R}/data/dataset_clean/Maxwell-Jia/AIME_2024",
    "AIME25": f"{R}/data/dataset_clean/yentinglin/aime_2025",
}
AIME_SOURCES = ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]

_results = []


def check(ok, label, detail=""):
    _results.append((ok, label))
    print(f"  {'✓' if ok else '✗'} {label}")
    if detail:
        print(f"      → {detail}")


def main():
    ds_map = {}
    for name, d in DIRS.items():
        print(f"\n===== {name} =====")
        src = "/".join(d.split("/")[-2:])
        print(f"目录 data_source = {src!r}")
        try:
            ds = datasets.load_dataset(d)["train"]
            ds_map[name] = ds
            check(True, f"{name} load_dataset(目录)", f"行数={len(ds):,}")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name} load_dataset(目录)", f"{type(exc).__name__}: {exc}")
            continue

        # 路由判定
        is_aime = src in AIME_SOURCES
        route = "map_fn(AIME)" if is_aime else "map_fn2(DAPO)"
        expect_aime = name.startswith("AIME")
        check(is_aime == expect_aime, f"{name} 路由={route}", f"期望 {'AIME' if expect_aime else 'DAPO'}")

        print(f"  列名：{ds.column_names}")

    # AIME 字段访问 + map_fn 真实首行
    print("\n===== map_fn（AIME）真实首行 =====")
    for name, src in [("AIME24", "Maxwell-Jia/AIME_2024"), ("AIME25", "yentinglin/aime_2025")]:
        ds = ds_map.get(name)
        if ds is None:
            continue
        row = dict(ds[0])
        # 字段是否存在
        if src == "Maxwell-Jia/AIME_2024":
            need = ["Problem", "Answer"]
        else:
            need = ["problem", "answer"]
        missing = [c for c in need if c not in row]
        check(not missing, f"{name} 有字段 {need}", f"缺失={missing}" if missing else "齐全")
        if missing:
            continue
        try:
            out = map_fn(row, data_source=src)
            check(
                True,
                f"{name} map_fn 成功",
                f"data_source={out['data_source']!r} gt={out['reward_model']['ground_truth']!r}",
            )
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name} map_fn 失败", f"{type(exc).__name__}: {exc}")

    # DAPO map_fn2 真实首行
    print("\n===== map_fn2（DAPO）真实首行 =====")
    ds = ds_map.get("DAPO")
    if ds is not None:
        row = dict(ds[0])
        print(f"  prompt 类型 = {type(row.get('prompt')).__name__}")
        try:
            out = map_fn2(row)
            check(True, "DAPO map_fn2 成功", f"agent_name={out.get('agent_name')!r}")
        except Exception as exc:  # noqa: BLE001
            check(False, "DAPO map_fn2 失败", f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    passed = sum(1 for ok, _ in _results if ok)
    print(f"结果：{passed}/{len(_results)} 通过")
    for ok, label in _results:
        if not ok:
            print(f"  ✗ {label}")
    return 1 if any(not ok for ok, _ in _results) else 0


if __name__ == "__main__":
    sys.exit(main())
