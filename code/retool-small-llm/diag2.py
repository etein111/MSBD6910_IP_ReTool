"""搞清楚 verl 的 data_files 到底是什么形态，以及 load_dataset 拿它怎么用。

之前两个误判的澄清：
  - data_check.py 里我用「单文件路径」调 load_dataset，报找不到文件；
  - 但 retool.py 的 data_source = "/".join(path.split("/")[-2:]) 要等于
    "Maxwell-Jia/AIME_2024"，说明 parquet_file 其实是「目录路径」而非文件路径。
这里直接把 verl 的源码读出来，看 self.data_files 的构造，再测真正的调用。
"""

import inspect
import os

import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset

print("=" * 70)
print("1. RLHFDataset 怎么构造 data_files")
print("=" * 70)
for meth in ["__init__", "_get_data_files", "_read_files_and_tokenize"]:
    fn = getattr(RLHFDataset, meth, None)
    if fn is None:
        print(f"\n[{meth}] 不存在")
        continue
    print(f"\n--- {meth} ---")
    print(inspect.getsource(fn))

print("=" * 70)
print("2. 用真实 config 看 self.data_files 的值")
print("=" * 70)
R = "/scratch/mscbdtspod2026/cyeah/retool"
cfg = {
    "train_files": [f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k"],
    "val_files": [
        f"{R}/data/dataset_clean/Maxwell-Jia/AIME_2024",
        f"{R}/data/dataset_clean/yentinglin/aime_2025",
    ],
}
# 只调 _get_data_files，不真正读数据、不 tokenize
for key in ["train_files", "val_files"]:
    try:
        files = RLHFDataset._get_data_files(cfg[key])
        print(f"\n{key}:")
        for f in files:
            print(f"   {f}")
            print(f"       → data_source = {os.path.join(*f.split('/')[-2:])!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{key} 报错: {type(exc).__name__}: {exc}")

print("=" * 70)
print("3. datasets.load_dataset 对这些真实路径的调用结果")
print("=" * 70)
dirs = [
    f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k",
    f"{R}/data/dataset_clean/Maxwell-Jia/AIME_2024",
    f"{R}/data/dataset_clean/yentinglin/aime_2025",
]
for d in dirs:
    print(f"\n--- {d}")
    print(f"    is_dir={os.path.isdir(d)}")
    try:
        ds = datasets.load_dataset(d)["train"]
        print(f"    ✓ load_dataset(目录) 成功，行数={len(ds):,}")
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ load_dataset(目录): {type(exc).__name__}: {str(exc)[:120]}")
        # 退回：目录里具体有哪些文件，逐个试
        try:
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                try:
                    ds = datasets.load_dataset(fp)["train"]
                    print(f"       ✓ load_dataset(文件 {f}) 成功，行数={len(ds):,}")
                except Exception as e2:  # noqa: BLE001
                    print(f"       ✗ load_dataset(文件 {f}): {type(e2).__name__}: {str(e2)[:100]}")
        except Exception as e3:  # noqa: BLE001
            print(f"       listdir 也失败: {e3}")
