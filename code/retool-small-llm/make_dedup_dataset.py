#!/usr/bin/env python3
"""DAPO-Math-17k 去重：179 万行 → ~17k 独特 prompt。

数据集发布的 bug：实际 ~17k 独特题目被误复制 ~100 次（官方卡片承认"无法重训，按原样发布"）。
用全量会训练时间暴增 + 同一 epoch 重复相同题 ~100 次严重过拟合。按 prompt[0]["content"] 去重，
字段格式与 recipe/retool/retool.py 的 map_fn2 完全兼容（prompt/reward_model.ground_truth 原样保留）。

用法（login node，CPU 即可，约 1-2 分钟）：
  python3 make_dedup_dataset.py
"""
import os

import pandas as pd

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/scratch/mscbdtspod2026/cyeah/retool")
SRC = f"{PROJECT_ROOT}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k/dapo-math-17k.parquet"
DST = f"{PROJECT_ROOT}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-dedup/train.parquet"

df = pd.read_parquet(SRC)
print(f"original rows: {len(df)}")

# 去重键 = 题目文本（prompt[0]["content"]）
df["_key"] = df["prompt"].map(lambda p: p[0]["content"])
df = df.drop_duplicates(subset="_key").drop(columns="_key")
print(f"dedup rows:   {len(df)}")

os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_parquet(DST, index=False)
print(f"saved: {DST}")
print("--- columns ---")
print(list(df.columns))
print("--- sample[0] ---")
print(df.iloc[0].to_dict())
