"""给冒烟测试切一个 DAPO 子集 parquet。

为什么需要它
============
训练集 DAPO-Math-17k 有 1,791,700 行。CustomRLHFDataset._read_files_and_tokenize
重写时没调用父类，所以 data.train_max_samples 采样键对它无效——不切子集就会
每次冒烟都先 map_fn2 全量 179 万行，白白烧时间。

这里用 pyarrow 直读直写，切前 N 行，输出到 dataset_clean 下的新目录
`BytedTsinghua-SIA/DAPO-Math-17k-smoke/`。目录名故意不匹配 AIME 那串，
保证 data_source 路由仍然落到 map_fn2（与全量 DAPO 完全一致）。

用法（login node 即可，不占 GPU）：
    python make_smoke_subset.py 64
"""

import os
import sys

import pyarrow.parquet as pq

R = "/scratch/mscbdtspod2026/cyeah/retool"
SRC = f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k/dapo-math-17k.parquet"
DST_DIR = f"{R}/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-smoke"
DST = f"{DST_DIR}/dapo-math-17k.parquet"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    print(f"源：{SRC}")
    table = pq.read_table(SRC)
    print(f"全量行数：{table.num_rows:,}")
    print(f"列/schema：\n{table.schema}")

    subset = table.slice(0, n)
    os.makedirs(DST_DIR, exist_ok=True)
    pq.write_table(subset, DST)
    print(f"\n✓ 已写 {n} 行到：{DST}")

    # 回读校验
    check = pq.read_table(DST)
    print(f"回读校验：{check.num_rows:,} 行，列 = {check.column_names}")
    # 数据完整性：首行 prompt 结构不变
    print(f"首行 prompt 类型：{type(check['prompt'][0].as_py()).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
