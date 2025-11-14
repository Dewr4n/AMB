import os
import pandas as pd
import numpy as np
from collections import deque

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"📂 发现 {len(csv_files)} 个 CSV 文件待处理。")


def find_linear_combination_bfs(r, t, graph):
    """BFS 搜索路径并返回 (nf_sum, nfx_sum, path)"""
    queue = deque([(r, 0.0, 0.0, [r])])
    visited = set([r])

    while queue:
        current, nf_sum, nfx_sum, path = queue.popleft()

        if current == t:
            return nf_sum, nfx_sum, path

        for (a, b, nf, nfx, _, _) in graph:
            if a == current and b not in visited:
                visited.add(b)
                queue.append((b, nf_sum + nf, nfx_sum + nfx, path + [b]))

            elif b == current and a not in visited:
                visited.add(a)
                queue.append((a, nf_sum - nf, nfx_sum - nfx, path + [a]))

    return None


for file in csv_files:
    print(f"\n🚀 正在处理文件: {file}")
    df = pd.read_csv(file)

    required_cols = [
        "time", "label",
        "Sat_ref", "Sat_tgt", "Nfloat_orig",
        "Sat_ref_shk", "Sat_tgt_shk", "Nfloat_shk", "Nfixed_shk"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{file} 缺少必要列: {c}")

    df["hour_block"] = df["time"].str.extract(r"\s(\d{2}):")[0].astype(int)
    unique_hours = sorted(df["hour_block"].unique())

    # === 🔥 修复 Warning：初始化为字符串列 ===
    df["Nfloat_matched"] = ""           # 原来是 np.nan
    df["Nfixed_matched"] = ""
    df["Ndiff_int"] = ""
    df["Matched_pair_id"] = ""
    df["Matched_pair_time"] = ""

    for h in unique_hours:
        seg = df[df["hour_block"] == h]
        idxs = seg.index.to_list()
        if len(seg) == 0:
            continue

        label1_pairs = []
        for idx, row in seg[seg["label"] == 1].iterrows():
            ref2s = row["Sat_ref_shk"].split(";")
            tgt2s = row["Sat_tgt_shk"].split(";")
            nfloat_vals = [float(x) for x in row["Nfloat_shk"].split(";")]
            nfixed_vals = [float(x) for x in row["Nfixed_shk"].split(";")]

            for rr, tt, nfo, nfx in zip(ref2s, tgt2s, nfloat_vals, nfixed_vals):
                label1_pairs.append((rr, tt, nfo, nfx, idx, row["time"]))

        for i in idxs:
            if df.loc[i, "label"] != 0:
                continue

            refs = df.loc[i, "Sat_ref"].split(";")
            tgts = df.loc[i, "Sat_tgt"].split(";")
            floats = [float(x) for x in df.loc[i, "Nfloat_orig"].split(";")]

            matched_floats = []
            matched_fixeds = []
            diff_ints = []
            matched_ids = []
            matched_times = []

            for r, t, nf in zip(refs, tgts, floats):
                found_float = None
                found_fixed = None
                matched_id = ""
                matched_time = ""

                # === ① 直接匹配 ===
                for rr, tt, nfo, nfx, row_idx, row_time in label1_pairs:
                    if rr == r and tt == t:
                        if abs(nfx - nf) <= 10:
                            found_float, found_fixed = round(nfo, 4), round(nfx, 4)
                            matched_id = f"{r}-{t}"
                            matched_time = str(row_time)
                        break

                # === ② 反向匹配 ===
                if found_fixed is None:
                    for rr, tt, nfo, nfx, row_idx, row_time in label1_pairs:
                        if rr == t and tt == r:
                            if abs(-nfx - nf) <= 10:
                                found_float, found_fixed = -round(nfo, 4), -round(nfx, 4)
                                matched_id = f"{t}-{r}"
                                matched_time = str(row_time)
                            break

                # === ③ BFS 匹配 ===
                if found_fixed is None:
                    val = find_linear_combination_bfs(r, t, label1_pairs)
                    if val is not None:
                        nf_bfs, nfx_bfs, path = val
                        if abs(nfx_bfs - nf) <= 10:
                            found_float, found_fixed = round(nf_bfs, 4), round(nfx_bfs, 4)
                            matched_id = "→".join(path)
                            matched_time = "BFS"

                diff_int = int(round(found_fixed - nf)) if found_fixed is not None else "None"

                matched_floats.append(str(found_float) if found_float else "None")
                matched_fixeds.append(str(found_fixed) if found_fixed else "None")
                diff_ints.append(str(diff_int))
                matched_ids.append(str(matched_id))
                matched_times.append(str(matched_time))

            df.loc[i, "Nfloat_matched"] = ";".join(matched_floats)
            df.loc[i, "Nfixed_matched"] = ";".join(matched_fixeds)
            df.loc[i, "Ndiff_int"] = ";".join(diff_ints)
            df.loc[i, "Matched_pair_id"] = ";".join(matched_ids)
            df.loc[i, "Matched_pair_time"] = ";".join(matched_times)

    df.drop(columns=["hour_block"], inplace=True)
    df.to_csv(file, index=False)
    print(f"✅ {file} 处理完成（Warning 已修复，无 dtype 冲突）。")

print("\n🎯 所有 CSV 文件处理完成。")