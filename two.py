import os
import pandas as pd
import numpy as np

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"📂 发现 {len(csv_files)} 个 CSV 文件待处理。")

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

    # === 字符串类型（不用 float64），避免 dtype Warning ===
    df["Nfloat_matched"] = ""
    df["Nfixed_matched"] = ""
    df["Ndiff_int"] = ""
    df["Matched_pair_id"] = ""
    df["Matched_pair_time"] = ""

    for h in unique_hours:
        seg = df[df["hour_block"] == h]
        idxs = seg.index.to_list()
        if len(seg) == 0:
            continue

        # 构建 label = 1 的模糊度对（带时间）
        label1_pairs = []
        for idx, row in seg[seg["label"] == 1].iterrows():
            ref2s = row["Sat_ref_shk"].split(";")
            tgt2s = row["Sat_tgt_shk"].split(";")
            nfloat_vals = [float(x) for x in row["Nfloat_shk"].split(";")]
            nfixed_vals = [float(x) for x in row["Nfixed_shk"].split(";")]

            for rr, tt, nfo, nfx in zip(ref2s, tgt2s, nfloat_vals, nfixed_vals):
                label1_pairs.append((rr, tt, nfo, nfx, row["time"]))

        # 处理 label = 0 的记录
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
                for rr, tt, nfo, nfx, row_time in label1_pairs:
                    if rr == r and tt == t:
                        if abs(nfx - nf) <= 10:
                            found_float = round(nfo, 4)
                            found_fixed = round(nfx, 4)
                            matched_id = f"{r}-{t}"
                            matched_time = str(row_time)
                        break

                # === ② 反向匹配 ===
                if found_fixed is None:
                    for rr, tt, nfo, nfx, row_time in label1_pairs:
                        if rr == t and tt == r:
                            if abs(-nfx - nf) <= 10:
                                found_float = -round(nfo, 4)
                                found_fixed = -round(nfx, 4)
                                matched_id = f"{t}-{r}"
                                matched_time = str(row_time)
                            break

                # === 整数差 ===
                if found_fixed is not None:
                    diff_int = int(round(found_fixed - nf))
                else:
                    diff_int = "None"

                matched_floats.append(str(found_float) if found_float is not None else "None")
                matched_fixeds.append(str(found_fixed) if found_fixed is not None else "None")
                diff_ints.append(str(diff_int))
                matched_ids.append(matched_id)
                matched_times.append(matched_time)

            df.loc[i, "Nfloat_matched"] = ";".join(matched_floats)
            df.loc[i, "Nfixed_matched"] = ";".join(matched_fixeds)
            df.loc[i, "Ndiff_int"] = ";".join(diff_ints)
            df.loc[i, "Matched_pair_id"] = ";".join(matched_ids)
            df.loc[i, "Matched_pair_time"] = ";".join(matched_times)

    df.drop(columns=["hour_block"], inplace=True)
    df.to_csv(file, index=False)
    print(f"✅ {file} 处理完成（仅保留前两种匹配方式，无 BFS）。")

print("\n🎯 所有 CSV 文件处理完成。")
