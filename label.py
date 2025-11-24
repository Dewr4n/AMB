import os
import pandas as pd
import numpy as np

sinex_file = "IGS0OPSSNX_20233050000_01D_01D_SOL.SNX"

def read_sinex_coordinates(snx_path):
    coords = {}
    with open(snx_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_block = False
    for line in lines:
        if "+SOLUTION/ESTIMATE" in line:
            in_block = True
            continue
        if "-SOLUTION/ESTIMATE" in line:
            in_block = False
            continue
        if in_block and line.strip() and not line.startswith("*"):
            parts = line.split()
            if len(parts) < 9:
                continue
            typ, code = parts[1], parts[2].upper()
            if typ in ["STAX", "STAY", "STAZ"]:
                try:
                    value = float(parts[8])
                except:
                    continue
                if code not in coords:
                    coords[code] = {"X": None, "Y": None, "Z": None}
                if typ == "STAX":
                    coords[code]["X"] = value
                elif typ == "STAY":
                    coords[code]["Y"] = value
                elif typ == "STAZ":
                    coords[code]["Z"] = value
    return coords


def fmt(num):
    return round(num, 4) if num is not None else None


def ecef_to_enu_matrix(x, y, z):
    a = 6378137.0
    e2 = 6.69437999014e-3

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))

    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    R = np.array([
        [-slon,  clon, 0],
        [-slat * clon, -slat * slon, clat],
        [clat * clon,  clat * slon,  slat]
    ])
    return R


coords_dict = read_sinex_coordinates(sinex_file)
print(f"✅ 已从 SINEX 读取 {len(coords_dict)} 个站的坐标")

for file in os.listdir("."):
    if file.startswith("flt-") and file.endswith(".csv"):
        station = file.replace("flt-", "").replace(".csv", "").upper()
        if station not in coords_dict:
            print(f"⚠️ 未在 SINEX 中找到 {station}，跳过。")
            continue

        coord = coords_dict[station]
        if None in (coord["X"], coord["Y"], coord["Z"]):
            print(f"⚠️ 坐标不完整，跳过 {station}")
            continue

        df = pd.read_csv(file)

        # 写入真值
        x_gt, y_gt, z_gt = fmt(coord["X"]), fmt(coord["Y"]), fmt(coord["Z"])
        df.loc[0, "X_GT"], df.loc[0, "Y_GT"], df.loc[0, "Z_GT"] = x_gt, y_gt, z_gt

        if all(c in df.columns for c in ["X_ECEF", "Y_ECEF", "Z_ECEF"]):
            df["dX"] = (df["X_ECEF"] - x_gt).round(4)
            df["dY"] = (df["Y_ECEF"] - y_gt).round(4)
            df["dZ"] = (df["Z_ECEF"] - z_gt).round(4)

            R = ecef_to_enu_matrix(x_gt, y_gt, z_gt)
            enu = np.dot(R, df[["dX", "dY", "dZ"]].T).T

            df["dE"] = enu[:, 0].round(4)
            df["dN"] = enu[:, 1].round(4)
            df["dU"] = enu[:, 2].round(4)

            df["2D_error"] = np.sqrt(df["dE"]**2 + df["dN"]**2).round(4)
            df["3D_error"] = np.sqrt(df["dE"]**2 + df["dN"]**2 + df["dU"]**2).round(4)

            # ================================================
            # 🔥 修改点：失稳在第一个坏点开始标记 0
            # ================================================
            df["label"] = 0
            window = 5       # 原来是 10 连续收敛，你说 5 就改这里
            th_c = 0.05      # <5cm 收敛
            th_b = 0.10      # >10cm 立即失稳
            bad_k = 3        # 连续 3 个 5~10cm 失稳

            is_conv = False
            bad_count = 0

            for i in range(len(df)):
                e3 = df.loc[i, "3D_error"]

                # 未收敛 → 检查连续 window 个 <5cm
                if not is_conv:
                    if i + window <= len(df) and np.all(df["3D_error"].iloc[i:i+window] < th_c):
                        is_conv = True
                        df.loc[i:i+window-1, "label"] = 1
                    continue

                # ======================
                #   已收敛 → 检查失稳
                # ======================

                # 一旦出现 >10cm → 从当下立刻失稳
                if e3 > th_b:
                    is_conv = False
                    df.loc[i, "label"] = 0
                    bad_count = 0
                    continue

                # 5~10cm 区间
                if th_c < e3 <= th_b:
                    bad_count += 1
                else:
                    bad_count = 0

                # 这里修改 —— 连续 bad_k 个点出现时：
                # 旧逻辑：在 i（第三个坏点）标 0
                # 新逻辑：在 i-bad_k+1（第一个坏点）标 0
                if bad_count >= bad_k:
                    start = i - bad_k + 1
                    end = i
                    is_conv = False
                    df.loc[start:end, "label"] = 0
                    bad_count = 0
                    continue

                # 仍然稳定
                if is_conv:
                    df.loc[i, "label"] = 1

        df.to_csv(file, index=False)
        print(f"✅ 已更新 {file}，写入真值、误差、标签")

print("🎯 所有文件处理完成。")
