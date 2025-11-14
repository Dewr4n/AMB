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
    """使用地理纬度（geodetic latitude）计算 ECEF→ENU 旋转矩阵"""
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

        # 写入真值（只在第一行）
        x_gt, y_gt, z_gt = fmt(coord["X"]), fmt(coord["Y"]), fmt(coord["Z"])
        df.loc[0, "X_GT"], df.loc[0, "Y_GT"], df.loc[0, "Z_GT"] = x_gt, y_gt, z_gt

        # 检查必要列
        if all(c in df.columns for c in ["X_ECEF", "Y_ECEF", "Z_ECEF"]):
            # ECEF 差值
            df["dX"] = (df["X_ECEF"] - x_gt).round(4)
            df["dY"] = (df["Y_ECEF"] - y_gt).round(4)
            df["dZ"] = (df["Z_ECEF"] - z_gt).round(4)

            # === ENU & 误差计算 ===
            R = ecef_to_enu_matrix(x_gt, y_gt, z_gt)
            enu = np.dot(R, df[["dX", "dY", "dZ"]].T).T
            df["dE"] = enu[:, 0].round(4)
            df["dN"] = enu[:, 1].round(4)
            df["dU"] = enu[:, 2].round(4)

            df["2D_error"] = np.sqrt(df["dE"]**2 + df["dN"]**2).round(4)
            df["3D_error"] = np.sqrt(df["dE"]**2 + df["dN"]**2 + df["dU"]**2).round(4)

            # === 收敛与失稳判定（通过 time 列小时判断） ===
            df["label"] = 0
            window = 10           # 连续10个历元判定收敛
            threshold = 0.05      # 3D < 5cm 判定为收敛
            re_duration = 3       # 连续3个历元 > 5cm 判为失稳

            if "time" in df.columns:
                # 从字符串中提取小时（格式示例：2023-11-01 22:00:30[GPS]）
                df["hour_block"] = df["time"].str.extract(r"\s(\d{2}):")[0].astype(int)
                unique_hours = sorted(df["hour_block"].unique())
            else:
                print(f"⚠️ {file} 中未找到 time 列，默认整段处理。")
                df["hour_block"] = 0
                unique_hours = [0]

            for h in unique_hours:
                segment = df[df["hour_block"] == h]
                if segment.empty:
                    continue

                is_converged = False
                bad_count = 0
                idxs = segment.index.to_list()

                for i, idx in enumerate(idxs):
                    if not is_converged:
                        if i + window <= len(segment) and np.all(segment["3D_error"].iloc[i:i+window] < threshold):
                            is_converged = True
                            df.loc[idxs[i]:idxs[-1], "label"] = 1
                    else:
                        if segment["3D_error"].iloc[i] > threshold:
                            bad_count += 1
                        else:
                            bad_count = 0
                        if bad_count >= re_duration:
                            is_converged = False
                        if is_converged:
                            df.loc[idx, "label"] = 1

            df.drop(columns=["hour_block"], inplace=True)

        else:
            print(f"⚠️ {file} 中未找到 X_ECEF/Y_ECEF/Z_ECEF 列，跳过差值与 ENU 计算。")

        df.to_csv(file, index=False)
        print(f"✅ 已更新 {file}，写入真值、差值、ENU 与收敛标签列")

print("🎯 所有文件处理完成。")
