import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from glob import glob
import random

# ==============================
# 固定随机种子
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed fixed to {seed}")

# ==============================
# 数据解析函数
# ==============================
def parse_vector_1xn(s, n_expected, max_dim):
    """解析 'a;b;c;...' 这样的 1×n 向量，补零到 max_dim"""
    if not isinstance(s, str):
        return np.zeros(max_dim, dtype=np.float32)
    vals = [x for x in s.split(";") if x.strip() != ""]
    # 转成 float，不可转的当成 0
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    out = out[:max(0, n_expected)]
    if len(out) < max_dim:
        out += [0.0] * (max_dim - len(out))
    else:
        out = out[:max_dim]
    return np.array(out, dtype=np.float32)

def parse_vector_1xn_with_none(s, n_expected, max_dim):
    """
    解析可能包含 'None' 的 1×n 向量:
      - 返回 fixed_vec: float32[max_dim]
      - 返回 valid_mask: int64[max_dim], 1 表示该维有有效固定值, 0 表示 None/无效
    """
    fixed = np.zeros(max_dim, dtype=np.float32)
    valid_mask = np.zeros(max_dim, dtype=np.int64)

    if not isinstance(s, str):
        return fixed, valid_mask

    vals = [v for v in s.split(";") if v.strip() != ""]
    # 只看前 n_expected 维
    for i in range(min(len(vals), max_dim, max(0, n_expected))):
        raw = vals[i].strip()
        if raw.lower() == "none":
            # 无对应固定模糊度 → 不学习
            fixed[i] = 0.0
            valid_mask[i] = 0
        else:
            try:
                fixed[i] = float(raw)
                valid_mask[i] = 1
            except Exception:
                fixed[i] = 0.0
                valid_mask[i] = 0
    return fixed, valid_mask

def parse_matrix_nxn(s, n_expected, max_dim):
    """
    解析 'a;b;cI...' 这样的 n×n 矩阵，裁剪成 n×n，再补零到 max_dim×max_dim。
    返回 2D 矩阵 (max_dim, max_dim)。
    """
    if not isinstance(s, str):
        return np.zeros((max_dim, max_dim), dtype=np.float32)
    rows = [r for r in s.split("I") if r.strip() != ""]
    mat = []
    for r in rows:
        parts = [p for p in r.split(";") if p.strip() != ""]
        row_vals = []
        for p in parts:
            try:
                row_vals.append(float(p))
            except Exception:
                row_vals.append(0.0)
        mat.append(row_vals)
    arr = np.array(mat, dtype=np.float32) if len(mat) > 0 else np.zeros((0, 0), dtype=np.float32)
    n = min(n_expected, arr.shape[0], arr.shape[1])
    cut = arr[:n, :n]
    padded = np.zeros((max_dim, max_dim), dtype=np.float32)
    padded[:cut.shape[0], :cut.shape[1]] = cut
    return padded

# ==============================
# 数据集（以文件为单位）
# 方案 B：每个历元 → 一个样本
#        样本内部：每颗卫星 = 一个 token
# ==============================
class AmbiguityDataset(Dataset):
    def __init__(self, file_list, max_dim=20, window=5):
        self.samples = []
        self.max_dim = max_dim
        self.window = window

        for file in file_list:
            try:
                df = pd.read_csv(file, sep=",")
            except Exception:
                continue

            df.columns = [c.strip() for c in df.columns]
            if "label" not in df.columns:
                continue

            has_shk = all(c in df.columns for c in ["NumSD_shk", "Nfloat_shk", "Nfixed_shk", "Q_shk"])
            has_orig = all(c in df.columns for c in ["NumSD_orig", "Nfloat_orig", "Nfixed_matched", "Q_orig"])

            for _, row in df.iterrows():
                try:
                    lbl = row["label"]
                    try:
                        lbl = int(lbl)
                    except Exception:
                        # 有些可能是字符串 '0'/'1'
                        lbl = int(str(lbl).strip())

                    # ==========================
                    # label = 1: 用 *_shk 特征
                    # ==========================
                    if lbl == 1:
                        if not has_shk:
                            continue

                        n_raw = int(float(row["NumSD_shk"]))
                        if n_raw <= 4:
                            continue
                        n = max(0, min(n_raw, self.max_dim))

                        nfloat_vec = parse_vector_1xn(row["Nfloat_shk"], n, self.max_dim)
                        nfixed_vec = parse_vector_1xn(row["Nfixed_shk"], n, self.max_dim)
                        q_mat      = parse_matrix_nxn(row["Q_shk"],     n, self.max_dim)  # (max_dim, max_dim)

                        # 每个 token 的特征: [Nfloat_i, Q[i,:], NumSD, pos_i]
                        feat_dim = self.max_dim + 3
                        tokens = np.zeros((self.max_dim, feat_dim), dtype=np.float32)
                        for i in range(self.max_dim):
                            if i < n:
                                tokens[i, 0] = nfloat_vec[i]
                                tokens[i, 1:1+self.max_dim] = q_mat[i, :self.max_dim]
                                tokens[i, 1+self.max_dim] = float(n_raw)   # 原始 NumSD_shk
                                tokens[i, 2+self.max_dim] = float(i) / float(self.max_dim)
                            else:
                                pass

                        x_seq = tokens.astype(np.float32)

                        # 目标：delta = Nfixed - round(Nfloat)
                        base_int = np.round(nfloat_vec)
                        delta = (nfixed_vec - base_int).astype(int)
                        delta = np.clip(delta, -self.window, self.window)
                        y_cls = (delta + self.window).astype(np.int64)

                        # label=1 时，所有前 n 维都有有效固定值
                        valid_mask = np.zeros(self.max_dim, dtype=np.int64)
                        if n > 0:
                            valid_mask[:n] = 1

                        self.samples.append((x_seq, y_cls, n, nfloat_vec, nfixed_vec, valid_mask))

                    # ==========================
                    # label = 0: 用 *_orig 特征 + Nfixed_matched
                    # ==========================
                    elif lbl == 0:
                        if not has_orig:
                            continue

                        n_raw = int(float(row["NumSD_orig"]))
                        if n_raw <= 4:
                            continue
                        n = max(0, min(n_raw, self.max_dim))

                        nfloat_vec = parse_vector_1xn(row["Nfloat_orig"], n, self.max_dim)
                        q_mat      = parse_matrix_nxn(row["Q_orig"],     n, self.max_dim)
                        nfixed_vec, valid_mask = parse_vector_1xn_with_none(
                            row["Nfixed_matched"], n, self.max_dim
                        )  # 有 None 的地方 valid_mask=0

                        # 如果这个历元一个有效固定值都没有，就没法学，跳过
                        if valid_mask[:n].sum() == 0:
                            continue

                        feat_dim = self.max_dim + 3
                        tokens = np.zeros((self.max_dim, feat_dim), dtype=np.float32)
                        for i in range(self.max_dim):
                            if i < n:
                                tokens[i, 0] = nfloat_vec[i]
                                tokens[i, 1:1+self.max_dim] = q_mat[i, :self.max_dim]
                                tokens[i, 1+self.max_dim] = float(n_raw)   # NumSD_orig
                                tokens[i, 2+self.max_dim] = float(i) / float(self.max_dim)
                            else:
                                pass

                        x_seq = tokens.astype(np.float32)

                        base_int = np.round(nfloat_vec)
                        delta = (nfixed_vec - base_int).astype(int)
                        delta = np.clip(delta, -self.window, self.window)
                        y_cls = (delta + self.window).astype(np.int64)

                        self.samples.append((x_seq, y_cls, n, nfloat_vec, nfixed_vec, valid_mask))

                    else:
                        # 其他 label 忽略
                        continue

                except Exception:
                    continue

        print(f"✅ Loaded {len(self.samples)} samples from {len(file_list)} files "
              f"(max_dim={self.max_dim}, window=±{self.window}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, n, nfloat, nfixed, valid_mask = self.samples[idx]
        return (
            torch.from_numpy(x),            # [max_dim, feat_dim]
            torch.from_numpy(y),            # [max_dim]
            torch.tensor(n, dtype=torch.long),
            torch.from_numpy(nfloat),       # [max_dim]
            torch.from_numpy(nfixed),       # [max_dim]
            torch.from_numpy(valid_mask)    # [max_dim]
        )

# ==============================
# Transformer 分类模型（序列版）
# 输入: [B, max_dim, feat_dim]
# 输出: [B, max_dim, n_class]
# ==============================
class FixingTransformerCLS(nn.Module):
    def __init__(self, feat_dim, max_dim=20, n_class=11,
                 d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.max_dim = max_dim
        self.n_class = n_class

        self.input_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, n_class)

    def forward(self, x):
        """
        x: [B, max_dim, feat_dim]
        return: [B, max_dim, n_class]
        """
        z = self.input_proj(x)
        z = self.encoder(z)
        out = self.fc(z)
        return out

# ==============================
# 评估函数
# ==============================
@torch.no_grad()
def evaluate_classification(model, loader, device, max_dim, window):
    model.eval()
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    total_loss, correct, total = 0.0, 0.0, 0.0
    fix_ok, fix_tot = 0, 0

    for x, y, n, nfloat, nfixed, valid_mask in loader:
        x = x.to(device).float()
        y = y.to(device)
        n = n.to(device)
        nfloat = nfloat.to(device)
        nfixed = nfixed.to(device)
        valid_mask = valid_mask.to(device).float()  # [B, max_dim]

        logits = model(x)  # [B, max_dim, n_class]
        B, D, C = logits.shape

        # 有效维度: 前 n 颗卫星
        dim_mask = (torch.arange(D, device=device).unsqueeze(0) < n.unsqueeze(1)).float()
        full_mask = dim_mask * valid_mask  # 只有 valid_mask=1 的卫星才参与 loss/ACC/FSR

        loss_mat = ce_loss(logits.transpose(1, 2), y)  # [B, max_dim]
        if full_mask.sum() > 0:
            loss = (loss_mat * full_mask).sum() / full_mask.sum()
            total_loss += loss.item() * B

        pred_cls = logits.argmax(dim=-1)  # [B, max_dim]
        correct += ((pred_cls == y) * full_mask).sum().item()
        total   += full_mask.sum().item()

        delta_pred = pred_cls - window
        nfixed_pred = torch.round(nfloat) + delta_pred

        # FSR: 历元级别，只看 valid_mask=1 的维
        for j in range(B):
            dim = n[j].item()
            if dim <= 0:
                continue
            vm = valid_mask[j, :dim] > 0.5
            if vm.sum() == 0:
                # 这个历元虽然在 dataset 里，但有效模糊度 mask 全 0，就不计入 FSR
                continue
            if torch.allclose(nfixed_pred[j, :dim][vm], nfixed[j, :dim][vm], atol=1e-3):
                fix_ok += 1
            fix_tot += 1

    acc = correct / total if total > 0 else 0.0
    fsr = fix_ok / fix_tot if fix_tot > 0 else 0.0
    return total_loss / len(loader.dataset), acc, fsr

# ==============================
# 训练流程 + 差值统计输出
# ==============================
def train_model_by_files(folder, num_files, epochs, batch_size,
                         lr, max_dim, window, d_model, nhead, num_layers):
    all_files = glob(os.path.join(folder, "*.csv"))
    random.shuffle(all_files)
    all_files = all_files[:num_files]
    print(f"📦 Using {len(all_files)} CSV files for this experiment.")

    n_total = len(all_files)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]
    print(f"📂 File split → Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_set = AmbiguityDataset(train_files, max_dim=max_dim, window=window)
    val_set   = AmbiguityDataset(val_files,   max_dim=max_dim, window=window)
    test_set  = AmbiguityDataset(test_files,  max_dim=max_dim, window=window)

    # 🌟 差值统计，只统计 valid_mask=1 的模糊度
    def diff_statistics(dataset, name):
        diffs = []
        for _, _, n, nfloat, nfixed, valid_mask in dataset:
            n = int(n.item()) if torch.is_tensor(n) else int(n)
            if n <= 0:
                continue
            vm = np.array(valid_mask[:n], dtype=bool)
            if vm.sum() == 0:
                continue
            diff = nfixed[:n][vm] - np.round(nfloat[:n][vm])
            diffs.extend(diff.tolist())

        if len(diffs) == 0:
            print(f"\n⚠️ {name} 集在有效模糊度上为空，跳过统计。")
            return

        diffs = np.array(diffs, dtype=int)
        unique_vals, counts = np.unique(diffs, return_counts=True)
        total = counts.sum()

        print(f"\n📊 {name} diff stats (Nfixed - round(Nfloat)) 仅统计 valid_mask=1 的维")
        print(f"  范围: [{unique_vals.min()} , {unique_vals.max()}] | 总卫星数: {total}")
        print("  差值 | 计数 | 占比(%)")
        print("  ----------------------")
        for val, cnt in zip(unique_vals, counts):
            ratio = cnt / total * 100
            print(f"  {val:>4d} | {cnt:>8d} | {ratio:>6.2f}%")
        zero_ratio = (counts[unique_vals == 0].sum() / total * 100) if 0 in unique_vals else 0
        print("  ----------------------")
        print(f"  零差比例: {zero_ratio:.2f}%\n")

    diff_statistics(train_set, "Train")
    diff_statistics(val_set, "Val")
    diff_statistics(test_set, "Test")

    def collate_fn(batch):
        xs, ys, ns, nfs, nfx, masks = zip(*batch)
        return (
            torch.stack(xs),
            torch.stack(ys),
            torch.stack(ns),
            torch.stack(nfs),
            torch.stack(nfx),
            torch.stack(masks)
        )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    sample_x, _, _, _, _, _ = train_set[0]
    feat_dim = sample_x.shape[-1]
    n_class = 2 * window + 1

    model = FixingTransformerCLS(
        feat_dim=feat_dim,
        max_dim=max_dim,
        n_class=n_class,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y, n, _, _, valid_mask in train_loader:
            x = x.to(device).float()
            y = y.to(device)
            n = n.to(device)
            valid_mask = valid_mask.to(device).float()

            logits = model(x)  # [B, max_dim, n_class]
            B, D, C = logits.shape

            dim_mask = (torch.arange(D, device=device).unsqueeze(0) < n.unsqueeze(1)).float()
            full_mask = dim_mask * valid_mask  # [B, max_dim]

            loss_mat = criterion(logits.transpose(1, 2), y)  # [B, max_dim]
            if full_mask.sum() == 0:
                # 这一批刚好全是没有有效模糊度的样本（极少见），直接跳过
                continue
            loss = (loss_mat * full_mask).sum() / full_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc, val_fsr = evaluate_classification(
            model, val_loader, device, max_dim, window
        )

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"TrainLoss: {train_loss:.6f} | "
              f"ValLoss: {val_loss:.6f} | "
              f"ValAcc: {val_acc:.4f} | "
              f"ValFSR: {val_fsr:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model_cls.pt")

    # 最终测试
    model.load_state_dict(torch.load("best_model_cls.pt",
                                     map_location=device,
                                     weights_only=True))
    test_loss, test_acc, test_fsr = evaluate_classification(
        model, test_loader, device, max_dim, window
    )
    print("\n✅ Final Test Metrics:")
    print(f"Test Loss: {test_loss:.6f} | "
          f"Test Accuracy: {test_acc:.4f} | "
          f"Test FSR: {test_fsr:.4f}")

# ==============================
# 主入口：所有参数集中管理
# ==============================
if __name__ == "__main__":
    PARAMS = {
        "seed": 41,
        "folder": ".",
        "num_files": 10,
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-3,
        "max_dim": 30,
        "window": 20,      # 分类窗口 [-5,5]
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2
    }

    set_seed(PARAMS["seed"])
    train_model_by_files(
        folder=PARAMS["folder"],
        num_files=PARAMS["num_files"],
        epochs=PARAMS["epochs"],
        batch_size=PARAMS["batch_size"],
        lr=PARAMS["lr"],
        max_dim=PARAMS["max_dim"],
        window=PARAMS["window"],
        d_model=PARAMS["d_model"],
        nhead=PARAMS["nhead"],
        num_layers=PARAMS["num_layers"]
    )
