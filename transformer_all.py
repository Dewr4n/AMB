import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from glob import glob
import random

########################################
# 固定随机种子
########################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed fixed to {seed}")

########################################
# 工具函数
########################################
def parse_vector_1xn(s, n_expected, max_dim):
    if not isinstance(s, str):
        return np.zeros(max_dim, dtype=np.float32)
    vals = [x for x in s.split(";") if x.strip() != ""]
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except:
            out.append(0.0)
    out = out[:n_expected]
    if len(out) < max_dim:
        out += [0.0] * (max_dim - len(out))
    else:
        out = out[:max_dim]
    return np.array(out, dtype=np.float32)


def parse_vector_1xn_with_none(s, n_expected, max_dim):
    fixed = np.zeros(max_dim, dtype=np.float32)
    valid = np.zeros(max_dim, dtype=np.int64)
    if not isinstance(s, str):
        return fixed, valid
    vals = [v for v in s.split(";") if v.strip() != ""]
    for i in range(min(len(vals), max_dim, n_expected)):
        raw = vals[i].strip()
        if raw.lower() == "none":
            valid[i] = 0
        else:
            try:
                fixed[i] = float(raw)
                valid[i] = 1
            except:
                valid[i] = 0
    return fixed, valid


def parse_matrix_nxn(s, n_expected, max_dim):
    if not isinstance(s, str):
        return np.zeros((max_dim, max_dim), dtype=np.float32)
    rows = [r for r in s.split("I") if r.strip() != ""]
    mat = []
    for r in rows:
        parts = [p for p in r.split(";") if p.strip() != ""]
        rv = []
        for p in parts:
            try:
                rv.append(float(p))
            except:
                rv.append(0.0)
        mat.append(rv)
    arr = np.array(mat, dtype=np.float32) if len(mat) > 0 else np.zeros((0, 0), dtype=np.float32)
    n = min(n_expected, arr.shape[0], arr.shape[1])
    cut = arr[:n, :n]
    out = np.zeros((max_dim, max_dim), dtype=np.float32)
    out[:cut.shape[0], :cut.shape[1]] = cut
    return out


########################################
# Dataset（保持原样）
########################################
class AmbiguityDataset(Dataset):
    def __init__(self, file_list, max_dim=30, window=20):
        self.samples = []
        self.max_dim = max_dim
        self.window = window
        self.k_corr = 3

        for file in file_list:
            try:
                df = pd.read_csv(file)
            except:
                continue

            if "label" not in df.columns:
                continue

            df.columns = [c.strip() for c in df.columns]
            has_shk = all(c in df.columns for c in ["NumSD_shk","Nfloat_shk","Nfixed_shk","Q_shk"])
            has_orig= all(c in df.columns for c in ["NumSD_orig","Nfloat_orig","Nfixed_matched","Q_orig"])

            for _, row in df.iterrows():
                lbl = int(row["label"])

                if lbl == 1 and not has_shk:
                    continue
                if lbl == 0 and not has_orig:
                    continue

                if lbl == 1:
                    n_raw = int(float(row["NumSD_shk"]))
                    if n_raw <= 4:
                        continue
                    n = min(n_raw, max_dim)
                    nfloat_vec = parse_vector_1xn(row["Nfloat_shk"], n, max_dim)
                    nfixed_vec = parse_vector_1xn(row["Nfixed_shk"], n, max_dim)
                    q_mat = parse_matrix_nxn(row["Q_shk"], n, max_dim)
                else:
                    n_raw = int(float(row["NumSD_orig"]))
                    if n_raw <= 4:
                        continue
                    n = min(n_raw, max_dim)
                    nfloat_vec = parse_vector_1xn(row["Nfloat_orig"], n, max_dim)
                    nfixed_vec, valid_mask = parse_vector_1xn_with_none(row["Nfixed_matched"], n, max_dim)
                    q_mat = parse_matrix_nxn(row["Q_orig"], n, max_dim)

                if n > 0:
                    Q_real = q_mat[:n, :n]
                    Qii = np.diag(Q_real)
                    eps = 1e-6
                    sigma = np.sqrt(np.clip(Qii, eps, None))
                    denom = np.sqrt(np.outer(Qii, Qii)) + eps
                    corr_mat = Q_real / denom
                    trace_Q = float(Qii.sum())
                else:
                    sigma = np.zeros(n)
                    corr_mat = np.zeros((n, n))
                    trace_Q = 0.0

                feat_dim = 5 + self.k_corr
                tokens = np.zeros((max_dim, feat_dim), dtype=np.float32)

                for i in range(max_dim):
                    if i < n:
                        tokens[i, 0] = nfloat_vec[i]
                        tokens[i, 1] = sigma[i]

                        row_corr = corr_mat[i].copy() if n > 1 else np.zeros(n)
                        if n > 1:
                            row_corr[i] = 0.0

                        abs_row = np.abs(row_corr)
                        sorted_idx = np.argsort(abs_row)[::-1]
                        for k in range(min(self.k_corr, n-1)):
                            tokens[i, 2+k] = row_corr[sorted_idx[k]]

                        tokens[i, 2+self.k_corr] = trace_Q
                        tokens[i, 3+self.k_corr] = float(n_raw)
                        tokens[i, 4+self.k_corr] = float(i)/float(max_dim)

                x_seq = tokens.astype(np.float32)

                base_int = np.round(nfloat_vec)
                delta = (nfixed_vec - base_int).astype(int)
                delta = np.clip(delta, -self.window, self.window)
                y_cls = (delta + self.window).astype(np.int64)

                if lbl == 1:
                    valid_mask = np.ones(max_dim, dtype=np.int64)
                    valid_mask[n:] = 0

                self.samples.append((x_seq, y_cls, n, nfloat_vec, nfixed_vec, valid_mask, lbl))

        print(f"✅ Loaded {len(self.samples)} samples (max_dim={self.max_dim}, window=±{self.window}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, n, nf, nfx, mask, lbl = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(n, dtype=torch.long),
            torch.from_numpy(nf),
            torch.from_numpy(nfx),
            torch.from_numpy(mask),
            torch.tensor(lbl, dtype=torch.long)
        )


########################################
# Transformer 模型
########################################
class FixingTransformerCLS(nn.Module):
    def __init__(self, feat_dim, max_dim, n_class,
                 d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.max_dim = max_dim
        self.n_class = n_class

        self.proj = nn.Linear(feat_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_layer = nn.Linear(d_model, n_class)

    def forward(self, x):
        B, D, F = x.size()
        x_emb = self.proj(x) + self.pos_embedding[:, :D, :]
        h = self.transformer(x_emb)
        out = self.out_layer(h)
        return out


########################################
# evaluate（仅过滤 label=1）
########################################
def evaluate_classification(model, loader, device, max_dim, window):
    model.eval()
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    total_loss = 0.0
    total_correct = 0
    total_cnt = 0
    total_fsr_count = 0
    total_fsr_total = 0

    with torch.no_grad():
        for batch in loader:
            x, y, n, nfloat, nfixed, valid_mask, lbl = batch

            keep = (lbl == 0)
            if keep.sum() == 0:
                continue

            x = x[keep].to(device).float()
            y = y[keep].to(device)
            n = n[keep].to(device)
            valid_mask = valid_mask[keep].to(device).float()
            lbl = lbl[keep].to(device)

            logits = model(x)
            B, D, C = logits.shape

            dim_mask = (torch.arange(D, device=device).unsqueeze(0) < n.unsqueeze(1)).float()
            full_mask = dim_mask * valid_mask

            loss_mat = ce_loss(logits.transpose(1, 2), y)
            loss_per_sample = (loss_mat * full_mask).sum(dim=1) / (full_mask.sum(dim=1) + 1e-6)
            loss = loss_per_sample.mean()

            total_loss += loss.item() * B

            pred = logits.argmax(dim=-1)
            correct = ((pred == y) * full_mask.bool()).sum().item()
            cnt = full_mask.sum().item()

            total_correct += correct
            total_cnt += cnt

            for b in range(B):
                vm = full_mask[b] > 0
                if vm.sum() == 0:
                    continue
                if torch.all(pred[b][vm] == y[b][vm]):
                    total_fsr_count += 1
                total_fsr_total += 1

    if total_cnt == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_cnt
    fsr = total_fsr_count / (total_fsr_total + 1e-9)
    return avg_loss, accuracy, fsr


########################################
# 原始统计函数（不改）
########################################
def diff_statistics(dataset, name):

    def to_numpy(a):
        """统一转成 numpy，无论输入是 tensor / numpy / list 都能处理"""
        if isinstance(a, np.ndarray):
            return a
        if torch.is_tensor(a):
            return a.cpu().numpy()
        return np.array(a)

    diffs = []
    label0_epochs = 0
    all_round_epochs = 0

    for x, y, n, nfloat, nfixed, valid_mask, lbl in dataset:

        # ★ 只统计 label=0
        if lbl != 0:
            continue

        label0_epochs += 1
        n = int(n.item()) if torch.is_tensor(n) else int(n)
        if n <= 0:
            continue

        vm = to_numpy(valid_mask[:n]).astype(bool)
        if vm.sum() == 0:
            continue

        # ★ 统一转 numpy 避免类型冲突
        nf = to_numpy(nfloat[:n])[vm]
        fx = to_numpy(nfixed[:n])[vm]

        rounded = np.round(nf)
        diff = fx - rounded

        diffs.extend(diff.tolist())

        # ★ 全部 round
        if np.all(diff == 0):
            all_round_epochs += 1

    if len(diffs) == 0:
        print(f"\n⚠️ {name} 集 label=0 没有可用维度，跳过统计。")
        return

    diffs = np.array(diffs, dtype=int)
    uniq, cnts = np.unique(diffs, return_counts=True)
    total = cnts.sum()

    print(f"\n📊 {name} diff stats (仅 label=0)")
    print(f"  范围: [{uniq.min()} , {uniq.max()}] | 总维数: {total}")
    print("  差值 | 计数 | 占比(%)")
    print("  ----------------------")
    for v, c in zip(uniq, cnts):
        print(f"  {v:>4d} | {c:>8d} | {c/total*100:>6.2f}%")
    zero_ratio = cnts[uniq == 0].sum() / total * 100 if 0 in uniq else 0
    print("  ----------------------")
    print(f"  零差比例: {zero_ratio:.2f}%")

    print(f"\n  ✔ label=0 历元总数: {label0_epochs}")
    print(f"  ✔ 全部 round 正确的历元数: {all_round_epochs} "
          f"({all_round_epochs/label0_epochs*100:.2f}%)\n")

########################################
# 训练（唯一改动：过滤 label=1）
########################################
def train_model_by_files(folder, num_files, epochs, batch_size,
                         lr, max_dim, window, d_model, nhead, num_layers):

    all_files = glob(os.path.join(folder, "*.csv"))
    random.shuffle(all_files)
    all_files = all_files[:num_files]

    print(f"\n📦 Using {len(all_files)} CSV files.")

    n_train = int(len(all_files) * 0.7)
    n_val   = int(len(all_files) * 0.15)
    n_test  = len(all_files) - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train+n_val]
    test_files  = all_files[n_train+n_val:]

    print(f"Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")

    train_set = AmbiguityDataset(train_files, max_dim=max_dim, window=window)
    val_set   = AmbiguityDataset(val_files,   max_dim=max_dim, window=window)
    test_set  = AmbiguityDataset(test_files,  max_dim=max_dim, window=window)

    print("\n=========== Train diff stats ===========")
    diff_statistics(train_set, "Train")
    print("\n=========== Val diff stats ===========")
    diff_statistics(val_set, "Val")
    print("\n=========== Test diff stats ===========")
    diff_statistics(test_set, "Test")

    def collate_fn(batch):
        xs, ys, ns, nfs, nfxs, masks, lbls = zip(*batch)
        return (
            torch.stack(xs),
            torch.stack(ys),
            torch.stack(ns),
            torch.stack(nfs),
            torch.stack(nfxs),
            torch.stack(masks),
            torch.stack(lbls),
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_x, _, _, _, _, _, _ = train_set[0]
    feat_dim = sample_x.shape[-1]
    n_class = 2 * window + 1

    model = FixingTransformerCLS(
        feat_dim, max_dim, n_class,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Using device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, y, n, nfloat, nfixed, valid_mask, lbl = batch

            keep = (lbl == 0)
            if keep.sum() == 0:
                continue

            x = x[keep].to(device).float()
            y = y[keep].to(device)
            n = n[keep].to(device)
            valid_mask = valid_mask[keep].to(device).float()
            lbl = lbl[keep].to(device)

            logits = model(x)
            B, D, C = logits.shape

            dim_mask = (torch.arange(D, device=device).unsqueeze(0) < n.unsqueeze(1)).float()
            full_mask = dim_mask * valid_mask

            loss_mat = ce_loss(logits.transpose(1, 2), y)
            loss_per_sample = (loss_mat * full_mask).sum(1) / (full_mask.sum(1) + 1e-6)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

        train_loss = total_loss / len(train_loader.dataset)

        val_loss, val_acc, val_fsr = evaluate_classification(
            model, val_loader, device, max_dim, window
        )

        print(f"Epoch {ep:02d}/{epochs} | "
              f"TrainLoss: {train_loss:.6f} | "
              f"ValLoss: {val_loss:.6f} | ValAcc: {val_acc:.4f} | ValFSR: {val_fsr:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_label0_only.pt")

    model.load_state_dict(torch.load("best_label0_only.pt", map_location=device, weights_only=True))

    test_loss, test_acc, test_fsr = evaluate_classification(
        model, test_loader, device, max_dim, window
    )

    print("\n=========== Final Test Metrics (label=0 only) ===========")
    print(f"TestLoss: {test_loss:.6f} | TestAcc: {test_acc:.4f} | TestFSR: {test_fsr:.4f}")


########################################
# main（保持原样）
########################################
if __name__ == "__main__":
    PARAMS = {
        "seed": 41,
        "folder": ".",
        "num_files": 200,
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-3,
        "max_dim": 30,
        "window": 20,
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
    }

    set_seed(PARAMS["seed"])

    args = PARAMS.copy()
    args.pop("seed")

    train_model_by_files(**args)
