"""v2: жёсткий порог + нормализация по окну."""
import os, glob, sys, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"
W = 32
HOP = 4              # ещё чаще нарезаем — больше окон
THR = 0.05           # высокий порог — только реальные события
GAP = 25
SEED = 42

def parse_imu(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            _, p = line.split("|", 1)
            p = p.strip()
            if any(t in p for t in ("rst:","load:","boot:","configsip","clk_drv","mode:")):
                continue
            parts = p.split(",")
            if len(parts) != 5:
                continue
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            out.append(vals)
    return np.array(out, dtype=np.float32)

def find_segments(extras, thr=THR, gap=GAP, min_len=W):
    active = extras > thr
    segs = []
    i = 0; N = len(active)
    while i < N:
        if not active[i]:
            i += 1; continue
        j = i; last = i; g = 0
        while j < N and g <= gap:
            if active[j]:
                last = j; g = 0
            else:
                g += 1
            j += 1
        if last + 1 - i >= min_len:
            segs.append((i, last + 1))
        i = j
    return segs

imu_files = sorted(glob.glob(os.path.join(ROOT, "imu_data_*.txt")))
windows = []
src_amplitude = []  # пиковая |extra| для conditional / классификации
for p in imu_files:
    arr = parse_imu(p)
    if len(arr) < W:
        continue
    segs = find_segments(arr[:, 3])
    if not segs:
        continue
    for s, e in segs:
        for k in range(s, e - W + 1, HOP):
            w = arr[k:k+W].copy()
            src_amplitude.append(w[:, 3].max())
            # вычитаем средний по окну (DC) по каждому каналу
            w = w - w.mean(axis=0, keepdims=True)
            windows.append(w)

X = np.stack(windows, axis=0)
amp = np.array(src_amplitude, dtype=np.float32)
print(f"Окон: {X.shape[0]:,}  shape={X.shape}")
print(f"Распределение пик-extra: min={amp.min():.3f} med={np.median(amp):.3f} "
      f"max={amp.max():.3f}")

rng = np.random.default_rng(SEED)
idx = rng.permutation(X.shape[0])
n_val = max(128, X.shape[0] // 20)
X_val = X[idx[:n_val]]; X_tr = X[idx[n_val:]]
amp_tr = amp[idx[n_val:]]

mean = X_tr.reshape(-1, 4).mean(axis=0)
std = X_tr.reshape(-1, 4).std(axis=0)
std[std < 1e-6] = 1e-6
print(f"mean: {mean}")
print(f"std:  {std}")

X_tr_n = ((X_tr - mean) / std).transpose(0, 2, 1).astype(np.float32)
X_val_n = ((X_val - mean) / std).transpose(0, 2, 1).astype(np.float32)

np.savez(os.path.join(ROOT, "_dataset_v2.npz"),
         X_tr=X_tr_n, X_val=X_val_n,
         mean=mean.astype(np.float32), std=std.astype(np.float32),
         amp_tr=amp_tr,
         W=W, dc_removed=True)
print(f"Сохранено: _dataset_v2.npz   X_tr: {X_tr_n.shape}  X_val: {X_val_n.shape}")
