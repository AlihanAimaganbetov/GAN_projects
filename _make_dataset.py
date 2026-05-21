"""
Нарезает датасет окон для GAN.
- 4 канала: x, y, z, extra
- окно W=32 (0.64 сек @50Hz)
- активные окна по порогу extra > 0.015 (обнаружение с GAP=25 склейкой)
- сохраняем нормировку (mean/std по тренировке) и сами окна в .npz
"""
import os, glob, sys, io, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"
W = 32
HOP = 8           # 75% overlap
THR = 0.015
GAP = 25          # склейка сегментов
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
    return np.array(out, dtype=np.float32)  # shape (N, 4)

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
print(f"Файлов: {len(imu_files)}")

windows = []
files_with_data = 0
for p in imu_files:
    arr = parse_imu(p)
    if len(arr) < W:
        continue
    segs = find_segments(arr[:, 3])
    if not segs:
        continue
    files_with_data += 1
    for s, e in segs:
        # вычитаем средний z (гравитация) внутри окна — оставляем только динамику
        # Нет: лучше не вычитать локально, иначе теряем DC. Глобальной нормировки достаточно.
        for k in range(s, e - W + 1, HOP):
            windows.append(arr[k:k+W])

X = np.stack(windows, axis=0)        # (N, W, 4)
print(f"Окон собрано: {X.shape[0]:,}  shape={X.shape}")
print(f"Файлов внесли вклад: {files_with_data}/{len(imu_files)}")

# train/val
rng = np.random.default_rng(SEED)
idx = rng.permutation(X.shape[0])
n_val = max(64, X.shape[0] // 20)
val_idx = idx[:n_val]
tr_idx = idx[n_val:]
X_tr = X[tr_idx]
X_val = X[val_idx]

# нормировка по тренировке: mean/std на канал, по всем окнам и временным точкам
mean = X_tr.reshape(-1, 4).mean(axis=0)
std = X_tr.reshape(-1, 4).std(axis=0)
std[std < 1e-6] = 1e-6
print(f"\nКаналы (x, y, z, extra):")
print(f"  mean: {mean}")
print(f"  std:  {std}")

X_tr_n = (X_tr - mean) / std
X_val_n = (X_val - mean) / std

# transpose -> (N, C, T) для PyTorch
X_tr_n = X_tr_n.transpose(0, 2, 1).astype(np.float32)
X_val_n = X_val_n.transpose(0, 2, 1).astype(np.float32)

np.savez(os.path.join(ROOT, "_dataset.npz"),
         X_tr=X_tr_n, X_val=X_val_n,
         mean=mean.astype(np.float32), std=std.astype(np.float32),
         W=W)

print(f"\nСохранено: _dataset.npz")
print(f"  X_tr:  {X_tr_n.shape} dtype={X_tr_n.dtype}")
print(f"  X_val: {X_val_n.shape}")
