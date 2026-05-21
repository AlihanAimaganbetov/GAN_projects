"""
Сравнение реальных и сгенерированных окон.
Считаем:
  - per-channel mean/std реального vs синтетического
  - средний спектр (FFT amplitude) на канал
  - расстояние Wasserstein-1 на каждом канале/тайм-степ-маржинале
  - сохраняем PNG-картинки
"""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"

data = np.load(os.path.join(ROOT, "_dataset.npz"))
X_tr = data["X_tr"]      # (N, C, T) нормированное
mean = data["mean"]; std = data["std"]
W = int(data["W"])
# денормализуем X_tr для сопоставления с _samples.npy (там уже денормировано)
X_real = X_tr * std.reshape(1, -1, 1) + mean.reshape(1, -1, 1)

X_fake = np.load(os.path.join(ROOT, "_samples.npy"))   # (M, C, T) денорм
print(f"real: {X_real.shape}  fake: {X_fake.shape}")
C = X_real.shape[1]
chans = ["x", "y", "z", "extra"]

# подвыборка для скорости
rng = np.random.default_rng(0)
real_sub = X_real[rng.choice(X_real.shape[0], min(4096, X_real.shape[0]), replace=False)]
fake_sub = X_fake[rng.choice(X_fake.shape[0], min(4096, X_fake.shape[0]), replace=False)]

print(f"\n=== per-channel статистика (mean ± std) ===")
print(f"{'канал':>8s}  {'real_mean':>10s} {'fake_mean':>10s}  {'real_std':>10s} {'fake_std':>10s}  {'W1':>8s}")
for c in range(C):
    rv = real_sub[:, c, :].flatten()
    fv = fake_sub[:, c, :].flatten()
    rm, rs = rv.mean(), rv.std()
    fm, fs = fv.mean(), fv.std()
    # для wasserstein берём подвыборку
    idx_r = rng.choice(rv.size, min(20000, rv.size), replace=False)
    idx_f = rng.choice(fv.size, min(20000, fv.size), replace=False)
    w1 = wasserstein_distance(rv[idx_r], fv[idx_f])
    print(f"{chans[c]:>8s}  {rm:>10.4f} {fm:>10.4f}  {rs:>10.4f} {fs:>10.4f}  {w1:>8.4f}")

# спектры
def avg_spectrum(X):
    # X: (N, C, T) -> avg |FFT| на канал
    F = np.fft.rfft(X, axis=2)
    A = np.abs(F).mean(axis=0)  # (C, T/2+1)
    return A

Sr = avg_spectrum(real_sub)
Sf = avg_spectrum(fake_sub)
freqs = np.fft.rfftfreq(W, d=1/50.0)

print(f"\nГрафики сохраняю...")

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for c in range(C):
    ax = axes[0, c]
    # пример: 8 окон реальных и 8 синтетических
    for i in range(8):
        ax.plot(real_sub[i, c, :], "b-", alpha=0.4, lw=0.8)
        ax.plot(fake_sub[i, c, :], "r-", alpha=0.4, lw=0.8)
    ax.set_title(f"{chans[c]}  blue=real  red=fake")
    ax.set_xlabel("t (samples @ 50Hz)")
    ax.grid(alpha=0.3)

    ax = axes[1, c]
    ax.plot(freqs, Sr[c], "b-", label="real", lw=1.5)
    ax.plot(freqs, Sf[c], "r--", label="fake", lw=1.5)
    ax.set_title(f"{chans[c]} avg |FFT|")
    ax.set_xlabel("Hz")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(ROOT, "_compare.png")
plt.savefig(out, dpi=110)
print(f"  {out}")

# гистограммы амплитуд по каналам
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for c in range(C):
    rv = real_sub[:, c, :].flatten()
    fv = fake_sub[:, c, :].flatten()
    lo = min(rv.min(), fv.min())
    hi = max(rv.max(), fv.max())
    bins = np.linspace(lo, hi, 60)
    axes[c].hist(rv, bins=bins, alpha=0.5, label="real", density=True, color="b")
    axes[c].hist(fv, bins=bins, alpha=0.5, label="fake", density=True, color="r")
    axes[c].set_title(chans[c])
    axes[c].legend(fontsize=8)
    axes[c].grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(ROOT, "_hist.png")
plt.savefig(out, dpi=110)
print(f"  {out}")

print("\nГотово.")
