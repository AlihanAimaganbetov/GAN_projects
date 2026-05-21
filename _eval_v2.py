"""Сравнение v2: данные DC-вычтены и фильтр extr > 0.05."""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"

data = np.load(os.path.join(ROOT, "_dataset_v2.npz"))
X_tr_n = data["X_tr"]
mean = data["mean"]; std = data["std"]
W = int(data["W"])
X_real = X_tr_n * std.reshape(1, -1, 1) + mean.reshape(1, -1, 1)

X_fake = np.load(os.path.join(ROOT, "_samples_v2.npy"))
print(f"real {X_real.shape}  fake {X_fake.shape}")
C = X_real.shape[1]; chans = ["x","y","z","extra"]

rng = np.random.default_rng(0)
n_real = min(X_real.shape[0], 2000)
n_fake = min(X_fake.shape[0], 2000)
real_sub = X_real[rng.choice(X_real.shape[0], n_real, replace=False)]
fake_sub = X_fake[rng.choice(X_fake.shape[0], n_fake, replace=False)]

print(f"\n{'канал':>8s}  {'real_mean':>10s} {'fake_mean':>10s}  "
      f"{'real_std':>10s} {'fake_std':>10s}  {'W1':>8s}")
for c in range(C):
    rv = real_sub[:, c, :].flatten()
    fv = fake_sub[:, c, :].flatten()
    w1 = wasserstein_distance(rv, fv)
    print(f"{chans[c]:>8s}  {rv.mean():>10.4f} {fv.mean():>10.4f}  "
          f"{rv.std():>10.4f} {fv.std():>10.4f}  {w1:>8.4f}")

def avg_spectrum(X):
    F = np.fft.rfft(X, axis=2)
    return np.abs(F).mean(axis=0)

Sr = avg_spectrum(real_sub); Sf = avg_spectrum(fake_sub)
freqs = np.fft.rfftfreq(W, d=1/50.0)

# 4×3 grid: для каждого канала — real trajectories, fake trajectories, spectrum
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for c in range(C):
    ax = axes[0, c]
    for i in range(12):
        ax.plot(real_sub[i, c, :], "-", alpha=0.5, lw=0.8)
    ax.set_title(f"{chans[c]} — REAL (DC вычтен)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, W-1)

    ax = axes[1, c]
    for i in range(12):
        ax.plot(fake_sub[i, c, :], "-", alpha=0.5, lw=0.8, color="C3")
    ax.set_title(f"{chans[c]} — FAKE")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, W-1)
    # подгоним ylim к real-диапазону
    ax.set_ylim(axes[0, c].get_ylim())

    ax = axes[2, c]
    ax.plot(freqs, Sr[c], "b-", label="real", lw=1.5)
    ax.plot(freqs, Sf[c], "r--", label="fake", lw=1.5)
    ax.set_title(f"{chans[c]} avg |FFT|")
    ax.set_xlabel("Hz")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(ROOT, "_compare_v2.png")
plt.savefig(out, dpi=110)
print(f"Сохранено: {out}")

# histograms
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for c in range(C):
    rv = real_sub[:, c, :].flatten()
    fv = fake_sub[:, c, :].flatten()
    lo = min(np.percentile(rv, 0.5), np.percentile(fv, 0.5))
    hi = max(np.percentile(rv, 99.5), np.percentile(fv, 99.5))
    bins = np.linspace(lo, hi, 60)
    axes[c].hist(rv, bins=bins, alpha=0.5, label="real", density=True, color="b")
    axes[c].hist(fv, bins=bins, alpha=0.5, label="fake", density=True, color="r")
    axes[c].set_title(chans[c])
    axes[c].legend(fontsize=8); axes[c].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "_hist_v2.png"), dpi=110)
print(f"Сохранено: _hist_v2.png")
