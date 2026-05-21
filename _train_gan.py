"""
Маленький 1D DCGAN для генерации вибро-окон.
Вход: (B, 4, 32) — каналы x,y,z,extra; T=32 (0.64 сек @50Hz).
"""
import os, sys, io, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"
DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- данные ----
data = np.load(os.path.join(ROOT, "_dataset.npz"))
X_tr = torch.from_numpy(data["X_tr"])     # (N, 4, 32)
X_val = torch.from_numpy(data["X_val"])
mean = torch.from_numpy(data["mean"])
std = torch.from_numpy(data["std"])
W = int(data["W"])
C = X_tr.shape[1]
print(f"X_tr {tuple(X_tr.shape)}, X_val {tuple(X_val.shape)}, W={W}, C={C}")

BATCH = 256
EPOCHS = 8
Z_DIM = 32
LR_G = 2e-4
LR_D = 1e-4
loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=0)

# ---- модели ----
class Generator(nn.Module):
    def __init__(self, z=32, c=4, t=32):
        super().__init__()
        self.t0 = 4  # стартовая длина временного измерения
        self.proj = nn.Linear(z, 128 * self.t0)
        self.bn0 = nn.BatchNorm1d(128)
        # 4 -> 8 -> 16 -> 32
        self.up1 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.up2 = nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.up3 = nn.ConvTranspose1d(32, c, 4, stride=2, padding=1)
    def forward(self, z):
        x = self.proj(z).view(-1, 128, self.t0)
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        return self.up3(x)  # линейный выход — данные не bounded

class Discriminator(nn.Module):
    def __init__(self, c=4):
        super().__init__()
        self.c1 = nn.Conv1d(c, 32, 4, stride=2, padding=1)   # 32 -> 16
        self.c2 = nn.Conv1d(32, 64, 4, stride=2, padding=1)  # 16 -> 8
        self.c3 = nn.Conv1d(64, 128, 4, stride=2, padding=1) # 8  -> 4
        self.head = nn.Linear(128 * 4, 1)
    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.c2(x), 0.2)
        x = F.leaky_relu(self.c3(x), 0.2)
        return self.head(x.flatten(1))

G = Generator(z=Z_DIM, c=C, t=W).to(DEVICE)
D = Discriminator(c=C).to(DEVICE)
nG = sum(p.numel() for p in G.parameters())
nD = sum(p.numel() for p in D.parameters())
print(f"G params: {nG:,} | D params: {nD:,}")

optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

# label smoothing + instance noise (decay)
def noise_sigma(epoch):
    return max(0.0, 0.1 * (1 - epoch / EPOCHS))

steps_per_epoch = len(loader)
print(f"Шагов в эпоху: {steps_per_epoch} | всего: {steps_per_epoch * EPOCHS}\n")

t0 = time.time()
log = []
for epoch in range(EPOCHS):
    sigma = noise_sigma(epoch)
    g_losses, d_losses, d_real_acc, d_fake_acc = [], [], [], []
    for step, (xb,) in enumerate(loader):
        xb = xb.to(DEVICE)
        bs = xb.size(0)

        # ===== D =====
        optD.zero_grad()
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        fake = G(z).detach()
        if sigma > 0:
            xb_n = xb + sigma * torch.randn_like(xb)
            fake_n = fake + sigma * torch.randn_like(fake)
        else:
            xb_n, fake_n = xb, fake
        d_real = D(xb_n)
        d_fake = D(fake_n)
        loss_d_real = bce(d_real, torch.full_like(d_real, 0.9))
        loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optD.step()

        # ===== G =====
        optG.zero_grad()
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        fake = G(z)
        if sigma > 0:
            fake_in = fake + sigma * torch.randn_like(fake)
        else:
            fake_in = fake
        d_fake2 = D(fake_in)
        loss_g = bce(d_fake2, torch.ones_like(d_fake2))
        loss_g.backward()
        optG.step()

        g_losses.append(loss_g.item())
        d_losses.append(loss_d.item())
        d_real_acc.append((torch.sigmoid(d_real) > 0.5).float().mean().item())
        d_fake_acc.append((torch.sigmoid(d_fake) < 0.5).float().mean().item())

    elapsed = time.time() - t0
    msg = (f"epoch {epoch+1}/{EPOCHS} | "
           f"D={np.mean(d_losses):.3f} G={np.mean(g_losses):.3f} | "
           f"acc_real={np.mean(d_real_acc):.2f} acc_fake={np.mean(d_fake_acc):.2f} | "
           f"σ_noise={sigma:.3f} | {elapsed:.0f}s")
    print(msg)
    log.append(msg)

# ---- сохранение ----
ckpt = os.path.join(ROOT, "_gan.pt")
torch.save({
    "G": G.state_dict(),
    "D": D.state_dict(),
    "mean": mean.numpy(),
    "std": std.numpy(),
    "z_dim": Z_DIM, "W": W, "C": C,
    "log": log,
}, ckpt)
print(f"\nСохранено: {ckpt}")

# ---- генерация выборки ----
G.eval()
with torch.no_grad():
    z = torch.randn(2048, Z_DIM, device=DEVICE)
    fake = G(z).cpu().numpy()
fake_denorm = fake * std.numpy().reshape(1, C, 1) + mean.numpy().reshape(1, C, 1)
np.save(os.path.join(ROOT, "_samples.npy"), fake_denorm)
print(f"Сгенерировано 2048 окон: _samples.npy {fake_denorm.shape}")
