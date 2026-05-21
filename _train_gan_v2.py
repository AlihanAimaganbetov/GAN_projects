"""WGAN-GP 1D на «настоящих» вибро-окнах после DC-вычитания."""
import os, sys, io, time
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

data = np.load(os.path.join(ROOT, "_dataset_v2.npz"))
X_tr = torch.from_numpy(data["X_tr"])
X_val = torch.from_numpy(data["X_val"])
mean = data["mean"]; std = data["std"]
W = int(data["W"]); C = X_tr.shape[1]
print(f"X_tr {tuple(X_tr.shape)}  W={W} C={C}")

BATCH = 64
EPOCHS = 80
Z_DIM = 32
LAMBDA_GP = 10.0
N_CRITIC = 3        # на CPU экономим
LR = 1e-4
loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH, shuffle=True, drop_last=True)

class Generator(nn.Module):
    def __init__(self, z=32, c=4):
        super().__init__()
        self.proj = nn.Linear(z, 64 * 4)
        self.bn0 = nn.BatchNorm1d(64)
        self.up1 = nn.ConvTranspose1d(64, 48, 4, 2, 1)   # 4 -> 8
        self.bn1 = nn.BatchNorm1d(48)
        self.up2 = nn.ConvTranspose1d(48, 32, 4, 2, 1)   # 8 -> 16
        self.bn2 = nn.BatchNorm1d(32)
        self.up3 = nn.ConvTranspose1d(32, c, 4, 2, 1)    # 16 -> 32
    def forward(self, z):
        x = self.proj(z).view(-1, 64, 4)
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        return self.up3(x)

class Critic(nn.Module):
    """WGAN-критик — без BatchNorm, с LayerNorm. Sigmoid не нужен."""
    def __init__(self, c=4):
        super().__init__()
        self.c1 = nn.Conv1d(c, 32, 4, 2, 1);   self.n1 = nn.LayerNorm([32, 16])
        self.c2 = nn.Conv1d(32, 64, 4, 2, 1);  self.n2 = nn.LayerNorm([64, 8])
        self.c3 = nn.Conv1d(64, 128, 4, 2, 1); self.n3 = nn.LayerNorm([128, 4])
        self.head = nn.Linear(128 * 4, 1)
    def forward(self, x):
        x = F.leaky_relu(self.n1(self.c1(x)), 0.2)
        x = F.leaky_relu(self.n2(self.c2(x)), 0.2)
        x = F.leaky_relu(self.n3(self.c3(x)), 0.2)
        return self.head(x.flatten(1))

G = Generator(Z_DIM, C).to(DEVICE)
D = Critic(C).to(DEVICE)
print(f"G params: {sum(p.numel() for p in G.parameters()):,} | "
      f"D params: {sum(p.numel() for p in D.parameters()):,}")

optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
optD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

def gradient_penalty(real, fake):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, device=DEVICE)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(d_interp, interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

t0 = time.time()
log = []
for ep in range(EPOCHS):
    g_l, d_l, wd = [], [], []
    for step, (xb,) in enumerate(loader):
        xb = xb.to(DEVICE)
        bs = xb.size(0)
        # ===== D (critic) =====
        for _ in range(N_CRITIC):
            optD.zero_grad()
            z = torch.randn(bs, Z_DIM, device=DEVICE)
            fake = G(z).detach()
            d_real = D(xb)
            d_fake = D(fake)
            gp = gradient_penalty(xb, fake)
            loss_d = -(d_real.mean() - d_fake.mean()) + LAMBDA_GP * gp
            loss_d.backward()
            optD.step()
            wd.append((d_real.mean() - d_fake.mean()).item())
            d_l.append(loss_d.item())
        # ===== G =====
        optG.zero_grad()
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        fake = G(z)
        loss_g = -D(fake).mean()
        loss_g.backward()
        optG.step()
        g_l.append(loss_g.item())

    if (ep + 1) % 5 == 0 or ep == 0:
        msg = (f"ep {ep+1:3d}/{EPOCHS} | D={np.mean(d_l):.3f} G={np.mean(g_l):.3f} "
               f"| W-dist={np.mean(wd):.3f} | {time.time()-t0:.0f}s")
        print(msg); log.append(msg)

torch.save({"G": G.state_dict(), "D": D.state_dict(),
            "mean": mean, "std": std, "z_dim": Z_DIM, "W": W, "C": C,
            "dc_removed": True, "log": log},
           os.path.join(ROOT, "_gan_v2.pt"))

G.eval()
with torch.no_grad():
    z = torch.randn(2048, Z_DIM, device=DEVICE)
    fake = G(z).cpu().numpy()
fake_denorm = fake * std.reshape(1, C, 1) + mean.reshape(1, C, 1)
np.save(os.path.join(ROOT, "_samples_v2.npy"), fake_denorm)
print(f"Готово: _gan_v2.pt, _samples_v2.npy {fake_denorm.shape}")
