"""
Генератор синтетических вибро-окон.

Использование:
    python generate.py                       # 16 окон → samples.csv
    python generate.py --n 1000              # 1000 окон
    python generate.py --n 64 --out out.csv  # свой путь

CSV: одна колонка t, и по 4 колонки на окно (x_i,y_i,z_i,e_i, i=0..N-1).
Параметры модели: см. _train_gan_v2.py.
"""
import os, sys, io, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "_gan_v2.pt")

class Generator(nn.Module):
    def __init__(self, z=32, c=4):
        super().__init__()
        self.proj = nn.Linear(z, 64 * 4)
        self.bn0 = nn.BatchNorm1d(64)
        self.up1 = nn.ConvTranspose1d(64, 48, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(48)
        self.up2 = nn.ConvTranspose1d(48, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.up3 = nn.ConvTranspose1d(32, c, 4, 2, 1)
    def forward(self, z):
        x = self.proj(z).view(-1, 64, 4)
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        return self.up3(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "samples.csv"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    z_dim = ckpt["z_dim"]; W = ckpt["W"]; C = ckpt["C"]
    mean = np.asarray(ckpt["mean"]); std = np.asarray(ckpt["std"])

    G = Generator(z=z_dim, c=C)
    G.load_state_dict(ckpt["G"]); G.eval()

    with torch.no_grad():
        z = torch.randn(args.n, z_dim)
        fake = G(z).cpu().numpy()                      # (N, C, T) z-score
    fake = fake * std.reshape(1, C, 1) + mean.reshape(1, C, 1)  # (N, C, T) DC-вычтенные значения

    # сохранение CSV: широкий формат
    header = ["t_ms"]
    for i in range(args.n):
        header += [f"x_{i}", f"y_{i}", f"z_{i}", f"extra_{i}"]
    rows = []
    for t in range(W):
        row = [t * 20]  # 50 Hz -> 20 мс на шаг
        for i in range(args.n):
            row += [f"{fake[i, 0, t]:.5f}", f"{fake[i, 1, t]:.5f}",
                    f"{fake[i, 2, t]:.5f}", f"{fake[i, 3, t]:.5f}"]
        rows.append(",".join(str(x) for x in row))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"Сгенерировано {args.n} окон по {W} отсчётов → {args.out}")
    print(f"Внимание: значения — DC-вычтенные (отклонение от среднего по окну).")
    print(f"Чтобы получить «сырые» — прибавьте константу-смещение, например (x,y,z) = (0,0,1.14g).")

if __name__ == "__main__":
    main()
