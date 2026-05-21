"""Пробуем разные пороги/окна — чтобы понять сколько данных мы можем добыть."""
import os, glob, statistics, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"

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
    return out

imu_files = sorted(glob.glob(os.path.join(ROOT, "imu_data_*.txt")))

# собрать все extras в один поток (по файлам)
print("Распределение extra по всему корпусу (квантили):")
all_extras = []
for p in imu_files:
    rs = parse_imu(p)
    all_extras.extend(r[3] for r in rs)
all_extras.sort()
n = len(all_extras)
for q in (0.50, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999):
    print(f"  q{q:.4f}: {all_extras[int(q*n)]:.4f}")
print(f"  max:    {all_extras[-1]:.4f}")
print(f"  всего:  {n:,}\n")

# теперь — сколько окон при разных порогах/окнах
print(f"{'thr':>6s}  {'>thr%':>6s}  {'win=32':>7s}  {'win=64':>7s}  {'win=128':>7s}")
for thr in (0.015, 0.020, 0.025, 0.030, 0.040, 0.060, 0.100):
    cnt_above = sum(1 for x in all_extras if x > thr)
    pct = 100.0 * cnt_above / n
    # пройти по файлам и склеить сегменты
    def windows_for(win, hop):
        total = 0
        for p in imu_files:
            rs = parse_imu(p)
            active = [r[3] > thr for r in rs]
            # сегментация склеиванием с зазором <= 25 (~0.5 сек)
            segs = []
            i = 0
            N = len(active)
            GAP = 25
            while i < N:
                if not active[i]:
                    i += 1; continue
                j = i; gap = 0
                last = i
                while j < N and gap <= GAP:
                    if active[j]:
                        last = j; gap = 0
                    else:
                        gap += 1
                    j += 1
                end = last + 1
                if end - i >= win:
                    segs.append((i, end))
                i = j
            for s, e in segs:
                L = e - s
                if L >= win:
                    total += 1 + (L - win) // hop
        return total
    w32 = windows_for(32, 16)
    w64 = windows_for(64, 32)
    w128 = windows_for(128, 64)
    print(f"{thr:>6.3f}  {pct:>5.2f}%  {w32:>7d}  {w64:>7d}  {w128:>7d}")

print("\nТакже посчитаем с аугментацией (time-shift):")
print("Если использовать hop=8 (87% overlap) на тех же сегментах при thr=0.025:")
for win in (32, 64, 128):
    total = 0
    for p in imu_files:
        rs = parse_imu(p)
        active = [r[3] > 0.025 for r in rs]
        segs = []
        i = 0; N = len(active); GAP = 25
        while i < N:
            if not active[i]:
                i += 1; continue
            j = i; last = i; gap = 0
            while j < N and gap <= GAP:
                if active[j]:
                    last = j; gap = 0
                else: gap += 1
                j += 1
            if last + 1 - i >= win:
                segs.append((i, last + 1))
            i = j
        for s, e in segs:
            if e - s >= win:
                total += 1 + (e - s - win) // 8
    print(f"  win={win}: {total} окон (hop=8)")
