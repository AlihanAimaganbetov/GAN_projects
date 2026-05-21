"""
Разведка вибро-сегментов в IMU-логах.
Идея: для каждой строки есть 5-я колонка (extra) — судя по данным, это уже
магнитуда вибрации/jerk. В покое она ~0.008, при движении уходит к 0.1...1.5.

Алгоритм:
  1. Парсим файл.
  2. Считаем bgnd-уровень: медиана extra по первым ~1000 отсчётов (или по всем,
     если активности < 5%).
  3. Порог = max(3 * bgnd_mad + bgnd_median, 0.03).
  4. Группируем подряд идущие активные отсчёты с зазором <= 50 (1 сек).
  5. Считаем длительности и сколько окон длиной W можно нарезать.

Записываем активные интервалы в JSON для следующего шага.
"""
import os, glob, json, statistics, sys, io
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"
WINDOW = 128         # отсчётов на окно (2.56 сек @ 50Hz)
HOP = 64             # перекрытие 50%
MIN_SEG_LEN = 32     # отбрасываем «искры» короче 0.64 сек
GAP_MERGE = 50       # склеиваем сегменты с разрывом <= 1 сек
FLOOR_ABS = 0.025    # абсолютный пол шума (на случай если фон сам шумный)

def parse_imu(path):
    """Возвращает список (idx, x, y, z, extra), пропуская boot-сообщения."""
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            _, payload = line.split("|", 1)
            payload = payload.strip()
            if any(tag in payload for tag in ("rst:", "load:", "boot:", "configsip",
                                              "clk_drv", "mode:")):
                continue
            parts = payload.split(",")
            if len(parts) != 5:
                continue
            try:
                _t = int(parts[0])
                vals = tuple(float(p) for p in parts[1:])
            except ValueError:
                continue
            out.append((len(out), *vals))
    return out

def mad(vals):
    if not vals:
        return 0.0
    m = statistics.median(vals)
    dev = [abs(v - m) for v in vals]
    return statistics.median(dev)

def detect_events(rows):
    """rows = [(i,x,y,z,extra), ...]; возвращает list of (start_idx, end_idx)."""
    if len(rows) < 100:
        return [], 0.0, 0.0
    extras = [r[4] for r in rows]
    # фон: оцениваем по всему массиву, но робастно
    bg_med = statistics.median(extras)
    bg_mad = mad(extras) or 1e-6
    thr = max(bg_med + 6.0 * bg_mad, FLOOR_ABS)
    active = [r[4] > thr for r in rows]

    # склеиваем
    segs = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and (active[j] or
                          (j - i > 0 and any(active[k] for k in range(j, min(j + GAP_MERGE, n))))):
            j += 1
        # точная правая граница: последний True в [i..j)
        right = i
        for k in range(i, j):
            if active[k]:
                right = k
        if right - i + 1 >= MIN_SEG_LEN:
            segs.append((i, right + 1))
        i = max(j, right + 1)
    return segs, bg_med, thr

def windows_in_segments(segs, win=WINDOW, hop=HOP):
    n = 0
    for s, e in segs:
        L = e - s
        if L < win:
            continue
        n += 1 + (L - win) // hop
    return n

imu_files = sorted(glob.glob(os.path.join(ROOT, "imu_data_*.txt")))
report = {"files": [], "total_rows": 0, "total_active_rows": 0,
          "total_windows": 0, "window": WINDOW, "hop": HOP}

print(f"Сканирую {len(imu_files)} IMU-файлов...\n")
print(f"{'файл':45s}  {'строк':>7s} {'активн':>7s} {'%':>5s} {'сегм':>5s} {'окон':>5s} {'thr':>7s}")
print("-" * 90)

for p in imu_files:
    rows = parse_imu(p)
    segs, bg_med, thr = detect_events(rows)
    act_rows = sum(e - s for s, e in segs)
    win_cnt = windows_in_segments(segs)
    pct = 100.0 * act_rows / max(len(rows), 1)
    name = os.path.basename(p)
    print(f"{name:45s}  {len(rows):>7d} {act_rows:>7d} {pct:>5.1f} {len(segs):>5d} {win_cnt:>5d} {thr:>7.4f}")
    report["files"].append({
        "name": name,
        "rows": len(rows),
        "segments": segs,
        "active_rows": act_rows,
        "windows": win_cnt,
        "bg_median": bg_med,
        "threshold": thr,
    })
    report["total_rows"] += len(rows)
    report["total_active_rows"] += act_rows
    report["total_windows"] += win_cnt

print("-" * 90)
tot = report["total_rows"]
act = report["total_active_rows"]
wins = report["total_windows"]
print(f"\nВсего отсчётов:   {tot:,}")
print(f"Активных:         {act:,} ({100*act/max(tot,1):.2f}%)")
print(f"Окон по {WINDOW}:      {wins:,}  (overlap hop={HOP})")
print(f"\n=> для GAN это {'много' if wins > 5000 else 'мало (нужен оверсэмплинг/аугментация)' if wins > 500 else 'недостаточно — собирай данные'}")

with open(os.path.join(ROOT, "_segments.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nОтчёт сохранён: _segments.json")
