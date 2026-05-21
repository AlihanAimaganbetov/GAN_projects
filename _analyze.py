import os, glob, statistics, re, sys, io
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\userus\Downloads\esp"

def parse_line(line):
    # формат: "YYYY-MM-DD HH:MM:SS.mmm | payload"
    if "|" not in line:
        return None, None
    ts_str, payload = line.split("|", 1)
    ts_str = ts_str.strip()
    payload = payload.strip()
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None, payload
    return ts, payload

def analyze_imu(path):
    rows = []
    bad = 0
    boot = 0
    first_ts = last_ts = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, payload = parse_line(line)
            if payload is None:
                bad += 1
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts if ts else last_ts
            if "rst:" in payload or "load:" in payload or "boot:" in payload or "configsip" in payload or "clk_drv" in payload or "mode:" in payload:
                boot += 1
                continue
            parts = payload.split(",")
            if len(parts) != 5:
                bad += 1
                continue
            try:
                tms = int(parts[0])
                x, y, z, extra = (float(p) for p in parts[1:])
                rows.append((tms, x, y, z, extra))
            except ValueError:
                bad += 1
    return rows, bad, boot, first_ts, last_ts

def analyze_lora(path):
    rows = []
    bad = 0
    boot = 0
    has_ms = None
    first_ts = last_ts = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, payload = parse_line(line)
            if payload is None:
                bad += 1
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts if ts else last_ts
            if "rst:" in payload or "load:" in payload or "boot:" in payload or "configsip" in payload or "clk_drv" in payload or "mode:" in payload:
                boot += 1
                continue
            parts = payload.split(",")
            try:
                if len(parts) == 3:
                    tms = int(parts[0]); a = float(parts[1]); b = float(parts[2])
                    rows.append((tms, a, b))
                    has_ms = True
                elif len(parts) == 2:
                    a = float(parts[0]); b = float(parts[1])
                    rows.append((None, a, b))
                    has_ms = False if has_ms is None else has_ms
                else:
                    bad += 1
            except ValueError:
                bad += 1
    return rows, bad, boot, first_ts, last_ts, has_ms

def desc(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }

def fmt(d, prec=4):
    if d is None:
        return "—"
    return f"n={d['n']} min={d['min']:.{prec}f} max={d['max']:.{prec}f} mean={d['mean']:.{prec}f} sd={d['stdev']:.{prec}f}"

imu_files = sorted(glob.glob(os.path.join(ROOT, "imu_data_*.txt")))
lora_files = sorted(glob.glob(os.path.join(ROOT, "lora_angles_*.txt")))

print(f"=== IMU: {len(imu_files)} файлов ===")
sample = imu_files[:3] + imu_files[-2:]
for p in sample:
    rows, bad, boot, t0, t1 = analyze_imu(p)
    name = os.path.basename(p)
    if not rows:
        print(f"\n[{name}] пусто, bad={bad}, boot={boot}")
        continue
    tms = [r[0] for r in rows]
    xs = [r[1] for r in rows]; ys = [r[2] for r in rows]; zs = [r[3] for r in rows]; es = [r[4] for r in rows]
    dur_s = (tms[-1] - tms[0])/1000.0 if len(tms) > 1 else 0
    rate = len(rows)/dur_s if dur_s > 0 else 0
    # шаг во времени
    diffs = [tms[i+1]-tms[i] for i in range(len(tms)-1)]
    gaps = [d for d in diffs if d > 200]
    print(f"\n[{name}] строк_данных={len(rows)} bad={bad} boot={boot} длит={dur_s:.1f}s ~{rate:.1f}Hz пропусков>200мс={len(gaps)}")
    print(f"  x: {fmt(desc(xs))}")
    print(f"  y: {fmt(desc(ys))}")
    print(f"  z: {fmt(desc(zs))}")
    print(f"  ex:{fmt(desc(es))}")
    if diffs:
        print(f"  dt(ms): min={min(diffs)} max={max(diffs)} mean={statistics.fmean(diffs):.1f}")

print(f"\n=== LoRa: {len(lora_files)} файлов ===")
sample = lora_files[:1] + lora_files[len(lora_files)//2-1:len(lora_files)//2+1] + lora_files[-2:]
for p in sample:
    rows, bad, boot, t0, t1, has_ms = analyze_lora(p)
    name = os.path.basename(p)
    if not rows:
        print(f"\n[{name}] пусто, bad={bad}, boot={boot}")
        continue
    a = [r[1] for r in rows]; b = [r[2] for r in rows]
    extra = ""
    if rows[0][0] is not None:
        tms = [r[0] for r in rows if r[0] is not None]
        dur_s = (tms[-1]-tms[0])/1000.0 if len(tms) > 1 else 0
        rate = len(tms)/dur_s if dur_s > 0 else 0
        diffs = [tms[i+1]-tms[i] for i in range(len(tms)-1)]
        gaps = [d for d in diffs if d > 200]
        extra = f" длит={dur_s:.1f}s ~{rate:.1f}Hz проп>200мс={len(gaps)}"
    print(f"\n[{name}] строк_данных={len(rows)} bad={bad} boot={boot} ms_ts={has_ms}{extra}")
    print(f"  a: {fmt(desc(a),2)}")
    print(f"  b: {fmt(desc(b),2)}")

# глобальная статистика по boot/перезагрузкам
print("\n=== Сводка перезагрузок (boot:) ===")
total_boot_lora = 0
files_with_boot = 0
for p in lora_files:
    cnt = 0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "POWERON_RESET" in line:
                cnt += 1
    if cnt:
        files_with_boot += 1
        total_boot_lora += cnt
print(f"LoRa файлов с POWERON_RESET: {files_with_boot}/{len(lora_files)}, всего событий: {total_boot_lora}")

total_boot_imu = 0
files_with_boot_imu = 0
for p in imu_files:
    cnt = 0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "POWERON_RESET" in line:
                cnt += 1
    if cnt:
        files_with_boot_imu += 1
        total_boot_imu += cnt
print(f"IMU файлов с POWERON_RESET: {files_with_boot_imu}/{len(imu_files)}, всего событий: {total_boot_imu}")
