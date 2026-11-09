import gzip
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import eccodes
import numpy as np

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HOURS = list(range(0, 73, 3))   # 0〜72時間先、3時間ごと
OUT = "data"
NEW = "data_new"

# 波の向き(DIRPW)が「波が来る方向」なら True、「波が向かう方向」なら False
# 波の粒子が風と逆向きに流れて見えたら、ここを切り替える
WAVE_FROM = True

# 名前: (GRIBの変数名, 高さ, 期間つき(平均)か)
SPEC = {
    "u": ("UGRD", "10 m above ground", False),
    "v": ("VGRD", "10 m above ground", False),
    "t": ("TMP", "2 m above ground", False),
    "rh": ("RH", "2 m above ground", False),
    "pres": ("PRMSL", "mean sea level", False),
    "prate": ("PRATE", "surface", True),
    "csnow": ("CSNOW", "surface", False),
    "cloud": ("TCDC", "entire atmosphere", True),
    "cape": ("CAPE", "surface", False),
    "vis": ("VIS", "surface", False),
}
REQUIRED = {"u", "v", "t"}
OPTIONAL = ["rh", "pres", "prate", "csnow", "cloud", "cape", "vis"]


def fetch(url, headers=None, retries=3):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == retries - 1:
                raise
        except Exception:
            if k == retries - 1:
                raise
        time.sleep(1.5)


def url_of(cycle, h):
    ymd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")
    return f"{BASE}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p50.f{h:03d}"


def wave_url(cycle, h):
    ymd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")
    return f"{BASE}/gfs.{ymd}/{hh}/wave/gridded/gfswave.t{hh}z.global.0p25.f{h:03d}.grib2"


def pick_cycle():
    """最後の時刻(72時間先)まで公開済みの、最新のサイクルを探す"""
    now = datetime.now(timezone.utc) - timedelta(hours=4)
    c = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for i in range(5):
        cand = c - timedelta(hours=6 * i)
        print("確認するサイクル(UTC):", cand)
        try:
            fetch(url_of(cand, HOURS[-1]) + ".idx")
            return cand
        except urllib.error.HTTPError as e:
            print("  まだ公開されていません:", e.code)
    raise SystemExit("使えるサイクルが見つかりませんでした")


def choose(lines, var, level, h, ranged):
    """目次(.idx)から、使う行の番号を選ぶ。なければ None"""
    cands = []
    for i, line in enumerate(lines):
        p = line.split(":")
        if len(p) > 5 and p[3] == var and p[4].startswith(level):
            cands.append((i, p[5]))
    if not cands:
        return None
    if not ranged:
        return cands[0][0]
    best = None
    for i, t in cands:
        m = re.match(r"(\d+)-(\d+) hour", t)
        if m and int(m.group(2)) == h:
            span = int(m.group(2)) - int(m.group(1))
            if best is None or span < best[0]:
                best = (span, i)
    return best[1] if best else None


def get_hour(cycle, h):
    url = url_of(cycle, h)
    lines = [l for l in fetch(url + ".idx").decode().splitlines() if l.strip()]
    starts = [int(l.split(":")[1]) for l in lines]
    out = {}
    for name, (var, level, ranged) in SPEC.items():
        i = choose(lines, var, level, h, ranged)
        if i is None:
            out[name] = None
            continue
        end = starts[i + 1] - 1 if i + 1 < len(lines) else ""
        out[name] = fetch(url, {"Range": f"bytes={starts[i]}-{end}"})
    missing = [k for k in REQUIRED if out[k] is None]
    if missing:
        raise RuntimeError(f"f{h:03d}: 必要な項目が見つかりません {missing}")
    print(f"f{h:03d} 取得")
    return out


def get_wave(cycle, h):
    url = wave_url(cycle, h)
    lines = [l for l in fetch(url + ".idx").decode().splitlines() if l.strip()]
    starts = [int(l.split(":")[1]) for l in lines]
    out = {}
    # (キー, 変数名, 必須か)
    for key, var, need in (("hs", "HTSGW", True), ("dir", "DIRPW", True), ("per", "PERPW", False)):
        idx = None
        for i, line in enumerate(lines):
            p = line.split(":")
            if len(p) > 4 and p[3] == var and p[4] == "surface":
                idx = i
                break
        if idx is None:
            if need:
                raise RuntimeError(f"波 f{h:03d}: {var} が見つかりません")
            out[key] = None
            continue
        end = starts[idx + 1] - 1 if idx + 1 < len(lines) else ""
        out[key] = fetch(url, {"Range": f"bytes={starts[idx]}-{end}"})
    print(f"波 f{h:03d} 取得")
    return out


def decode(buf):
    gid = eccodes.codes_new_from_message(buf)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        a = eccodes.codes_get_values(gid).astype(np.float32).reshape(nj, ni)
        if eccodes.codes_get(gid, "jScansPositively"):
            a = a[::-1]          # 上が北になるようにそろえる
        return a
    finally:
        eccodes.codes_release(gid)


def decode_wave(buf, shape):
    """波のデータ(0.25度)を、風などと同じ0.5度の格子(上が北、経度0〜360)にそろえる"""
    gid = eccodes.codes_new_from_message(buf)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        a = eccodes.codes_get_values(gid).astype(np.float32).reshape(nj, ni)
        lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lat2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
    finally:
        eccodes.codes_release(gid)

    a[a >= 9998] = np.nan        # 陸地など値がない所
    if lat1 < lat2:
        a = a[::-1]
        top = lat2
    else:
        top = lat1
    dy = abs(lat1 - lat2) / (nj - 1)
    dx = 360.0 / ni
    nrows = int(round(180 / dy)) + 1
    step = int(round(0.5 / dx))
    shift = int(round((lon1 % 360) / dx)) % ni
    a = np.roll(a, shift, axis=1)
    full = np.full((nrows, ni), np.nan, np.float32)
    r0 = int(round((90 - top) / dy))
    r1 = min(nrows, r0 + nj)
    full[r0:r1] = a[: r1 - r0]
    out = full[::step, ::step]
    if out.shape != shape:
        raise RuntimeError(f"波の格子の大きさが合いません: {out.shape} / {shape}")
    return out


# 保存する細かさ(保存値をこの数ごとに丸める)。画面で見分けがつかない範囲で粗くして、ファイルを小さくする
STEP = {
    "wind": 10,    # 0.1 m/s
    "temp": 10,    # 0.1 ℃
    "rh": 100,     # 1 %
    "pres": 1,     # 0.1 hPa(等圧線に使うので細かいまま)
    "rain": 5,     # 0.05 mm/h
    "snow": 5,     # 0.05 mm/h
    "cloud": 100,  # 1 %
    "cape": 10,    # 10 J/kg
    "vis": 10,     # 100 m
    "waveh": 10,   # 0.1 m
    "wavev": 10,   # 見た目用の速さ 0.1
}


def save(name, arrays, scale):
    """整数(int16)にして、横方向の差分をとり、gzipで圧縮して書き出す
    (読む側は js/core.js の decodeDZ。差分を足し戻して step 倍すれば元の整数に戻る)"""
    step = STEP[name.split("_")[0]]
    lim = 32767 // step
    rows = np.concatenate([
        np.clip(np.rint(np.nan_to_num(a) * scale / step), -lim, lim).astype(np.int32).reshape(a.shape)
        for a in arrays
    ], axis=0)
    d = (np.diff(rows, axis=1, prepend=0) & 0xFFFF).astype("<u2")
    b = d.view(np.uint8).reshape(-1, 2)
    body = b[:, 0].tobytes() + b[:, 1].tobytes()       # 下位バイト→上位バイトの順に並べると縮みやすい
    with open(os.path.join(NEW, name), "wb") as f:
        f.write(gzip.compress(body, 9, mtime=0))


def fill_gaps(seq):
    """f000 には降水・雲などが無いので、近い時刻の値で埋める"""
    last = None
    for k in range(len(seq)):
        if seq[k] is not None:
            last = seq[k]
        elif last is not None:
            seq[k] = last
    nxt = None
    for k in reversed(range(len(seq))):
        if seq[k] is not None:
            nxt = seq[k]
        elif nxt is not None:
            seq[k] = nxt
    return seq


def main():
    cycle = pick_cycle()
    print("使用するサイクル(UTC):", cycle)

    shutil.rmtree(NEW, ignore_errors=True)
    os.makedirs(NEW)

    print("ダウンロード中…")
    with ThreadPoolExecutor(6) as ex:
        raw = list(ex.map(lambda h: get_hour(cycle, h), HOURS))

    wave_raw = None
    try:
        with ThreadPoolExecutor(6) as ex:
            wave_raw = list(ex.map(lambda h: get_wave(cycle, h), HOURS))
    except Exception as e:
        print("警告: 波のデータが取得できませんでした(波レイヤーは使えません):", e)

    grids = {name: [decode(d[name]) if d[name] is not None else None for d in raw]
             for name in SPEC}
    shape = grids["u"][0].shape

    have = {}
    for name in OPTIONAL:
        have[name] = any(g is not None for g in grids[name])
        if have[name]:
            grids[name] = fill_gaps(grids[name])
        else:
            print(f"警告: {name} が取得できませんでした(そのレイヤーは使えません)")

    have_wave = False
    if wave_raw:
        try:
            decode_wave(wave_raw[0]["hs"], shape)
            have_wave = True
        except Exception as e:
            print("警告: 波のデータを読み取れませんでした(波レイヤーは使えません):", e)

    for k, h in enumerate(HOURS):
        u, v = grids["u"][k], grids["v"][k]
        save(f"wind_{h:03d}.dz", [u, v], 100)                                   # m/s
        save(f"temp_{h:03d}.dz", [grids["t"][k] - 273.15], 100)                 # ℃
        if have["rh"]:
            save(f"rh_{h:03d}.dz", [np.clip(grids["rh"][k], 0, 100)], 100)      # %
        if have["pres"]:
            save(f"pres_{h:03d}.dz", [grids["pres"][k] / 100], 10)              # hPa
        if have["prate"]:
            rain = grids["prate"][k] * 3600                                       # mm/h
            save(f"rain_{h:03d}.dz", [rain], 100)
            if have["csnow"]:
                save(f"snow_{h:03d}.dz", [rain * np.clip(grids["csnow"][k], 0, 1)], 100)
        if have["cloud"]:
            save(f"cloud_{h:03d}.dz", [np.clip(grids["cloud"][k], 0, 100)], 100)  # %
        if have["cape"]:
            save(f"cape_{h:03d}.dz", [np.clip(grids["cape"][k], 0, 8000)], 1)     # J/kg
        if have["vis"]:
            save(f"vis_{h:03d}.dz", [grids["vis"][k]], 0.1)                       # 10m単位
        if have_wave:
            w = wave_raw[k]
            hs = decode_wave(w["hs"], shape)
            dr = decode_wave(w["dir"], shape)
            per = decode_wave(w["per"], shape) if w.get("per") else None
            land = np.isnan(hs)
            hs0 = np.nan_to_num(hs)
            toward = np.deg2rad(dr + (180 if WAVE_FROM else 0))
            # 粒子の進む速さ(見た目用)。周期が長いほど速い。周期がなければ波の高さから決める
            if per is not None and not np.all(np.isnan(per)):
                sp = np.clip(0.8 * np.nan_to_num(per, nan=8.0), 3, 14)
            else:
                sp = np.clip(3 + 1.5 * hs0, 3, 12)
            ok = ~(land | np.isnan(dr))
            wu = np.where(ok, np.sin(toward) * sp, 0)
            wv = np.where(ok, np.cos(toward) * sp, 0)
            save(f"waveh_{h:03d}.dz", [np.where(land, -0.5, hs)], 100)            # m(陸は-0.5)
            save(f"wavev_{h:03d}.dz", [wu, wv], 100)
        print(f"f{h:03d} 保存")

    fields = ["wind", "temp"]
    if have["rh"]:
        fields.append("rh")
    if have["pres"]:
        fields.append("pres")
    if have["prate"]:
        fields.append("rain")
        if have["csnow"]:
            fields.append("snow")
    if have["cloud"]:
        fields.append("cloud")
    if have["cape"]:
        fields.append("cape")
    if have["vis"]:
        fields.append("vis")
    if have_wave:
        fields.append("waveh")

    manifest = {
        "v": 5,
        "fmt": "dz",
        "step": STEP,
        "cycle": cycle.isoformat(),
        "hours": HOURS,
        "w": int(shape[1]),
        "h": int(shape[0]),
        "fields": fields,
    }
    with open(os.path.join(NEW, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # data/ に入れ替える(manifest は最後)
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in os.listdir(NEW) if n != "manifest.json"]
    for n in names:
        os.replace(os.path.join(NEW, n), os.path.join(OUT, n))
    os.replace(os.path.join(NEW, "manifest.json"), os.path.join(OUT, "manifest.json"))
    for n in os.listdir(OUT):
        if n != "manifest.json" and n not in names:
            os.remove(os.path.join(OUT, n))
    shutil.rmtree(NEW, ignore_errors=True)
    print("保存しました:", manifest)


if __name__ == "__main__":
    main()