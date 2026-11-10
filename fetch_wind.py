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
    "ju": ("UGRD", "250 mb", False),        # 上空の風(250hPa、高さ約1万m。ジェット気流)
    "jv": ("VGRD", "250 mb", False),
}
REQUIRED = {"u", "v", "t"}
OPTIONAL = ["rh", "pres", "prate", "csnow", "cloud", "cape", "vis", "ju", "jv"]


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


# ---------- PM2.5(NOAA GEFS-Aerosols。0.25°を0.5°に間引く) ----------
CHEM = "https://noaa-gefs-pds.s3.amazonaws.com"


def chem_url(cycle, h):
    ymd, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    return f"{CHEM}/gefs.{ymd}/{hh}/chem/pgrb2ap25/gefs.chem.t{hh}z.a2d_0p25.f{h:03d}.grib2"


def pick_chem(cycle):
    """風と同じ時刻に合わせられる、いちばん新しいエアロゾル予報の回(と時間のずれ)を探す"""
    for back in (0, 6, 12, 18):
        c = cycle - timedelta(hours=back)
        try:
            fetch(chem_url(c, HOURS[-1] + back) + ".idx")
            return c, back
        except Exception:
            pass
    return None, 0


# エアロゾル予報の中から使うもの: (キー, 変数名, 高さ, 種類, 波長550nmに限るか)
CHEM_SPEC = [
    ("pm25", "PMTF", "surface", "Total aerosol", False),                       # 地表の PM2.5(µg/m³)
    ("dust", "AOTK", "entire atmosphere", "Dust dry", True),                   # 黄砂・砂ぼこりの光学的厚さ
    ("bc", "AOTK", "entire atmosphere", "Black carbon dry", True),             # すす
    ("om", "AOTK", "entire atmosphere", "Particulate organic matter dry", True),   # 有機物(煙)
]


def get_chem(c, h):
    url = chem_url(c, h)
    lines = [l for l in fetch(url + ".idx").decode().splitlines() if l.strip()]
    starts = [int(l.split(":")[1]) for l in lines]
    out = {}
    for key, var, level, kind, w550 in CHEM_SPEC:
        for i, line in enumerate(lines):
            p = line.split(":")
            if (len(p) > 6 and p[3] == var and p[4] == level and f"aerosol={kind}" in p[6]
                    and (not w550 or (len(p) > 8 and "5.45e-07" in p[8]))):
                end = starts[i + 1] - 1 if i + 1 < len(lines) else ""
                out[key] = decode(fetch(url, {"Range": f"bytes={starts[i]}-{end}"}))
                break
    if "pm25" not in out:
        raise RuntimeError(f"PM2.5 f{h:03d}: 見つかりません")
    return out


# ---------- 海流(NOAA CoastWatch: 衛星の海面高度から求めた地衡流。1日1回・0.25°) ----------
CUR_URL = ("https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily.nc"
           "?u_current%5B(last)%5D%5B(-89.875):1:(89.875)%5D%5B(-179.875):1:(179.875)%5D"
           ",v_current%5B(last)%5D%5B(-89.875):1:(89.875)%5D%5B(-179.875):1:(179.875)%5D")


def get_currents(shape):
    """0.25°の格子(格子点は .125/.375…)を、2×2の平均で、風と同じ0.5°の格子(上が北、経度0〜360)にする"""
    import netCDF4
    path = os.path.join(NEW, "_cur.nc")
    with open(path, "wb") as f:
        f.write(fetch(CUR_URL))
    d = netCDF4.Dataset(path)
    lat = np.asarray(d["latitude"][:])
    t = netCDF4.num2date(d["time"][:], d["time"].units)[-1]
    out = []
    for name in ("u_current", "v_current"):
        a = np.ma.filled(d[name][-1].astype(np.float32), np.nan)      # (lat 720, lon 1440)
        if lat[0] < lat[-1]:
            a = a[::-1]                                                  # 上が北
        a = np.roll(a, 720, axis=1)                                      # 経度 -180〜 → 0〜
        # 目的の格子点(緯度 90,89.5,…/ 経度 0,0.5,…)を囲む4点の平均。端は1行ずつ足して作る
        p = np.full((722, 1441), np.nan, np.float32)
        p[1:721, :1440] = a
        p[:, 1440] = p[:, 0]
        q = np.stack([p[0:721, 0:1440], p[1:722, 0:1440], p[0:721, 1:1441], p[1:722, 1:1441]])
        with np.errstate(invalid="ignore"):
            m = np.nanmean(q, axis=0)                                    # (721, 1440): 緯度90→-90 の0.25°ごと
        g = m[::2, ::2]                                                  # 0.5°ごと
        if g.shape != shape:
            raise RuntimeError(f"海流の格子の大きさが合いません: {g.shape}")
        out.append(np.nan_to_num(g))
    # 地衡流は赤道付近で計算が成り立たず、値が暴れる。赤道±3°は0、速さは2.5m/sまでにする
    lat05 = np.linspace(90, -90, shape[0])[:, None]
    sp = np.hypot(out[0], out[1])
    k = np.where(np.abs(lat05) < 3, 0, np.minimum(1, 2.5 / np.maximum(sp, 1e-6)))
    out = [out[0] * k, out[1] * k]
    d.close()
    os.remove(path)
    return out, str(t)[:10]


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
    "jet": 10,     # 0.1 m/s
    "wavev": 10,   # 見た目用の速さ 0.1
    "pm25": 10,    # 1 µg/m³
    "dust": 10,    # 光学的厚さ 0.01
    "smoke": 10,   # 光学的厚さ 0.01
    "cur": 1,      # 0.01 m/s(海流は遅いので細かく)
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

    pm = None
    try:
        cc, back = pick_chem(cycle)
        if cc is None:
            raise RuntimeError("エアロゾル予報の回が見つかりません")
        print(f"PM2.5: {cc} の予報を使います(ずれ {back} 時間)")
        with ThreadPoolExecutor(6) as ex:
            pm = list(ex.map(lambda h: get_chem(cc, h + back), HOURS))
        a = pm[8]["pm25"][::2, ::2]
        at = lambda la, lo: float(a[int(round((90 - la) / 0.5)), int(round(lo / 0.5)) % a.shape[1]])
        print(f"  確認(24時間後): 中央値 {np.median(a):.1f} / 99% {np.percentile(a, 99):.1f} µg/m³ / "
              f"北京 {at(39.9, 116.4):.1f} / 東京 {at(35.7, 139.7):.1f} / デリー {at(28.6, 77.2):.1f}")
        for key in ("dust", "bc", "om"):
            if key in pm[8]:
                print(f"  確認 {key}: 最大 {pm[8][key].max():.2f} / 99% {np.percentile(pm[8][key], 99):.2f}")
    except Exception as e:
        pm = None
        print("警告: PM2.5 が取得できませんでした(そのレイヤーは使えません):", e)

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

    cur, cur_date = None, None
    try:
        cur, cur_date = get_currents(shape)
        save("cur_000.dz", cur, 100)
        print("海流:", cur_date)
        at = lambda a, la, lo: float(a[int(round((90 - la) / 0.5)), int(round(lo / 0.5)) % shape[1]])
        sp = np.hypot(cur[0], cur[1])
        print(f"  確認: 最大 {sp.max():.2f} m/s / 黒潮(33.5N,137E) u={at(cur[0], 33.5, 137):.2f} v={at(cur[1], 33.5, 137):.2f}"
              f" / 湾流(38N,65W) u={at(cur[0], 38, 295):.2f} / 陸(36N,100E) {at(cur[0], 36, 100):.2f}")
    except Exception as e:
        print("警告: 海流が取得できませんでした(海流レイヤーは使えません):", e)

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
        if have["ju"] and have["jv"]:
            save(f"jet_{h:03d}.dz", [grids["ju"][k], grids["jv"][k]], 100)        # 上空の風 m/s
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
        if pm is not None:
            half = lambda a: a[::2, ::2] if a.shape[0] > shape[0] else a        # 0.25° → 0.5°
            c = pm[k]
            save(f"pm25_{h:03d}.dz", [np.clip(half(c["pm25"]), 0, 3000)], 10)   # 値はすでに µg/m³
            if "dust" in c:
                save(f"dust_{h:03d}.dz", [np.clip(half(c["dust"]), 0, 30)], 1000)
            if "bc" in c and "om" in c:
                save(f"smoke_{h:03d}.dz", [np.clip(half(c["bc"] + c["om"]), 0, 30)], 1000)
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
    if have["ju"] and have["jv"]:
        fields.append("jet")
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
    if pm is not None:
        fields.append("pm25")
        if all("dust" in c for c in pm):
            fields.append("dust")
        if all("bc" in c and "om" in c for c in pm):
            fields.append("smoke")

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
    if cur is not None:
        manifest["cur"] = cur_date
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