"""国土地理院の「指定緊急避難場所」データ(ズーム10のタイル)を全国ぶん数えて、都道府県ごとの数を JSON にする(週1回)。"""
import json, math, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 wind-data"}
TYPES = {1: "洪水", 2: "土砂", 3: "高潮", 4: "地震", 5: "津波", 6: "大火事", 7: "内水氾濫", 8: "火山"}
PREF = re.compile(r"^(北海道|東京都|京都府|大阪府|.{2,3}県)")


def tile(lat, lon, z=10):
    n = 2 ** z; la = math.radians(lat)
    return int((lon + 180) / 360 * n), int((1 - math.log(math.tan(la) + 1 / math.cos(la)) / math.pi) / 2 * n)


def get(t, x, y):
    u = f"https://cyberjapandata.gsi.go.jp/xyz/skhb{t:02d}/10/{x}/{y}.geojson"
    for a in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40) as r:
                return json.loads(r.read()).get("features", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return []


def main(out):
    x0, y0 = tile(46.0, 122.0); x1, y1 = tile(20.0, 154.5)
    cells = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    print("タイル", len(cells))
    shel = {}
    with ThreadPoolExecutor(32) as ex:
        # まず地震(04)で、避難所のあるタイルを見つける
        have = set()
        for (x, y), fs in zip(cells, ex.map(lambda c: get(4, *c), cells)):
            if fs:
                have.add((x, y))
            for f in fs:
                p = f["properties"]; k = p.get("name", "") + json.dumps(f["geometry"]["coordinates"])
                shel.setdefault(k, {"a": p.get("address") or "", "t": set()})
                for n in TYPES:
                    if p.get(f"disaster{n}") == 1:
                        shel[k]["t"].add(n)
        print("避難所のあるタイル", len(have), "地震で", len(shel))
        # 洪水(01)も全国を見る(地震の指定のない避難所もあるため)、そのほかは見つかったタイルだけ
        jobs = [(1, x, y) for x, y in cells] + [(t, x, y) for t in (2, 3, 5, 6, 7, 8) for x, y in have]
        for fs in ex.map(lambda j: get(*j), jobs):
            for f in fs:
                p = f["properties"]; k = p.get("name", "") + json.dumps(f["geometry"]["coordinates"])
                s = shel.setdefault(k, {"a": p.get("address") or "", "t": set()})
                for n in TYPES:
                    if p.get(f"disaster{n}") == 1:
                        s["t"].add(n)
    pref = {}
    for s in shel.values():
        m = PREF.match(s["a"])
        name = m.group(1) if m else "不明"
        d = pref.setdefault(name, {"all": 0, **{str(n): 0 for n in TYPES}})
        d["all"] += 1
        for n in s["t"]:
            d[str(n)] += 1
    total = {"all": len(shel), **{str(n): sum(1 for s in shel.values() if n in s["t"]) for n in TYPES}}
    res = {"updated": time.strftime("%Y-%m-%d"), "types": TYPES, "total": total, "pref": pref}
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False)
    print("全国", total)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shelters.json")
