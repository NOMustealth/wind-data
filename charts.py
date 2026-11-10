"""気象庁の天気図(地上実況図・24時間予想図・48時間予想図)を、地図に描ける小さな JSON にする。
気象庁防災情報XML(無料。政府標準利用規約)の長期フィードから、それぞれ一番新しいものを使う。"""
import json, re, sys, urllib.request

FEED = "https://www.data.jma.go.jp/developer/xml/feed/regular_l.xml"
UA = {"User-Agent": "Mozilla/5.0 wind-data"}
KINDS = {"地上実況図": "now", "地上２４時間予想図": "f24", "地上４８時間予想図": "f48"}


def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8")


def line(s, step=1):
    pts = re.findall(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)", s)
    out = [[round(float(b), 2), round(float(a), 2)] for a, b in pts]
    return out[::step] + ([out[-1]] if step > 1 and len(out) % step != 1 else [])


def parse(xml):
    feats = []
    vt = re.search(r'<DateTime type="[^"]*">([^<]+)</DateTime>', xml)
    for it in re.findall(r"<Item>(.*?)</Item>", xml, re.S):
        types = re.findall(r"<Type>([^<]*)</Type>", it)
        t = types[0] if types else ""
        if t == "等圧線":
            p = re.search(r'unit="hPa">(\d+)<', it)
            ln = re.search(r'<jmx_eb:Line[^>]*>([^<]+)<', it)
            if p and ln:
                feats.append({"type": "Feature", "properties": {"k": "iso", "p": int(p.group(1))},
                              "geometry": {"type": "LineString", "coordinates": line(ln.group(1), 2)}})
        elif t.endswith("前線"):
            ln = re.search(r'<jmx_eb:Line[^>]*>([^<]+)<', it)
            k = {"寒冷前線": "cold", "温暖前線": "warm", "停滞前線": "stat", "閉塞前線": "occl"}.get(t, "stat")
            if ln:
                feats.append({"type": "Feature", "properties": {"k": k},
                              "geometry": {"type": "LineString", "coordinates": line(ln.group(1))}})
        elif t in ("高気圧", "低気圧", "台風", "熱帯低気圧"):
            c = re.search(r'type="中心位置（度）">([^<]+)<', it)
            p = re.search(r'type="中心気圧" unit="hPa">(\d+)<', it)
            sp = re.search(r'type="移動速度" unit="km/h"[^>]*>([^<]*)<', it)
            spd = re.search(r'description="([^"]*)" type="移動速度" unit="km/h"', it)
            dr = re.search(r'type="移動方向" unit="度（真方位）">(\d+)<', it)
            nm = re.search(r"<NameKana>([^<]+)</NameKana>", it)
            num = re.search(r"<Number>(\d+)</Number>", it)
            if not c:
                continue
            ll = line(c.group(1))[0]
            k = {"高気圧": "H", "低気圧": "L", "台風": "TY", "熱帯低気圧": "TD"}[t]
            feats.append({"type": "Feature", "properties": {
                "k": k, "p": int(p.group(1)) if p else None,
                "spd": (sp.group(1) if sp and sp.group(1) else ""), "sd": spd.group(1) if spd else "",
                "dir": int(dr.group(1)) if dr else None,
                "name": (nm.group(1) if nm else ""), "num": (num.group(1) if num else "")},
                "geometry": {"type": "Point", "coordinates": ll}})
    return {"valid": vt.group(1) if vt else "", "features": feats}


def main(out):
    feed = get(FEED)
    res = {}
    for e in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
        t = re.search(r"<title>(.*?)</title>", e).group(1)
        if t in KINDS and KINDS[t] not in res:
            u = re.search(r'<link[^>]*href="([^"]+)"', e).group(1)
            x = get(u)
            d = parse(x)
            d["issued"] = (re.search(r"<ReportDateTime>([^<]+)<", x) or [None, ""])[1]
            res[KINDS[t]] = d
            print(t, d["valid"], len(d["features"]), "本/点")
    if not res:
        sys.exit("天気図が見つかりませんでした")
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "charts.json")
