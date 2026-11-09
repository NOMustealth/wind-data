# wind-data

天気アプリ(wind-app)が読む予報データ。NOAA GFS / GFS-Wave(0.5°)を6時間ごとに取得し、GitHub Pagesで配信する。

- 配信先: https://nomustealth.github.io/wind-data/manifest.json
- 形式: `<種類>_<時間>.dz` = int16の格子を横方向に差分 → 下位/上位バイトに分けて → gzip。復元方法は `fetch_wind.py` の `save()` と、アプリ側 `js/core.js` の `decodeDZ()` を参照
- データはコミットしない(Pagesに最新分だけ置く)
