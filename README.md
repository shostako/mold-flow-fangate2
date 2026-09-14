# mold-flow-fangate2

12.3 インチ 額縁肉厚プレート（外周 20 幅 t=1 / 内側 t=4、**圧縮は内側だけ**）をファン状ランナ＋ホットランナーゲートで
射出圧縮成形するときの流動解析。[mold-flow-fangate](https://github.com/shostako/mold-flow-fangate) v0.7.3 を起点に、
この図面専用として別リポで立ち上げた。

- 形状仕様: `docs/spec.md`
- `core/`: fangate から移植した Hele-Shaw ソルバ／多層熱／二相ショートショット／可視化、ランナ builder `core/fan_runner.py`
- `app.py`: Streamlit UI

## 使い方

ローカル:

```
streamlit run app.py
```

左のサイドバーで形状（既定値は図面の実機）・材料・射出条件・圧縮条件を設定して「解析実行」。

## 開発

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && MPLBACKEND=Agg .venv/bin/pytest
```
