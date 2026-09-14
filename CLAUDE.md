# mold-flow-fangate2

12.3 インチ 額縁肉厚プレート（外周 20 幅 t=1 / 内側 t=4）＋ファン状ランナ＋ホットランナーゲート φ3 の射出圧縮成形流動解析。
[mold-flow-fangate](../mold-flow-fangate) v0.7.3 を起点に、**この図面専用の別リポ**として立ち上げた（fangate 側は触らない）。
顧客名は書かない（public リポ）。図面 PDF はリポに入れない。

## 現状（2026-09-14）
- 形状仕様は `docs/spec.md`（図面のベクタから実寸で読んだ寸法と、読み取りの根拠）
- fangate の形状非依存モジュールは `core/` に移植済み（v0.1.0）。`geometry.py` は `Geometry` dataclass と `build_demo_geometry` のみ
- builder は `core/fan_runner.py`（`FanRunnerPlateConfig` / `build_fan_runner_plate_geometry`、v0.2.0）。既定値が図面。
  **圧縮マスク＝内側 t4 だけ**、`product_mask`＝額縁＋内側（表示原点は製品エッジ）。ランナ形状は 1 種（三角形 ∪ 丸端の円、
  肉厚は深さだけの関数）、肉盗み `balancer_*` だけ fangate から残した。図は `docs/draft/geometry_draft.png`
- fangate の builder 依存テストは two_phase / compression_stroke / settings_record / fill_render を新 builder で書き直し済み。
  UI 依存の 3 本（`test_fan_gate_ui` / `test_two_phase_ui` / `test_weld_ui`）は UI と一緒に
- 環境: `uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`。テストは `MPLBACKEND=Agg .venv/bin/pytest`
- Streamlit Community Cloud: 配備予定（main を自動デプロイ）。`requirements.txt` は pyproject の deps のミラー、
  `runtime.txt` は `python-3.12`。deps を変えたら requirements.txt も同期

## fangate から持ち込まなかったもの
`core/fan_gate.py`（旧ゲート／ウイング／タブ／井戸／スプルーブッシュ込みの builder）と、それに依存するテスト
（`test_geometry_fan_gate` / `test_fan_gate_ui` / `test_two_phase` / `test_compression_stroke` / `test_settings_record` /
`test_fill_render` / `test_two_phase_ui` / `test_weld_ui`）。builder と UI が入った時点で新 builder で書き直す。

## builder の設計メモ
- 座標は fangate と同じ格子系（ランナが下、製品が上、y は上向き）。`display_origin_mm()` で製品エッジ y=0
- ランナのシルエット: 製品エッジで幅 300、フランク 14°（`fan_flank_deg`）の直線、スプルー軸（エッジから 30）に R12 の丸端。
  フランクは円に交差し、延長頂点（深さ 37.4）は円の内側なので「三角形 ∪ 円」。validate は三角形が円に届くことを要求する
- ランナ肉厚は深さ d だけの関数: d≤1 で 1.0、1→20 で 1.0→2.5、以降 2.5。井戸・コールドスラッグ無し
- 射出点は軸上の φ3 ディスク（Dirichlet）。粗メッシュで空になったら軸に最も近いキャビティセルへスナップ（fangate と同じ）
- `grid_shift_mm`（fangate 由来）で製品エッジをセルエッジに、`grid_shift_x_mm`（新設）で軸をセル境界／中心に揃える。
  製品幅 302.26 は軸を格子から 0.13 ずらすので、x の揃えが無いとランナが非対称にラスタされる。上辺と左右端はセルに揃わないので
  体積検算は許容幅つき（1 mm セルで解析値 +0.11%）

## 運用
- feature branch + PR + CI green + マージ前レビュー。マージは明示確認を取る
- push 前に `ruff check . && ruff format --check . && pytest`
- 作業ログは `logs/yyyy-MM.md`
