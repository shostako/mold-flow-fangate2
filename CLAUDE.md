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
- Streamlit UI `app.py`（v0.3.0）: fangate の app.py からソルバ設定とメインパネルを持ち込み、形状入力だけ差し替え。
  形状ウィジェットは `fg_<field>` キー。UI テストは `tests/ui_helpers.py` の `app(fast=True)`（4 mm セル）で回す。
  fangate の builder 依存テストは全部新 builder で書き直し済み（`test_fan_gate_ui` → `test_fan_runner_ui`）
- 射出条件は**実機のスクリュー設定から**（v0.4.0、sim #88/#89/#90 の横展開）。`core/injection_profile.py` の
  `InjectionProfile` がスクリュー径・計量位置・各段の速度切替位置と速度から体積 → 時刻の区分線形写像を作り、
  `HeleShawSolver.injection_profile` に渡すと体積 CDF 写像がそこを通る（`injection_volume_flow_cm3s` に優先。
  無指定なら従来の定率で既存結果と bit 一致）。段の注入体積は速度によらずストロークだけで決まるので、
  折れ点の体積は固定で傾きだけが段ごとに変わる。**既定は機械の設定（φ50 / V/P 18 / 3 段・全段 200 mm/s）だが、
  計量位置だけ本リポ固有の 150 mm** — 既定形状が 219.4 cm³ あって sim の 30 mm（理論射出量 23.6 cm³）では
  全然足りず、既定画面が常に外挿警告を出すため。キャビティを覆う最小ストロークに 15% の余裕を足して 10 mm 丸めた
  **逆算値**で、実機の設定値ではない。切替位置も sim の 28/22 は別案件のものなので持ち込まずストロークを等分割。
  V/P を越える体積は最終段の射出率で外挿し、理論射出量がキャビティ体積や計量体積を下回るときは警告を出す。
  スキン層の時計の UI 既定も `constant_rate`（速度制御）に変えた（ライブラリ既定は `constant_pressure` のまま）
- `tests/test_injection_ui.py` の期待既定値は先頭の定数ブロック（`DEF_METER` / `DEF_SWITCHES` 等）1 箇所に集約してある。
  sim / fangate と共有するファイルなので、値を変えるときは `app.py` の定数と両方を直す
- sim v0.39.0 の取り込み（v0.5.0）: スキン層の反復上限 1〜100・既定 20、未収束警告は結果ペインの先頭（スキン／層別／
  領域絞り込み／二相の射出相）、式解説 §2 を体積 CDF 写像に。**既定形状は圧力一定時計だと上限 80 でやっと収束し
  （最終領域は 37 反復）、封止がキャビティの 25%**（上限 5 の「封止 0」は打ち切りの産物。上限ごとに数字が単調でないのは
  領域パスの候補が同じ予算で解かれるため）。速度制御時計（UI 既定）は 2 反復で収束・封止 0。2 つの時計は別の絵を出す。
  途中候補の打ち切りが黙る件は Issue #10
- sim との core 差分はロジック無し（`_restricted_to` の `product_mask` 引き継ぎだけ本リポが先行。sim へ逆移植の候補）
- ゲート形状の選択（v0.7.0）: `runner_shape` が `"triangle"`＝Gate 1（客先図面）／`"pentagon"`＝Gate 2（五角形、既定は
  `GATE2_DEFAULTS`）。Gate 2 は側辺 `side_len_mm`（**額縁の下端から測る**。額縁込みは 20 + 入力）＋ R12 に接する斜辺。
  肉厚の規則（エッジ帯の肉厚・幅、傾斜の有無と開始位置、円の外の肉厚 t_o、円の深さ）は両方の形で共通。
  **Gate 1 の既定は v0.5.0 とビット一致**（`tests/test_geometry_gate2.py` の指紋。新しい変数の既定が従来の規則そのもの）。
  Gate 2 の図面は `docs/draft/gate2_drawing.pdf`（`gate2_drawing.py` が builder から描く）。UI の Gate 2 専用ウィジェットは
  `fg_side_len_mm` / `fg_g2_*`、Gate 1 の円の深さは `fg_g1_*`
- 環境: `uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`。テストは `MPLBACKEND=Agg .venv/bin/pytest -n auto --dist loadscope`（CI と同じ。直列だと 8 分、4 並列で 5 分半）
- Streamlit Community Cloud: <https://mold-flow-fangate2.streamlit.app>（main を自動デプロイ）。`requirements.txt` は pyproject の deps のミラー、
  `runtime.txt` は `python-3.12`。deps を変えたら requirements.txt も同期

## fangate から持ち込まなかったもの
`core/fan_gate.py`（旧ゲート／ウイング／タブ／井戸／スプルーブッシュ込みの builder）と `test_geometry_fan_gate`。
UI のゲート形状 radio（fangate の旧／ウイング。本リポの Gate 1／Gate 2 の選択は v0.7.0 で別に作った）・タブ・井戸／スプルーの expander。

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
