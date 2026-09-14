# 変更履歴

[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) 準拠、[セマンティック バージョニング](https://semver.org/lang/ja/) に従う。
`0.x` 系のため、マイナー版の更新に後方非互換の変更を含むことがある。

## [0.3.0] — 2026-09-14

**Streamlit UI。** fangate v0.7.3 の `app.py` からソルバ設定とメインパネルを持ち込み、形状サイドバーだけ
`FanRunnerPlateConfig` 向けに差し替えた（`_fan_runner_sidebar()`、ウィジェットキーは `fg_<field>`）。

### 追加

- サイドバー: 製品（額縁プレート、圧縮は内側だけの注記）／ランナ（幅・フランク角・長さ・丸端 φ、深さ別の肉厚 4 欄、ゲート φ）／
  肉盗み（上限は `balancer_limits_mm` に追従、置けない形状は警告して OFF）／メッシュ。フランク角の欄にフランク延長の頂点深さと
  丸端上端の深さを併記
- プレビューの厚みマップに圧縮部（内側）の赤枠。説明文（ICM の膨張対象、モデル化している現象、ゲート圧損）をこの金型に合わせた
- テスト: `test_fan_runner_ui` 新設、fangate の `test_two_phase_ui` / `test_weld_ui` / `ui_helpers` を移植（既定幅 302.26 に追従）。
  402 passed

### 持ち込まなかったもの

ゲート形状 radio、タブ、井戸・スプルーの expander（builder に無い）

## [0.2.0] — 2026-09-14

**図面の builder。** `core/fan_runner.py` に `FanRunnerPlateConfig` / `build_fan_runner_plate_geometry` を新設。既定値は図面
（額縁 20 × t1.0、内側 262.26 × 180.6 × t4.0、ランナ幅 300・フランク 14°・軸まで 30・R12 丸端、肉厚 1.0 → 2.5、ゲート φ3）。

### 追加

- ランナのシルエットは「製品エッジから下がる三角形 ∪ 軸上の円」。フランクが円に接線でない図面どおりの形を、クリップ無しで表す
  （`validate` は三角形が円の上端に届くことだけ要求）
- ランナ肉厚は深さだけの関数（エッジ帯 → 直線傾斜 → 均一）。井戸・コールドスラッグ無し
- **圧縮マスク＝内側 t4 だけ**（`compression_mask`）、`product_mask`＝額縁＋内側で表示原点は製品エッジ
- 射出点＝軸上の φ3 ディスク（Dirichlet）。粗メッシュではスナップ、マーカーは実寸 1 個
- 肉盗み `balancer_*`（fangate v0.5.0 と同型、底辺は製品エッジ線、`min()` で増肉しない）
- `grid_shift_x_mm`: 製品幅 302.26 で軸が格子から 0.13 ずれて非対称にラスタされるのを、左パッドを 1 セル未満伸ばして
  軸をセル境界（偶数列）／セル中心（奇数列）に揃えて解消。y 方向の `grid_shift_mm` は fangate 由来
- テスト: `test_geometry_fan_runner`（図面の交点・幅・肉厚・体積・対称性・validate）、fangate の builder 依存テスト 4 本
  （two_phase / compression_stroke / settings_record / fill_render）を新 builder で書き直し。364 passed
- `docs/draft/geometry_draft.py` → `geometry_draft.png`（厚みマップ 3 面）
- Codex P2 × 2（PR #4）: フランクが円にぎりぎり届く形状を粗メッシュでラスタすると三角形と円が 2 島に割れてゲートが
  製品に届かない → ラスタ後に `scipy.ndimage.label` で連結を確認して builder が明示エラー。丸端が製品幅＋パッドより
  広いと格子外で黙って切れる → `runner_end_d_mm ≤ runner_w_mm` を `validate` に追加
- Claude レビュー（PR #4）: `HeleShawSolver._restricted_to` が部分ジオメトリに `compression_mask` しか引き継がず、
  `product_mask` が `compression_mask` と別物になったこのリポでは部分ジオメトリの表示原点が内側の下端に落ちる
  （現状その経路で表示はしないが地雷）→ `product_mask` / `valve_axis_x_mm` / `valve_marker_mm` も引き継ぐ。
  鏡映対称テストに `cell_size_mm ∈ {1, 0.5, 2}` を追加

## [0.1.0] — 2026-09-14

**リポ立ち上げ。** [mold-flow-fangate](https://github.com/shostako/mold-flow-fangate) v0.7.3 から形状非依存の `core/`
（Hele-Shaw ソルバ、多層熱、二相ショートショット、可視化、材料 DB、設定記録、版管理）とそのテストを移植。
形状仕様 `docs/spec.md` を図面から起こした。builder と UI は次の版。
