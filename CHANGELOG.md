# 変更履歴

[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) 準拠、[セマンティック バージョニング](https://semver.org/lang/ja/) に従う。
`0.x` 系のため、マイナー版の更新に後方非互換の変更を含むことがある。

## [0.4.1] — 2026-09-15

**Codex P2 4 件の修正**（本 PR で検出、sim v0.42.3 と同一パッチ）。

- `injection_profile` を dataclass の途中に挿したせいで位置引数がずれていた。7 番目の位置引数は
  `compression_molding` だったので、そこに bool を渡していた呼び出しは profile に bool を束縛し
  `solve()` の中で `bool.time_at_volume_mm3` に到達する。`field(kw_only=True)` で
  ソース上の位置は保ったまま `__init__` の末尾へ移した。層別も同じ。
- 層別ソルバーの metadata に `injection_Q_effective_cm3s` が無く、`FlowResult` /
  `MultilayerFlowResult` の契約が非対称だった。
- 二相の外挿フラグが、計量がストローク内でも開きキャビティが超えていれば立っていた。
  `T_open_total` も写像を通るが到着時刻場の正規化係数として約分され、計量より先のセルは
  `T_inj` 以降に着いてプールからもスキンの時計からも外れる。外挿値は報告に乗らないので、
  判定を `V_shot` だけにして文言も直した。
- メイン結果ペインの外挿警告が最終キャビティ体積としか比べていなかった。ICM ON では写像が
  読むのは**開いた隙間**の体積なので、ストロークが最終形状を覆っていても V/P より先を読む帯が
  ある。判定を `solve()` の metadata（`injection_extrapolated_past_vp` /
  `injection_swept_volume_cm3`）に移し、UI はそのフラグを読む。

## [0.4.0] — 2026-09-15

**射出率を成形機の設定そのもの（スクリュー径・位置・速度）から出せるようにし、多段射出で
充填時間の時間軸が段ごとに折れるようにした。** mold-flow-sim #88 / PR #89・#90 の横展開（本リポの Issue #7）。

射出速度スライダーは代表せん断速度＝粘度にしか効かず、総充填時間は別入力の射出率が決めていた。
その既定 589 cm³/s は取説の**最大**射出率で、どんな条件も機械が全開で走っている前提の絵になっていた。

### 追加

- `core/injection_profile.py`: `InjectionProfile` / `InjectionStage`。スクリュー径・計量位置・
  各段の速度切替位置と速度から、段ごとの射出率 `Q_i = πD²/4·v_i`・時間・注入体積と、
  **体積 → 時刻の区分線形写像**を出す。各段の注入体積 `πD²/4·L_i` は速度によらずストロークだけで
  決まるので、折れ点の体積は固定で傾き（射出率）だけが段ごとに変わる。V/P を越える体積は
  最終段の射出率で外挿する。
- `HeleShawSolver.injection_profile` / `MultilayerHeleShawSolver.injection_profile`。設定すると
  `injection_volume_flow_cm3s` に**優先**し、体積 CDF 写像がこの写像を通る。`T_fill` への
  正規化は従来どおりなので、スキン膨張・ICM 短縮との合成は変わらない — 変わるのは途中の形だけ。
  プロファイル無しの経路は式もそのままで既存結果と bit 一致する。
- UI「射出条件」に入力方式ラジオ。**既定は「実機条件から計算」**。射出段数 1〜5、2 以上で各段の
  速度切替位置と速度の欄が開く。平均射出率・V/P までの射出時間・V/P 時点の理論射出量を表示。
- 結果ペインに射出条件のキャプションと、**理論射出量がキャビティ体積を下回るときの警告**。
  二相ショートショットも `injection_extrapolated_past_vp` を metadata に持ち、計量または
  開きギャップ体積が理論射出量を超えたらパネルが警告する。
- `tests/test_injection_profile.py`（49 件）と `tests/test_injection_ui.py`（20 件）。

### 変更

- 「射出速度 [mm/s] (代表)」→「代表流動速度 [mm/s]」。**スクリュー速度とは別量**で、
  キャビティ内のギャップ平均流速として `6V/h` の代表せん断速度にだけ入る（薄板では 1 桁大きい）。
- **スキン層の時計の UI 既定が `constant_rate`（速度制御）**。射出入力の既定が実機条件に
  なった以上、時計だけ「機械が速度を保てない」側に倒れていると、機械の設定どおりの射出時間を
  入れた画面が設定どおりでない充填時間を返す。ライブラリ既定は `constant_pressure` のまま。
- `metadata` に `injection_profile` と `injection_Q_effective_cm3s`、`settings.json` の
  `injection` に `rate_input_mode` と `injection_profile`。
- 二相ショートショットの `T_inj` と開きキャビティ充填時間もこの写像を通る。

### 既定値について

機械は sim と同じ（スクリュー径 50 mm、V/P 18 mm、3 段・全段 200 mm/s）だが、**計量位置だけ
本リポ固有で 150 mm**。この既定形状のキャビティは 219.4 cm³ あり、φ50 では 111.7 mm の
ストロークが要る。sim の計量位置 30 mm（理論射出量 23.6 cm³）のままだと既定画面が常に
外挿警告を出すので、キャビティを覆う最小ストロークに 15% の余裕を足して 10 mm 丸めた。
**実機の設定値ではなく既定形状からの逆算値**で、切替位置も sim の 28/22 は別の案件のものなので
持ち込まず、ストロークを等分割している。

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
