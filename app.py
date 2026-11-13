"""Streamlit UI — 12.3-inch frame plate (body-only compression) fed by a fan runner
from a hot-runner gate (mold-flow-fangate2).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core import (
    DIRECT_LAYOUTS,
    DIRECT_OPEN_MODES,
    FAN_SHAPES,
    GATE2_DEFAULTS,
    GATE3_DEFAULTS,
    OVERLAP_TAB_DEFAULTS,
    RUNNER_SHAPES,
    TAB_SHAPES,
    FanRunnerPlateConfig,
    HeleShawSolver,
    MaterialDB,
    MultilayerHeleShawSolver,
    build_fan_runner_plate_geometry,
    build_fill_player_html,
    draw_gate_markers,
    export_frames,
    fill_frame_fractions,
    fill_frame_times,
    fill_player_height_px,
    packing_reach,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
    render_core_layer_map,
    render_fill_animation,
    render_packing_reach_map,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
    summarize_by_thickness,
    wrap_standalone_html,
)
from core.flash import flash_risk
from core.geometry import Geometry
from core.injection_profile import InjectionProfile, InjectionStage
from core.packing_solve import (
    coarsening_factor,
    solve_packing_coarsened,
    summarize_solve_by_thickness,
)
from core.settings_record import config_settings, settings_json
from core.solver import WELD_MIN_ANGLE_DEG
from core.transient_fill import enclosed_seams
from core.two_phase import solve_two_phase_short_shot
from core.version import build_label
from core.visualizer import (
    ISOCHRONE_LEVELS,
    THICKNESS_CMAP,
    export_two_phase_frames,
    render_flash_risk_map,
    render_layer_grid,
    render_packing_late_fill_map,
    render_packing_pressure_history,
    render_packing_shrinkage_map,
    render_short_shot_map,
    render_two_phase_animation,
    render_two_phase_map,
    two_phase_frame_labels,
)

APP_DIR = Path(__file__).parent

#: Default machine conditions for the screw-side injection block. The rate is
#: Q = pi*D^2/4 * v; the spec-sheet 589 cm^3/s of the direct input is the
#: machine's *maximum*, Phi50 x 200 mm/s gives 392.7 cm^3/s.
DEF_SCREW_DIAMETER_MM = 50.0
DEF_METERING_POSITION_MM = 150.0
DEF_VP_POSITION_MM = 18.0
DEF_SCREW_VELOCITY_MMS = 200.0
#: Same machine as the sibling repos, but this part is two orders of
#: magnitude larger, so the metering position is the one setting that
#: differs: once the stage count is 2 or more, the stroke has to displace
#: the cavity as drawn (plus a margin) before the volume-to-time map stops
#: extrapolating past V/P.
#: v0.8.0 (sim v0.55.0): one stage by default. At one speed the metering and
#: V/P positions only bound a stroke the solver never reads (the shot volume
#: is the two-phase input, not metering minus V/P), so a single stage asks
#: for the diameter and the speed and nothing else. The positions, the
#: switch points and the per-stage speeds appear once the stage count is 2
#: or more. v0.4.0-v0.7.0 opened on three equal-speed stages.
DEF_INJECTION_STAGES = 1
#: The machine's own switch points, pre-filled at ``len(...) + 1`` stages.
#: The other job's (28 / 22) do not carry over, so none are known here and
#: every stage count splits the stroke evenly.
DEF_SWITCH_POSITIONS_MM = ()
#: Cap on the stage count the UI will unfold. Machines go further; this is
#: the point past which the sidebar stops being readable.
MAX_INJECTION_STAGES = 5


#: Label recorded in ``settings.json`` for the one geometry input this app has.
GEOMETRY_SOURCE = "Fan runner plate (parametric)"

st.set_page_config(page_title="12.3 インチ 額縁プレート 流動解析", layout="wide")
st.title("12.3 インチ 額縁プレート 流動解析")
st.caption(
    "12.3 インチ 額縁肉厚プレート（外周 20 幅 t1.0／内側 t4.0、圧縮は内側だけ）を、ホットランナーゲート φ3 から"
    "通常成形または射出圧縮成形するときの樹脂流動を簡易解析するツール。ゲート側の形は、ファン状のランナ（Gate 1・2・3）、"
    "中央タブ、重ねタブ、厚いタブ、本体への直接ゲートから選ぶ。"
    "ゲートの形と寸法・肉盗み・成形条件の方向性検討を、実機評価前の初期検討段階で迅速に行うことを目的とする。"
    "商用 CAE（Moldflow 等）の代替を意図したものではない。"
)

with st.expander("📐 使用している方程式と適用範囲"):
    st.markdown("### 1. 全体モデル：Hele-Shaw 近似（薄板潤滑流れ）")
    st.markdown(
        "金型キャビティが**薄板（厚み $h \\ll$ 平面サイズ）**であることを前提に、"
        "面内 2D + 厚み方向の解析積分という簡略モデル。"
        "射出成形の薄肉製品では商用 CAE（Moldflow / Moldex3D）の中間面ソルバーも"
        "本質的に同じ Hele-Shaw 系を使う。"
    )
    st.latex(
        r"\nabla \cdot \left( S \, \nabla p \right) = 0,"
        r"\quad S = \frac{h^3}{12\,\eta_{\text{eff}}}"
    )
    st.markdown(
        "- $h(x,y)$: 局所キャビティ厚み [m]\n"
        "- $\\eta_{\\text{eff}}$: コンダクタンス計算用の有効粘度 [Pa·s]\n"
        "- $S$: コンダクタンス（流れやすさ）。$h^3$ で効くので**厚み変化が支配的**\n"
        "- 圧力 $p$ はゲートで一定、流動先端で 0 を境界条件にして解く"
    )

    st.markdown("### 2. 充填時間場：Pseudo-Conduction 法")
    st.markdown(
        "Hele-Shaw の圧力場を時間ステップで進めず、**楕円型（熱伝導型）方程式に置き換えて 1 発で解く**"
        "高速化テクニック。$\\tau$ は擬似的な「ゲートからの到達時間場」。"
    )
    st.latex(r"-\,\nabla \cdot \left( S\, \nabla \tau \right) = 1 \quad \text{in cavity}")
    st.markdown(
        "- ゲートで $\\tau = 0$（ディリクレ境界）、キャビティ壁で no-flux（ノイマン境界）\n"
        "- 解いた $\\tau$ は**順序**だけを使い、絶対時間は体積 CDF 写像で与える（v0.25.0）: "
        "セルの充填時刻 ＝ 総充填時間 × （そのセル以下の $\\tau$ を持つセル群の体積 / 解いた領域の体積）。"
        "定率射出なら先端は注入体積に比例して進むので、順序を体積で刻むのが物理そのもの\n"
        r"$t_{\text{fill}}(\mathbf{x}) = T_{\text{fill}} \cdot \dfrac{V(\tau \le \tau(\mathbf{x}))}{V_{\text{solved}}}$"
        "\n- $V_{\\text{solved}}$ は充填可能なセルの体積。封止で切られた未充填セルは含まないので、"
        "ショートショット時は最後に届くセルが $T_{\\text{fill}}$ を持つ\n"
        "- $T_{\\text{fill}} = V_{\\text{solved}} / Q$ は**定率射出・圧縮なし**のとき"
        "（壁面冷却なし、またはスキン層の「速度制御」時計）。"
        "射出圧縮成形なら §7 の等価モデルで短縮され、スキン層の「圧力一定」時計と層別の熱結合は"
        "体積重み付き $\\tau$ の比（抵抗比）で再スケールする — スキン層は膨張のみ、層別は層の粘度分布"
        "次第で 1 を挟んでどちらにも動く（そのとき $t_{\\text{fill}}$ は $V/Q$ で割った値ではない）\n"
        "- 多段射出（スクリュー径と射出速度から計算で段数 2 以上）では $Q$ が段ごとに変わるので、"
        "体積 → 時刻の写像が段の切替体積で折れる区分線形になる。"
        "各段の注入体積 $\\pi D^2/4 \\cdot L_i$ は速度によらずストロークだけで決まるので、"
        "折れ点の体積は固定で、傾き（＝射出率）だけが段ごとに変わる\n"
        "- 流動先端の進行は $\\tau$ の等値線として可視化"
    )

    st.markdown("### 3. 粘度モデル：Cross-WLF")
    st.markdown(
        "剪断速度依存の擬塑性 + 温度依存（WLF型）を組み合わせた業界標準モデル。"
        "`data/materials.json` に PP / PP_T10 / PP_T20 / PP_T30 / ABS / PC / PA66 / PMMA の代表値を保持。"
    )
    st.latex(
        r"\eta(\dot\gamma, T) = \frac{\eta_0(T)}"
        r"{1 + \left( \dfrac{\eta_0(T)\,\dot\gamma}{\tau^{*}} \right)^{1-n}}"
    )
    st.latex(r"\eta_0(T) = D_1 \exp\!\left[\,-\,\frac{A_1 (T - T^{*})}{A_2 + (T - T^{*})}\,\right]")
    st.markdown(
        "**温度・剪断速度の評価は壁面冷却モデルで変わる**:\n"
        "- **なし**: バルク温度 $T_{\\text{bulk}} = 0.7\\,T_{\\text{melt}} + 0.3\\,T_{\\text{mold}}$ と"
        " 代表剪断速度 $\\dot\\gamma = 6V/h$ で 1 回だけ評価\n"
        "- **スキン層 / 層別**: 厚み方向に **層別** $T_k$・$\\dot\\gamma_k$ を割り振り、各層で別個に $\\eta_k$"
        " を評価（後述 §4・§5）"
    )

    st.markdown("### 4. 壁面冷却モデル A：スキン層（Stefan / Neumann 1 層近似）")
    st.markdown(
        "金型壁で樹脂が固化して育つ「スキン層」を熱拡散の Stefan 解で表現し、"
        "流動はコア層 $h_{\\text{core}} = h - 2s$ のみを通る。**コア温度は melt のまま固定**。"
    )
    st.latex(r"s(t) = c_{\text{skin}} \sqrt{\alpha\,t}")
    st.markdown(
        "- $\\alpha$: 樹脂の熱拡散率（材料 DB から取得）\n"
        "- $c_{\\text{skin}}$: 成長定数（$\\sim 1.0$ が物理的代表値、UI で調整）\n"
        "- $\\tau$ と $s$ が相互依存するので fixed-point 反復で釣り合わせる\n"
        "- 時計は**露光時計**: 壁は先端が通過した瞬間から老化し続け、求解にはセルの役務期間の時間平均スキン $\\tfrac{2}{3}s$ を当てる\n"
        "- スキンが出会う年齢 $t_c$ に役務が届いたセル＝**封止**（充填後に閉じた、赤マーク）。閉じた後に届くセルは**未充填**（充填時間なし）"
    )

    st.markdown("### 5. 壁面冷却モデル B：層別 N 層離散化（選択式。二相モデルの射出相にも乗る）")
    st.markdown(
        "厚み方向を $N$ 層に離散化し、**各層に固有の温度・粘度・剪断速度** を持たせる。"
        "スキン層モデルが「壁面凍結フロント」しか扱わないのに対し、こちらは**コア内部の温度・粘度プロファイル**"
        "まで解像する。極薄プレート（$t < 0.5$ mm）で必須。"
    )

    st.markdown("**5-1. 厚み離散化**")
    st.latex(r"\zeta_k \in [0, 1],\quad h_k(x,y) = (\zeta_k - \zeta_{k-1}) \cdot h(x,y)")
    st.markdown(
        "- **wall_refined**（既定）: Chebyshev-Lobatto 点 "
        r"$\zeta_k = \tfrac{1}{2}(1 - \cos(\pi k / N))$ で壁近傍を細かく"
        "\n- **uniform**: 等間隔"
    )

    st.markdown("**5-2. 層別温度（Neumann 1D 重ね合わせ）**")
    st.latex(
        r"T_k(x,y) = T_{\text{mold}} + (T_{\text{melt}} - T_{\text{mold}}) \cdot "
        r"\left[\,\operatorname{erf}\!\left(\tfrac{z_k}{2\sqrt{\alpha t_{\text{arr}}}}\right)"
        r"+ \operatorname{erf}\!\left(\tfrac{h - z_k}{2\sqrt{\alpha t_{\text{arr}}}}\right) - 1\,\right]"
    )
    st.markdown(
        "両壁から育つ熱境界層の重ね合わせ。長時間極限の数値発散を避けるため "
        r"$T_k \ge T_{\text{mold}}$ で clamp。$t_{\text{arr}}(x,y)$"
        " は §2 の体積 CDF 写像によるセル到達時間。"
    )

    st.markdown("**5-3. 層別剪断速度（Poiseuille 解析微分）**")
    st.latex(r"\dot\gamma_k(x,y) = \frac{6 V}{h(x,y)} \cdot |2\zeta_k - 1|")
    st.markdown(
        "壁で最大、中央でゼロ。中央層は Cross-WLF の $D_1$ 発散を避けるため "
        "$\\dot\\gamma_{\\text{floor}} = 0.01 \\cdot 6V/h$ でクリップ。"
    )

    st.markdown("**5-4. 層の粘度から導通へ（潤滑の積分）**")
    st.latex(
        r"S_{\text{total}}(x,y) = \int_0^{h} \frac{(z - h/2)^2}{\eta(z)}\,dz"
        r" = h(x,y)^3 \sum_{k=1}^{N} \frac{m_k}{\eta_k(x,y)}"
    )
    st.latex(
        r"m_k = \left[\frac{(\zeta - 1/2)^3}{3}\right]_{\zeta_{k-1}}^{\zeta_k},"
        r"\quad \sum_k m_k = \frac{1}{12}"
    )
    st.markdown(
        "重みは中央面からの距離の 2 乗で、壁際の層ほど大きい。流速は厚み方向に連続なので、"
        "壁際の層が粘ると、その内側の層もまとめて遅くなる。"
        r"粘度が一様なら $S = h^3/(12\eta)$ に戻り、$N=1$ では従来と厳密に一致する。"
        "v0.10.0 までは層を独立した流路として扱い、等粘度の流速分布 $\\zeta(1-\\zeta)$ で重み付けしていた。"
        "壁際が固まると導通を大きく見積もる式で、溶けた芯が厚みの割合 $c$ のとき $1.5/c^2 - 0.5$ 倍"
        "（$c$ = 0.5 で 5.5 倍）になっていた。"
    )

    st.markdown("**5-5. 固定点反復で $\\tau$ と層フィールドを結合（充填の解き方「一発で解く」）**")
    st.markdown(
        r"$\tau \to t_{\text{arr}} \to T_k \to \eta_k \to S_{\text{total}} \to \tau_{\text{new}}$"
        " を $\\|\\Delta\\tau\\|_2 / \\|\\tau\\|_2 < $ tol まで反復。"
        "発散時のみ $\\omega = 0.7$ で適応的 damping。**ショートショット判定**は最終 iteration の中央層温度ベース:"
        r" $T_{\text{solid}} = T_{\text{mold}} + f_{\text{solid}} (T_{\text{melt}} - T_{\text{mold}})$。"
    )

    st.markdown("**5-6. 先端を時間で進める（充填の解き方「時間で追う」、既定）**")
    st.markdown(
        "固定点の代わりに、射出した量だけ先端を時間で進める。満杯のセルの圧力を解き、"
        "満杯のセルに接する先端のセル（$p = 0$）を流入で満たす。刻みごとに、機械が送った量がちょうどゲートから出ていく。"
        "各セルの層は樹脂が着いた時刻 $t_{\\text{arr}}$ で読む（5-2〜5-4 の式は同じ）。"
        "到着時刻は因果的に決まるので反復は無い。総充填時間は射出率どおり（体積 ÷ 射出率。射出圧縮成形では §7 の短縮込み）になる。"
        "「一発で解く」は $\\tau$ の大小で充填の順番を決めるので、厚い所を満たしきってから薄い所に入る。"
        "「時間で追う」は、厚い所が埋まりきる前に、先端が届いた薄い所へも入る。"
    )

    st.markdown("### 6. 剪断発熱（viscous dissipation, 段階1）")
    st.markdown(
        "粘性散逸による発熱を**層別モード内で**取り込む補正。"
        "極薄プレート + 高速射出で Brinkman 数 $Br \\gg 1$ になりがちな領域で必須。"
    )
    st.latex(
        r"\Delta T_{\text{shear},k}(x,y) = \frac{\eta_k \,\dot\gamma_k^{\,2}}{\rho \, c_p}"
        r"\cdot \min\!\left(t_{\text{arr}},\; \tau_{\text{thermal}}\right)"
    )
    st.latex(r"\tau_{\text{thermal}} = \frac{h^2}{\pi^2 \, \alpha}")
    st.markdown(
        "$\\tau_{\\text{thermal}}$ は厚み方向 1D 拡散の最低モード時定数で頭打ち。"
        "実態は **粘性散逸 vs 1D 壁面冷却** の準定常バランス近似。"
        " $T_k \\leftarrow T_{k,\\text{Neumann}} + \\Delta T_{\\text{shear},k}$ で Cross-WLF を再評価する。"
        "$\\eta_k$ は加熱後の $T_k$ で読み、$\\Delta T_{\\text{shear},k}$ と自己整合させてセルごとに解く"
        "（上昇が大きいほど $\\eta_k$ が下がり発熱が減るので、解は 1 つに決まる）。"
    )

    st.markdown("**Brinkman 数（剪断発熱の必要性診断、補正 OFF でも常時計算）**")
    st.latex(
        r"Br = \frac{\eta \,\dot\gamma^{\,2}\, h^2}{k \, (T_{\text{melt}} - T_{\text{mold}})},"
        r"\quad k = \alpha \cdot \rho \cdot c_p"
    )
    st.markdown(
        "- 🟢 $Br < 0.5$: 熱伝導支配、剪断発熱無視可\n"
        "- 🟡 $0.5 \\le Br < 2$: 同程度、補正 ON 推奨\n"
        "- 🔴 $Br \\ge 2$: 剪断発熱支配、本来は **段階2（1D FDM 陰解法）** が必要"
    )

    st.markdown("### 7. 射出圧縮成形（ICM、オプション）：等価厚み膨張モデル")
    st.markdown(
        "圧縮位相を時間ステッピングで解かず、**圧縮部の厚みを膨張**させた等価モデルとして扱う。"
        "流路抵抗 $S \\propto h^3$ が一気に下がる効果を擬似再現。"
        "膨張対象は**内側 t4.0 の矩形だけ**（この金型の圧縮ブロック）。額縁 t1.0 とランナ（ゲート φ3 まで）は"
        "射出時の肉厚のまま不変。"
        "UI は **stroke モード**（金型シム量の物理に整合）で統一。"
        "倍率指定の **factor モード**は CLI / solver 引数で後方互換のためだけに残す。"
    )

    st.markdown("**stroke モード（絶対加算、段差保存、UI 既定）**")
    st.markdown(
        "全 target セルに同じ絶対量（ストローク $s$）を**加算**する。"
        "**金型シム量**が設計指標のとき（＝実機の射出圧縮成形そのもの）に使う。"
        "段差プレート（例: 薄肉部 $t_0=0.35$ mm ／ 厚肉部 $t_0=0.50$ mm）に "
        "$s=0.70$ mm を加算すると薄肉部 $1.05$ mm ／ 厚肉部 $1.20$ mm となり、"
        "**段差 $0.15$ mm が圧縮位相中も保存される**（factor モードだと段差が $0.45$ mm に膨らんで非物理）。"
    )
    st.latex(r"h_{\text{eff}}(x,y) = h(x,y) + s \quad \text{on compression cells}")
    st.latex(
        r"T_{\text{fill}}^{\text{ICM}} = T_{\text{fill}}^{\text{base}} \cdot "
        r"\left[\,\frac{f_{\text{cmp}}}{1 + s \cdot A_{\text{cm}} / V_{\text{total}}}\,"
        r"+\,(1 - f_{\text{cmp}})\,\right]"
    )
    st.markdown(
        "- $s$: `compression_stroke_mm`（圧縮ストローク [mm]、絶対加算量）\n"
        "- $A_{\\text{cm}}$: 圧縮対象セルの**面積** [mm²]（`compression_area_mm2()`）\n"
        "- $V_{\\text{total}}$: 全キャビティ体積 [mm³]\n"
        "- $f_{\\text{cmp}}$: 充填占有率（圧縮開始時にキャビティ何%まで充填されているか）"
    )

    st.markdown("### 8. 保圧が届く時間（見積り、通常成形のとき）")
    st.markdown(
        "保圧の流れは解かない。充填の結果から、**ゲートからその場所まで溶けた通り道がいつまで残るか**だけを読む。"
        "温度の扱いは 5-2 と同じで、充填の終わりにそのセルにいる樹脂は、到達時刻 $t_{\\text{arr}}$ だけ"
        "金型温度の壁にはさまれて冷えている。充填のあとは流れを止め、同じ板の熱伝導で冷やす。"
    )
    st.latex(
        r"t_f(h) = \mathrm{Fo}^* \, \frac{h^2}{\alpha}, \qquad "
        r"s = \max\bigl(t_f(h) - t_{\text{arr}},\, 0\bigr), \qquad "
        r"R(x) = \max_{\text{path}:\,\text{gate} \to x} \; \min_{c \,\in\, \text{path}} s(c)"
    )
    st.markdown(
        "- $t_f(h)$: 厚さ $h$ の板の中心が固化温度 $T_{\\text{mold}} + f\\,(T_{\\text{melt}} - T_{\\text{mold}})$ まで下がる樹脂の年齢。"
        "$\\mathrm{Fo}^*$ は両面冷却の板の厳密解（級数）から求める（$f = 0.3$ で 0.1465）。**厚さの 2 乗に比例する**\n"
        "- $s$: 充填の終わりから数えて、そのセルの中心が固まるまでの時間（0 ＝ 充填の終わりに固まっている）\n"
        "- $R$: 保圧が届く時間。通り道は、その上でいちばん早く固まるセルと同じだけもつ。"
        "いちばん長くもつ通り道を取る（ゲートは固まらないとする）\n"
        "- 割合 $R / s$: そのセルが固まるまでの時間のうち、保圧が届いている割合\n\n"
        "見ていないもの: 保圧中に流れ込む樹脂の熱（入れると $R$ は延びる）、充填中にゲートの近くの壁にできる固化層"
        "（入れると縮む）、固化層で通り道が狭くなること、樹脂の縮み（pvT）。2 つの効果は逆向きなので秒は目安で、"
        "形どうしの順位と比に使う。射出圧縮成形のときは出さない。"
    )

    st.markdown("### 9. 保圧の計算（通常成形で「保圧を解く」が ON のとき）")
    st.markdown(
        "体積の VP 切替の充填率まで射出率で埋めた時点から、ゲートを保圧 $P_h$ で保持時間 $t_h$ だけ保持し、"
        "そのあとゲートを閉じる。樹脂は縮む（密度は圧力と温度で決まる）ので、保圧のあいだ樹脂が流れ込み続ける。"
        "セル（面積 $A$、厚さ $h$）の質量の保存を、時間を追って解く。"
    )
    st.latex(
        r"\frac{\partial}{\partial t}\bigl(\bar\rho(p)\, h\bigr) = "
        r"\nabla \cdot \bigl(\rho\, S\, \nabla p\bigr), \qquad "
        r"S = \int_{-h/2}^{h/2} \frac{z^2}{\eta(T(z),\, \tau)}\, dz, \qquad \tau = |\nabla p|\,|z|"
    )
    st.latex(
        r"v(T, p) = v_0(T)\left[1 - C \ln\!\left(1 + \frac{p}{B(T)}\right)\right], \qquad "
        r"S_v = 1 - \frac{M\, v(T_{\text{room}},\, 0)}{A\, h}"
    )
    st.markdown(
        "- $\\bar\\rho(p)$: 厚さで平均した密度。厚さ方向の温度は §8 と同じ板の冷え方（樹脂の年齢 $t_{\\text{arr}} + t$）で、"
        "各深さの密度を 2 領域 Tait の pvT（$v$ は比容積、$C = 0.0894$）から出す\n"
        "- $S$: 流れやすさ。各深さの Cross-WLF 粘度を、その場の剪断応力 $\\tau$ で下げて積分する。"
        "固化温度より冷えた層は流れない。**通り道が冷えて細るほど、保圧は届かなくなる**\n"
        "- 圧力は 0 より下がらない。質量が足りなくなったセルは壁から離れた（引けた）とみなし、圧力 0 のまま不足分を持つ\n"
        "- **VP 切替**: 充填の到達時刻の順に体積を積み、切替の充填率に達した時刻を切替とする。そこまでに樹脂が届いた"
        "セルは満杯、残りは空として始め、空のセルは満杯の隣から樹脂を受け取って埋まる。100 % にすると、全部埋まった"
        "ものとして保圧だけを解く\n"
        "- **機械が送る量の上限**: 保圧に切り替えても、機械は射出率より速くは送れない。上限を超える間は"
        "ゲートの圧力を保圧より低く保つ\n"
        "- $S_v$: 室温・大気圧まで冷えたあとの体積の縮み。$M$ は流れが止まったときのセルの質量。"
        "負は詰め込みすぎ（成形品がキャビティより大きくなる側）\n\n"
        "入っていないもの: ノズルとホットランナーの圧力損失（保圧はゲートの位置の圧力として扱う）、"
        "保圧中に流れ込む樹脂が運ぶ熱、型のたわみ、粘度の圧力依存、切替の時点の圧力分布（満杯のセルは一様な 1 MPa から始める）。"
        "pvT データのある材料（PMMA）でだけ解く。既定は充填と同じ格子で解き、「保圧を粗い格子で解く（検算用）」で"
        "2 万マス以下にまとめて解ける。"
    )
    st.warning(
        "**（層の温度を到達時刻から読む方のとき）薄い所が保圧で埋まるかの判定は当てにならない。** 厚さ方向の温度を解かず、"
        "「樹脂がそこにあった時間」と「その場所の肉厚」だけで決めている。切替の時点で空のセルにあとから入る樹脂も、"
        "充填の開始からその場所の肉厚で冷えてきたものとして扱うので、1 mm の額縁のような薄い所は、厚い所を通ってきた"
        "熱い樹脂が入る場合でも、最初から固まっている扱いになり、埋まらないと出る。厚い所の縮みと圧力は、"
        "形どうしの比較に使える。市販の解析（充填と保圧を通してエネルギーの式を解く）との大きな違いはここにある。"
        "層の温度を「刻みごとに解く」にすると、§10 の温度を切替の時点から保圧へ渡すので、この弱点は無くなる。"
    )

    st.markdown("### 10. 層の温度を刻みごとに解く（層別・時間で追う解き方で選んだとき）")
    st.markdown(
        "厚さの半分を 16 層（Gauss 点）に分け、各層の温度 $\\theta = (T - T_{\\text{mold}}) / (T_{\\text{melt}} - T_{\\text{mold}})$ を"
        "マスごとの状態として持ち、充填の刻みごとに 3 つの効果を順に足す。"
    )
    st.latex(
        r"\frac{\partial \theta}{\partial t} = \alpha \frac{\partial^2 \theta}{\partial z^2}"
        r" \;-\; \underbrace{\nabla\!\cdot(\mathbf{q}_k\, \theta_k)}_{\text{流れが運ぶ熱}}"
        r" \;+\; \frac{\text{剪断の仕事}}{\rho c_p\, \Delta T}"
    )
    st.markdown(
        "- **壁への熱伝導**: 樹脂の表面と金型の間に熱伝達係数 $h_c$ を置く（表面の熱の流れ $= h_c\\,(T_{\\text{表面}} - T_{\\text{金型}})$）。"
        "充填・保圧・壁から離れたあと（保圧中に圧力が 0 になり、質量が足りなくなったマス）の 3 つで値を替える。既定は "
        "5000／2500／1250 W/(m²·K) で、Moldflow の既定と同じ。効き方はビオ数 $Bi = h_c h / k$ で決まり、薄い所ほど小さく、"
        "冷えるのが遅くなる（PMMA の 1 mm で充填中 $Bi \\approx 25$）。「表面＝金型温度」を選ぶと $h_c \\to \\infty$（v0.15 までの扱い）。"
        "熱伝導の行列を固有分解して持ち、刻みの長さによらず厳密に進める（ビオ数は 2 % 刻みでまとめる）\n"
        "- **流れが運ぶ熱**: 面を通った樹脂の量を、層ごとの流量の配分（固まった層は 0）で分け、上流の層の温度を運ぶ。"
        "先端が入るマスは、入ってくる樹脂の流量平均の温度をすべての層に入れる（ファウンテンフロー）。ゲートからは溶融温度\n"
        "- **剪断発熱**: 面ごとの「流量 × 圧力差」を両側のマスに抵抗の比で分けて、層ごとの発熱にする。合計は圧力の仕事"
        "（ゲート圧 × 送った体積）と一致する\n"
        "- **流れやすさ**: その場所の流量から壁の剪断応力を解き、Cross-WLF の粘度を層ごとに下げて積分する"
        "（代表流動速度は使わない）\n"
        "- **射出圧の上限**: ゲートの圧力が上限に達すると流量が落ちる。流れが止まれば、そこで充填を止める。"
        "止まった理由は結果の先頭に出す: 上限に達して流量が落ちたのか、上限で流量が落ちる前に残りへ入る所のマスが中心まで固まったのか"
        "（どちらとも示せないときは、そう書く）。"
        "\n"
        "- **先端のマス**: 流れやすさは、そのマスに入っている樹脂の温度で決める。先に少し入って固まったマスは、"
        "その樹脂を送ってきた隣のマスからは、もう入れない（先に入った樹脂を押して進むことになる）。"
        "別の向きから後で届いた先端は、マスの空いた側に当たるので、入ってくる樹脂の温度と、そのマスの肉厚で決めた流れやすさで入る"
        "（先端のマスが樹脂を受け始めたあとに満ちた隣のマスを、後から届いた先端とみなす）。"
        "この扱いが無いと、先に止まって固まった先端と後から来た先端の合わせ目に、幅 1 マスの線が埋まらずに残る\n"
        "- **保圧へ**: VP 切替の時点の状態をそのまま保圧の計算（§9）へ渡し、保圧中も同じ 3 つの効果を刻みごとに解く\n\n"
        "入っていないもの: 面内（板の広がる向き）の熱伝導、金型の中の温度の分布（金型の表面は一様に金型温度）、"
        "射出圧縮成形との組み合わせ（射出圧縮成形では到達時刻から読む方で解く）。"
        "格子は 2 mm 以下で使う。4 mm 格子は軸に対して左右対称に切れず、流れが集まる通り道が片側に寄って、結果が左右で違って出る。"
        "熱伝導と運びは刻みの中で小分けにして解く（1 刻みにマスを通り過ぎる流れで冷えすぎないように）。\n\n"
        "**薄いゲート（入口 1 mm のファン、2 mm のタブなど）の読み方**: 保圧のあいだ流れ続ける樹脂が熱を運ぶので、"
        "薄い首は開いたままになりやすく、ゲートの形による縮みの差が小さく出る。到達時刻から読む方は逆に、首が保圧の初めに"
        "固まる扱いで差が大きく出る。試作の重量で確かめるまで、答えは 2 つの間にあると読む。"
    )

    st.markdown("### 11. バリ注意（PL、保圧を解いたとき）")
    st.markdown(
        "キャビティの輪郭（製品とランナの外周のマス）を PL の合わせ目とみなし、輪郭の各マスについて、"
        "厚さの中心が溶けている間に受けた最大の圧力 $p_{\\text{molten}} = \\max_{t < t_{\\text{freeze}}} p(t)$ を出す。"
        "固まったあとの圧力は数えない（固まった樹脂はすき間に入らない）。製品の輪郭とランナの輪郭は別々に順位を付け、"
        "上位から 20 mm 以上離れた所を 5 か所ずつ選ぶ。型を開く力（$\\int p\\, dA$ の最大）と機械の型締力も並べる。\n\n"
        "入っていないもの: すき間の深さ（ベント・摩耗）、型のたわみ、すき間に入る樹脂の流れ。何 MPa でバリが出るかは型ごとに判断する。"
        "試作でバリが出始めた所の圧力が分かったら、保圧の欄の「バリの目安の圧力」に入れる。その圧力以上の輪郭に紫の枠が付き、長さが出る"
        "（空欄なら線は引かない。既定の値は持たない）。"
        "層の温度を到達時刻から読む方では、薄い額縁が切替の時点で固まった扱いになり、製品の輪郭が低く（0 に）出る。"
    )

    st.markdown("---")
    st.markdown("### ✅ モデル化している現象")
    st.markdown(
        "- 薄板キャビティ内の 2D 流動（局所剪断と局所抵抗の効果）\n"
        "- 樹脂物性（密度・比熱・熱拡散率 → 熱伝導率派生・Cross-WLF 粘度パラメータ）\n"
        "- 額縁プレートの 2 段肉厚、ランナ（Gate 1 逆三角形／Gate 2 五角形／Gate 3 逆三角形＋フランク沿いの腕。幅・R 丸端・深さ方向の肉厚プロファイル）、肉盗み（▽）\n"
        "- 流動先端の到達順、ウェルドライン、エアトラップ\n"
        "- 圧力分布の相対値（ゲート＝1、最終充填点＝0 の正規化）\n"
        "- 充填時間（射出率 $Q$ から逆算した絶対時間）\n"
        "- **壁面冷却**: スキン層 1 層モデル または **層別 $N$ 層モデル**（厚み方向温度・粘度プロファイル）\n"
        "- **剪断発熱（段階1, 層別モード内）**: 粘性散逸による局所温度上昇、Brinkman 数診断\n"
        "- **ショートショット予測**: スキン層モデルでは露光時計の封止（$t_{\\text{close}} = t_{\\text{arr}} + t_c$）で切られたセル、"
        "層別モデルでは中央層温度ベース判定\n"
        "- 射出圧縮成形による等価流路拡大（stroke モード、CLI に factor 後方互換あり、オプション）\n"
        "- **保圧が届く時間の見積り**（通常成形のとき）: 充填のあと流れを止めて冷やし、ゲートからの溶けた通り道が残る時間を読む\n"
        "- **保圧の計算**（通常成形で「保圧を解く」ON、pvT のある材料）: 圧力の時間変化、保圧で入る樹脂の量、"
        "体積の縮み、型を開く力（§9）\n"
        "- **層の温度を刻みごとに解く**（選んだとき）: 熱伝導・流れが運ぶ熱・剪断発熱を厚さ 16 層で、充填から保圧まで通して解く（§10）\n"
        "- **バリ注意（PL）**（保圧を解いたとき）: 輪郭が溶けている間に受けた最大の圧力、型を開く力と機械の型締力（§11）"
    )

    st.markdown("### ❌ モデル化していない現象（重要）")
    st.markdown(
        "- **面内 3D 流れ・ジェッティング・噴流・コーナー渦**: あくまで 2D Hele-Shaw（厚み方向は層別化済みだが、"
        "**面内**の 3D 性は完全 3D FVM / FEM ソルバーでないと出ない。Hele-Shaw 系の根本限界）\n"
        "- **剪断発熱の自己整合（段階2）**: 段階1 は閉形式の局所近似のみ。"
        r"$\rho c_p \partial_t T = k \partial_z^2 T + \eta \dot\gamma^2$ を厚み方向 1D FDM で陰解法積分する"
        "段階2 は別ロードマップ。$Br \\ge 2$ 領域では段階1 がズレる\n"
        "- **保圧中の熱と型のたわみ**: §9 の保圧の計算は、流れ込む樹脂が運ぶ熱、型のたわみ、"
        "ノズルとホットランナーの圧力損失を入れていない\n"
        "- **反り・残留応力**: 体積の縮み（§9）までで、縮みの向き・反り・残留応力は出ない。結晶化も入っていない\n"
        "- **層内対流項**: 1D Neumann は純粋拡散のみ（薄板では妥当な近似だが、極厚 $h > 4$ mm では破綻）\n"
        "- **ベント・脱気挙動**: エアトラップ位置は予測するが圧抜けは考慮しない\n"
        "- **ホットランナー／ゲートの圧損**: ゲート φ3 の円を射出点として扱い、ノズル・ゲートの縦方向抵抗は入っていない\n"
        "- **STL/STEP 直接読み込み**: パラメトリック形状（額縁プレート＋ランナ 2 種）のみ\n"
        "- **非構造格子・中立面メッシュ**: 構造格子（正方形セル）固定\n"
        "- **充填中の絶対圧力場**: 充填の圧力マップは正規化値（ゲート=1 / フロント=0）のみ。"
        "充填中の必要型締力は出ない（保圧中の型を開く力は §9 で出る）"
    )

    st.markdown("### 用途と適用範囲")
    st.markdown(
        "本ツールは**初期スクリーニング・概念検証**用途。商用 CAE "
        "（Moldflow / Moldex3D 等）の置き換えではない。\n\n"
        "- ◯ 向く：ゲート位置候補の比較、ランナー形状の方向性決定、"
        "**極薄プレート（$t < 0.5$ mm）の壁面冷却・剪断発熱の効きの可視化**、"
        "ランナ寸法・肉盗みの変更が充填順序に与える効果、圧縮ストロークと計量体積の当たり\n"
        "- × 向かない：寸法精度予測、保圧設計、収縮反り、面内コーナー流動詳細、最終肉厚分布の精密計算"
    )


# NOTE: Material DB schema version is embedded as a cache key so that
# Streamlit Cloud invalidates the @st.cache_resource entry whenever the
# Material dataclass shape changes (e.g. new field added). Without this,
# old Material instances pickled in the deploy's persistent cache lack
# the new attribute and the solver raises AttributeError.
# Bump this string when you add/remove fields on `core.materials.Material`.
# The parameter must NOT start with an underscore: Streamlit leaves
# underscore-prefixed arguments out of the cache hash (Codex P2, PR #5).
_MATERIAL_DB_SCHEMA_VERSION = "v3_pvt"


@st.cache_resource
def _load_db(schema_version: str = _MATERIAL_DB_SCHEMA_VERSION) -> MaterialDB:
    return MaterialDB()


db = _load_db()
material_keys = list(db.keys())

# 「保圧が届く時間」を出さないときに結果ペインへ出す理由
PACK_SKIP_ICM = (
    "射出圧縮成形の解析では出しません（圧縮が保圧の役をし、"
    "キャビティの厚さも充填中に変わるため）。"
    "成形方法を通常成形にして解析すると出ます。"
)
PACK_SKIP_FRACTION = (
    "固化判定 fraction が 0 のときは出しません（厚さの中心が金型温度まで下がる時刻は"
    "決まらないため）。0 より大きい値で解析すると出ます。"
)


# 保圧の計算を解かないときに結果ペインへ出す理由
HOLD_SKIP_OFF = "サイドバー「保圧」で「保圧を解く」を ON にして解析すると出ます。"
HOLD_SKIP_ICM = (
    "射出圧縮成形の解析では解きません（圧縮が保圧の役をし、キャビティの厚さも充填中に変わるため）。"
)
HOLD_SKIP_PVT = "この材料には pvT データが無いので解けません（pvT があるのは PMMA）。"
HOLD_SKIP_SHORT = "充填が全体に届いていない（ショートショット）ので解きません。"
# VP 切替のあと保圧で埋める計算の、いちばん大きい弱点。結果ペインに必ず出す
HOLD_THIN_CAVEAT = (
    "薄い所（1 mm の額縁など）が保圧で埋まるかの判定は当てにならない。切替のあとに入る樹脂の"
    "温度を、充填の開始からその場所の肉厚で冷えてきたものとして決めているので、厚い所を"
    "通ってきた熱い樹脂でも、薄い所では最初から固まっている扱いになる。"
    "そのため、薄い所は実際より埋まりにくく出る。厚い所の縮みと圧力の比較には使える。"
)
# 保圧の計算に使うセル数の上限。これを超える格子は、保圧だけ粗い格子に写して解く
# （直接法の連立方程式を 1 刻みに何度も解くので、セル数で時間が決まる）。
HOLD_MAX_CELLS = 20_000


# ----------------------- fan-runner plate: parametric inputs -----------------------
_D = FanRunnerPlateConfig()  # Gate 1 = the drawing (docs/spec.md)
_D2 = FanRunnerPlateConfig(**GATE2_DEFAULTS)  # Gate 2 = the pentagon (docs/spec.md)
_D3 = FanRunnerPlateConfig(**GATE3_DEFAULTS)  # Gate 3 = Gate 1 + flank arms (docs/spec.md)
_DOV = FanRunnerPlateConfig(**OVERLAP_TAB_DEFAULTS)  # overlap tab (the t3.5 drawing)
_SHAPE_LABELS = {
    "triangle": "Gate 1（逆三角形・客先図面）",
    "pentagon": "Gate 2（五角形・側辺つき）",
    "triangle_arms": "Gate 3（逆三角形＋フランク沿いの腕）",
    "tab": "中央タブ（幅の狭いゲート、額縁から入れる）",
    "overlap_tab": "重ねタブ（額縁の下に重ねて本体へつなぐ）",
    "thick_tab": "厚いタブ（額縁を切り、本体の肉厚でつなぐ）",
    "direct": "直接ゲート（本体にバルブゲート）",
}
#: 成形方法: the sidebar's first choice
# the first one is the default: the trials run as conventional molding
MOLDING_METHODS = ("conventional", "icm")
_MOLDING_LABELS = {"icm": "射出圧縮成形", "conventional": "通常成形"}
_RUNNER_EXPANDER = "ゲートブロック（ランナ、製品エッジ直結）"
_GATE_EXPANDER = dict.fromkeys(RUNNER_SHAPES, _RUNNER_EXPANDER) | {
    "direct": "直接ゲート（位置と径）"
}
# what each shape does to the gate-side band of the rim (the fourth side)
_RIM_KEPT = "ゲート側の額縁の帯は、ほかの 3 辺と同じ額縁のまま。"
_GATE_SIDE_NOTE = dict.fromkeys(RUNNER_SHAPES, _RIM_KEPT) | {
    "overlap_tab": "ゲート側の額縁の帯は、タブの幅だけ厚くなる（額縁の下にタブが重なる）。ほかは額縁のまま。",
    "thick_tab": "ゲート側の額縁の帯は、中央がタブの肉厚、その外は切り取り（樹脂が入らない）、角だけ額縁。",
    "direct": "ランナは無い。" + _RIM_KEPT,
}


def _fan_runner_sidebar() -> dict:
    """Draw the geometry widgets and return the raw values (keys = config fields).

    Every widget carries a ``fg_<field>`` key so AppTest can drive it."""
    v: dict = {}

    def num(
        field: str, label: str, lo: float, hi: float, step: float, *, fmt: str = "%.2f"
    ) -> None:
        v[field] = st.number_input(
            label,
            min_value=lo,
            max_value=hi,
            value=float(getattr(_D, field)),
            step=step,
            format=fmt,
            key=f"fg_{field}",
        )

    v["runner_shape"] = st.radio(
        "ゲート側の形",
        RUNNER_SHAPES,
        format_func=_SHAPE_LABELS.__getitem__,
        key="fg_runner_shape",
        help="Gate 1 は客先図面の逆三角形（フランク 14°）。Gate 2 は製品エッジの両端から側辺を"
        "下ろし、斜辺を丸端の円に接して結ぶ五角形。Gate 3 は Gate 1 のフランクに沿って厚い溝"
        "（腕）を彫った形。この 3 つは肉厚の決め方（エッジ帯・傾斜・円の外・円の中）と肉盗みが共通。"
        "中央タブは、額縁の中央に幅の狭いタブをつなぐ形。重ねタブは、そのタブを額縁の下に重ねて本体の段まで"
        "伸ばす形。厚いタブは、ゲート側の額縁を切り、中央だけ本体の肉厚でつなぐ形。"
        "直接ゲートは、ランナを使わず本体にバルブゲートを置く形。",
    )
    gate2 = v["runner_shape"] == "pentagon"
    gate3 = v["runner_shape"] == "triangle_arms"
    with st.expander("製品（額縁プレート）", expanded=False):
        st.caption(
            "製品部は、内側の厚肉部（本体）と、左右・天側の 3 辺の額縁。ゲート側の 1 辺の額縁の帯は"
            "「ゲート側の形」で決まる。圧縮されるのは本体だけで、額縁・ゲート側の帯・ランナは固定。"
        )
        num("plate_w_mm", "製品幅（長辺、ランナ側）[mm]", 50.0, 600.0, 1.0)
        num("plate_h_mm", "製品奥行 [mm]", 30.0, 600.0, 1.0)
        num("frame_w_mm", "額縁幅 [mm]", 0.0, 100.0, 0.5, fmt="%.1f")
        num("frame_thk_mm", "額縁肉厚 [mm]", 0.1, 10.0, 0.05)
        num("inner_thk_mm", "内側肉厚（圧縮部）[mm]", 0.1, 10.0, 0.05)
    fan = v["runner_shape"] in FAN_SHAPES
    with st.expander(_GATE_EXPANDER[v["runner_shape"]], expanded=False):
        st.caption(_GATE_SIDE_NOTE[v["runner_shape"]])
        if fan:
            _fan_gate_block(v, num, gate2, gate3)
        elif v["runner_shape"] in TAB_SHAPES:
            _tab_gate_block(v, num)
        elif v["runner_shape"] == "thick_tab":
            _thick_tab_gate_block(v, num)
        else:
            _direct_gate_block(v, num)
    if fan:
        _balancer_block(v)
    else:
        v["balancer_on"] = False
    with st.expander("メッシュ", expanded=False):
        v["cell_size_mm"] = st.slider(
            "メッシュ粗さ [mm/cell]", 0.5, 4.0, float(_D.cell_size_mm), 0.25, key="fg_cell_size_mm"
        )
        st.caption("既定 1.0 mm で約 85k セル。0.5 mm は 4 倍重い。")
        v["pad_mm"] = float(_D.pad_mm)
    return v


def _fan_gate_block(v: dict, num, gate2: bool, gate3: bool) -> None:
    """The fan runners' widgets (Gate 1 / 2 / 3): silhouette, thickness profile, arms."""
    num("runner_w_mm", "ランナ幅（製品エッジ）[mm]", 5.0, 600.0, 1.0, fmt="%.1f")
    if gate2:
        v["fan_flank_deg"] = float(_D.fan_flank_deg)
        v["side_len_mm"] = st.number_input(
            "側辺の長さ（額縁の下端＝製品エッジから）[mm]",
            min_value=0.0,
            max_value=300.0,
            value=float(_D2.side_len_mm),
            step=1.0,
            format="%.1f",
            key="fg_side_len_mm",
        )
    else:
        num(
            "fan_flank_deg",
            "フランク角（製品エッジ線に対して）[deg]",
            1.0,
            89.0,
            0.5,
            fmt="%.1f",
        )
        v["side_len_mm"] = float(_D.side_len_mm)
    num("runner_len_mm", "ランナ長（製品エッジ → ゲート軸）[mm]", 1.0, 300.0, 1.0, fmt="%.1f")
    num("runner_end_d_mm", "丸端の径 φ（ゲート軸中心）[mm]", 1.0, 200.0, 1.0, fmt="%.1f")
    r_top = v["runner_len_mm"] - v["runner_end_d_mm"] / 2
    if gate2:
        _g2 = FanRunnerPlateConfig(
            runner_shape="pentagon",
            runner_w_mm=v["runner_w_mm"],
            side_len_mm=v["side_len_mm"],
            runner_len_mm=v["runner_len_mm"],
            runner_end_d_mm=v["runner_end_d_mm"],
        )
        _cap = (
            f"額縁込みの側辺 h = {v['frame_w_mm'] + v['side_len_mm']:.1f} mm"
            f"（額縁 {v['frame_w_mm']:.1f} ＋ 側辺 {v['side_len_mm']:.1f}）。"
        )
        try:
            _g2.validate()
        except ValueError:
            pass  # the builder reports it; the caption only adds numbers when they exist
        else:
            _xt, _dt = _g2.pentagon_tangent_mm
            _cap += (
                f"斜辺は丸端の円に接する（傾き {_g2.pentagon_slant_deg:.1f}°、接点は軸から "
                f"±{_xt:.1f} mm・深さ {_dt:.1f} mm）。側辺は丸端の下端 "
                f"{v['runner_len_mm'] + v['runner_end_d_mm'] / 2:.1f} mm より浅いこと。"
            )
        st.caption(_cap)
    else:
        _apex = FanRunnerPlateConfig(
            runner_w_mm=v["runner_w_mm"], fan_flank_deg=v["fan_flank_deg"]
        ).apex_depth_mm
        st.caption(
            f"フランクの延長が軸で交わる深さ {_apex:.1f} mm（丸端の上端 "
            f"{r_top:.1f} mm より深いこと）。"
            "図面ではフランクが丸端の円に交差する（接線ではない）。"
        )
    st.markdown("**肉厚（製品エッジからの深さで決まる）**")
    num("runner_edge_thk_mm", "エッジ帯の肉厚 [mm]", 0.05, 10.0, 0.05)
    num(
        "runner_edge_flat_mm",
        "エッジ帯の幅（製品エッジから。0 で帯なし）[mm]",
        0.0,
        50.0,
        0.5,
        fmt="%.1f",
    )
    v["runner_ramp_on"] = st.checkbox(
        "傾斜を付ける（エッジ帯の肉厚 → 円の外の肉厚 t_o）",
        value=_D.runner_ramp_on,
        key="fg_runner_ramp_on",
        help="外すと、エッジ帯の先で段差になって t_o に上がる。エッジ帯の幅が 0 なら製品エッジから t_o。",
    )
    if not v["runner_ramp_on"]:
        v["runner_ramp_end_mm"] = (_D2 if gate2 else _D).runner_ramp_end_mm
    elif gate2:
        follow = st.checkbox(
            "傾斜の開始位置を丸端の円の上端に合わせる",
            value=True,
            key="fg_g2_ramp_at_end_top",
            help="ランナ長と丸端の径を変えると開始位置も動く。外すと深さを数値で指定する。",
        )
        if follow:
            v["runner_ramp_end_mm"] = None
            st.caption(f"傾斜の開始位置（下限）= {r_top:.1f} mm（丸端の円の上端）")
        else:
            v["runner_ramp_end_mm"] = st.number_input(
                "傾斜の開始位置（下限、製品エッジからの深さ）[mm]",
                min_value=0.0,
                max_value=300.0,
                value=float(max(r_top, 0.0)),
                step=1.0,
                format="%.1f",
                key="fg_g2_runner_ramp_end_mm",
            )
    else:
        num(
            "runner_ramp_end_mm",
            "傾斜の開始位置（下限、製品エッジからの深さ）[mm]",
            0.0,
            300.0,
            1.0,
            fmt="%.1f",
        )
    num("runner_thk_mm", "円の外の肉厚 t_o（傾斜ありは厚い側）[mm]", 0.1, 10.0, 0.05)
    if gate2:
        v["runner_end_thk_mm"] = st.number_input(
            "円の深さ（丸端の円の中の肉厚）[mm]",
            min_value=0.1,
            max_value=10.0,
            value=float(_D2.runner_end_thk_mm),
            step=0.05,
            format="%.2f",
            key="fg_g2_runner_end_thk_mm",
        )
    elif st.checkbox(
        "丸端の円の中を別の肉厚にする",
        value=False,
        key="fg_g1_end_thk_on",
        help="外したままなら、円の中も上の深さの決め方に従う（客先図面のとおり）。",
    ):
        v["runner_end_thk_mm"] = st.number_input(
            "円の深さ（丸端の円の中の肉厚）[mm]",
            min_value=0.1,
            max_value=10.0,
            value=float(_D2.runner_end_thk_mm),
            step=0.05,
            format="%.2f",
            key="fg_g1_runner_end_thk_mm",
        )
    else:
        v["runner_end_thk_mm"] = None
    if gate3:
        st.markdown("**腕（フランク沿いの厚い溝）**")
        # the bound comes from the config itself (single source with validate())
        _w_sup = FanRunnerPlateConfig(
            runner_w_mm=v["runner_w_mm"], fan_flank_deg=v["fan_flank_deg"]
        ).arm_w_sup_mm
        # the largest 0.5 multiple that validate() accepts, i.e. strictly below
        # the bound minus its 1e-6 tolerance: 36.29 → 36.0, 75.0 (sin 30°) → 74.5
        # (Codex P2 on PR #14: flooring after subtracting a step lost 36.0)
        _w_hi = max((math.ceil((_w_sup - 1e-6) * 2.0) - 1) / 2.0, 0.5)
        v["arm_w_mm"] = st.number_input(
            "腕の幅（フランクから直角に測る）[mm]",
            min_value=0.5,
            max_value=float(_w_hi),
            value=min(float(_D3.arm_w_mm), float(_w_hi)),
            step=0.5,
            format="%.1f",
            key="fg_g3_arm_w_mm",
        )
        v["arm_thk_mm"] = st.number_input(
            "腕の肉厚（エッジ帯より厚く）[mm]",
            min_value=0.1,
            max_value=10.0,
            value=float(_D3.arm_thk_mm),
            step=0.05,
            format="%.2f",
            key="fg_g3_arm_thk_mm",
        )
        st.caption(
            "腕はエッジ帯の先から丸端の円の手前まで、両方のフランクに沿って彫る。"
            "溝は削るだけなので、今の肉厚より薄い所は変えない。エッジ帯と丸端の円の肉厚は"
            f"そのまま。腕の幅はフランクから軸までの距離 {_w_sup:.1f} mm 未満。"
        )
    else:
        v["arm_w_mm"] = float(_D.arm_w_mm)
        v["arm_thk_mm"] = float(_D.arm_thk_mm)
    num("gate_d_mm", "ゲート径 φ（ホットランナー、射出点）[mm]", 0.5, 50.0, 0.5, fmt="%.1f")


def _balancer_block(v: dict) -> None:
    """The balancer's widgets (fan runners only)."""
    with st.expander("肉盗み（▽ フローバランサー）", expanded=False):
        v["balancer_on"] = st.checkbox(
            "肉盗みを有効化（製品エッジに底辺を置く逆三角形の薄肉部）",
            value=_D.balancer_on,
            key="fg_balancer_on",
            help="底辺は製品エッジ線、頂点はゲート側。ランナ幅中央に置く。ランナの内側だけ削る"
            "（圧縮部には入らない。エッジ帯の t1.0 より薄くしない限り、そこは削れない）。",
        )
        # the bounds come from the config itself (single source with validate());
        # v holds every field the limits depend on by this point
        w_max, h_max, thk_sup = FanRunnerPlateConfig(**v).balancer_limits_mm
        W_MIN, H_MIN, THK_MIN, THK_STEP = 1.0, 1.0, 0.05, 0.05
        thk_hi = round(thk_sup - THK_STEP, 2)
        if v["balancer_on"] and (h_max < H_MIN or w_max < W_MIN or thk_hi < THK_MIN):
            st.warning(
                "この形状には肉盗みを置けない（ランナ長 − 丸端半径 = "
                f"{h_max:.1f} mm、エッジのランナ幅 {w_max:.1f} mm、ランナ肉厚 {thk_sup:.2f} mm）。"
                "ランナを長く／広く／厚くするか、肉盗みを OFF にする。"
            )
            v["balancer_on"] = False
        if v["balancer_on"]:
            # the upper bounds follow the runner; the defaults are clamped so a
            # short runner does not start above its own maximum
            for field, label, lo, hi, step, fmt in (
                ("balancer_w_mm", "底辺幅（製品エッジ）[mm]", W_MIN, w_max, 1.0, "%.1f"),
                (
                    "balancer_h_mm",
                    "高さ（製品エッジ → 頂点、丸端の手前まで）[mm]",
                    H_MIN,
                    h_max,
                    1.0,
                    "%.1f",
                ),
                (
                    "balancer_thk_mm",
                    "薄肉部の厚み（ランナ肉厚より薄く）[mm]",
                    THK_MIN,
                    thk_hi,
                    THK_STEP,
                    "%.2f",
                ),
            ):
                v[field] = st.number_input(
                    label,
                    min_value=lo,
                    max_value=float(hi),
                    value=min(float(getattr(_D, field)), float(hi)),
                    step=step,
                    format=fmt,
                    key=f"fg_{field}",
                )
        else:
            for f in ("balancer_w_mm", "balancer_h_mm", "balancer_thk_mm"):
                v[f] = float(getattr(_D, f))


def _runner_end_inputs(v: dict, num) -> None:
    """Runner length, round end and gate orifice: shared by the tabs."""
    num("runner_len_mm", "ランナ長（製品エッジ → ゲート軸）[mm]", 1.0, 300.0, 1.0, fmt="%.1f")
    num("runner_end_d_mm", "丸端の径 φ（ゲート軸中心）[mm]", 1.0, 200.0, 1.0, fmt="%.1f")
    num("gate_d_mm", "ゲート径 φ（ホットランナー、射出点）[mm]", 0.5, 50.0, 0.5, fmt="%.1f")


def _tab_gate_block(v: dict, num) -> None:
    """Centre tab and overlap tab: a trapezoid from the edge line to the round end."""
    overlap = v["runner_shape"] == "overlap_tab"
    num("tab_w_mm", "タブの幅（製品エッジ）[mm]", 1.0, 600.0, 1.0, fmt="%.1f")
    if overlap:
        v["tab_thk_mm"] = st.number_input(
            "タブの肉厚（製品の外、均一）[mm]",
            min_value=0.1,
            max_value=10.0,
            value=float(_DOV.tab_thk_mm),
            step=0.05,
            format="%.2f",
            key="fg_ov_tab_thk_mm",
        )
        v["lap_thk_mm"] = st.number_input(
            "額縁に重なる所の肉厚（額縁＋タブ）[mm]",
            min_value=float(v["frame_thk_mm"]),
            max_value=max(10.0, float(v["frame_thk_mm"])),
            value=max(float(_DOV.lap_thk_mm), float(v["frame_thk_mm"])),
            step=0.05,
            format="%.2f",
            key="fg_ov_lap_thk_mm",
        )
        _runner_end_inputs(v, num)
        return
    num("tab_end_thk_mm", "入口の肉厚（製品エッジ）[mm]", 0.1, 10.0, 0.05)
    num("tab_thk_mm", "奥の肉厚（ゲート付近と丸端の中）[mm]", 0.1, 10.0, 0.05)
    _runner_end_inputs(v, num)
    ramp_max = max(v["runner_len_mm"] - v["runner_end_d_mm"] / 2, 0.0)
    v["tab_ramp_len_mm"] = st.number_input(
        "傾斜の長さ（製品エッジから、奥の肉厚になるまで）[mm]",
        min_value=0.0,
        max_value=float(ramp_max),
        value=min(float(_D.tab_ramp_len_mm), float(ramp_max)),
        step=1.0,
        format="%.1f",
        key="fg_tab_ramp_len_mm",
        help=f"0 で段差。上限は丸端の円の上端までの {ramp_max:.1f} mm（ランナ長 − 丸端の半径）。",
    )


def _thick_tab_gate_block(v: dict, num) -> None:
    """Thick tab: the gate-side band is cut away but for the tab and the corners."""
    num("thick_tab_w_mm", "タブの幅（製品エッジ）[mm]", 1.0, 600.0, 1.0, fmt="%.1f")
    num("thick_tab_thk_mm", "タブの肉厚（ランナと、額縁の帯の中央）[mm]", 0.1, 10.0, 0.05)
    num(
        "corner_keep_mm",
        "角に残す額縁の長さ（本体の横の縁から内側へ）[mm]",
        0.0,
        300.0,
        1.0,
        fmt="%.1f",
    )
    inner_half = 0.5 * v["plate_w_mm"] - v["frame_w_mm"]
    cut = inner_half - v["corner_keep_mm"] - 0.5 * v["thick_tab_w_mm"]
    st.caption(
        f"額縁の帯を切る長さは、片側 {max(cut, 0.0):.1f} mm（タブの端から、角に残す額縁まで）。"
        + ("" if cut > 0 else "切る所が無く、帯はタブと額縁でつながる。")
    )
    _runner_end_inputs(v, num)


_DIRECT_LAYOUT_LABELS = {
    "center": "本体の中心に 1 点",
    "corners4": "本体の 4 つの隅に 1 点ずつ",
    "diagonal2": "対角の 2 つの隅に 1 点ずつ",
}


_DIRECT_OPEN_LABELS = {
    "together": "同時に開く（既定）",
    "delay": "時間で遅らせる",
    "arrival": "樹脂が届いてから開く",
}
_DIRECT_DELAY_LABELS = {
    "delay": "射出の開始から開くまでの時間 [s]",
    "arrival": "樹脂が届いてから開くまでの時間 [s]",
}


def _direct_gate_block(v: dict, num) -> None:
    """Direct gates: valve gates on the body, no runner."""
    v["direct_layout"] = st.radio(
        "ゲートの位置",
        DIRECT_LAYOUTS,
        format_func=_DIRECT_LAYOUT_LABELS.__getitem__,
        key="fg_direct_layout",
    )
    num("gate_d_mm", "ゲート径 φ（バルブゲート）[mm]", 0.5, 50.0, 0.5, fmt="%.1f")
    if v["direct_layout"] != "center":
        # the orifice stays on the body: the inset is at least the gate's radius
        inset_lo = v["gate_d_mm"] / 2
        inset_sup = 0.5 * (min(v["plate_w_mm"], v["plate_h_mm"]) - 2 * v["frame_w_mm"])
        inset_hi = max(math.ceil(inset_sup - 1e-6) - 1.0, inset_lo)
        v["direct_inset_mm"] = st.number_input(
            "隅からゲートの中心までの距離（縦・横とも）[mm]",
            min_value=float(inset_lo),
            max_value=float(inset_hi),
            value=min(max(float(_D.direct_inset_mm), float(inset_lo)), float(inset_hi)),
            step=1.0,
            format="%.1f",
            key="fg_direct_inset_mm",
            help=f"下限はゲートの半径 {inset_lo:.2f} mm（ゲートの穴が本体からはみ出さない）。",
        )
    v["direct_open_mode"] = "together"
    v["direct_delay_s"] = 0.0
    if v["direct_layout"] != "center":
        if st.session_state.get("molding_method", MOLDING_METHODS[0]) == "icm":
            st.caption(
                "射出圧縮成形では、ゲートは全部同時に開く（遅らせて開く計算は通常成形だけ）。"
            )
        else:
            v["direct_open_mode"] = st.radio(
                "2 つ目からのゲートを開く時",
                DIRECT_OPEN_MODES,
                format_func=_DIRECT_OPEN_LABELS.__getitem__,
                key="fg_direct_open_mode",
                help="1 つ目のゲートは左下の隅（ゲート側・左）で、射出の開始から開いている。"
                "「時間で遅らせる」は、残りのゲートを射出の開始から決めた秒数のあとに一斉に開く。"
                "「樹脂が届いてから開く」は、それぞれのゲートに先端が届いたあとに開く（届かなければ開かない）。"
                "遅らせるときは、充填の解き方が「時間で追う」（既定）でないと解けない。",
            )
            if v["direct_open_mode"] != "together":
                v["direct_delay_s"] = st.number_input(
                    _DIRECT_DELAY_LABELS[v["direct_open_mode"]],
                    min_value=0.0,
                    max_value=120.0,
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key="fg_direct_delay_s",
                )
    st.caption(
        "開いているゲートは同じ圧力を共有する。射出率は合計で、ゲートの間の配分は流れやすさで決まる"
        "（マニホールドの圧力損失は見ない）。保圧の計算は、全部のゲートを開いて解く。"
    )


#: Internal field names of ``InjectionProfile`` mapped to the sidebar's own
#: wording. ``validate()`` names ``stages[1].end_position_mm``; the widget
#: above it is labelled ``第2段 速度切替位置``. Leaving the raw name on screen
#: makes the user translate a 0-indexed English path into a 1-indexed Japanese
#: label to work out which box to fix (@claude on PR #89).
_STAGE_FIELD_JA = {
    "end_position_mm": "速度切替位置",
    "velocity_mms": "射出速度",
}


def _injection_error_ja(message: str) -> str:
    """Render an ``InjectionProfile.validate()`` message in the sidebar's words.

    Both halves need translating, not just the field path: leaving "must be
    below" next to 第2段 produces a sentence that is half English and reads
    as an internal error rather than as "you set these two boxes the wrong
    way round".
    """

    def _stage(m: re.Match[str]) -> str:
        field = _STAGE_FIELD_JA.get(m.group(2), m.group(2))
        return f"第{int(m.group(1)) + 1}段の{field}"

    out = re.sub(r"stages\[(\d+)\]\.(\w+)", _stage, message)
    out = out.replace("metering_position_mm", "計量位置")
    out = out.replace("screw_diameter_mm", "スクリュー径")
    out = re.sub(
        r"^(.*?) must be below (.*?)(?: --.*)?$",
        r"\1 は \2 より小さくしてください",
        out,
    )
    out = out.replace("must be a positive finite value", "は正の有限値である必要があります")
    out = out.replace("must be finite and >= 0", "は 0 以上の有限値である必要があります")
    out = out.replace("at least one injection stage is required", "射出段数は 1 以上必要です")
    return out


def _injection_rate_inputs(
    mode: str,
) -> tuple[InjectionProfile | None, float | None, str | None, dict | None]:
    """The rate half of the injection block: a flat Q, or one from the screw.

    Returns ``(profile, Q_cm3s, error, machine)``. Exactly one of the first
    two is set when ``error`` is None. ``machine`` is the screw-side record
    for settings.json (None on the direct rate).

    - ``direct``: the rate slider.
    - ``machine``, one stage (the default): screw diameter and speed only,
      ``Q = pi*D^2/4 * v`` handed to the solver as a flat rate. A single
      speed has no switch to place, and the stroke between metering and V/P
      does not set how much resin goes in (that is the two-phase shot
      volume), so the positions are not asked for.
    - ``machine``, two stages or more: metering and V/P positions, the switch
      points and a speed per stage build an ``InjectionProfile``.

    On a rejected machine condition the first two are None and ``error``
    carries the message -- the caller stops the run rather than this function
    calling ``st.stop()``, which sits above the version caption and would take
    it off the screen with the error still on it.

    The stage widgets are laid out with static bounds (the metering and V/P
    positions of the run) and the ordering is checked afterwards instead of
    being enforced by nesting each stage's ``max_value`` under the previous
    one's value. Chained dynamic bounds re-create the widget on every edit,
    and the value snaps back to the default while the user is still typing --
    the same trap the shot-volume follow already fell into (v0.41.1).
    """
    if mode == "direct":
        Q = st.slider(
            "射出率 [cm³/s]",
            1.0,
            800.0,
            589.0,
            step=1.0,
            key="inj_Q_direct",
            help="成形機の取説の射出率をそのまま使う。取説の値は機械の最大射出率なので、"
            "実際より速く充填される。実機に合わせるなら「スクリュー径と射出速度から計算」。",
        )
        return None, float(Q), None, None

    screw_d = st.number_input(
        "スクリュー径 [mm]",
        min_value=5.0,
        max_value=200.0,
        value=DEF_SCREW_DIAMETER_MM,
        step=1.0,
        key="inj_screw_d",
        help="バレル内でスクリューが押しのける断面の直径。Q = πD²/4 × v。",
    )
    n_stages = int(
        st.number_input(
            "射出段数",
            min_value=1,
            max_value=MAX_INJECTION_STAGES,
            value=DEF_INJECTION_STAGES,
            step=1,
            key="inj_num_stages",
            help="1 なら射出速度だけで射出率が決まる。2 以上にすると計量位置・"
            "V/P切替位置・各段の速度切替位置と速度の欄が出る。",
        )
    )
    area_mm2 = math.pi * float(screw_d) ** 2 / 4.0

    if n_stages == 1:
        v = float(
            st.number_input(
                "射出速度 [mm/s]",
                min_value=0.1,
                max_value=1000.0,
                # Coming back from several stages, start from the first
                # stage's speed rather than the default.
                value=float(st.session_state.get("inj_velocity_0", DEF_SCREW_VELOCITY_MMS)),
                step=1.0,
                key="inj_velocity",
                help="スクリューの前進速度（成形機の画面の射出速度）。",
            )
        )
        Q = area_mm2 * v / 1000.0
        st.caption(f"射出率: **{Q:.1f} cm³/s**（π × {float(screw_d):g}² / 4 × {v:g} mm/s）")
        return None, Q, None, {"screw_diameter_mm": float(screw_d), "stages": 1, "velocity_mms": v}

    meter_pos = st.number_input(
        "計量位置 [mm]",
        min_value=0.5,
        max_value=1000.0,
        value=DEF_METERING_POSITION_MM,
        step=1.0,
        key="inj_meter_pos",
        help="射出開始時のスクリュー位置。ここから前進する。",
    )
    vp_pos = st.number_input(
        "V/P切替位置 [mm]",
        min_value=0.0,
        max_value=1000.0,
        value=DEF_VP_POSITION_MM,
        step=0.5,
        key="inj_vp_pos",
        help="速度制御から保圧に切り替わる位置。ここまでが充填。",
    )
    machine = {"screw_diameter_mm": float(screw_d), "stages": n_stages}

    if vp_pos >= meter_pos:
        return (
            None,
            None,
            (
                f"V/P切替位置 ({vp_pos:g} mm) が計量位置 ({meter_pos:g} mm) 以上です。"
                "スクリューは前進するので、V/P は計量位置より小さい必要があります。"
            ),
            machine,
        )

    # Only the boundaries between stages are asked for; the last stage always
    # ends at V/P, which is what makes it the V/P transfer. Defaults are the
    # machine's own switch points at the stage count it is set up in, an even
    # split of the stroke otherwise.
    span = meter_pos - vp_pos
    switches: list[float] = []
    for i in range(n_stages - 1):
        default = meter_pos - span * (i + 1) / n_stages
        if (
            n_stages == len(DEF_SWITCH_POSITIONS_MM) + 1
            and vp_pos < DEF_SWITCH_POSITIONS_MM[i] < meter_pos
        ):
            # The machine's own switch points, as long as they still sit
            # inside the stroke the user has dialled in.
            default = DEF_SWITCH_POSITIONS_MM[i]
        switches.append(
            float(
                st.number_input(
                    f"第{i + 1}段 速度切替位置 [mm]",
                    min_value=float(vp_pos),
                    max_value=float(meter_pos),
                    value=float(default),
                    step=0.5,
                    key=f"inj_switch_{i}",
                )
            )
        )
    # A new stage starts at the single-stage speed the user had, so going
    # from one stage to several keeps the speed instead of resetting it.
    v_seed = float(st.session_state.get("inj_velocity", DEF_SCREW_VELOCITY_MMS))
    velocities = [
        float(
            st.number_input(
                f"第{i + 1}段 射出速度 [mm/s]",
                min_value=0.1,
                max_value=1000.0,
                value=v_seed,
                step=1.0,
                key=f"inj_velocity_{i}",
            )
        )
        for i in range(n_stages)
    ]

    ends = [*switches, float(vp_pos)]
    try:
        profile = InjectionProfile(
            screw_diameter_mm=float(screw_d),
            metering_position_mm=float(meter_pos),
            stages=tuple(
                InjectionStage(end_position_mm=e, velocity_mms=v) for e, v in zip(ends, velocities)
            ),
        )
    except ValueError as exc:
        return (
            None,
            None,
            (
                f"射出条件が成立しません: {_injection_error_ja(str(exc))}。"
                "速度切替位置は計量位置から V/P 位置へ向かって降順に並べてください"
                "（最終段は V/P 位置で終わります）。"
            ),
            machine,
        )

    lines = [
        f"射出率（平均）: **{profile.mean_rate_cm3s:.1f} cm³/s**",
        f"射出時間 (V/P まで): **{profile.total_time_s:.3f} s**",
        f"理論射出量 (V/P 時点): **{profile.total_volume_cm3:.2f} cm³**",
    ]
    for i, (rate, t, vol) in enumerate(
        zip(profile.stage_rates_cm3s(), profile.stage_times_s(), profile.stage_volumes_mm3())
    ):
        lines.append(f"　第{i + 1}段: {rate:.1f} cm³/s / {t:.3f} s / {vol / 1000.0:.2f} cm³")
    st.caption("  \n".join(lines))
    return profile, None, None, machine


def build_geometry(v: dict) -> tuple[Geometry, dict]:
    """Assemble the config from the sidebar values and rasterise it.

    Returns the geometry and the ``settings.json`` record of the inputs."""
    cfg = FanRunnerPlateConfig(**v)
    try:
        geom = build_fan_runner_plate_geometry(cfg)
    except ValueError as e:
        st.error(f"形状パラメータが不正です: {e}")
        st.stop()
    return geom, config_settings(GEOMETRY_SOURCE, cfg)


with st.sidebar:
    _hdr_col, _run_col = st.columns([1.4, 1], vertical_alignment="bottom")
    with _hdr_col:
        st.header("成形品設計")
    with _run_col:
        do_run = st.button("解析実行", type="primary", use_container_width=True)
    molding_method = st.radio(
        "成形方法",
        MOLDING_METHODS,
        format_func=_MOLDING_LABELS.__getitem__,
        key="molding_method",
        horizontal=True,
        help="通常成形は型を閉じたまま充填する。保圧の計算と、層の温度を刻みごとに解く充填は通常成形だけで、"
        "「保圧」の欄も通常成形のときだけ出る。"
        "射出圧縮成形は、型を少し開いたまま充填して、あとから閉じる。圧縮されるのは内側の厚肉部（本体）だけで、"
        "額縁・ゲート側の形・ランナは固定。条件は下の「射出圧縮成形 (ICM)」で決める。",
    )
    icm = molding_method == "icm"
    fan_inputs = _fan_runner_sidebar()

    with st.expander("材料", expanded=False):
        material_key = st.selectbox("樹脂", material_keys, index=material_keys.index("PMMA"))
        mat = db[material_key]
        st.caption(f"{mat.name}")
        st.caption(
            f"推奨 melt: {mat.T_melt_recommended[0] - 273.15:.0f}–{mat.T_melt_recommended[1] - 273.15:.0f} ℃, "
            f"mold: {mat.T_mold_recommended[0] - 273.15:.0f}–{mat.T_mold_recommended[1] - 273.15:.0f} ℃"
        )

    with st.expander("射出条件", expanded=False):
        _melt_min = int(mat.T_melt_recommended[0] - 273.15) - 20
        _melt_max = int(mat.T_melt_recommended[1] - 273.15) + 20
        melt_C = st.slider(
            "樹脂温度 [℃]",
            _melt_min,
            _melt_max,
            max(_melt_min, min(260, _melt_max)),
        )
        _mold_min = int(mat.T_mold_recommended[0] - 273.15) - 10
        _mold_max = int(mat.T_mold_recommended[1] - 273.15) + 30
        mold_C = st.slider(
            "金型温度 [℃]",
            _mold_min,
            _mold_max,
            max(_mold_min, min(50, _mold_max)),
        )
        inj_v = st.slider(
            "代表流動速度 [mm/s]",
            5.0,
            400.0,
            200.0,
            step=5.0,
            key="inj_rep_velocity",
            help="キャビティ内のギャップ平均流速。代表せん断速度 6V/h に入り、"
            "Cross-WLF の粘度だけを決める。**スクリューの射出速度とは別物**で、"
            "薄板では 1 桁大きい（バレルの断面積とキャビティの断面積が違う）。"
            "充填時間の絶対値はこの値ではなく下の射出率が決める。",
        )
        inj_mode = st.radio(
            "射出率の指定",
            options=("direct", "machine"),
            # 既定はスクリュー径と射出速度から計算（Φ50 × 200 mm/s、単段 = 392.7 cm³/s）。
            # v0.8.0 で sim v0.55.0・v0.57.0 に揃えた。v0.4.0〜v0.7.0 は同じ計算側の
            # 3 段（全段同速）が既定だった。直接入力の 589 は取説の最大射出率。
            index=1,
            key="inj_mode",
            format_func=lambda m: {
                "direct": "射出率を直接入力",
                "machine": "スクリュー径と射出速度から計算",
            }[m],
            help="直接入力: 取説の射出率（機械の最大値）をそのまま使う。"
            "スクリュー径と射出速度から計算: Q = πD²/4 × v。射出段数を 2 以上にすると"
            "計量位置・V/P切替位置・各段の速度を入れる多段射出になる。",
        )
        inj_profile, inj_Q, inj_error, inj_machine = _injection_rate_inputs(inj_mode)

    with st.expander("壁面冷却モデル", expanded=False):
        wall_model = st.radio(
            "壁面冷却の表現",
            options=("none", "skin", "multilayer"),
            index=2,
            key="wall_model",
            format_func=lambda m: {
                "none": "なし（等温・代表粘度のみ）",
                "skin": "スキン層 (1層 + Stefan/Neumann)",
                "multilayer": "層別 (N 層離散化 + Cross-WLF 結合)",
            }[m],
            help=(
                "なし: 既存 HeleShawSolver 相当、温度結合なし。\n"
                "スキン層: 壁面で固化するスキン層を s(t)=c_skin·√(αt) で取り込み、"
                "コア層 h_core=h-2s だけが流れる（露光時計、役務平均）。封止と未充填も検出。\n"
                "層別: 厚み方向を N 層に分割、Neumann 1D 温度プロファイルから "
                "層別粘度を Cross-WLF で評価。fixed-point で τ ↔ T_k ↔ η_k を結合。\n"
                "既定は層別（N=7）。温度まで効くのは層別だけ。スキン層・層別とも二相"
                "ショートショットの射出相に乗る。"
            ),
        )

        # default container (so downstream `solver = HeleShawSolver(...)` /
        # `MultilayerHeleShawSolver(...)` always has the kwargs it expects).
        # 既定モードは『層別』(index=2、v0.8.0、sim v0.50.0 の横展開)。v0.1.0〜v0.7.0 は
        # 『スキン層』(index=1)だった。層別は v0.8.0（sim v0.48.0）から二相ショートショットの
        # 射出相に乗るので、既定 ON の二相と両立する。『なし』は η が定数で
        # S ∝ h³ になり、材料も温度も充填順序に効かない（形状と Q だけで決まる）。
        # スキン層で効くのは材料の熱拡散率 α（s = c·√(αt)）で、T_melt / T_mold は
        # 依然として効かない — 温度まで効かせるには層別。
        # 層別を選んだときの既定値 (極薄 t0.35〜0.50 向け):
        #   層数 N: 7 (壁勾配が急なので N=5 から増量)
        #   反復上限: 12 (収束が遅くなりがちなので上限緩め)
        skin_on = wall_model == "skin"
        c_skin = 0.0
        skin_max_iter = 20
        skin_tol = 1e-3
        skin_clock_mode = "constant_pressure"
        multilayer_on = wall_model == "multilayer"
        num_layers = 7
        layer_distribution = "wall_refined"
        multilayer_max_iter = 12
        multilayer_tol = 1e-3
        solid_fraction = 0.3
        fill_method = "tau"
        thermal_model = "arrival"
        inj_p_limit_mpa = 280.0
        wall_contact_model = "htc"
        htc_fill, htc_pack, htc_detached = 5000.0, 2500.0, 1250.0

        if wall_model == "skin":
            c_skin = st.slider(
                "スキン層成長定数 c_skin",
                0.0,
                2.0,
                1.0,
                step=0.05,
                help="0で OFF と同等。1.0 付近が物理的代表値。薄肉ほど効果大。",
            )
            skin_max_iter = st.slider(
                "fixed-point 反復上限",
                1,
                100,
                20,
                help=(
                    "τ ↔ h_core 結合の反復回数。ソルバ既定と同じ 20。圧力一定時計では封止が"
                    "雪崩（スキン↑→抵抗↑→T_fill↑→スキン↑）になることがあり、途中で打ち切ると"
                    "半分凍った絵が収束したように見える（sim v0.39.0）。この金型の既定形状は"
                    "圧力一定時計だと 80 で収束、速度制御時計なら数回で収束する。"
                    "未収束なら結果ペインに警告が出る。"
                ),
            )
            skin_tol_log10 = st.slider(
                "収束判定 log10(tol)",
                -5,
                -1,
                -3,
                help="τ場の相対L2変化が 10^tol を下回ったら収束。",
            )
            skin_tol = 10.0 ** float(skin_tol_log10)
            skin_clock_mode = st.radio(
                "スキン層の時計",
                options=("constant_pressure", "constant_rate"),
                # v0.42.2: 速度制御が既定。実機条件（スクリュー位置と速度）が
                # 既定の射出入力になった以上、時計だけ「機械が速度を保てない」
                # 側に倒れていると、設定どおりの射出時間を入れた画面が設定
                # どおりでない充填時間を返す。圧力一定は既存結果の再現用に残す。
                index=1,
                key="skin_clock",
                format_func=lambda m: {
                    "constant_pressure": "圧力一定（従来）: 抵抗増で流量が細り T_fill が伸びる",
                    "constant_rate": "速度制御（既定）: 射出時間 V/Q 固定、圧力が上がる",
                }[m],
                help=(
                    "スキンで流路が痩せたとき機械がどう応えるか。速度制御で射出する"
                    "実機（充填時間が設定どおりに出る）なら「速度制御」。従来の圧力一定は"
                    "T_fill を体積重み付き τ 比で膨らませる近似で、既存結果の再現用に残す。"
                    "二相ショートショットの射出相は計量 V/Q の定義上つねに速度制御。"
                ),
            )
            if inj_mode == "machine" and skin_clock_mode == "constant_pressure":
                # A screw speed prescribes a rate. Holding pressure instead
                # is the "machine could not keep up" scenario; the profile's
                # shape survives (every stage stretches by the same factor)
                # but the times are no longer the ones on the machine.
                st.caption(
                    "スクリュー径と射出速度で決めた射出は速度制御そのものです。"
                    "圧力一定を選ぶと、抵抗増のぶん**射出全体が同じ倍率で**"
                    "引き伸ばされ、射出時間は設定値と一致しなくなります。"
                    "設定どおりの充填時間を見たいなら「速度制御」を選んでください。"
                )
        elif wall_model == "multilayer":
            fill_method = st.radio(
                "充填の解き方",
                options=("tau", "march"),
                # 既定は時間で追う方（v0.12.0）。v0.11.0 までは一発で解く方（index=0）だった。
                # sim の既定は一発で解く方のまま。
                index=1,
                key="fill_method",
                format_func=lambda m: {
                    "tau": "一発で解く（v0.11.0 までの既定）: 充填の順番を 1 回の計算で決める",
                    "march": "時間で追う（既定）: 先端を時間で進める",
                }[m],
                help=(
                    "一発で解く方は、充填の順番を 1 回の楕円型の計算（τ）で決める。速いが、"
                    "ランナの厚い所を満たしきってから薄い所に入る癖がある。\n"
                    "時間で追う方は、射出した量だけ先端を時間で進める。厚い所が埋まりきる前に、"
                    "先端が届いた薄い所へも入り始める。総充填時間は射出率どおり（体積 ÷ 射出率。射出圧縮成形ではその短縮込み）になる。"
                    "mold-flow-sim の別の製品では、ショートショットの写真の到達線に近づいた"
                    "（sim v0.62.0 の変更履歴）。本リポの形状の写真とはまだ照合していない。"
                    "計算は 1.0 mm で主解析と二相を合わせて 8〜10 分、一発で解く方は 1〜2 分。"
                    "急ぐときはメッシュを粗くするか、一発で解く方を選ぶ。"
                    "層の温度の読み方（到着時刻で冷える）は同じで、変わるのは順番の決め方だけ。"
                    "時計は射出率どおり（速度制御）で、圧力一定の引き伸ばしは無い。"
                    "下の反復上限と収束判定は、一発で解く方だけに効く。"
                ),
            )
            thermal_model = st.radio(
                "層の温度の解き方",
                options=("arrival", "energy"),
                index=0,
                key="thermal_model",
                disabled=fill_method != "march",
                format_func=lambda m: {
                    "arrival": "到達時刻から読む（既定）: 届いてからの時間、その場所の肉厚で冷えたとみなす",
                    "energy": "刻みごとに解く: 厚さ方向の温度を状態として持ち、流れが熱を運ぶ",
                }[m],
                help=(
                    "時間で追う解き方のときだけ選べる。射出圧縮成形のときは到達時刻から読む方で解く。\n"
                    "到達時刻から読む方は、厚い所を通ってきた熱い樹脂が薄い所に入っても、最初からその薄い肉厚で"
                    "冷えてきたものとして扱う。1 mm の額縁のような薄い所を冷たく見すぎる。\n"
                    "刻みごとに解く方は、厚さ方向 16 層の温度を毎刻み解く（壁への熱伝導、流れが運ぶ熱、剪断発熱）。"
                    "流れやすさはその場所の流量から決める。保圧を解くときは、VP 切替の時点の温度をそのまま保圧へ渡す。"
                    "射出圧が下の上限に達すると流量が落ち、流れが止まればそこで充填を止める（理由は結果の先頭に出る）。"
                    "面内（板の広がる向き）の熱の伝わりは入っていない。格子は 2 mm 以下で使う"
                    "（4 mm は軸に対して左右対称に切れず、結果が左右で違って出る）。"
                ),
            )
            if fill_method != "march":
                thermal_model = "arrival"
            inj_p_limit_mpa = st.number_input(
                "射出圧の上限 [MPa]",
                min_value=20.0,
                max_value=400.0,
                value=280.0,
                step=10.0,
                key="inj_p_limit_mpa",
                disabled=thermal_model != "energy",
                help=(
                    "刻みごとに解くときだけ効く。ゲートの圧力がこの値に達すると、射出率を保てずに流量が落ちる。"
                    "ノズルとホットランナーの圧力損失は入っていないので、機械の最大射出圧よりその分低く置く。"
                ),
            )
            wall_contact_model = st.radio(
                "金型との熱の伝わり方",
                options=("htc", "ideal"),
                index=0,
                key="wall_contact_model",
                disabled=thermal_model != "energy",
                format_func=lambda m: {
                    "htc": "熱伝達係数（既定。Moldflow と同じ扱い）",
                    "ideal": "表面＝金型温度（v0.15 までの扱い）",
                }[m],
                help=(
                    "刻みごとに解くときだけ効く。熱伝達係数は、樹脂の表面と金型の間の熱の伝わりやすさ。"
                    "小さいほど表面が冷えにくく、薄い所が固まるのが遅くなる。"
                    "「表面＝金型温度」は、樹脂の表面が触れた瞬間に金型温度になる扱いで、薄い所の固まりを早めに見る。"
                ),
            )
            _htc_off = thermal_model != "energy" or wall_contact_model != "htc"
            _hc0, _hc1, _hc2 = st.columns(3)
            htc_fill = _hc0.number_input(
                "充填中 [W/m²K]",
                min_value=100.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0,
                key="htc_fill",
                disabled=_htc_off,
                help="充填中の熱伝達係数。既定 5000 は Moldflow の既定と同じ。",
            )
            htc_pack = _hc1.number_input(
                "保圧中 [W/m²K]",
                min_value=100.0,
                max_value=100000.0,
                value=2500.0,
                step=250.0,
                key="htc_pack",
                disabled=_htc_off,
                help="VP 切替のあとの熱伝達係数。既定 2500 は Moldflow の既定と同じ。保圧を解くときに効く。",
            )
            htc_detached = _hc2.number_input(
                "離れたあと [W/m²K]",
                min_value=100.0,
                max_value=100000.0,
                value=1250.0,
                step=250.0,
                key="htc_detached",
                disabled=_htc_off,
                help=(
                    "保圧中に圧力が 0 になり、樹脂が縮んで壁から離れたマスに使う熱伝達係数。"
                    "既定 1250 は Moldflow の既定と同じ。保圧を解くときに効く。"
                ),
            )
            num_layers = st.slider(
                "層数 N",
                3,
                9,
                7,
                help=(
                    "厚み方向の離散化数。奇数で中央層がショートショット判定の代表セルに。"
                    "極薄プレートでは壁勾配が急なので N=7 を推奨。"
                ),
            )
            layer_distribution = st.radio(
                "層分布",
                options=("wall_refined", "uniform"),
                index=0,
                format_func=lambda m: {
                    "wall_refined": "壁近傍密 (Chebyshev-Lobatto)",
                    "uniform": "等間隔",
                }[m],
                help=(
                    "wall_refined: ζ_k = 0.5·(1 - cos(πk/N))。Neumann 勾配の急な"
                    "壁面で解像度を稼ぐ。layer 数が同じなら推奨。\n"
                    "uniform: 等間隔。デバッグ・解析比較用。"
                ),
            )
            multilayer_max_iter = st.slider(
                "fixed-point 反復上限",
                1,
                20,
                12,
                help=(
                    "τ ↔ T_k ↔ η_k 結合の反復回数。極薄プレートでは収束が遅くなりがちなので 12 を推奨。"
                ),
            )
            multilayer_tol_log10 = st.slider(
                "収束判定 log10(tol)",
                -5,
                -1,
                -3,
                help="τ場の相対L2変化が 10^tol を下回ったら収束。",
            )
            multilayer_tol = 10.0 ** float(multilayer_tol_log10)
            solid_fraction = st.slider(
                "固化判定 fraction",
                0.0,
                0.9,
                0.3,
                step=0.05,
                help=(
                    "中央層温度が T_mold + fraction·(T_melt - T_mold) を下回るセルを"
                    "ショートショットにマーク。PP は 0.3 が目安。"
                ),
            )
            shear_heating_enabled = st.checkbox(
                "剪断発熱補正 (viscous dissipation, 段階1)",
                value=True,
                help=(
                    "ON で Neumann 温度に剪断発熱補正項 ΔT_k = (η_k·γ̇_k²)·min(t_arr, τ_thermal)/(ρ·cp) を加算。"
                    "τ_thermal = h²/(π²·α) で頭打ち。"
                    "極薄プレート (t<0.5mm) では Brinkman 数 Br ≫ 1 になりがちなので推奨。"
                    "OFF でも Br 数は結果ペインに表示されるので、必要性を事前判定できる。"
                ),
            )
        else:
            shear_heating_enabled = False

    if icm:
        with st.expander("射出圧縮成形 (ICM)", expanded=False):
            # ストローク (絶対加算) モードに統一。圧縮 mask 内の全セルに同じ絶対量を加算
            # するので段差プレートでも段差が保存される (金型シム量の物理に整合)。
            # 旧倍率モードは solver / CLI には後方互換で残しているが UI には出さない。
            comp_stroke = st.slider(
                "圧縮ストローク [mm]",
                0.0,
                2.0,
                0.50,
                step=0.05,
                help=(
                    "金型シム量。圧縮 mask セル全てに加算される絶対量。"
                    "段差プレートでも段差が圧縮位相中も保存される (実機の挙動と整合)。"
                ),
            )
            comp_factor = 1.0  # unused, kept for solver kwargs symmetry
            comp_frac = st.slider(
                "圧縮位相の充填占有率",
                0.1,
                1.0,
                0.60,
                step=0.05,
                help="充填全体に対し、圧縮位相 (型開き状態) で占める時間比率。",
            )
    else:
        comp_factor = 1.0
        comp_stroke = None
        comp_frac = 0.0

    if icm:
        # 射出圧縮成形では保圧を解かないので、欄ごと出さない（値は使われない）
        hold_on = False
        hold_p_mpa, hold_t_s, hold_vp_pct = 60.0, 10.0, 99.0
        machine_clamp_kN = 4413.0
        flash_threshold_mpa = None
        hold_coarse = False
    else:
        with st.expander("保圧", expanded=False):
            hold_on = st.checkbox(
                "保圧を解く",
                value=False,
                key="hold_on",
                help=(
                    "充填のあと、ゲートを保圧で保持して、圧力の時間変化と体積の縮みを解く"
                    "（圧縮性つき Hele-Shaw、2 領域 Tait の pvT）。"
                    "充填と同じ格子で解くので、解析時間が大きく延びる（数分〜数十分。形と VP 切替の値で変わる）。"
                ),
            )
            hold_p_mpa = st.number_input(
                "保圧 [MPa]",
                min_value=0.0,
                max_value=250.0,
                value=60.0,
                step=5.0,
                key="hold_p_mpa",
                disabled=not hold_on,
                help="ゲートにかかる圧力。ノズルとホットランナーの圧力損失は入らない。",
            )
            hold_t_s = st.number_input(
                "保圧時間 [s]",
                min_value=0.0,
                max_value=120.0,
                value=10.0,
                step=0.5,
                key="hold_t_s",
                disabled=not hold_on,
                help="この時間のあとゲートを閉じる（バルブゲート。逆流はしない）。",
            )
            hold_vp_pct = st.number_input(
                "VP 切替の充填率 [%]",
                min_value=50.0,
                max_value=100.0,
                value=99.0,
                step=1.0,
                key="hold_vp_pct",
                disabled=not hold_on,
                help=(
                    "体積のこの割合まで射出率で埋め、そこから保圧に切り替える。残りは保圧で埋める"
                    "（機械が送る量は射出率が上限）。100 にすると、全部埋まったものとして保圧だけを見る。"
                    "薄い所が保圧で埋まるかの判定は、温度の扱いが粗いので当てにならない（式解説 §9）。"
                ),
            )
            machine_clamp_kN = st.number_input(
                "機械の型締力 [kN]",
                min_value=100.0,
                max_value=50000.0,
                value=4413.0,
                step=100.0,
                key="machine_clamp_kN",
                disabled=not hold_on,
                help=(
                    "保圧の結果の「バリ注意」で、型を開く力のピークと比べる。既定はソディック 450t（4,413 kN）。"
                ),
            )
            flash_threshold_mpa = st.number_input(
                "バリの目安の圧力 [MPa]（空欄＝線を引かない）",
                min_value=1.0,
                max_value=400.0,
                value=None,
                step=5.0,
                key="flash_threshold_mpa",
                disabled=not hold_on,
                placeholder="試作のバリの位置から決める",
                help=(
                    "「バリ注意」の図で、輪郭が溶けている間に受けた最大の圧力がこの値以上の所に紫の枠を付け、長さを出す。"
                    "何 MPa でバリが出るかは型のすき間とたわみで決まるので、既定の値は持たない。"
                    "試作でバリが出始めた所を図の圧力と見比べて決める。"
                ),
            )
            hold_coarse = st.checkbox(
                "保圧を粗い格子で解く（検算用）",
                value=False,
                key="hold_coarse",
                disabled=not hold_on,
                help=(
                    "OFF（既定）は充填と同じ格子で解く。1 mm 格子では数十分かかることがある。"
                    "ON はマスを 2 万以下にまとめて解く（検算用。それでも数分〜数十分）。"
                ),
            )

    # ショートショット欄はこの位置に出すが、中身は版表示の後で埋める:
    # build_geometry() は不整合で st.stop() するので、サイドバー内で先に呼ぶと
    # 「出力」欄と版表示がエラーのたびに消える（版表示をサイドバーに置いた理由
    # そのもの）。container は生成位置に描画されるので見た目の順序は変わらない。
    _short_shot_slot = st.container()

    with st.expander("出力", expanded=False):
        num_frames = st.slider("アニメーションフレーム数", 12, 60, 60)
        # 既定の turbo は商用 CAE と同じ虹配色。色相コントラストで等時線が
        # 読めるのが狙いで、赤=最後に充填=リスク箇所という意味とも一致する。
        # 赤緑色覚に配慮するときは cividis / viridis を選ぶ。
        fill_cmap = st.selectbox(
            "充填アニメの配色",
            options=["turbo", "jet", "viridis", "cividis"],
            index=0,
            format_func=lambda c: {
                "turbo": "turbo（既定・虹／偽の縞が出ない）",
                "jet": "jet（従来の商用 CAE と同じ虹）",
                "viridis": "viridis（知覚均等・色覚配慮）",
                "cividis": "cividis（色覚配慮を最優先）",
            }[c],
            help="虹系は等時線の形が読みやすく、viridis / cividis は量の大小比較と色覚配慮に向く。",
        )
        iso_levels = st.slider(
            "等時線の本数",
            0,
            24,
            ISOCHRONE_LEVELS,
            help="同時に充填される位置を結んだ線。線が詰まる=流れが遅い、"
            "ぶつかる=ウェルド、途切れた先=最後に充填。0 で非表示。",
        )
        # ウェルドは「2つの流れが出会う角度」で描く。45° 以上（商用 CAE の
        # 合流角 135° 境界）は濃い赤＝ウェルド、そこから下限までは薄い赤＝
        # メルド（ほぼ平行に合流する痕、強度欠陥より外観）。下限を下げるほど
        # 長く描かれ、数値ノイズも拾う。
        weld_min_angle = st.slider(
            "メルド表示の下限角 [deg]",
            0,
            40,
            int(WELD_MIN_ANGLE_DEG),
            step=5,
            key="weld_min_angle",
            help="2 つの流れが出会うときの開き角。45° 以上は濃い赤（ウェルド）、"
            "この角度から 45° までは薄い赤（メルド）、未満は描かない。"
            "穴の後ろに残る遅れ帯の痕を追いたいときは下げる。",
        )

    # Version / build label.
    # Rendered here (end of the sidebar) rather than at the end of the script
    # because the main flow has several ``st.stop()`` calls for parameter
    # validation — a footer placed after them would vanish exactly when a user
    # screenshots an error and asks which build produced it.
    st.divider()
    st.caption(build_label())

    with _short_shot_slot:
        geom, geom_settings = build_geometry(fan_inputs)

        with st.expander("ショートショット（計量制限）", expanded=False):
            # 二相モデル: (1) 射出相 = 型開きギャップで計量体積ぶん充填、
            # (2) 圧縮相 = 型閉じで溶融プールを等圧ソースとして前進（体積保存）。
            # 線形求解2回・時間積分なし。実機の計量値をそのまま入れて
            # 段階ショートショットの現物形状と直接比較する用途。
            two_phase_on = st.checkbox(
                "二相ショートショット解析ON",
                value=True,
                key="two_phase_on",
                help=(
                    "計量を意図的に絞ったショートショットの最終形状を予測する。"
                    "射出相（型開きギャップで計量体積まで充填）→ 圧縮相（型閉じで"
                    "溶融プールを前進、体積保存）の二相。壁面冷却モデルは射出相に乗る:"
                    "スキン層は開いた薄板が射出中に痩せてゲート部が先に埋まる順番を、"
                    "層別は層ごとの温度・粘度で決まる順番を、どちらも計量 V/Q の時計で出す。"
                    "圧縮相はどのモデルでも等温。"
                ),
            )
            if two_phase_on:
                # 既定値は現在の形状の最終キャビティ体積。形状を変えると追従するが、
                # ユーザーが値を触った後は（前回の自動値から動いているので）触らない。
                # 丸めない: solver は素の体積と比較するので、下に丸めた既定は
                # 「完全充填ちょうど」でなく極小のショートショットになる（Codex P2）。
                # 「触った」の判定は on_change（計量欄そのものへのユーザー操作）で
                # 立つフラグに限る。「前回の自動値と今の値が一致するか」で判定すると、
                # 形状ウィジェットの連打で前の rerun の新値がブラウザに届く前に次の
                # rerun が始まったとき、ブラウザが持つ古い値が session_state に書き
                # 戻されて「不一致＝触った」と誤判定し、以後の追従が黙って止まる
                # （2026-09-11、翼 t0.65 の計量のまま t0.9 を解析して 0.4% ショート）。
                # ただし Streamlit は script 本体より先に「前回の状態と違う widget 値」の
                # on_change を発火するので、stale echo でもコールバックは呼ばれる。
                # 届いた値が過去の自動値のどれかと一致するなら echo であって編集では
                # ない（13 桁の浮動小数を手で打つことはない）— それだけを除外する。
                _v_cav = float(geom.volume_cm3())
                # 履歴の深さ 32 は「echo が届くより先に積もる形状変更の回数」の上限。
                # 1 rerun は形状ビルドで 0.5〜1 s かかり、ブラウザの echo 遅れは
                # 高々数 rerun なので 32 で桁余り。超えれば echo を編集と誤判定する
                # が、その場合もリセットボタンで追従を再開できる。
                _auto_hist: list[float] = st.session_state.setdefault(
                    "mfs_shot_volume_auto_history", []
                )
                if _v_cav not in _auto_hist:
                    _auto_hist.append(_v_cav)
                    del _auto_hist[:-32]
                # 二相を OFF にした rerun で widget が描かれないと Streamlit は
                # `two_phase_shot_volume` を session_state から落とすが、非 widget の
                # 編集フラグは残る。そのまま ON に戻すと「編集済みなので触らない」で
                # 初期化を飛ばし、number_input が min の 0.01 cm³ で再登場して次の
                # 実行がほぼ空の計量になる（Codex P2 on PR #87）。編集値は widget と
                # 別のキーに写しておき、widget が消えていたらそこから復元する。
                _edited = st.session_state.get("mfs_shot_volume_user_edited", False)
                if "two_phase_shot_volume" not in st.session_state:
                    _kept = st.session_state.get("mfs_shot_volume_user_value")
                    if _edited and _kept is not None:
                        st.session_state["two_phase_shot_volume"] = _kept
                    else:
                        st.session_state["mfs_shot_volume_user_edited"] = _edited = False
                        st.session_state["two_phase_shot_volume"] = _v_cav
                elif not _edited:
                    st.session_state["two_phase_shot_volume"] = _v_cav
                st.session_state["mfs_shot_volume_auto"] = _v_cav

                def _mark_shot_volume_edited() -> None:
                    _v = st.session_state["two_phase_shot_volume"]
                    if _v not in st.session_state.get("mfs_shot_volume_auto_history", []):
                        st.session_state["mfs_shot_volume_user_edited"] = True
                        st.session_state["mfs_shot_volume_user_value"] = _v

                def _reset_shot_volume() -> None:
                    st.session_state["mfs_shot_volume_user_edited"] = False
                    st.session_state.pop("mfs_shot_volume_user_value", None)
                    st.session_state["two_phase_shot_volume"] = st.session_state[
                        "mfs_shot_volume_auto"
                    ]

                shot_volume_cm3 = st.number_input(
                    "計量体積 V_shot [cm³]",
                    min_value=0.01,
                    step=0.1,
                    key="two_phase_shot_volume",
                    on_change=_mark_shot_volume_edited,
                    help=(
                        "実機の計量値（ショット体積）。既定は現在の形状の最終キャビティ"
                        "体積（完全充填ちょうど）で、欄を編集するまで形状変更に追従する。"
                        "減らすとショートショットになる。"
                    ),
                )
                _hint = f"最終キャビティ体積 {_v_cav:.2f} cm³"
                if icm and comp_stroke is not None:
                    _v_open = _v_cav + comp_stroke * geom.compression_area_mm2() / 1000.0
                    _hint += f" / 開きギャップ体積 ≈ {_v_open:.2f} cm³"
                _short = _v_cav - float(shot_volume_cm3)
                if _short > 1e-9:
                    _hint += (
                        f"。**計量がキャビティ体積を {_short:.2f} cm³ 下回る → ショートショット**"
                    )
                else:
                    _hint += "。計量が最終キャビティ体積以上だと完全充填になる"
                st.caption(_hint)
                if st.session_state.get("mfs_shot_volume_user_edited", False):
                    st.button(
                        "計量をキャビティ体積に戻す（形状追従を再開）",
                        key="two_phase_shot_volume_reset",
                        on_click=_reset_shot_volume,
                    )
            else:
                shot_volume_cm3 = None


# ----------------------- main panel -----------------------


def _gate_open_note(md: dict) -> str | None:
    """When the gates that open later opened (direct gates), from the run's record."""
    late = md.get("delayed_gates") or []
    if not late:
        return None
    parts = []
    opened = set()
    for e in md.get("gate_events") or []:
        opened.add(int(e["gate"]))
        parts.append(
            f"{int(e['gate']) + 2} つ目のゲートは {float(e['t_s']):.2f} 秒で開いた"
            + ("（樹脂が届いたあと）" if e.get("melt_already_there") else "（樹脂が届く前）")
        )
    for k, d in enumerate(late):
        if k not in opened:
            why = (
                "樹脂が届かなかった"
                if d.get("on_arrival")
                else f"開く時刻 {float(d['delay_s']):.2f} 秒より先に充填が終わった"
            )
            parts.append(f"{k + 2} つ目のゲートは開かなかった（{why}）")
    return "遅らせて開くゲート: " + "、".join(parts) + "。"


def _stall_note(md: dict, unreached: np.ndarray | None, geom: Geometry) -> str | None:
    """Say why a fill that carries the temperature stopped short, and what is left.

    The march stops either at the injection pressure limit with the rate fallen
    away, or with the way to the last cells frozen shut before the limit ever
    slowed the melt. The second leaves, where a front that stopped and froze
    meets a later one whose melt is solid as well, a line one or two cells wide
    with melt on both sides (a later front that still flows fills the cell:
    ``MultilayerHeleShawSolver.late_front_inflow``). That line is a seam, not a
    region the melt did not reach. A rim or a region left unfilled stays a short
    shot, and a stall the record cannot attribute is reported as such.
    """
    if not md.get("march_stalled") or unreached is None:
        return None
    un = np.asarray(unreached, dtype=bool) & geom.mask
    if not un.any():
        return None
    n_cells = int(un.sum())
    vol_cm3 = float(geom.thickness_mm[un].sum()) * geom.cell_size_mm**2 / 1000.0
    head = (
        f"充填は体積の {100.0 * float(md.get('march_volume_fraction', 1.0)):.2f} % で止まった。"
        f"届かなかった所は {n_cells} マス・{vol_cm3:.2f} cm³。"
    )
    limit = md.get("injection_pressure_limit_MPa")
    at_limit = f"射出圧の上限（{limit:g} MPa）" if limit is not None else "射出圧の上限"
    cause = md.get("march_stall_cause")
    if cause == "undetermined":
        return head + (
            f"ゲートの圧力が{at_limit}に達したところで、流量がほぼ 0 になった。"
            "上限が足りないのか、残りへ入る道が固まったのかは、この計算の記録からは分けられない。"
        )
    if cause != "frozen_path":
        return head + f"ゲートの圧力が{at_limit}に達して流量が落ち、流れが止まった。"
    p_free = float(md.get("march_gate_pressure_free_max_MPa", 0.0))
    why = (
        f"{at_limit}が理由ではない（止まる前のゲートの圧力は最大 {p_free:.0f} MPa）。"
        "残りへ入る所のマスが、中心まで固まっている。"
    )
    if not enclosed_seams(un, geom.mask):
        return head + why + "樹脂が届く前に道が固まった所（ショート）と読む。"
    return (
        head
        + why
        + (
            "残りは幅 1〜2 マスの線で、線の両側には樹脂が届いている。2 つの先端の合わせ目にあたり、"
            "どちらの側の樹脂も固まっていて、間が埋まらない。面で欠ける予測ではなく、"
            "ウェルドが弱く出やすい場所の目安として読む。"
        )
    )


def _fixed_point_warning(
    md: dict,
    *,
    converged_key: str,
    what: str,
    advice: str,
    no_flow_key: str | None = "no_flow",
) -> bool:
    """Warn when a fixed-point loop recorded in ``md`` was cut off.

    A truncated loop draws a half-frozen field that looks converged (sim
    v0.39.0), so the warning goes wherever the numbers are read, not only
    inside a collapsed expander. ``no_flow`` (every path but the gate sealed)
    is the loop's terminal state, not a cut-off. Returns True when something
    was shown."""
    if md.get(converged_key) is not False:
        return False
    if no_flow_key and md.get(no_flow_key):
        st.info(
            f"{what}: ゲート以外の流路が全て封止した状態で終了（反復の打ち切りではない）。"
            "これ以上反復しても場は変わらないので、この表示が最終解。"
        )
    else:
        st.warning(
            f"{what}の fixed-point 反復が上限で打ち切られた。表示は途中状態で、"
            f"封止・未充填が実際より少なく見えることがある。{advice}"
        )
    return True


_SKIN_ADVICE = "反復上限を上げるか、時計を「速度制御」にして再実行を。"


col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader("成形品設計図")
    fig_data = np.where(geom.mask, geom.thickness_mm, np.nan)
    _v_product = float(geom.thickness_mm[geom.product_mask].sum()) * geom.cell_size_mm**2 / 1000.0
    st.write(
        f"格子: {geom.nx} × {geom.ny}, セル {geom.cell_size_mm} mm, "
        f"キャビティ体積 {geom.volume_cm3():.2f} cm³（うち製品 {_v_product:.2f} cm³）"
    )
    fig_buf = io.BytesIO()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    # Product-referenced frame: x = 0 on the gate axis (= product center),
    # y = 0 on the product's runner-side edge (``product_mask`` bottom — the
    # compression zone is only the body, 20 mm further in). The runner reads
    # y < 0. Same convention is shared by every result-time map in
    # core/visualizer.py and the 3D views.
    x0_mm, y0_mm = geom.display_origin_mm()
    extent = [
        -x0_mm,
        geom.nx * geom.cell_size_mm - x0_mm,
        -y0_mm,
        geom.ny * geom.cell_size_mm - y0_mm,
    ]
    im = ax.imshow(fig_data, origin="lower", extent=extent, cmap=THICKNESS_CMAP)
    # the compression zone is the body alone; outline it so the rim reads as fixed
    if geom.compression_mask is not None and geom.compression_mask.any():
        ax.contour(
            geom.compression_mask.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=0.8,
            extent=extent,
            origin="lower",
        )
    draw_gate_markers(ax, geom)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title("thickness map [mm]")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="h [mm]")
    fig.tight_layout()
    fig.savefig(fig_buf, format="png")
    plt.close(fig)
    st.image(fig_buf.getvalue())
    st.caption(
        "y = 0 は製品エッジ（ランナ側長辺）、x = 0 はゲート軸（赤丸＝射出点 φ3）。"
        "ランナは y < 0 側。赤枠＝圧縮部（内側の厚肉部だけ、額縁とランナは固定）。"
    )


# Signature of every input that shapes the analysis or its rendered assets
# (the weld threshold is excluded: it is re-thresholded live from the cached
# result). Stored with a run and compared on every rerun so the result pane
# can say when it no longer matches the sidebar (Codex P1 on PR #5).
analysis_inputs = {
    "geometry": geom_settings,
    "material": material_key,
    "injection": (melt_C, mold_C, inj_v, inj_Q, inj_profile),
    "wall_cooling": (
        ("skin", c_skin, skin_max_iter, skin_tol, skin_clock_mode)
        if skin_on
        else (
            "multilayer",
            num_layers,
            layer_distribution,
            multilayer_max_iter,
            multilayer_tol,
            solid_fraction,
            shear_heating_enabled,
            fill_method,
            thermal_model,
            inj_p_limit_mpa if thermal_model == "energy" else None,
            (
                (wall_contact_model, htc_fill, htc_pack, htc_detached)
                if thermal_model == "energy" and wall_contact_model == "htc"
                else (wall_contact_model if thermal_model == "energy" else None)
            ),
        )
        if multilayer_on
        else ("none",)
    ),
    "compression_molding": (icm, comp_stroke, comp_frac),
    "two_phase_short_shot": (two_phase_on, shot_volume_cm3),
    "holding": (
        (
            hold_on,
            hold_p_mpa,
            hold_t_s,
            hold_vp_pct,
            hold_coarse,
            machine_clamp_kN,
            flash_threshold_mpa,
        )
        if hold_on
        else (False,)
    ),
    "output": (num_frames, fill_cmap, iso_levels),
}
STALE_RESULT_MSG = (
    "入力が前回の解析から変更されています。表示中の結果は前回の設定のものです。"
    "「解析実行」で更新してください。"
)

if do_run:
    # --- セル数ガード（メッシュ下限 0.2mm 開放に伴う安全弁）---
    # 最大寸法のプレート × 0.2mm では cavity が 100 万セルを超え得る。
    # 行列組立は Python ループ＋直接 spsolve、さらに層別モードは (N, ny, nx) の
    # 場を複数確保するため、Streamlit Cloud のメモリ枠を食い潰してプロセスごと
    # 落ちる恐れがある（Codex P1）。ソルバー起動前に cavity セル数を見て、
    # 安全上限を超えるなら明示して止める（クラッシュさせず綺麗に停止）。
    n_cavity = int(geom.mask.sum())
    if multilayer_on:
        # 層別はメモリが層数 N に比例するため上限を N で割る。
        cell_limit = max(120_000, 1_500_000 // num_layers)
    else:
        cell_limit = 500_000
    if n_cavity > cell_limit:
        _hint = (
            " 層別モードは厚み方向 N 層分のメモリを使うため上限が低めです。"
            "「なし」/「スキン層」に切り替えるか N を下げると緩和されます。"
            if multilayer_on
            else ""
        )
        st.error(
            f"この設定は格子が大きすぎます（cavity {n_cavity:,} セル ＞ 上限 "
            f"{cell_limit:,} セル）。メッシュ粗さを上げるか、製品・ランナー寸法を"
            f"小さくしてください。{_hint}"
            "（メモリ枯渇によるアプリのクラッシュを防ぐためのガードです。）"
        )
        st.stop()
    if inj_error is not None:
        st.error(inj_error)
        st.stop()
    if geom.delayed_gates and not (multilayer_on and fill_method == "march"):
        st.error(
            "直接ゲートを遅らせて開く計算は、壁面冷却の表現が「層別」で、充填の解き方が「時間で追う」のときだけ解ける。"
            "どちらかを直すか、「2 つ目からのゲートを開く時」を「同時に開く」に戻す。"
        )
        st.stop()
    # 刻みごとに解く温度は、射出圧縮では試していないので到達時刻から読む方に戻す
    energy_on = multilayer_on and thermal_model == "energy" and not icm
    if multilayer_on and thermal_model == "energy" and icm:
        st.info(
            "射出圧縮成形なので、層の温度は到達時刻から読む方で解いた（刻みごとに解く方は通常成形だけ）。"
        )
    with st.spinner("Hele-Shaw方程式を解いている…"):
        if multilayer_on:
            solver = MultilayerHeleShawSolver(
                geometry=geom,
                material=mat,
                melt_temperature_K=melt_C + 273.15,
                mold_temperature_K=mold_C + 273.15,
                injection_velocity_mms=inj_v,
                injection_volume_flow_cm3s=inj_Q,
                injection_profile=inj_profile,
                compression_molding=icm,
                compression_factor=comp_factor,
                compression_stroke_mm=comp_stroke,
                compression_fraction=comp_frac,
                num_layers=num_layers,
                layer_distribution=layer_distribution,
                thermal_coupling=True,
                max_iterations=multilayer_max_iter,
                convergence_tol=multilayer_tol,
                solidification_temperature_fraction=solid_fraction,
                shear_heating_enabled=shear_heating_enabled,
                fill_method=fill_method,
                thermal_model="energy" if energy_on else "arrival",
                injection_pressure_limit_Pa=float(inj_p_limit_mpa) * 1e6,
                mold_htc_fill_W_m2K=(
                    float(htc_fill) if energy_on and wall_contact_model == "htc" else None
                ),
                # 保圧へ渡す VP 切替の時点の温度
                thermal_snapshot_fractions=(
                    (float(hold_vp_pct) / 100.0,) if energy_on and hold_on else ()
                ),
            )
        else:
            solver = HeleShawSolver(
                geometry=geom,
                material=mat,
                melt_temperature_K=melt_C + 273.15,
                mold_temperature_K=mold_C + 273.15,
                injection_velocity_mms=inj_v,
                injection_volume_flow_cm3s=inj_Q,
                injection_profile=inj_profile,
                compression_molding=icm,
                compression_factor=comp_factor,
                compression_stroke_mm=comp_stroke,
                compression_fraction=comp_frac,
                skin_layer_enabled=skin_on,
                skin_growth_constant=c_skin,
                skin_max_iterations=skin_max_iter,
                skin_convergence_tol=skin_tol,
                skin_clock_mode=skin_clock_mode,
            )
        try:
            result = solver.solve(num_frames=num_frames)
        except ValueError as exc:
            # The solver rejects a cavity whose cells cannot all be reached
            # from a gate (Issue #58). The builders catch the shapes they can
            # name, but a combination they do not model still has to reach the
            # user as a message -- an uncaught exception here renders as a raw
            # traceback in the app.
            st.error(f"解析できない形状: {exc}")
            st.stop()

        # 二相ショートショット。壁面冷却モデル（なし／スキン層／層別）は射出相に
        # 乗る。層別は固定点を計量 V/Q の時計（膨張なし）で解き直す。
        two_phase_result = None
        two_phase_skip_reason: str | None = None
        if two_phase_on:
            try:
                two_phase_result = solve_two_phase_short_shot(solver, shot_volume_cm3)
            except ValueError as e:
                # 例: 計量がゲート群の開ギャップ体積を下回る。メッセージは
                # モデル側の固定文言 + 体積数値のみで、パス等の秘匿情報は
                # 含まない。
                two_phase_result = None
                two_phase_skip_reason = str(e)
                st.warning(f"二相ショートショット解析をスキップしました: {e}")

        # 入力の記録。metadata.json は解いた結果しか持たないので、これが無いと
        # ダウンロードした ZIP から設定を復元できない (画像から寸法を測って
        # 体積と tau_max で逆算する羽目になる)。
        run_settings = {
            "app_version": build_label(),
            "geometry": geom_settings,
            "material": material_key,
            "injection": {
                "melt_temperature_C": melt_C,
                "mold_temperature_C": mold_C,
                "injection_velocity_mms": inj_v,
                "rate_input_mode": inj_mode,
                "machine": inj_machine,
                "injection_volume_flow_cm3s": inj_Q,
                "injection_profile": (None if inj_profile is None else inj_profile.as_record()),
            },
            "wall_cooling": (
                {
                    "model": "skin",
                    "skin_growth_constant": c_skin,
                    "skin_max_iterations": skin_max_iter,
                    "skin_convergence_tol": skin_tol,
                    "skin_clock_mode": skin_clock_mode,
                }
                if skin_on
                else {
                    "model": "multilayer",
                    "num_layers": num_layers,
                    "layer_distribution": layer_distribution,
                    "max_iterations": multilayer_max_iter,
                    "convergence_tol": multilayer_tol,
                    "solidification_temperature_fraction": solid_fraction,
                    "shear_heating_enabled": shear_heating_enabled,
                    "fill_method": fill_method,
                    # what was solved: "energy" falls back to "arrival" with compression on
                    "thermal_model": "energy" if energy_on else "arrival",
                    "injection_pressure_limit_MPa": float(inj_p_limit_mpa) if energy_on else None,
                    # None: the melt's surface at the mold temperature
                    "mold_htc_W_m2K": (
                        {
                            "filling": float(htc_fill),
                            "packing": float(htc_pack),
                            "detached": float(htc_detached),
                        }
                        if energy_on and wall_contact_model == "htc"
                        else None
                    ),
                }
                if multilayer_on
                else {"model": "none"}
            ),
            "compression_molding": (
                {
                    "enabled": True,
                    "mode": "stroke",
                    "stroke_mm": comp_stroke,
                    "fraction": comp_frac,
                }
                if icm
                else {"enabled": False}
            ),
            "two_phase_short_shot": (
                {
                    "enabled": True,
                    "shot_volume_cm3": shot_volume_cm3,
                    "skin_layer": bool(skin_on),
                    "wall_model": wall_model,
                }
                if two_phase_result is not None
                else {"enabled": False}
            ),
            "holding": (
                {
                    "enabled": True,
                    "pressure_MPa": float(hold_p_mpa),
                    "time_s": float(hold_t_s),
                    "vp_switch_fill_percent": float(hold_vp_pct),
                    "coarse_grid": bool(hold_coarse),
                    "machine_clamp_kN": float(machine_clamp_kN),
                    "flash_threshold_MPa": (
                        None if flash_threshold_mpa is None else float(flash_threshold_mpa)
                    ),
                }
                if hold_on
                else {"enabled": False}
            ),
            "output": {
                "num_frames": num_frames,
                "fill_cmap": fill_cmap,
                "isochrone_levels": iso_levels,
                "weld_min_angle_deg": float(weld_min_angle),
            },
        }

        # 重い PNG/GIF レンダリングはここで一回だけやる。後段の widget 操作で
        # rerun が走っても再生成しないよう、すべて session_state に置く。
        _tmp_dir = Path(tempfile.mkdtemp(prefix="mfs_"))
        _gif_path = render_fill_animation(
            result,
            _tmp_dir / "fill.gif",
            num_frames=num_frames,
            fps=8,
            cmap=fill_cmap,
            isochrone_levels=iso_levels,
        )
        # 各フレームの PNG 連番も書き出す。GIF と同じ ZIP に frames/ で同梱して、
        # ユーザーが GIF とフレーム画像を 1 ダウンロードで両取りできるようにする。
        _frame_paths = export_frames(
            result,
            _tmp_dir / "frames",
            num_frames=num_frames,
            cmap=fill_cmap,
            isochrone_levels=iso_levels,
        )
        # スクラバ用プレイヤーの HTML はここで一度だけ組む。フレーム PNG を
        # data URI で埋め込むので、後段の再生・シーク操作はサーバに戻らない。
        _player_html = build_fill_player_html(
            _frame_paths,
            fill_frame_times(result, num_frames),
            fill_frame_fractions(result, num_frames),
            fps=8,
        )
        _press_path = render_pressure_map(result, _tmp_dir / "pressure.png")
        _weld_path = render_weldlines(
            result, _tmp_dir / "weld.png", weld_min_angle_deg=float(weld_min_angle)
        )
        # 保圧が届く時間（見積り）。キャビティの厚さが動かない通常成形でだけ出す:
        # 射出圧縮は圧縮そのものが保圧の役をし、厚さも充填中に変わる。
        # 出せないときは理由を残して充填の結果はそのまま見せる: ここで例外を出すと、
        # 済んだ解析の結果も ZIP も画面に届かなくなる（Codex P2 on PR #18）。
        _pack_path: Path | None = None
        _pack_rows: list[dict] | None = None
        _pack_skip: str | None = None
        if icm:
            _pack_skip = PACK_SKIP_ICM
        elif not 0.0 < float(solid_fraction) < 1.0:
            # 層別の「固化判定 fraction」は 0 まで選べる。0 は「金型温度まで下がったら固化」で、
            # 板の中心がそこへ届く時刻は決まらない（無限大）
            _pack_skip = PACK_SKIP_FRACTION
        else:
            _pack = packing_reach(
                geom.mask,
                geom.thickness_mm,
                result.fill_time_s,
                geom.gates,
                float(mat.thermal_diffusivity_m2_s),
                float(solid_fraction),
            )
            _pack_path = render_packing_reach_map(result, _pack, _tmp_dir / "packing_reach.png")
            _pack_region = geom.mask if geom.product_mask is None else geom.mask & geom.product_mask
            _pack_rows = summarize_by_thickness(
                _pack, geom.thickness_mm, _pack_region, geom.cell_size_mm
            )
        # 保圧の計算（圧力の時間変化と体積の縮み）。頼まれたときだけ解く。
        _hold_path: Path | None = None
        _hold_hist_path: Path | None = None
        _hold_late_path: Path | None = None
        _hold_rows: list[dict] | None = None
        _hold_info: dict | None = None
        _flash_path: Path | None = None
        _flash_info: dict | None = None
        _hold_skip: str | None = None
        if icm:  # the sidebar hides the holding inputs with ICM
            _hold_skip = HOLD_SKIP_ICM
        elif not hold_on:
            _hold_skip = HOLD_SKIP_OFF
        elif mat.pvt is None:
            _hold_skip = HOLD_SKIP_PVT
        elif not 0.0 < float(solid_fraction) < 1.0:
            _hold_skip = PACK_SKIP_FRACTION
        elif not np.all(np.isfinite(np.asarray(result.fill_time_s, dtype=float)[geom.mask])):
            _hold_skip = HOLD_SKIP_SHORT
        else:
            _hold_factor = (
                coarsening_factor(int(geom.mask.sum()), HOLD_MAX_CELLS) if hold_coarse else 1
            )
            # 機械が送れる量の上限。射出率が決まっていればその値、プロファイルのときは
            # 設定した射出がキャビティの体積を送る平均（解いた充填時間は、射出圧の上限で
            # 遅れたり止まったりすると機械の速さを表さないので使わない）
            _hold_rate = (
                float(inj_Q) * 1e-6
                if inj_Q
                else getattr(solver, "_base", solver).programmed_mean_rate_cm3s(
                    float(geom.thickness_mm[geom.mask].sum()) * geom.cell_size_mm**2
                )
                * 1e-6
            )
            # 充填で温度を刻みごとに解いたときは、VP 切替の時点の温度を保圧へ渡す
            _arrival = np.asarray(result.fill_time_s, dtype=float)
            _vp = float(hold_vp_pct) / 100.0
            _hold_thermal: dict = {}
            _hold_note: str | None = None
            _hold_blocked: str | None = None
            if energy_on:
                _md = result.metadata
                _snaps = getattr(result, "thermal_snapshots", None) or {}
                _unreached = getattr(result, "unreached_mask", None)
                _stalled = bool(_md.get("march_stalled", False))
                _delivered = float(_md.get("march_volume_fraction", 1.0))
                _theta = None
                if _vp in _snaps and not (_stalled and _vp >= _delivered):
                    _theta = _snaps[_vp][1]
                elif _stalled and _unreached is not None and np.any(_unreached & geom.mask):
                    # 充填が VP 切替より前で止まった: 止まった時点で切り替え、届かなかった
                    # マスは空として保圧で埋めにいく（到達時刻を充填の終わりの後ろへ送る）
                    _theta = getattr(result, "thermal_theta_end", None)
                    _cell_v = geom.thickness_mm * geom.cell_size_mm**2
                    _reached_v = float(_cell_v[geom.mask & ~_unreached].sum())
                    _vp = _reached_v / float(_cell_v[geom.mask].sum()) * (1.0 - 1e-9)
                    _arrival = np.where(_unreached, np.nanmax(_arrival[geom.mask]) + 1.0, _arrival)
                    _hold_note = (
                        f"充填が体積の {100.0 * _delivered:.2f} % で止まったので、VP 切替の設定より前の"
                        "その時点で保圧に切り替えた。届かなかった所は空として、保圧で埋めにいく。"
                    )
                elif _vp >= 1.0:
                    _theta = getattr(result, "thermal_theta_end", None)
                if _theta is None:
                    _hold_blocked = (
                        "充填の VP 切替の時点の温度が残っていないので、保圧を解けませんでした"
                        "（解き直すと出ます）。"
                    )
                else:
                    _hold_thermal = {
                        "thermal": "layers",
                        "initial_theta": _theta,
                        "shear_heating_enabled": bool(shear_heating_enabled),
                    }
                    if wall_contact_model == "htc":
                        _hold_thermal["htc_packing_W_m2K"] = float(htc_pack)
                        _hold_thermal["htc_detached_W_m2K"] = float(htc_detached)
            _hold = None
            if _hold_blocked is not None:
                _hold_skip = _hold_blocked
                st.warning(_hold_skip)
            else:
                if _hold_note is not None:
                    st.info(_hold_note)
                try:
                    with st.spinner("保圧を解いている…"):
                        _hold = solve_packing_coarsened(
                            geom.mask,
                            geom.thickness_mm,
                            _arrival,
                            geom.gates,
                            geom.cell_size_mm,
                            mat,
                            melt_C + 273.15,
                            mold_C + 273.15,
                            float(hold_p_mpa) * 1e6,
                            float(hold_t_s),
                            solid_fraction=float(solid_fraction),
                            factor=_hold_factor,
                            switch_fill_fraction=_vp,
                            max_inflow_m3_s=_hold_rate,
                            **_hold_thermal,
                        )
                except (ValueError, RuntimeError) as e:
                    # 充填の結果は済んでいる。保圧だけ理由を残して飛ばす
                    _hold_skip = f"保圧の計算を完了できませんでした: {e}"
                    st.warning(_hold_skip)
            if _hold is not None:
                _hold_region = (
                    geom.mask if geom.product_mask is None else geom.mask & geom.product_mask
                )
                _hold_path = render_packing_shrinkage_map(
                    result, _hold, _tmp_dir / "packing_solve.png"
                )
                _hold_late_path = render_packing_late_fill_map(
                    result, _hold, _tmp_dir / "packing_late_fill.png"
                )
                _hold_hist_path = render_packing_pressure_history(
                    _hold, geom.thickness_mm, _hold_region, _tmp_dir / "packing_pressure.png"
                )
                _hold_rows = summarize_solve_by_thickness(
                    _hold, geom.thickness_mm, _hold_region, geom.cell_size_mm
                )
                _hold_info = {
                    "pressure_MPa": float(hold_p_mpa),
                    "time_s": float(hold_t_s),
                    "vp_switch_fill_percent": round(100.0 * _vp, 6),
                    "vp_switch_note": _hold_note,
                    "vp_switch_time_s": float(_hold.switch_time_s),
                    "unfilled_volume_percent": 100.0 * _hold.unfilled_volume_fraction,
                    "max_inflow_cm3_s": _hold_rate * 1e6,
                    "solid_fraction": float(solid_fraction),
                    "grid_mm": float(_hold.cell_size_mm),
                    "added_mass_percent": 100.0 * _hold.added_mass_fraction,
                    "clamp_force_peak_kN": float(_hold.clamp_force_N.max()) / 1e3,
                    "end_time_s": float(_hold.times_s[-1]),
                    "unconverged_steps": int(_hold.unconverged_steps),
                    "rate_cap_exceeded_steps": int(_hold.rate_cap_exceeded_steps),
                    "thermal": "layers" if _hold_thermal else "series",
                }
                # バリ注意（PL）: 輪郭のセルが溶けている間に受けた最大の圧力
                _flash = flash_risk(
                    geom,
                    _hold,
                    machine_clamp_N=float(machine_clamp_kN) * 1e3,
                    threshold_Pa=(
                        None if flash_threshold_mpa is None else float(flash_threshold_mpa) * 1e6
                    ),
                )
                _flash_path = render_flash_risk_map(result, _flash, _tmp_dir / "flash_risk.png")

                def _spot_row(tag, sp):
                    return {
                        "印": tag,
                        "x [mm]": round(sp.x_mm, 1),
                        "y [mm]": round(sp.y_mm, 1),
                        "肉厚 [mm]": round(sp.thickness_mm, 2),
                        "最大圧力 [MPa]": round(sp.pressure_Pa / 1e6, 1),
                        "時刻 [s]": round(sp.time_s, 2),
                    }

                _flash_info = {
                    "product_spots": [
                        _spot_row(str(k), sp) for k, sp in enumerate(_flash.spots, start=1)
                    ],
                    "runner_spots": [
                        _spot_row(f"R{k}", sp) for k, sp in enumerate(_flash.runner_spots, start=1)
                    ],
                    "clamp_peak_kN": _flash.clamp_peak_N / 1e3,
                    "clamp_peak_time_s": _flash.clamp_peak_time_s,
                    "machine_clamp_kN": float(machine_clamp_kN),
                    "thermal": _hold_info["thermal"],
                    "threshold_MPa": (
                        None if _flash.threshold_Pa is None else _flash.threshold_Pa / 1e6
                    ),
                    "over_length_product_mm": _flash.over_length_product_mm,
                    "over_length_runner_mm": _flash.over_length_runner_mm,
                }
        _skin_path: Path | None = None
        _core_path: Path | None = None
        _layer_T_grid_path: Path | None = None
        _layer_eta_grid_path: Path | None = None
        _layer_short_shot_path: Path | None = None
        if skin_on and result.skin_thickness_mm is not None:
            _skin_path = render_skin_layer_map(result, _tmp_dir / "skin.png")
            _core_path = render_core_layer_map(result, _tmp_dir / "core.png")
        if multilayer_on and getattr(result, "layer_temperature_K", None) is not None:
            _layer_T_grid_path = render_layer_grid(
                result, _tmp_dir / "layer_temperature_grid.png", field="temperature"
            )
            _layer_eta_grid_path = render_layer_grid(
                result, _tmp_dir / "layer_viscosity_grid.png", field="viscosity"
            )
            _layer_short_shot_path = render_short_shot_map(
                result, _tmp_dir / "multilayer_short_shot.png"
            )

        _two_phase_path: Path | None = None
        _two_phase_gif_path: Path | None = None
        _two_phase_player_html: str | None = None
        _two_phase_player_height: int | None = None
        if two_phase_result is not None:
            _two_phase_path = render_two_phase_map(
                two_phase_result, _tmp_dir / "two_phase_short_shot.png"
            )
            _two_phase_gif_path = render_two_phase_animation(
                two_phase_result,
                _tmp_dir / "two_phase.gif",
                num_frames=num_frames,
                fps=8,
            )
            # 充填先端と同じスクラバ。フレーム系列は frame_states が単一ソース
            # なので GIF のコマ k とプレイヤーのコマ k は同じ状態を指す。
            _two_phase_frame_paths = export_two_phase_frames(
                two_phase_result, _tmp_dir / "two_phase_frames", num_frames=num_frames
            )
            _n_tp = len(_two_phase_frame_paths)
            _two_phase_player_html = build_fill_player_html(
                _two_phase_frame_paths,
                [0.0] * _n_tp,
                [0.0] * _n_tp,
                fps=8,
                labels=two_phase_frame_labels(two_phase_result, num_frames),
            )
            _two_phase_player_height = fill_player_height_px(_two_phase_frame_paths)

        _zip_buf_run = io.BytesIO()
        with zipfile.ZipFile(_zip_buf_run, "w", zipfile.ZIP_DEFLATED) as _zf_run:
            for _p in (
                _gif_path,
                _press_path,
                _weld_path,
                _pack_path,
                _hold_path,
                _hold_hist_path,
                _hold_late_path,
                _flash_path,
                _skin_path,
                _core_path,
                _layer_T_grid_path,
                _layer_eta_grid_path,
                _layer_short_shot_path,
                _two_phase_path,
                _two_phase_gif_path,
            ):
                if _p is not None and _p.exists():
                    _zf_run.write(_p, _p.name)
            # 各フレーム PNG を frames/ 配下に同梱
            for _fp in _frame_paths:
                if _fp.exists():
                    _zf_run.write(_fp, f"frames/{_fp.name}")
            # ZIP を渡された相手が、Streamlit も追加ソフトも無しに
            # 画面と同じコマ送りを使えるようプレイヤーを単体 HTML で同梱する。
            # フレームは data URI で埋まっているのでオフラインで完結し、
            # HTML が受信側のフィルタで弾かれても frames/ の連番 PNG が残る。
            _zf_run.writestr(
                "player.html",
                wrap_standalone_html(
                    _player_html,
                    title="充填アニメーション",
                    note=build_label(),
                ),
            )
            _zf_run.writestr("settings.json", settings_json(run_settings))
            _zf_run.writestr(
                "metadata.json",
                json.dumps(result.metadata, indent=2, ensure_ascii=False, default=str),
            )
            if _pack_rows is not None:
                _zf_run.writestr(
                    "packing_reach.json",
                    json.dumps(
                        {"solid_fraction": float(solid_fraction), "by_thickness": _pack_rows},
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            if _hold_rows is not None:
                _zf_run.writestr(
                    "packing_solve.json",
                    json.dumps(
                        {**(_hold_info or {}), "by_thickness": _hold_rows},
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            if _flash_info is not None:
                _zf_run.writestr(
                    "flash_risk.json", json.dumps(_flash_info, indent=2, ensure_ascii=False)
                )
            if two_phase_result is not None:
                if _two_phase_player_html is not None:
                    _zf_run.writestr(
                        "two_phase_player.html",
                        wrap_standalone_html(
                            _two_phase_player_html,
                            title="二相ショートショット アニメーション",
                            note=build_label(),
                        ),
                    )
                _zf_run.writestr(
                    "two_phase_metadata.json",
                    json.dumps(
                        two_phase_result.metadata, indent=2, ensure_ascii=False, default=str
                    ),
                )

        # 解析結果の一式を session_state に格納。次回 rerun（3D スライダー操作等）
        # でも下のブロックがこれを拾って表示する。
        st.session_state["mfs_result"] = result
        st.session_state["mfs_settings"] = run_settings
        st.session_state["mfs_inputs"] = analysis_inputs
        st.session_state["mfs_geom"] = geom
        st.session_state["mfs_skin_on"] = skin_on
        st.session_state["mfs_multilayer_on"] = multilayer_on
        st.session_state["mfs_num_frames"] = num_frames
        # The previous run's assets are unreachable once the paths below are
        # replaced; drop them so repeated runs on a long-lived server do not
        # fill the temp filesystem (Codex P2 on PR #5). Done here, after the
        # new assets exist, so a failed render keeps the old results intact.
        _prev_tmp = st.session_state.get("mfs_tmp_dir")
        if _prev_tmp is not None and Path(_prev_tmp) != _tmp_dir:
            shutil.rmtree(_prev_tmp, ignore_errors=True)
        st.session_state["mfs_tmp_dir"] = _tmp_dir
        st.session_state["mfs_gif_path"] = _gif_path
        st.session_state["mfs_player_html"] = _player_html
        st.session_state["mfs_player_height"] = fill_player_height_px(_frame_paths)
        st.session_state["mfs_press_path"] = _press_path
        st.session_state["mfs_weld_path"] = _weld_path
        st.session_state["mfs_weld_min_angle"] = float(weld_min_angle)
        st.session_state["mfs_pack_path"] = _pack_path
        st.session_state["mfs_pack_rows"] = _pack_rows
        st.session_state["mfs_pack_solid_fraction"] = float(solid_fraction)
        st.session_state["mfs_pack_skip"] = _pack_skip
        st.session_state["mfs_hold_path"] = _hold_path
        st.session_state["mfs_hold_hist_path"] = _hold_hist_path
        st.session_state["mfs_hold_late_path"] = _hold_late_path
        st.session_state["mfs_hold_rows"] = _hold_rows
        st.session_state["mfs_hold_info"] = _hold_info
        st.session_state["mfs_hold_skip"] = _hold_skip
        st.session_state["mfs_flash_path"] = _flash_path
        st.session_state["mfs_flash_info"] = _flash_info
        st.session_state["mfs_skin_path"] = _skin_path
        st.session_state["mfs_core_path"] = _core_path
        st.session_state["mfs_layer_T_grid_path"] = _layer_T_grid_path
        st.session_state["mfs_layer_eta_grid_path"] = _layer_eta_grid_path
        st.session_state["mfs_layer_short_shot_path"] = _layer_short_shot_path
        st.session_state["mfs_two_phase_path"] = _two_phase_path
        st.session_state["mfs_two_phase_gif_path"] = _two_phase_gif_path
        st.session_state["mfs_two_phase_player_html"] = _two_phase_player_html
        st.session_state["mfs_two_phase_player_height"] = _two_phase_player_height
        st.session_state["mfs_two_phase_result"] = two_phase_result
        st.session_state["mfs_two_phase_skip"] = two_phase_skip_reason
        st.session_state["mfs_zip_bytes"] = _zip_buf_run.getvalue()


# 結果が session_state にある間は、do_run=False のときも（3D 倍率スライダー
# などのウィジェット操作で rerun が走った場合も）表示を維持する。
def _refresh_weld_assets(min_angle: float) -> None:
    """Re-threshold the cached weld map when the slider moves after a run.

    The solver keeps ``weld_angle_deg`` precisely so this does not need a
    re-solve: redraw weld.png from the cached result, update the recorded
    setting, and swap both entries inside the cached ZIP so a download taken
    after moving the slider matches what the screen shows (Codex P2).
    """
    cached = st.session_state["mfs_result"]
    tmp_dir = st.session_state["mfs_tmp_dir"]
    new_path = render_weldlines(cached, tmp_dir / "weld.png", weld_min_angle_deg=min_angle)
    st.session_state["mfs_weld_path"] = new_path
    st.session_state["mfs_weld_min_angle"] = min_angle
    settings = st.session_state["mfs_settings"]
    settings["output"]["weld_min_angle_deg"] = min_angle
    old_zip = zipfile.ZipFile(io.BytesIO(st.session_state["mfs_zip_bytes"]))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in old_zip.infolist():
            if info.filename in (new_path.name, "settings.json"):
                continue
            zf.writestr(info, old_zip.read(info.filename))
        zf.write(new_path, new_path.name)
        zf.writestr("settings.json", settings_json(settings))
    st.session_state["mfs_zip_bytes"] = buf.getvalue()


if "mfs_result" in st.session_state:
    if st.session_state.get("mfs_weld_min_angle") != float(weld_min_angle):
        _refresh_weld_assets(float(weld_min_angle))
    result_is_stale = st.session_state.get("mfs_inputs") != analysis_inputs
    result = st.session_state["mfs_result"]
    geom = st.session_state["mfs_geom"]
    skin_on = st.session_state["mfs_skin_on"]
    multilayer_on = st.session_state.get("mfs_multilayer_on", False)
    num_frames = st.session_state["mfs_num_frames"]
    gif_path = st.session_state["mfs_gif_path"]
    player_html = st.session_state.get("mfs_player_html")
    player_height = st.session_state.get("mfs_player_height")
    press_path = st.session_state["mfs_press_path"]
    weld_path = st.session_state["mfs_weld_path"]
    skin_path = st.session_state["mfs_skin_path"]
    core_path = st.session_state["mfs_core_path"]
    layer_T_grid_path = st.session_state.get("mfs_layer_T_grid_path")
    layer_eta_grid_path = st.session_state.get("mfs_layer_eta_grid_path")
    layer_short_shot_path = st.session_state.get("mfs_layer_short_shot_path")
    two_phase_path = st.session_state.get("mfs_two_phase_path")
    two_phase_gif_path = st.session_state.get("mfs_two_phase_gif_path")
    two_phase_player_html = st.session_state.get("mfs_two_phase_player_html")
    two_phase_player_height = st.session_state.get("mfs_two_phase_player_height")
    two_phase_result = st.session_state.get("mfs_two_phase_result")
    two_phase_skip = st.session_state.get("mfs_two_phase_skip")
    _zip_bytes = st.session_state["mfs_zip_bytes"]

    with col_right:
        st.subheader("結果")
        if result_is_stale:
            st.warning(STALE_RESULT_MSG)
        c1, c2, c3 = st.columns(3)
        c1.metric("総充填時間 T_fill", f"{result.total_fill_time_s:.3f} s")
        c2.metric("代表粘度 η_eff", f"{result.viscosity_Pa_s:.1f} Pa·s")
        c3.metric("キャビティ体積", f"{geom.volume_cm3():.2f} cm³")
        # A cut-off loop makes the headline T_fill and every map below it a
        # snapshot; say so here, next to the number, not only in an expander
        _fixed_point_warning(
            result.metadata, converged_key="skin_converged", what="スキン層", advice=_SKIN_ADVICE
        )
        _fixed_point_warning(
            result.metadata,
            converged_key="multilayer_converged",
            what="層別モデル",
            advice="反復上限を上げて再実行を。",
            no_flow_key=None,
        )
        _stall_msg = _stall_note(result.metadata, getattr(result, "unreached_mask", None), geom)
        if _stall_msg is not None:
            st.info(_stall_msg)
        _gate_msg = _gate_open_note(result.metadata)
        if _gate_msg is not None:
            st.info(_gate_msg)
        if result.metadata.get("domain_converged") is False:
            st.warning(
                "封止で切れたセルを外す領域の絞り込み（二分探索）が上限回数で打ち切られた。"
                "充填可能な領域が確定していないので、封止・未充填の数は目安。"
            )
        # Read the run's own record, not the sidebar: the pane also renders
        # from a cached result, and the sidebar may have moved since.
        _prof_rec = result.metadata.get("injection_profile")
        if _prof_rec is not None:
            _V_shot_cm3 = float(_prof_rec["total_volume_cm3"])
            st.caption(
                f"射出条件: 平均 {_prof_rec['mean_rate_cm3s']:.1f} cm³/s、"
                f"V/P まで {_prof_rec['total_time_s']:.3f} s、"
                f"理論射出量 {_V_shot_cm3:.2f} cm³"
            )
            if result.metadata.get("injection_extrapolated_past_vp"):
                # Past V/P the volume-to-time map extrapolates at the last
                # stage's rate — the fill time is then "if the machine kept
                # going", not a fill this shot achieves. The comparison is the
                # solver's, against the volume it actually read the map
                # through: with ICM that is the *open-gap* volume of the cells
                # that fill, which a stroke can clear the final cavity and
                # still fall short of (Codex P2 on the port PRs).
                _V_swept_cm3 = float(result.metadata.get("injection_swept_volume_cm3", 0.0))
                st.warning(
                    f"理論射出量 {_V_shot_cm3:.2f} cm³ が、この解析が掃く体積 "
                    f"{_V_swept_cm3:.2f} cm³ を下回っています"
                    + ("（射出圧縮成形なので開いた隙間の体積です）。" if icm else "。")
                    + "V/P 以降も最終段の射出率で射出し続けた前提の時間軸になって"
                    f"いるので、{_V_shot_cm3:.2f} cm³ を超えた分の充填時刻は外挿です。"
                    "計量律速のショートショットは「ショートショット（計量制限）」"
                    "で見てください。"
                )

        def _download(label: str, path: Path, mime: str, key: str) -> None:
            with open(path, "rb") as _f:
                st.download_button(
                    label,
                    data=_f.read(),
                    file_name=path.name,
                    mime=mime,
                    key=key,
                )

        st.markdown("**充填先端アニメーション**")
        if player_html:
            # 高さはフレーム PNG の実寸から導出する。プレイヤー側が画像を
            # ネイティブ幅で頭打ちにするので、列がどれだけ広くても操作列が
            # この高さから溢れない（scrolling=False で切れると操作不能になる）。
            components.html(player_html, height=player_height, scrolling=False)
        else:
            st.image(str(gif_path))
        st.download_button(
            "⬇ GIF + フレーム画像をダウンロード",
            data=_zip_bytes,
            file_name="mold_flow_results.zip",
            mime="application/zip",
            key="dl_zip_all",
            help=(
                "GIF（fill.gif）・各フレーム PNG（frames/frame_NNN.png）・"
                "各マップ PNG・metadata.json（解析結果）・settings.json（入力設定）を"
                "1つの ZIP にまとめてダウンロード"
            ),
        )

        with st.expander("圧力マップ"):
            st.image(str(press_path))
            st.caption("0=ゲート遠端、1=ゲート。実圧力スケールではなく相対分布。")
            _download("⬇ PNGをダウンロード", press_path, "image/png", "dl_press_png")

        with st.expander("等値線・ウェルドライン候補・エアトラップ"):
            st.image(str(weld_path))
            st.caption(
                "濃い赤=ウェルド（開き角 45° 以上）、薄い赤=メルド（下限角〜45°）、"
                "黄×=最終充填位置（エアトラップ候補）。下限角はサイドバー「出力」で変えられる"
            )
            _download("⬇ PNGをダウンロード", weld_path, "image/png", "dl_weld_png")

        with st.expander("保圧が届く時間（見積り）"):
            _pack_path_shown = st.session_state.get("mfs_pack_path")
            _pack_rows_shown = st.session_state.get("mfs_pack_rows")
            if _pack_path_shown is None or _pack_rows_shown is None:
                st.info(st.session_state.get("mfs_pack_skip") or PACK_SKIP_ICM)
            else:
                st.image(str(_pack_path_shown))
                _sf = float(st.session_state.get("mfs_pack_solid_fraction", 0.3))
                st.caption(
                    "色＝充填の終わりから数えて、ゲートからその場所まで溶けた通り道が"
                    "残っている時間。灰色＝充填の終わりに厚さの中心まで固まっている所。"
                    f"固化の判定は T_mold + {_sf:g}·(T_melt − T_mold)。"
                )
                st.dataframe(
                    [
                        {
                            "肉厚 [mm]": r["thickness_mm"],
                            "面積 [cm²]": round(r["area_cm2"], 1),
                            "充填の終わりに固化 [%]": round(100 * r["frozen_at_fill_end"], 1),
                            "固まるまで [s]": None
                            if r["solidify_s"] is None
                            else round(r["solidify_s"], 2),
                            "保圧が届く [s]": None
                            if r["reach_s"] is None
                            else round(r["reach_s"], 2),
                            "割合 [%]": None if r["ratio"] is None else round(100 * r["ratio"], 1),
                        }
                        for r in _pack_rows_shown
                    ],
                    hide_index=True,
                )
                st.caption(
                    "製品の中を肉厚ごとに集計。「固まるまで」「保圧が届く」「割合」は、"
                    "充填の終わりにまだ溶けているセルの平均。割合＝保圧が届く ÷ 固まるまで。"
                    "流れを止めて冷やした見積りで、保圧中に流れ込む樹脂の熱と樹脂の縮みは"
                    "見ていない。秒は目安で、形どうしの比較に使う。"
                )
                _download("⬇ PNGをダウンロード", _pack_path_shown, "image/png", "dl_pack_png")

        with st.expander("保圧の計算（体積の縮みと圧力）"):
            _hold_path_shown = st.session_state.get("mfs_hold_path")
            _hold_hist_shown = st.session_state.get("mfs_hold_hist_path")
            _hold_rows_shown = st.session_state.get("mfs_hold_rows")
            _hold_info_shown = st.session_state.get("mfs_hold_info")
            if (
                _hold_path_shown is None
                or _hold_hist_shown is None
                or _hold_rows_shown is None
                or _hold_info_shown is None
            ):
                st.info(st.session_state.get("mfs_hold_skip") or HOLD_SKIP_OFF)
            else:
                _hi = _hold_info_shown
                _hc0, _hc1, _hc2, _hc3 = st.columns(4)
                _hc0.metric(
                    "埋まらなかった体積",
                    f"{_hi['unfilled_volume_percent']:.1f} %",
                    help="保圧をかけても最後まで満杯にならなかったマスの体積の割合。",
                )
                _hc1.metric("保圧で入った樹脂", f"{_hi['added_mass_percent']:.1f} %")
                _hc2.metric(
                    "型を開く力の最大",
                    f"{_hi['clamp_force_peak_kN'] / 9.80665:.0f} tf",
                    help="キャビティの圧力 × 投影面積の最大（製品＋ランナ）。",
                )
                _hc3.metric("解いた格子", f"{_hi['grid_mm']:g} mm")
                _hold_late_shown = st.session_state.get("mfs_hold_late_path")
                if _hi.get("vp_switch_note"):
                    st.info(_hi["vp_switch_note"])
                if _hi["vp_switch_fill_percent"] < 100.0 and _hold_late_shown is not None:
                    st.image(str(_hold_late_shown))
                    st.caption(
                        f"体積の {_hi['vp_switch_fill_percent']:g} % を射出率で埋めた時点"
                        f"（充填開始から {_hi['vp_switch_time_s']:.2f} 秒）で保圧に切り替えた。"
                        "灰色＝切替の時点で埋まっていた所、色＝保圧で埋まった時刻（切替から）、"
                        "赤＝最後まで埋まらなかった所。機械が送る量は "
                        f"{_hi['max_inflow_cm3_s']:.1f} cm³/s を上限にしている。"
                    )
                    if _hi.get("thermal", "series") == "series":
                        st.warning(HOLD_THIN_CAVEAT)
                else:
                    st.caption("VP 切替 100 %：全部埋まったものとして、保圧だけを解いた。")
                st.image(str(_hold_path_shown))
                st.caption(
                    f"保圧 {_hi['pressure_MPa']:g} MPa を {_hi['time_s']:g} 秒かけてゲートを閉じ、"
                    "室温・大気圧まで冷えたあとの体積の縮み。赤＝縮む、青＝詰め込みすぎ（縮みが負）。"
                )
                st.dataframe(
                    [
                        {
                            "肉厚 [mm]": r["thickness_mm"],
                            "面積 [cm²]": round(r["area_cm2"], 1),
                            "縮み 平均 [%]": round(100 * r["shrinkage_mean"], 2),
                            "縮み 最小 [%]": round(100 * r["shrinkage_min"], 2),
                            "縮み 最大 [%]": round(100 * r["shrinkage_max"], 2),
                            "圧力の最大 [MPa]": round(r["peak_pressure_Pa"] / 1e6, 1),
                            "固まるときの圧力 [MPa]": round(r["freeze_pressure_Pa"] / 1e6, 1),
                        }
                        for r in _hold_rows_shown
                    ],
                    hide_index=True,
                )
                st.image(str(_hold_hist_shown))
                st.caption(
                    "製品の中を肉厚ごとに集計（埋まらなかった所は除く）。圧力はゲートにかけた値で、ノズルとホットランナーの"
                    "損失は入っていない。"
                    + (
                        "温度は、充填で刻みごとに解いた VP 切替の時点の温度から、保圧中も刻みごとに解いた"
                        "（流れ込む樹脂が運ぶ熱を含む）。型のたわみは入っていない。"
                        if _hi.get("thermal") == "layers"
                        else "温度は充填と同じ板の冷え方で、保圧中に流れ込む樹脂の熱と型のたわみは入っていない。"
                    )
                    + "「固まるときの圧力」は厚さの中心が固まった時点の圧力。"
                )
                if _hi["unconverged_steps"]:
                    st.warning(
                        f"{_hi['unconverged_steps']} 刻みで反復が収束しないまま進めています。"
                        "数字は目安として見てください。"
                    )
                if _hi.get("rate_cap_exceeded_steps"):
                    st.warning(
                        f"{_hi['rate_cap_exceeded_steps']} 刻みで、送る量の上限より多く流れ込んだまま進めています"
                        "（上限に収まるゲート圧が見つからなかった）。保圧の始めの圧力の上がり方は目安として見てください。"
                    )
                _download("⬇ 縮みの地図 PNG", _hold_path_shown, "image/png", "dl_hold_png")

        with st.expander("バリ注意（PL）"):
            _flash_path_shown = st.session_state.get("mfs_flash_path")
            _fi = st.session_state.get("mfs_flash_info")
            if _flash_path_shown is None or _fi is None:
                st.info(
                    "保圧の計算の結果から出す。"
                    + (st.session_state.get("mfs_hold_skip") or HOLD_SKIP_OFF)
                )
            else:
                _fc0, _fc1 = st.columns(2)
                _fc0.metric(
                    "型を開く力の最大",
                    f"{_fi['clamp_peak_kN']:,.0f} kN",
                    help=f"切替から {_fi['clamp_peak_time_s']:.2f} 秒。キャビティの圧力 × 投影面積（製品＋ランナ）。",
                )
                _ratio = _fi["clamp_peak_kN"] / max(_fi["machine_clamp_kN"], 1e-9)
                _fc1.metric(
                    "機械の型締力に対して",
                    f"{100 * _ratio:.0f} %",
                    help=f"機械の型締力 {_fi['machine_clamp_kN']:,.0f} kN（サイドバーの保圧の欄）。",
                )
                if _ratio >= 1.0:
                    st.error(
                        "型を開く力が機械の型締力を超えている。型が開いて、PL の全周にバリが出る。"
                        "保圧を下げるか、VP 切替を早めるか、型締力の大きい機械にする。"
                    )
                elif _ratio >= 0.8:
                    st.warning(
                        "型を開く力が機械の型締力の 8 割を超えている。型のたわみで PL が開きやすい。"
                    )
                if _fi["thermal"] == "series":
                    st.warning(
                        "温度を到達時刻から読む解き方なので、薄い所（1 mm の額縁など）は切替の時点で中心まで"
                        "固まった扱いになり、製品の輪郭の圧力が実際より低く（0 に）出る。製品の輪郭を見るときは、"
                        "サイドバーの層別の設定で「層の温度の解き方」を「刻みごとに解く」にして解き直す。"
                    )
                if _fi.get("threshold_MPa") is not None:
                    _ft0, _ft1 = st.columns(2)
                    _ft0.metric(
                        f"目安 {_fi['threshold_MPa']:.0f} MPa 以上（製品の輪郭）",
                        f"{_fi['over_length_product_mm']:,.0f} mm",
                        help="輪郭のマスの数 × マスの大きさ。斜めや曲線の輪郭は実際より長めに出る。",
                    )
                    _ft1.metric(
                        f"目安 {_fi['threshold_MPa']:.0f} MPa 以上（ランナの輪郭）",
                        f"{_fi['over_length_runner_mm']:,.0f} mm",
                    )
                st.image(str(_flash_path_shown))
                st.caption(
                    (
                        "紫の枠＝バリの目安の圧力以上の所。"
                        if _fi.get("threshold_MPa") is not None
                        else ""
                    )
                    + "キャビティの輪郭（製品とランナの外周）を PL の合わせ目とみなし、輪郭の各マスが"
                    "厚さの中心が溶けている間に受けた最大の圧力を色で示す。固まったあとの圧力は数えない"
                    "（固まった樹脂はすき間に入らない）。黒の丸＝製品の輪郭の上位、灰色の丸＝ランナの輪郭の上位。"
                    "表の「最大圧力」はこの溶けている間の最大、「時刻」は VP 切替からの秒。"
                    "ゲートに近いランナは、形によらずほぼ保圧そのものになるので、製品とは別に順位を付けている。"
                    "すき間の深さ（ベント・摩耗）と型のたわみは入っていないので、何 MPa でバリが出るかは"
                    "型ごとに判断する。値は順位と形どうしの比較に使う。"
                )
                st.markdown("**製品の輪郭**")
                st.dataframe(_fi["product_spots"], hide_index=True)
                st.markdown("**ランナの輪郭**")
                st.dataframe(_fi["runner_spots"], hide_index=True)
                _download("⬇ PNGをダウンロード", _flash_path_shown, "image/png", "dl_flash_png")
                _download("⬇ 圧力の時間変化 PNG", _hold_hist_shown, "image/png", "dl_hold_hist_png")

        if two_phase_skip is not None:
            # 実行時の警告は rerun で流れるので、結果ペイン側にも理由を残す
            st.info(f"二相ショートショット解析はスキップされました: {two_phase_skip}")

        if two_phase_path is not None and two_phase_result is not None:
            with st.expander("二相ショートショット（計量制限 + 圧縮前進）", expanded=True):
                md2 = two_phase_result.metadata
                st.image(str(two_phase_path))
                st.caption(
                    "青=射出相で充填（白線=射出等時線）、橙=圧縮相で前進、"
                    "灰=未充填。実機の計量値・ストロークをそのまま入れて"
                    "段階ショートショットの現物形状と比較する。"
                )
                if md2.get("skin_layer_enabled"):
                    st.caption(
                        "スキン層を射出相に乗せた結果（時計は計量 V/Q 固定）: "
                        f"反復={md2.get('skin_iterations')}, 収束={md2.get('skin_converged')}, "
                        f"射出終了時のスキン最大 {md2.get('injection_skin_max_mm', 0.0):.3f} mm、"
                        f"封止 {md2.get('injection_sealed_cells', 0)} セル、"
                        f"封止で届かず {md2.get('injection_unfillable_cells', 0)} セル。"
                        "圧縮相は等温（プールは等圧ソースなので内部のスキンは前進に効かない）。"
                    )
                    # the injection phase's own fixed-point loop, same cap as the main solve
                    _fixed_point_warning(
                        md2,
                        converged_key="skin_converged",
                        what="二相の射出相のスキン層",
                        advice="反復上限を上げて再実行を。",
                        no_flow_key=None,
                    )
                    if md2.get("injection_sealed_cells", 0) > 0:
                        _short = md2["shot_volume_cm3"] - md2["achieved_volume_final_cm3"]
                        st.warning(
                            "射出中に封止したセルがある（濃赤）。射出時間が長すぎるか、"
                            "モデルがゲート部の剪断発熱を持っていないためランナの薄い入口が早く閉じている。"
                            "実機の入口が開いたままなら、この封止は模型の限界と読む。"
                            "封止は圧縮相でも閉じたまま（封止の奥へは前進しない、"
                            f"届かないセル {md2.get('compression_unreachable_cells', 0)}）。"
                            + (
                                f" 計量のうち {_short:.2f} cm³ はキャビティに入らない。"
                                if _short > 1e-9
                                else ""
                            )
                        )
                if md2.get("wall_model") == "multilayer":
                    if md2.get("fill_method") == "march":
                        _how = f"時間で追う解き方、{md2.get('multilayer_iterations')} 刻み"
                    else:
                        _how = f"固定点 {md2.get('multilayer_iterations')} 回" + (
                            "で収束" if md2.get("multilayer_converged") else "、未収束"
                        )
                    st.caption(
                        f"層別 {md2.get('num_layers')} 層を射出相に乗せた結果（時計は計量 V/Q 固定、"
                        f"{_how}）。圧縮相は等温（層ごとの温度は圧縮での前進に効かない）。"
                    )
                    # the injection phase's own fixed-point loop, same cap as the main solve
                    _fixed_point_warning(
                        md2,
                        converged_key="multilayer_converged",
                        what="二相の射出相の層別モデル",
                        advice="反復上限を上げて再実行を。",
                        no_flow_key=None,
                    )
                    if md2.get("injection_center_solid_cells", 0) > 0:
                        st.warning(
                            f"射出終了時のプールに中央層が固化温度を下回ったセルが "
                            f"{md2['injection_center_solid_cells']} ある。層別モデルは封止の"
                            "時刻を持たないので、このセルも圧縮相では流路として扱っている。"
                        )
                if md2.get("injection_extrapolated_past_vp"):
                    # The sidebar's own extrapolation check compares the
                    # stroke against the *final* cavity; neither the metered
                    # shot nor the open-gap cavity is that number, so this
                    # panel has to say it for itself (@claude on PR #89).
                    st.warning(
                        "この二相解析は V/P より先を外挿しています。計量 "
                        f"{md2['shot_volume_cm3']:.2f} cm³ が理論射出量 "
                        f"{md2.get('injection_profile_volume_cm3', 0.0):.2f} cm³ を"
                        "超えているためで、射出時間は最終段の射出率で射出し続けた"
                        "前提の値です。計量位置と V/P 位置を実機に合わせるか、"
                        "計量体積を理論射出量以下にしてください。"
                    )
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("計量体積 V_shot", f"{md2['shot_volume_cm3']:.2f} cm³")
                tc2.metric(
                    "射出終了時 充填率",
                    f"{md2['injection_fill_fraction'] * 100:.1f} %",
                )
                tc3.metric(
                    "圧縮後 充填率",
                    f"{md2['final_fill_fraction'] * 100:.1f} %",
                )
                if md2["final_complete"]:
                    st.info("この計量では圧縮後に完全充填する（ショートショットにならない）。")
                if two_phase_gif_path is not None:
                    st.markdown("**二相アニメーション**")
                    if two_phase_player_html:
                        components.html(
                            two_phase_player_html,
                            height=two_phase_player_height,
                            scrolling=False,
                        )
                    else:
                        st.image(str(two_phase_gif_path))
                    st.caption(
                        "射出相は実時間で進む。圧縮相は前進の順序のみ"
                        "（モデルは圧縮の時間スケールを持たない）。"
                    )
                _download(
                    "⬇ PNGをダウンロード",
                    two_phase_path,
                    "image/png",
                    "dl_two_phase_png",
                )
                if two_phase_gif_path is not None:
                    _download(
                        "⬇ GIFをダウンロード",
                        two_phase_gif_path,
                        "image/gif",
                        "dl_two_phase_gif",
                    )

        if skin_path is not None and core_path is not None:
            with st.expander("スキン層 / コア層 / ショートショット"):
                # skin_path exists only for a run solved with skin_layer_enabled
                # (the same skin_on that built the solver), and such a run always
                # records skin_iterations / skin_converged / no_flow in metadata
                md = result.metadata
                st.caption(
                    f"反復={md.get('skin_iterations')}, 収束={md.get('skin_converged')}, "
                    f"T_fill_inflation={md.get('T_fill_inflation', 1.0):.3f}, "
                    f"封止セル={md.get('short_shot_cells', 0)}, "
                    f"未充填セル={md.get('unfillable_cells', 0)}"
                    "（未収束の警告は結果ペインの先頭）"
                )
                st.image(str(skin_path))
                st.caption("スキン層厚さ s(x,y) [mm]。流動が遅いほど・薄肉ほど s が大きい。")
                _download("⬇ スキン層 PNGをダウンロード", skin_path, "image/png", "dl_skin_png")
                st.image(str(core_path))
                st.caption(
                    "コア層 h_core = h - 2s。赤マーク = スキン同士が会合したショートショット候補。"
                )
                _download("⬇ コア層 PNGをダウンロード", core_path, "image/png", "dl_core_png")

        if multilayer_on and layer_T_grid_path is not None:
            with st.expander("層別プロファイル (Multi-layer N=...)"):
                md = result.metadata
                _solve_how = (
                    f"解き方=時間で追う（{md.get('march_steps')} 刻み）, "
                    if md.get("fill_method") == "march"
                    else f"反復={md.get('multilayer_iterations')}, 収束={md.get('multilayer_converged')}, "
                )
                st.caption(
                    f"層数 N={md.get('num_layers')}, 分布={md.get('layer_distribution')}, "
                    + _solve_how
                    + f"T_fill_inflation={md.get('T_fill_inflation', 1.0):.3f}, "
                    f"ショートショット率={md.get('short_shot_fraction', 0.0):.3f}"
                )
                _br_max = md.get("brinkman_number_max", 0.0)
                _br_mean = md.get("brinkman_number_mean", 0.0)
                _sh_max = md.get("shear_heating_max_K", 0.0)
                _sh_mean = md.get("shear_heating_mean_K", 0.0)
                _sh_enabled = md.get("shear_heating_enabled", False)
                # Brinkman number sanity-band: < 0.5 negligible, < 2 moderate, >= 2 strong.
                if _br_max < 0.5:
                    _br_emoji = "🟢"
                elif _br_max < 2.0:
                    _br_emoji = "🟡"
                else:
                    _br_emoji = "🔴"
                _badge = "✅ ON" if _sh_enabled else "OFF"
                st.caption(
                    f"剪断発熱 {_badge}: ΔT_max={_sh_max:.1f}K / mean={_sh_mean:.2f}K　"
                    f"{_br_emoji} Brinkman数 Br_max={_br_max:.2f} / mean={_br_mean:.3f}"
                    "（Br>1 で剪断発熱が支配的）"
                )
                st.markdown(
                    "**各層の温度マップ T_k(x,y)** — 壁層は T_mold へ、中央層は T_melt 寄り"
                )
                st.image(str(layer_T_grid_path))
                _download(
                    "⬇ 温度グリッド PNG",
                    layer_T_grid_path,
                    "image/png",
                    "dl_layer_T_png",
                )
                if layer_eta_grid_path is not None:
                    st.markdown(
                        "**各層の粘度マップ η_k(x,y)** — 対数スケール、壁層は剪断高でも低温で η 大"
                    )
                    st.image(str(layer_eta_grid_path))
                    _download(
                        "⬇ 粘度グリッド PNG",
                        layer_eta_grid_path,
                        "image/png",
                        "dl_layer_eta_png",
                    )
                if layer_short_shot_path is not None:
                    st.markdown(
                        "**ショートショット予測** — 中央層温度が T_solid を切ったセルを赤マーク"
                        + (
                            "。充填が止まって届かなかったマスも赤（理由は結果の先頭）"
                            if result.metadata.get("march_stalled")
                            else ""
                        )
                    )
                    st.image(str(layer_short_shot_path))
                    _download(
                        "⬇ ショートショット PNG",
                        layer_short_shot_path,
                        "image/png",
                        "dl_layer_short_png",
                    )

        with st.expander("3D表示（plotly）"):
            st.caption(
                "PL（パーティングライン）= Z=0 を底面とし、各セルを厚み h(x,y) 分だけ"
                "立ち上げたソリッド表示。x / y / z すべて同じ mm スケール（実物等倍）"
                "で描画。**天面と側壁の両方が物理量で着色**され、1つのカラーバーで"
                "読める（PLの薄グレー床は形状参照用）。ドラッグで回転、スクロール"
                "でズーム。物理は 2D Hele-Shaw のまま（表現上の3D化のみ）。"
            )
            t3d_h, t3d_fill, t3d_press = st.tabs(["厚み h(x,y)", "充填時間", "圧力"])
            with t3d_h:
                st.plotly_chart(
                    render_3d_thickness_map(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with t3d_fill:
                st.plotly_chart(
                    render_3d_fill_time(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with t3d_press:
                st.plotly_chart(
                    render_3d_pressure(result),
                    use_container_width=True,
                    config={"displaylogo": False},
                )

        with st.expander("この結果を出した設定"):
            st.caption(
                "ZIP の settings.json と同じ内容。metadata.json は解いた結果しか"
                "持たないので、設定を辿るならこちら。"
                "アップロードしたスペック JSON は名前と SHA-256 だけを記録する"
                "（ZIP は人に渡す前提なので、図面由来の寸法は載せない）。"
            )
            st.json(st.session_state.get("mfs_settings", {}))

        with st.expander("生データ"):
            st.json(result.metadata)
else:
    with col_right:
        st.info("左側でパラメータを設定し、「解析実行」を押してください。")
