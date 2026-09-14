"""Streamlit UI — 12.3-inch frame plate (body-only compression) fed by a fan runner
from a hot-runner gate (mold-flow-fangate2).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from core import (
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
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
    wrap_standalone_html,
)
from core.geometry import Geometry
from core.injection_profile import InjectionProfile, InjectionStage
from core.settings_record import config_settings, settings_json
from core.solver import WELD_MIN_ANGLE_DEG
from core.two_phase import solve_two_phase_short_shot
from core.version import build_label
from core.visualizer import (
    ISOCHRONE_LEVELS,
    THICKNESS_CMAP,
    export_two_phase_frames,
    render_layer_grid,
    render_short_shot_map,
    render_two_phase_animation,
    render_two_phase_map,
    two_phase_frame_labels,
)

APP_DIR = Path(__file__).parent

#: Default machine conditions for the screw-side injection block. A real
#: shot is a screw diameter plus positions and speeds; the rate follows from
#: them. The previous default -- a flat 589 cm^3/s -- was a spec-sheet
#: *maximum*, so every condition ran as if the screw were flat out.
DEF_SCREW_DIAMETER_MM = 50.0
DEF_METERING_POSITION_MM = 150.0
DEF_VP_POSITION_MM = 18.0
DEF_SCREW_VELOCITY_MMS = 200.0
#: Same machine as the sibling repos, but this part is two orders of
#: magnitude larger, so the metering position is the one setting that
#: differs: the stroke has to displace the cavity as drawn (plus a margin)
#: before the volume-to-time map stops extrapolating past V/P. The machine
#: runs three equal-speed steps -- arithmetically a single stage, kept as
#: the default because it is what its screen says -- but the switch points
#: of the other job do not carry over, so the stroke is split evenly.
DEF_INJECTION_STAGES = 3
DEF_SWITCH_POSITIONS_MM = ()
#: Cap on the stage count the UI will unfold. Machines go further; this is
#: the point past which the sidebar stops being readable.
MAX_INJECTION_STAGES = 5


#: Label recorded in ``settings.json`` for the one geometry input this app has.
GEOMETRY_SOURCE = "Fan runner plate (parametric)"

st.set_page_config(page_title="12.3 インチ 額縁プレート 流動解析", layout="wide")
st.title("12.3 インチ 額縁プレート 流動解析")
st.caption(
    "12.3 インチ 額縁肉厚プレート（外周 20 幅 t1.0／内側 t4.0、圧縮は内側だけ）をホットランナーゲート φ3 →"
    "ファン状ランナ（幅 300、R12 丸端、肉厚 1.0 → 2.5）で射出圧縮成形するときの樹脂流動を簡易解析するツール。"
    "ランナ寸法・肉盗み・圧縮条件の方向性検討を、実機評価前の初期検討段階で迅速に行うことを目的とする。"
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
        "- 解いた $\\tau$ を最大値で正規化し、絶対時間に換算: "
        r"$t_{\text{fill}}(x,y) = \dfrac{\tau(x,y)}{\tau_{\max}} \cdot T_{\text{fill}}$"
        "\n- $T_{\\text{fill}} = V_{\\text{cavity}} / Q$（射出率一定）\n"
        "- 多段射出（実機条件モードで段数 2 以上）では $Q$ が段ごとに変わるので、"
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

    st.markdown("### 5. 壁面冷却モデル B：層別 N 層離散化（推奨・極薄向け既定）")
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
        r"$T_k \ge T_{\text{mold}}$ で clamp。$t_{\text{arr}}(x,y) = (\tau/\tau_{\max}) \cdot T_{\text{fill}}$"
        " はセル到達時間。"
    )

    st.markdown("**5-3. 層別剪断速度（Poiseuille 解析微分）**")
    st.latex(r"\dot\gamma_k(x,y) = \frac{6 V}{h(x,y)} \cdot |2\zeta_k - 1|")
    st.markdown(
        "壁で最大、中央でゼロ。中央層は Cross-WLF の $D_1$ 発散を避けるため "
        "$\\dot\\gamma_{\\text{floor}} = 0.01 \\cdot 6V/h$ でクリップ。"
    )

    st.markdown("**5-4. 並列流路統合（Poiseuille モーメント積分）**")
    st.latex(r"S_{\text{total}}(x,y) = \frac{h(x,y)^3}{2} \sum_{k=1}^{N} \frac{m_k}{\eta_k(x,y)}")
    st.latex(
        r"m_k = \left[\frac{\zeta^2}{2} - \frac{\zeta^3}{3}\right]_{\zeta_{k-1}}^{\zeta_k},"
        r"\quad \sum_k m_k = \frac{1}{6}"
    )
    st.markdown(
        r"$\sum m_k = 1/6$ が保存するので $N=1$ では従来 $S = h^3/(12\eta)$ と厳密一致（後方互換）。"
    )

    st.markdown("**5-5. 固定点反復で $\\tau$ と層フィールドを結合**")
    st.markdown(
        r"$\tau \to t_{\text{arr}} \to T_k \to \eta_k \to S_{\text{total}} \to \tau_{\text{new}}$"
        " を $\\|\\Delta\\tau\\|_2 / \\|\\tau\\|_2 < $ tol まで反復。"
        "発散時のみ $\\omega = 0.7$ で適応的 damping。**ショートショット判定**は最終 iteration の中央層温度ベース:"
        r" $T_{\text{solid}} = T_{\text{mold}} + f_{\text{solid}} (T_{\text{melt}} - T_{\text{mold}})$。"
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
        " $T_k \\leftarrow T_{k,\\text{Neumann}} + \\Delta T_{\\text{shear},k}$ で Cross-WLF を再評価、"
        "粘度低下→流動加速→発熱低下の負のフィードバックは fixed-point 反復で自然収束。"
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

    st.markdown("---")
    st.markdown("### ✅ モデル化している現象")
    st.markdown(
        "- 薄板キャビティ内の 2D 流動（局所剪断と局所抵抗の効果）\n"
        "- 樹脂物性（密度・比熱・熱拡散率 → 熱伝導率派生・Cross-WLF 粘度パラメータ）\n"
        "- 額縁プレートの 2 段肉厚、ファン状ランナ（幅・フランク角・R 丸端・深さ方向の肉厚プロファイル）、肉盗み（▽）\n"
        "- 流動先端の到達順、ウェルドライン、エアトラップ\n"
        "- 圧力分布の相対値（ゲート＝1、最終充填点＝0 の正規化）\n"
        "- 充填時間（射出率 $Q$ から逆算した絶対時間）\n"
        "- **壁面冷却**: スキン層 1 層モデル または **層別 $N$ 層モデル**（厚み方向温度・粘度プロファイル）\n"
        "- **剪断発熱（段階1, 層別モード内）**: 粘性散逸による局所温度上昇、Brinkman 数診断\n"
        "- **ショートショット予測**: スキン層モデルでは露光時計の封止（$t_{\\text{close}} = t_{\\text{arr}} + t_c$）で切られたセル、"
        "層別モデルでは中央層温度ベース判定\n"
        "- 射出圧縮成形による等価流路拡大（stroke モード、CLI に factor 後方互換あり、オプション）"
    )

    st.markdown("### ❌ モデル化していない現象（重要）")
    st.markdown(
        "- **面内 3D 流れ・ジェッティング・噴流・コーナー渦**: あくまで 2D Hele-Shaw（厚み方向は層別化済みだが、"
        "**面内**の 3D 性は完全 3D FVM / FEM ソルバーでないと出ない。Hele-Shaw 系の根本限界）\n"
        "- **剪断発熱の自己整合（段階2）**: 段階1 は閉形式の局所近似のみ。"
        r"$\rho c_p \partial_t T = k \partial_z^2 T + \eta \dot\gamma^2$ を厚み方向 1D FDM で陰解法積分する"
        "段階2 は別ロードマップ。$Br \\ge 2$ 領域では段階1 がズレる\n"
        "- **保圧（パッキング段階）**: 充填までしかモデル化しない\n"
        "- **収縮・反り・残留応力**: 熱固化収縮も結晶化も入っていない\n"
        "- **層内対流項**: 1D Neumann は純粋拡散のみ（薄板では妥当な近似だが、極厚 $h > 4$ mm では破綻）\n"
        "- **ベント・脱気挙動**: エアトラップ位置は予測するが圧抜けは考慮しない\n"
        "- **ホットランナー／ゲートの圧損**: ゲート φ3 の円を射出点として扱い、ノズル・ゲートの縦方向抵抗は入っていない\n"
        "- **STL/STEP 直接読み込み**: パラメトリック形状（額縁プレート＋ファン状ランナ）のみ\n"
        "- **非構造格子・中立面メッシュ**: 構造格子（正方形セル）固定\n"
        "- **絶対圧力場の出力**: 圧力は正規化値（ゲート=1 / フロント=0）のみ。"
        "実機の必要型締力評価には未対応"
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
_MATERIAL_DB_SCHEMA_VERSION = "v2_shear_heating"


@st.cache_resource
def _load_db(schema_version: str = _MATERIAL_DB_SCHEMA_VERSION) -> MaterialDB:
    return MaterialDB()


db = _load_db()
material_keys = list(db.keys())


# ----------------------- fan-runner plate: parametric inputs -----------------------
_D = FanRunnerPlateConfig()  # the drawing (docs/spec.md)


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

    with st.expander("製品（額縁プレート）", expanded=False):
        st.caption("圧縮されるのは内側の厚肉部だけ。額縁とランナは固定。")
        num("plate_w_mm", "製品幅（長辺、ランナ側）[mm]", 50.0, 600.0, 1.0)
        num("plate_h_mm", "製品奥行 [mm]", 30.0, 600.0, 1.0)
        num("frame_w_mm", "額縁幅 [mm]", 0.0, 100.0, 0.5, fmt="%.1f")
        num("frame_thk_mm", "額縁肉厚 [mm]", 0.1, 10.0, 0.05)
        num("inner_thk_mm", "内側肉厚（圧縮部）[mm]", 0.1, 10.0, 0.05)
    with st.expander("ランナ（ファン状、製品エッジ直結）", expanded=False):
        num("runner_w_mm", "ランナ幅（製品エッジ）[mm]", 5.0, 600.0, 1.0, fmt="%.1f")
        num("fan_flank_deg", "フランク角（製品エッジ線に対して）[deg]", 1.0, 89.0, 0.5, fmt="%.1f")
        num("runner_len_mm", "ランナ長（製品エッジ → ゲート軸）[mm]", 1.0, 300.0, 1.0, fmt="%.1f")
        num("runner_end_d_mm", "丸端の径 φ（ゲート軸中心）[mm]", 1.0, 200.0, 1.0, fmt="%.1f")
        _apex = FanRunnerPlateConfig(
            runner_w_mm=v["runner_w_mm"], fan_flank_deg=v["fan_flank_deg"]
        ).apex_depth_mm
        st.caption(
            f"フランクの延長が軸で交わる深さ {_apex:.1f} mm（丸端の上端 "
            f"{v['runner_len_mm'] - v['runner_end_d_mm'] / 2:.1f} mm より深いこと）。"
            "図面ではフランクが丸端の円に交差する（接線ではない）。"
        )
        st.markdown("**肉厚（製品エッジからの深さで決まる）**")
        num("runner_edge_thk_mm", "エッジ帯の肉厚（額縁と同厚）[mm]", 0.05, 10.0, 0.05)
        num("runner_edge_flat_mm", "エッジ帯の長さ [mm]", 0.0, 50.0, 0.5, fmt="%.1f")
        num("runner_ramp_end_mm", "傾斜の終端深さ [mm]", 0.0, 300.0, 1.0, fmt="%.1f")
        num("runner_thk_mm", "ランナ肉厚（傾斜終端 → 丸端）[mm]", 0.1, 10.0, 0.05)
        num("gate_d_mm", "ゲート径 φ（ホットランナー、射出点）[mm]", 0.5, 50.0, 0.5, fmt="%.1f")
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
    with st.expander("メッシュ", expanded=False):
        v["cell_size_mm"] = st.slider(
            "メッシュ粗さ [mm/cell]", 0.5, 4.0, float(_D.cell_size_mm), 0.25, key="fg_cell_size_mm"
        )
        st.caption("既定 1.0 mm で約 85k セル。0.5 mm は 4 倍重い。")
        v["pad_mm"] = float(_D.pad_mm)
    return v


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
) -> tuple[InjectionProfile | None, float | None, str | None]:
    """The rate half of the injection block: a screw profile, or a flat Q.

    Returns ``(profile, Q_cm3s, error)``. Exactly one of the first two is set
    when ``error`` is None. On a rejected machine condition both are None and
    ``error`` carries the message -- the caller stops the run rather than
    this function calling ``st.stop()``, which sits above the version caption
    and would take it off the screen with the error still on it.

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
            help="ソディック等の成形機取説の射出率に対応。"
            "取説の値は機械の最大射出率なので、実際より速く充填されることがある。",
        )
        return None, float(Q), None

    screw_d = st.number_input(
        "スクリュー径 [mm]",
        min_value=5.0,
        max_value=200.0,
        value=DEF_SCREW_DIAMETER_MM,
        step=1.0,
        key="inj_screw_d",
        help="バレル内でスクリューが押しのける断面の直径。Q = πD²/4 × v。",
    )
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
    n_stages = int(
        st.number_input(
            "射出段数",
            min_value=1,
            max_value=MAX_INJECTION_STAGES,
            value=DEF_INJECTION_STAGES,
            step=1,
            key="inj_num_stages",
            help="1 なら計量位置から V/P まで一定速度。2 以上にすると"
            "各段の速度切替位置と速度の欄が出る。",
        )
    )

    if vp_pos >= meter_pos:
        return (
            None,
            None,
            (
                f"V/P切替位置 ({vp_pos:g} mm) が計量位置 ({meter_pos:g} mm) 以上です。"
                "スクリューは前進するので、V/P は計量位置より小さい必要があります。"
            ),
        )

    # Only the boundaries between stages are asked for; the last stage always
    # ends at V/P, which is what makes it the V/P transfer. Defaults are the
    # machine's own switch points at the default stage count, an even split of
    # the stroke otherwise.
    span = meter_pos - vp_pos
    switches: list[float] = []
    for i in range(n_stages - 1):
        default = meter_pos - span * (i + 1) / n_stages
        if (
            n_stages == DEF_INJECTION_STAGES
            and i < len(DEF_SWITCH_POSITIONS_MM)
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
    velocities = [
        float(
            st.number_input(
                f"第{i + 1}段 射出速度 [mm/s]",
                min_value=0.1,
                max_value=1000.0,
                value=DEF_SCREW_VELOCITY_MMS,
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
        )

    lines = [
        f"射出率（平均）: **{profile.mean_rate_cm3s:.1f} cm³/s**",
        f"射出時間 (V/P まで): **{profile.total_time_s:.3f} s**",
        f"理論射出量 (V/P 時点): **{profile.total_volume_cm3:.2f} cm³**",
    ]
    if profile.num_stages > 1:
        for i, (rate, t, vol) in enumerate(
            zip(profile.stage_rates_cm3s(), profile.stage_times_s(), profile.stage_volumes_mm3())
        ):
            lines.append(f"　第{i + 1}段: {rate:.1f} cm³/s / {t:.3f} s / {vol / 1000.0:.2f} cm³")
    st.caption("  \n".join(lines))
    return profile, None, None


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
            options=("machine", "direct"),
            index=0,
            key="inj_mode",
            format_func=lambda m: {
                "machine": "実機条件から計算（スクリュー径・位置・速度）",
                "direct": "射出率を直接入力",
            }[m],
            help="実機条件: Q = πD²/4 × v。成形機に実際に入れる数字から射出率と"
            "V/P までの射出時間を出す。直接入力: 取説の射出率をそのまま使う。",
        )
        inj_profile, inj_Q, inj_error = _injection_rate_inputs(inj_mode)

    with st.expander("壁面冷却モデル", expanded=False):
        wall_model = st.radio(
            "壁面冷却の表現",
            options=("none", "skin", "multilayer"),
            index=1,
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
                "極薄プレート (t<0.5mm) では層別を推奨。"
            ),
        )

        # default container (so downstream `solver = HeleShawSolver(...)` /
        # `MultilayerHeleShawSolver(...)` always has the kwargs it expects).
        # 既定モードは『スキン層』(index=1) — 既定 ON の二相ショートショットと併用でき、
        # 射出相で薄い額縁が痩せる順番まで出る（層別は二相と併用不可）。
        # 層別を選んだときの既定値 (極薄 t0.35〜0.50 向け):
        #   層数 N: 7 (壁勾配が急なので N=5 から増量)
        #   反復上限: 12 (収束が遅くなりがちなので上限緩め)
        skin_on = wall_model == "skin"
        c_skin = 0.0
        skin_max_iter = 5
        skin_tol = 1e-3
        skin_clock_mode = "constant_pressure"
        multilayer_on = wall_model == "multilayer"
        num_layers = 7
        layer_distribution = "wall_refined"
        multilayer_max_iter = 12
        multilayer_tol = 1e-3
        solid_fraction = 0.3

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
                10,
                5,
                help="τ ↔ h_core 結合の反復回数。3〜5で十分なケースが多い。",
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
            if inj_profile is not None and skin_clock_mode == "constant_pressure":
                # A screw profile prescribes a rate. Holding pressure instead
                # is the "machine could not keep up" scenario; the profile's
                # shape survives (every stage stretches by the same factor)
                # but the times are no longer the ones on the machine.
                st.caption(
                    "実機条件（スクリュー位置と速度）は速度制御そのものです。"
                    "圧力一定を選ぶと、抵抗増のぶん**プロファイル全体が同じ倍率で**"
                    "引き伸ばされ、V/P までの射出時間は設定値と一致しなくなります。"
                    "設定どおりの充填時間を見たいなら「速度制御」を選んでください。"
                )
        elif wall_model == "multilayer":
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

    with st.expander("射出圧縮成形 (ICM)", expanded=False):
        icm = st.checkbox("圧縮成形ON", value=True, key="icm_on")
        if icm:
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
                    "溶融プールを前進、体積保存）の二相。壁面冷却モデルは『なし』か"
                    "『スキン層』で実行（スキン層は射出相に乗る: 開いた薄板が射出中に"
                    "痩せてゲート部が先に埋まる順番を出す）。『層別』とは併用不可。"
                ),
            )
            if two_phase_on and wall_model == "multilayer":
                # 実行時の一過性警告だけだと rerun で消えて「ON にしたのに何も
                # 出ない」に見える。設定と同じ場所に常時出す。
                st.warning(
                    "壁面冷却モデルが『なし』または『スキン層』のときだけ実行される。"
                    "現在の設定（層別）では二相解析はスキップされる。"
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
        )
        if multilayer_on
        else ("none",)
    ),
    "compression_molding": (icm, comp_stroke, comp_frac),
    "two_phase_short_shot": (two_phase_on, shot_volume_cm3),
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

        # 二相ショートショット。HeleShawSolver 専用（等温、またはスキン層を
        # 射出相に乗せる）— 層別ソルバーには射出相の時計が無い。
        two_phase_result = None
        two_phase_skip_reason: str | None = None
        if two_phase_on:
            if multilayer_on:
                two_phase_skip_reason = "壁面冷却モデルが『層別』に設定されている（併用不可）"
                st.warning(
                    "二相ショートショット解析は壁面冷却モデル『なし』または『スキン層』専用です。"
                    "今回はスキップしました。"
                )
            else:
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
                }
                if two_phase_result is not None
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
        # Read the run's own record, not the sidebar: the pane also renders
        # from a cached result, and the sidebar may have moved since.
        _prof_rec = result.metadata.get("injection_profile")
        if _prof_rec is not None:
            _V_shot_cm3 = float(_prof_rec["total_volume_cm3"])
            _V_cav_cm3 = geom.volume_cm3()
            st.caption(
                f"射出条件: 平均 {_prof_rec['mean_rate_cm3s']:.1f} cm³/s、"
                f"V/P まで {_prof_rec['total_time_s']:.3f} s、"
                f"理論射出量 {_V_shot_cm3:.2f} cm³"
            )
            if _V_shot_cm3 < _V_cav_cm3:
                # Past V/P the volume-to-time map extrapolates at the last
                # stage's rate — the fill time is then "if the machine kept
                # going", not a fill this shot achieves. Say so rather than
                # letting the number pass as a prediction.
                st.warning(
                    f"理論射出量 {_V_shot_cm3:.2f} cm³ がキャビティ体積 "
                    f"{_V_cav_cm3:.2f} cm³ を下回っています。V/P 以降も最終段の"
                    "射出率で射出し続けた前提の時間軸になっているので、"
                    f"{_V_shot_cm3:.2f} cm³ を超えた分の充填時刻は外挿です。"
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
                        f"射出終了時のスキン最大 {md2.get('injection_skin_max_mm', 0.0):.3f} mm、"
                        f"封止 {md2.get('injection_sealed_cells', 0)} セル、"
                        f"封止で届かず {md2.get('injection_unfillable_cells', 0)} セル。"
                        "圧縮相は等温（プールは等圧ソースなので内部のスキンは前進に効かない）。"
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
                if md2.get("injection_extrapolated_past_vp"):
                    # The sidebar's own extrapolation check compares the
                    # stroke against the *final* cavity; neither the metered
                    # shot nor the open-gap cavity is that number, so this
                    # panel has to say it for itself (@claude on PR #89).
                    st.warning(
                        "この二相解析は V/P より先を外挿しています。理論射出量 "
                        f"{md2.get('injection_profile_volume_cm3', 0.0):.2f} cm³ に対し、"
                        f"計量 {md2['shot_volume_cm3']:.2f} cm³ ／ 開きキャビティ "
                        f"{md2.get('cavity_volume_open_cm3', float('nan')):.2f} cm³ が"
                        "それを超えているためで、射出時間は最終段の射出率で射出し続けた"
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
                st.caption(
                    f"層数 N={md.get('num_layers')}, 分布={md.get('layer_distribution')}, "
                    f"反復={md.get('multilayer_iterations')}, "
                    f"収束={md.get('multilayer_converged')}, "
                    f"T_fill_inflation={md.get('T_fill_inflation', 1.0):.3f}, "
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
