"""Gate 2（五角形）の図面（A3 横、平面 1:1、断面 3:1）。2026-10-06 にユーザーと詰めた形。

寸法は ``GATE2_DEFAULTS`` の既定値から取り、接点は builder の ``pentagon_tangent_mm`` を使う::

    .venv/bin/python docs/draft/gate2_drawing.py

日本語は Windows の BIZ UDGothic（``/mnt/c/Windows/Fonts``）で埋め込む。CI では走らせない。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from core.fan_runner import GATE2_DEFAULTS, FanRunnerPlateConfig  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).with_suffix(".pdf")

# ---- parameters (mm), from the builder's Gate 2 defaults ----
CFG = FanRunnerPlateConfig(**GATE2_DEFAULTS)
PLATE_W = CFG.plate_w_mm
FRAME_W = CFG.frame_w_mm
W = CFG.runner_w_mm  # runner width on the product edge
H_SIDE = CFG.side_len_mm  # side length below the rim
L = CFG.runner_len_mm  # product edge -> sprue axis
R = CFG.runner_end_d_mm / 2  # round end
GATE_D = CFG.gate_d_mm
T_EDGE = CFG.runner_edge_thk_mm
LAND = CFG.runner_edge_flat_mm  # edge band width
T_BODY = CFG.runner_thk_mm  # t_o: thickness outside the circle
T_CIRCLE = CFG.runner_end_thk_mm  # circle depth
D_START = CFG.ramp_end_depth_mm  # ramp start (deep end) = top of R12
FLANK = FanRunnerPlateConfig().fan_flank_deg  # Gate 1

RED = "#d0021b"
INK = "#222222"
GRAY = "#888888"
MM = 1 / 25.4
FIG_W, FIG_H = 420.0, 297.0


TX, _TD = CFG.pentagon_tangent_mm  # right tangent point (x, depth)
TY = -_TD
TH = math.atan2(TY + L, TX)  # its angle about the axis centre
SLANT_DEG = CFG.pentagon_slant_deg


def pentagon_vertices(n_arc: int = 60) -> np.ndarray:
    pts = [(-W / 2, 0.0), (W / 2, 0.0), (W / 2, -H_SIDE)]
    th_l = math.pi - TH  # mirror of TH about the vertical axis
    if th_l > 0:
        th_l -= 2 * math.pi
    for th in np.linspace(TH, th_l, n_arc):
        pts.append((R * math.cos(th), -L + R * math.sin(th)))
    pts.append((-W / 2, -H_SIDE))
    return np.array(pts)


def triangle_flank_hit() -> tuple[float, float]:
    """Where the current 14° flank meets the R12 circle (right side)."""
    tan = math.tan(math.radians(FLANK))
    best = None
    for x in np.linspace(0, R, 200001):
        y = -(W / 2 - x) * tan
        if abs(math.hypot(x, y + L) - R) < 2e-4:
            best = (x, y)
            break
    return best


def thickness(xx: np.ndarray, yy: np.ndarray, ramp_on: bool) -> np.ndarray:
    d = -yy
    if ramp_on:
        t = np.clip((d - LAND) / (D_START - LAND), 0.0, 1.0)
        thk = T_EDGE + (T_BODY - T_EDGE) * t
    else:
        thk = np.where(d > LAND, T_BODY, T_EDGE)
    in_c = xx**2 + (yy + L) ** 2 <= R**2
    return np.where(in_c, T_CIRCLE, thk)


def add_axes_mm(fig, left, bottom, xr, yr, scale):
    w = (xr[1] - xr[0]) * scale
    h = (yr[1] - yr[0]) * scale
    ax = fig.add_axes((left / FIG_W, bottom / FIG_H, w / FIG_W, h / FIG_H))
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def dim(ax, p1, p2, off, text, color=INK, fs=7, tside=1, ext=True, tpos=0.5, rot=None):
    """Linear dimension between p1 and p2, offset perpendicular by `off`."""
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    nx, ny = -dy / ln, dx / ln
    a = (x1 + nx * off, y1 + ny * off)
    b = (x2 + nx * off, y2 + ny * off)
    if ext:
        s = 1 if off >= 0 else -1
        for (px, py), (qx, qy) in ((p1, a), (p2, b)):
            ax.plot(
                [px + nx * s * 0.8, qx + nx * s * 1.5],
                [py + ny * s * 0.8, qy + ny * s * 1.5],
                color=color,
                lw=0.35,
            )
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(
            arrowstyle="<|-|>", lw=0.5, color=color, mutation_scale=5, shrinkA=0, shrinkB=0
        ),
    )
    mx = a[0] + (b[0] - a[0]) * tpos
    my = a[1] + (b[1] - a[1]) * tpos
    ang = math.degrees(math.atan2(dy, dx)) if rot is None else rot
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    ax.text(
        mx + nx * 1.2 * tside,
        my + ny * 1.2 * tside,
        text,
        color=color,
        fontsize=fs,
        rotation=ang,
        ha="center",
        va="bottom" if tside > 0 else "top",
        rotation_mode="anchor",
    )


def leader(ax, xy, xytext, text, color=INK, fs=6.5, ha="left"):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fs,
        color=color,
        ha=ha,
        va="center",
        arrowprops=dict(
            arrowstyle="-|>", lw=0.4, color=color, mutation_scale=4, shrinkA=1, shrinkB=0
        ),
    )


CMAP = plt.get_cmap("Blues")
VMIN, VMAX = 0.0, 5.0


def tcolor(t):
    return CMAP((t - VMIN) / (VMAX - VMIN))


def plan_view(fig):
    xr, yr = (-200.0, 185.0), (-68.0, 42.0)
    ax = add_axes_mm(fig, 12, 160, xr, yr, 1.0)

    # thickness raster: product (to the break line) + pentagon
    res = 0.1
    xs = np.arange(xr[0], xr[1], res) + res / 2
    ys = np.arange(yr[0], 36.0, res) + res / 2
    xx, yy = np.meshgrid(xs, ys)
    pent = pentagon_vertices()
    in_pent = MPath(pent).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    in_prod = (yy > 0) & (np.abs(xx) <= PLATE_W / 2)
    in_inner = in_prod & (yy > FRAME_W) & (np.abs(xx) < PLATE_W / 2 - FRAME_W)
    thk = np.full(xx.shape, np.nan)
    thk[in_pent] = thickness(xx, yy, True)[in_pent]
    thk[in_prod] = T_EDGE
    thk[in_inner] = 4.0
    ax.imshow(
        thk,
        origin="lower",
        extent=(xr[0], xr[1], yr[0], 36.0),
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        rasterized=True,
        zorder=1,
    )

    # product outline with break line at y≈36
    hw = PLATE_W / 2
    zig_x = np.linspace(-hw, hw, 41)
    zig_y = 36.0 + np.where(np.arange(41) % 2 == 0, -0.9, 0.9)
    ax.plot([-hw, -hw], [0, 36], color=INK, lw=0.6, zorder=3)
    ax.plot([hw, hw], [0, 36], color=INK, lw=0.6, zorder=3)
    ax.plot(zig_x, zig_y, color=INK, lw=0.4, zorder=3)
    ax.plot([-hw + FRAME_W, -hw + FRAME_W], [FRAME_W, 36], color=INK, lw=0.35, zorder=3)
    ax.plot([hw - FRAME_W, hw - FRAME_W], [FRAME_W, 36], color=INK, lw=0.35, zorder=3)
    ax.plot([-hw + FRAME_W, hw - FRAME_W], [FRAME_W, FRAME_W], color=INK, lw=0.35, zorder=3)
    ax.plot([-hw, hw], [0, 0], color=INK, lw=0.6, zorder=3)
    ax.text(
        0,
        28,
        "製品 内側 t4.0（圧縮部）",
        fontsize=7,
        ha="center",
        va="center",
        color="white",
        zorder=4,
    )
    ax.text(
        -75,
        10,
        "額縁 t1.0（ゲートブロックの一部・変えない）",
        fontsize=6.5,
        ha="center",
        va="center",
        color=INK,
        zorder=4,
    )

    # pentagon outline
    ax.add_patch(Polygon(pent, closed=True, fill=False, ec=RED, lw=0.9, zorder=5))

    # current triangle (gray dashed) for reference
    hx, hy = triangle_flank_hit()
    for s in (-1, 1):
        ax.plot([s * W / 2, s * hx], [0, hy], color=GRAY, lw=0.6, ls=(0, (4, 2)), zorder=4)
    ax.text(
        -100,
        -10.3,
        "Gate 1 の逆三角形（14°）",
        fontsize=6,
        color=GRAY,
        rotation=-14,
        ha="center",
        va="center",
        rotation_mode="anchor",
        zorder=6,
    )

    # R12 circle, gate, centre lines
    ax.add_patch(Circle((0, -L), R, fill=False, ec=INK, lw=0.5, zorder=5))
    ax.add_patch(Circle((0, -L), GATE_D / 2, fill=False, ec=INK, lw=0.5, zorder=6))
    ax.plot([0, 0], [-L - R - 6, 40], color=INK, lw=0.3, ls=(0, (8, 2, 1, 2)), zorder=4)
    ax.plot([-R - 5, R + 5], [-L, -L], color=INK, lw=0.3, ls=(0, (8, 2, 1, 2)), zorder=4)

    # land line and ramp-start line
    ax.plot([-W / 2, W / 2], [-LAND, -LAND], color=INK, lw=0.3, zorder=5)
    ax.plot([-W / 2, W / 2], [-D_START, -D_START], color=RED, lw=0.6, ls=(0, (5, 2)), zorder=5)

    # tangent points
    for s in (-1, 1):
        ax.plot(s * TX, TY, marker="o", ms=1.6, color=RED, zorder=7)

    # balancer (optional, default OFF)
    bal = np.array([(-50.0, 0.0), (50.0, 0.0), (0.0, -15.0)])
    ax.add_patch(
        Polygon(bal, closed=True, fill=False, ec="#555555", lw=0.45, ls=(0, (1.5, 1.5)), zorder=5)
    )

    # section line A-A on the axis
    for y0, va in ((40.0, "bottom"), (-L - R - 7.0, "top")):
        ax.annotate(
            "",
            xy=(-6, y0),
            xytext=(0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6),
        )
        ax.text(-8, y0, "A", fontsize=8, ha="right", va="center", color=INK, zorder=7)

    # ---- dimensions ----
    dim(ax, (-W / 2, -H_SIDE), (W / 2, -H_SIDE), -36.0, "300（ランナ幅・既存）", tside=-1)
    dim(ax, (-W / 2, 0.0), (-W / 2, -H_SIDE), -14.0, f"入力 {H_SIDE:g}", color=RED, fs=7, tside=-1)
    dim(
        ax,
        (-W / 2, FRAME_W),
        (-W / 2, -H_SIDE),
        -30.0,
        f"h = {FRAME_W + H_SIDE:g}（額縁込み）",
        color=RED,
        fs=7,
        tside=-1,
    )
    ax.text(
        -198,
        -37,
        "側辺の長さ（新）：入力は額縁の下端（製品エッジ）から測る。\nh = 20 + 入力。入力 10 なら h = 30",
        fontsize=6,
        color=RED,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    dim(ax, (28.0, 0.0), (28.0, -L), 0.0, "30（軸の深さ）", ext=False, tside=-1, fs=6.5)
    ax.plot([R + 1, 30], [-L, -L], color=INK, lw=0.3)
    dim(
        ax,
        (-60.0, 0.0),
        (-60.0, -D_START),
        0.0,
        f"d_s = {D_START:g}",
        ext=False,
        color=RED,
        tside=1,
        fs=7,
    )
    dim(ax, (hw, 0.0), (hw, FRAME_W), -10.0, "20", tside=-1, fs=6.5)

    # leaders
    leader(
        ax,
        (R * math.cos(math.radians(-40)), -L + R * math.sin(math.radians(-40))),
        (48, -46),
        "R12（スプルー軸の受け・既存）",
    )
    leader(
        ax,
        (GATE_D / 2 * 0.7, -L + GATE_D / 2 * 0.7),
        (-40, -25),
        "ゲート φ3（射出点）",
        fs=6,
        ha="right",
    )
    leader(
        ax,
        (TX, TY),
        (30, -55),
        f"接点 (±{TX:.1f}, −{-TY:.1f})：斜辺は R12 の円に接する。斜辺の傾き {SLANT_DEG:.1f}° は h と R12 から決まる",
        color=RED,
        fs=6,
    )
    leader(
        ax,
        (-40, -D_START),
        (-120, -48),
        "傾斜の開始位置（下限）d_s。既定は R12 の上端（30 − 12 = 18）",
        color=RED,
        fs=6,
    )
    leader(ax, (75, -LAND / 2), (85, -4.5), "エッジ帯 t1.0 × 幅 1（変数・幅 0 で帯なし）", fs=6)
    leader(ax, (35, -4.5), (45, -12), "肉盗み（任意・既定 OFF）", fs=6)

    # zone labels
    ax.text(122, -11.0, "傾斜 1.0 → t_o", fontsize=6, ha="center", va="center", color=INK)
    ax.text(118, -22.0, "平坦 t_o = 2.5", fontsize=6, ha="center", va="center", color="white")
    ax.text(
        0,
        -36.5,
        "円\nt2.5",
        fontsize=5.5,
        ha="center",
        va="center",
        color="white",
        linespacing=0.95,
        zorder=7,
    )

    ax.text(
        xr[0] + 15,
        yr[1] - 1,
        "平面図（1:1）  色は肉厚。赤 = 新しく加える形と変数",
        fontsize=8,
        ha="left",
        va="top",
        color=INK,
    )
    return ax


def section(fig, left, bottom, ramp_on: bool, title: str):
    xr, yr = (-8.0, 47.0), (-5.5, 7.5)
    ax = add_axes_mm(fig, left, bottom, xr, yr, 3.0)
    d = np.linspace(-8.0, L + R, 2001)
    xx = np.zeros_like(d)
    yy = -d
    t = thickness(xx, yy, ramp_on)
    t = np.where(d < 0, T_EDGE, t)
    poly = np.r_[np.c_[d, t], np.c_[d[::-1], np.zeros_like(d)]]
    ax.add_patch(Polygon(poly, closed=True, fc=tcolor(2.0), ec="none", zorder=1))
    ax.plot(d, t, color=INK, lw=0.7, zorder=3)
    ax.plot([-8, L + R], [0, 0], color=INK, lw=0.7, zorder=3)
    ax.plot([L + R, L + R], [0, t[-1]], color=INK, lw=0.7, zorder=3)
    # product edge
    ax.plot([0, 0], [-1.0, 3.6], color=INK, lw=0.35, ls=(0, (4, 2)), zorder=2)
    ax.text(0, 3.9, "製品エッジ", fontsize=6, ha="center", va="bottom")
    ax.text(-4.0, 1.3, "額縁 t1.0", fontsize=5.5, ha="center", va="bottom")
    # circle range
    ax.plot([L - R, L - R], [-0.3, -2.8], color=INK, lw=0.3)
    ax.plot([L + R, L + R], [-0.3, -2.8], color=INK, lw=0.3)
    ax.annotate(
        "",
        xy=(L + R, -2.4),
        xytext=(L - R, -2.4),
        arrowprops=dict(
            arrowstyle="<|-|>", lw=0.45, color=INK, mutation_scale=4, shrinkA=0, shrinkB=0
        ),
    )
    ax.text(L, -2.7, "R12 の円の範囲（中は「円の深さ」）", fontsize=5.8, ha="center", va="top")
    # gate
    for gx in (L - GATE_D / 2, L + GATE_D / 2):
        ax.plot([gx, gx], [T_CIRCLE, 5.5], color=INK, lw=0.5, zorder=3)
    ax.text(L + 2.5, 5.6, "ゲート φ3（固定側から）", fontsize=5.8, ha="left", va="center")
    # thickness callouts
    ax.text(
        L + R - 3.5,
        T_CIRCLE + 0.25,
        f"円の深さ {T_CIRCLE:g}（新）",
        fontsize=5.8,
        color=RED,
        ha="right",
        va="bottom",
    )
    ax.text(
        L - R - 0.6,
        T_BODY + 0.12,
        f"{'傾斜の厚い側 = ' if ramp_on else ''}円の外の肉厚 t_o = {T_BODY:g}（新）",
        fontsize=5.8,
        color=RED,
        ha="right",
        va="bottom",
    )
    if ramp_on:
        # example: circle deeper
        ex = 3.5
        ax.plot(
            [L - R, L - R, L + R, L + R],
            [T_BODY, ex, ex, T_CIRCLE],
            color=RED,
            lw=0.5,
            ls=(0, (1.5, 1.2)),
            zorder=4,
        )
        ax.text(
            L - R - 0.5,
            ex + 0.35,
            "例：円の深さを 3.5 にしたとき",
            fontsize=5.5,
            color=RED,
            ha="right",
            va="bottom",
        )
        ax.plot([D_START, D_START], [T_BODY, -1.0], color=RED, lw=0.35, zorder=2)
        ax.plot([LAND, LAND], [T_EDGE, -1.0], color=INK, lw=0.3, zorder=2)
        ax.annotate(
            "",
            xy=(D_START, -0.8),
            xytext=(LAND, -0.8),
            arrowprops=dict(
                arrowstyle="<|-|>", lw=0.45, color=RED, mutation_scale=4, shrinkA=0, shrinkB=0
            ),
        )
        ax.text(
            (LAND + D_START) / 2,
            -1.0,
            "傾斜 1.0 → t_o（d = 1 〜 d_s = 18）",
            fontsize=5.8,
            color=RED,
            ha="center",
            va="top",
        )
    else:
        ax.plot([LAND, LAND], [T_EDGE, -1.0], color=INK, lw=0.3, zorder=2)
        ax.text(
            LAND + 0.4,
            -0.8,
            "d = 1 で段差（エッジ帯 t1.0・幅 1 の先は t_o）",
            fontsize=5.8,
            color=RED,
            ha="left",
            va="top",
        )
        ax.plot(
            [LAND, L - R, L - R],
            [T_EDGE, T_EDGE, T_CIRCLE],
            color=RED,
            lw=0.5,
            ls=(0, (1.5, 1.2)),
            zorder=4,
        )
        ax.text(
            9.5,
            T_EDGE + 0.15,
            "例：t_o = 1.0 にしたとき（円だけが深い）",
            fontsize=5.3,
            color=RED,
            ha="center",
            va="bottom",
            zorder=5,
        )
    ax.text(xr[0], yr[1], title, fontsize=7.5, ha="left", va="top")
    ax.text(
        xr[1],
        yr[0] + 0.2,
        "製品側 ← 深さ d → スプルー側",
        fontsize=5.5,
        ha="right",
        va="bottom",
        color=GRAY,
    )
    return ax


def table(fig):
    ax = fig.add_axes((25 / FIG_W, 8 / FIG_H, 200 / FIG_W, 82 / FIG_H))
    ax.axis("off")
    rows = [
        ("項目", "既定値", "区分"),
        ("ゲート形状", "Gate 1（逆三角形）／Gate 2（五角形）", "新：選択肢"),
        ("ランナ幅（製品エッジ）", "300", "既存"),
        (
            "側辺の長さ（額縁の下端から）",
            f"{H_SIDE:g}（額縁込み h = {FRAME_W + H_SIDE:g}）",
            "新：五角形だけ",
        ),
        ("軸の深さ（製品エッジ → スプルー軸）", f"{L:g}", "既存"),
        ("丸端 R（スプルー軸の受け）", f"R{R:g}", "既存"),
        ("円の深さ（円の中の肉厚）", f"{T_CIRCLE:g}", "新"),
        ("円の外の肉厚 t_o（傾斜ありは厚い側）", f"{T_BODY:g}", "新：今のランナ肉厚を充てる"),
        ("傾斜", "付ける／付けない", "新"),
        ("傾斜の開始位置（下限）d_s", "18（R12 の上端）", "新（現行の値は 20）"),
        ("エッジ帯 肉厚 × 幅（幅 0 で帯なし）", f"{T_EDGE:g} × {LAND:g}", "変数：両方の形"),
        ("肉盗み（▽）", "OFF", "既存：両方の形で使う"),
    ]
    col_x = (0.0, 0.50, 0.76)
    n = len(rows)
    for i, row in enumerate(rows):
        y = 1.0 - (i + 0.5) / n
        for j, cell in enumerate(row):
            color = RED if (i > 0 and row[2].startswith(("新", "変数"))) else INK
            ax.text(
                col_x[j],
                y,
                cell,
                fontsize=6.8 if i else 7,
                color=color,
                ha="left",
                va="center",
                fontweight="bold" if i == 0 else "normal",
                transform=ax.transAxes,
            )
        ax.plot([0, 1], [1.0 - (i + 1) / n] * 2, color="#bbbbbb", lw=0.3, transform=ax.transAxes)
    ax.text(
        0,
        1.04,
        "変数（赤 = 新しく足すもの）",
        fontsize=7.5,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )


def notes(fig):
    ax = fig.add_axes((240 / FIG_W, 8 / FIG_H, 165 / FIG_W, 82 / FIG_H))
    ax.axis("off")
    lines = [
        "決めたこと（2026-10-06）",
        "1. 側辺の長さは額縁の下端（製品エッジ）から測る。入力 28 なら額縁込みの h は 48。",
        "    入力 10 なら h = 30。",
        "2. 斜辺は側辺の下端から R12 の円の下側に接する。底は R12 の円弧でつながる。",
        "3. 肉厚は深さ d だけで決まる。円の中は常に「円の深さ」。",
        "4. 傾斜を付けるときは、d = 1 の t1.0 から d_s で t_o まで上げる。",
        "    d_s より深いところは、円の外も t_o の平坦。",
        "5. 傾斜を付けないときは、エッジ帯（t1.0・幅 1）の先を t_o にする。",
        "    t_o = 1.0 にすれば、円だけが深い形になる。",
        "6. Gate 1（逆三角形）は値も結果も今のまま。エッジ帯・傾斜の有無・円の深さ・t_o・肉盗みは両方の形で使う。",
        "7. 額縁（下辺の t1.0・幅 20）は製品の一部として残し、形は変えない。",
    ]
    for i, s in enumerate(lines):
        ax.text(
            0,
            1.0 - i * 0.089,
            s,
            fontsize=7.2 if i == 0 else 6.6,
            ha="left",
            va="top",
            color=INK,
            transform=ax.transAxes,
        )


def main() -> None:
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    fig.text(
        25 / FIG_W,
        288 / FIG_H,
        "12.3 インチ プレート  ゲートブロック Gate 2（五角形）",
        fontsize=11,
        ha="left",
        va="top",
    )
    fig.text(395 / FIG_W, 288 / FIG_H, "2026-10-06  単位 mm", fontsize=8, ha="right", va="top")
    plan_view(fig)
    section(fig, 25, 100, True, "断面 A-A（3:1）  傾斜を付ける（既定）")
    section(fig, 225, 100, False, "断面 A-A（3:1）  傾斜を付けない")
    table(fig)
    notes(fig)
    fig.savefig(OUT)
    print(OUT)
    print(f"tangent = ({TX:.3f}, {TY:.3f}), slant = {SLANT_DEG:.2f} deg")
    print(f"triangle flank hit = {triangle_flank_hit()}")


if __name__ == "__main__":
    main()
