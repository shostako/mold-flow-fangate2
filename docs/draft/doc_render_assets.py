"""Re-solve a run's ``settings.json`` (repo venv) and render the pieces the
A4 "図面と解析結果" sheet needs at high resolution: thickness map, final fill
frame, two-phase short-shot map, plus ``info.json`` with the arrival-time
numbers the reading text quotes. Asserts τ_max / injection fraction against the
run's metadata so the re-solve is the same run. Adapted from fangate's
``docs/draft/doc_render_assets.py`` for the fan-runner plate (no tab, no
wings; the runner is the only non-product region).

    MPLBACKEND=Agg .venv/bin/python docs/draft/doc_render_assets.py <settings_dir> <asset_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import visualizer as V  # noqa: E402
from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry  # noqa: E402
from core.materials import MaterialDB  # noqa: E402
from core.solver import HeleShawSolver  # noqa: E402
from core.two_phase import solve_two_phase_short_shot  # noqa: E402

sdir = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
s = json.loads((sdir / "settings.json").read_text(encoding="utf-8"))
m = json.loads((sdir / "metadata.json").read_text(encoding="utf-8"))
m2 = json.loads((sdir / "two_phase_metadata.json").read_text(encoding="utf-8"))

cfg = FanRunnerPlateConfig(**s["geometry"]["config"])
geom = build_fan_runner_plate_geometry(cfg)
inj, wc, cm, tp = s["injection"], s["wall_cooling"], s["compression_molding"], s["two_phase_short_shot"]
assert wc["model"] in ("none", "skin") and cm["enabled"] and cm["mode"] == "stroke" and tp["enabled"]
skin_on = wc["model"] == "skin"
skin_kw = (
    dict(
        skin_growth_constant=wc["skin_growth_constant"],
        skin_max_iterations=wc["skin_max_iterations"],
        skin_convergence_tol=wc["skin_convergence_tol"],
        skin_clock_mode=wc["skin_clock_mode"],
    )
    if skin_on
    else {}
)
solver = HeleShawSolver(
    geometry=geom,
    material=MaterialDB().get(s["material"]),
    melt_temperature_K=inj["melt_temperature_C"] + 273.15,
    mold_temperature_K=inj["mold_temperature_C"] + 273.15,
    injection_velocity_mms=inj["injection_velocity_mms"],
    injection_volume_flow_cm3s=inj["injection_volume_flow_cm3s"],
    compression_molding=True,
    compression_factor=m["compression_factor"],
    compression_stroke_mm=cm["stroke_mm"],
    compression_fraction=cm["fraction"],
    skin_layer_enabled=skin_on,
    **skin_kw,
)
r = solver.solve(num_frames=s["output"]["num_frames"])
T = V.fill_time_max(r)
assert abs(r.metadata["tau_max"] - m["tau_max"]) < 1e-3 * max(1.0, abs(m["tau_max"]) * 1e-9), "metadata.json と不一致"
if skin_on:
    assert abs(T - m["T_fill_baseline_s"] * m["T_fill_inflation"]) < 1e-9, "T_fill が metadata.json と不一致"
r2 = solve_two_phase_short_shot(solver, tp["shot_volume_cm3"])
assert abs(r2.metadata["injection_fill_fraction"] - m2["injection_fill_fraction"]) < 1e-9, "two_phase_metadata と不一致"
assert r2.metadata["final_fill_fraction"] == m2["final_fill_fraction"]

x0, y0 = geom.display_origin_mm()
dx = geom.cell_size_mm
extent = [-x0, geom.nx * dx - x0, -y0, geom.ny * dx - y0]
FIG = (6.4, 4.4)

fig, ax = plt.subplots(figsize=FIG, dpi=220)
im = ax.imshow(np.where(geom.mask, geom.thickness_mm, np.nan), origin="lower", extent=extent, cmap=V.THICKNESS_CMAP)
ax.contour(geom.compression_mask.astype(float), levels=[0.5], colors="red", linewidths=0.8, extent=extent, origin="lower")
V.draw_gate_markers(ax, geom)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.set_title("thickness map [mm]")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="h [mm]")
fig.tight_layout()
fig.savefig(out / "thickness.png")
plt.close(fig)

ext = V._base_extent(r)
norm = mcolors.Normalize(0, T)
cmap = s["output"]["fill_cmap"]
rgba = V._fill_field_rgb(r, cmap)
fig, ax = plt.subplots(figsize=FIG, dpi=220)
ov = V._draw_fill_state(ax, r, rgba, np.zeros_like(geom.mask), smooth=True, isochrone_levels=s["output"]["isochrone_levels"])
V._draw_gate_markers(ax, r)
ax.set_xlim(ext[0], ext[1])
ax.set_ylim(ext[2], ext[3])
ax.set_aspect("equal")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
filled = r.fill_time_s <= T
ov.set_array(V._unfilled_overlay(r, filled))
ax.set_title(V._fill_title(r, T, filled[geom.mask].sum() / geom.mask.sum()))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
cb.set_label("fill time [s]")
fig.tight_layout()
fig.savefig(out / "final_frame.png")
plt.close(fig)

V.render_two_phase_map(r2, out / "two_phase.png")

# ---- numbers for the reading text ----
ft = r.fill_time_s
pm = geom.product_mask
rows = np.where(pm.any(axis=1))[0]
cols = np.where(pm.any(axis=0))[0]
xs = (np.arange(cols[0], cols[-1] + 1) + 0.5) * dx - x0
ys = (np.arange(rows[0], rows[-1] + 1) + 0.5) * dx - y0


def stats(row):
    i = int(np.nanargmin(row))
    j = int(np.nanargmax(row))
    c = len(row) // 2
    return dict(min_t=float(row[i]), min_x=float(xs[i]), max_t=float(row[j]), max_x=float(xs[j]), center=float(row[c]))


bottom = ft[rows[0], cols[0] : cols[-1] + 1]  # product edge row (runner side)
top = ft[rows[-1], cols[0] : cols[-1] + 1]
left = ft[rows[0] : rows[-1] + 1, cols[0]]
right = ft[rows[0] : rows[-1] + 1, cols[-1]]
side = 0.5 * (left + right)
i_side_min = int(np.nanargmin(side))
frame = pm & (geom.thickness_mm <= cfg.frame_thk_mm + 1e-9)
inner = pm & ~frame
runner = geom.mask & ~pm
# the runner: arrival at its far corners (x = ±runner_w/2 on the edge line)
runner_rows = np.where(runner.any(axis=1))[0]
runner_top = ft[runner_rows[-1]]  # runner row just below the product edge
rc = np.where(runner[runner_rows[-1]])[0]
runner_corner_t = float(np.nanmax(ft[runner_rows[-1], rc]))
runner_corner_x = float((rc[np.nanargmax(ft[runner_rows[-1], rc])] + 0.5) * dx - x0)
# inner body: first arrival (the step from the rim into the body) and the last cell
inner_rows = np.where(inner.any(axis=1))[0]
inner_first_t = float(np.nanmin(ft[inner]))
ii = np.unravel_index(int(np.nanargmin(np.where(inner, ft, np.inf))), ft.shape)
inner_first_xy = [float((ii[1] + 0.5) * dx - x0), float((ii[0] + 0.5) * dx - y0)]
inner_last_t = float(np.nanmax(ft[inner]))
jj = np.unravel_index(int(np.nanargmax(np.where(inner, ft, -np.inf))), ft.shape)
inner_last_xy = [float((jj[1] + 0.5) * dx - x0), float((jj[0] + 0.5) * dx - y0)]
# rim: the bottom strip (runner side) vs the sides vs the top strip
XX = (np.arange(geom.nx)[None, :] + 0.5) * dx - x0
YY = (np.arange(geom.ny)[:, None] + 0.5) * dx - y0
XXb = np.broadcast_to(XX, ft.shape)
YYb = np.broadcast_to(YY, ft.shape)
rim_bottom = frame & (YYb < cfg.frame_w_mm)
rim_top = frame & (YYb > cfg.plate_h_mm - cfg.frame_w_mm)
rim_sides = frame & ~rim_bottom & ~rim_top
# two-phase: which region did compression advance
adv = r2.final_mask & ~r2.injection_mask
adv_frame = int((adv & frame).sum())
adv_inner = int((adv & inner).sum())
adv_runner = int((adv & runner).sum())
adv_rows = np.where(adv.any(axis=1))[0]
ai = adv & inner
ai_absx_min = float(np.abs(XXb[ai]).min()) if ai.any() else None
af = adv & frame


def bands(mask_rows):
    ys_ = np.where(mask_rows)[0]
    if not ys_.size:
        return []
    cuts = np.where(np.diff(ys_) > 1)[0]
    starts = np.r_[ys_[0], ys_[cuts + 1]]
    ends = np.r_[ys_[cuts], ys_[-1]]
    return [[float(YY[a_, 0] - dx / 2), float(YY[b_, 0] + dx / 2)] for a_, b_ in zip(starts, ends, strict=True)]


info = dict(
    T=T,
    nx=geom.nx,
    ny=geom.ny,
    cell=dx,
    volume_cm3=geom.volume_cm3(),
    product_cm3=float(geom.thickness_mm[pm].sum()) * dx**2 / 1000.0,
    bottom=stats(bottom),
    top=stats(top),
    side_min_t=float(side[i_side_min]),
    side_min_y=float(ys[i_side_min]),
    side_top_t=float(side[-1]),
    side_bottom_t=float(side[0]),
    frame_mean_t=float(np.nanmean(ft[frame])),
    inner_mean_t=float(np.nanmean(ft[inner])),
    rim_bottom_mean_t=float(np.nanmean(ft[rim_bottom])),
    rim_sides_mean_t=float(np.nanmean(ft[rim_sides])),
    rim_top_mean_t=float(np.nanmean(ft[rim_top])),
    runner_corner_t=runner_corner_t,
    runner_corner_x=runner_corner_x,
    runner_max_t=float(np.nanmax(ft[runner])),
    inner_first_t=inner_first_t,
    inner_first_xy=inner_first_xy,
    inner_last_t=inner_last_t,
    inner_last_xy=inner_last_xy,
    two_phase=dict(
        injection_fraction=float(r2.metadata["injection_fill_fraction"]),
        final_fraction=float(r2.metadata["final_fill_fraction"]),
        injection_time_s=float(r2.injection_time_s),
        shot_cm3=float(r2.shot_volume_cm3),
        adv_cells=int(adv.sum()),
        adv_frame=adv_frame,
        adv_inner=adv_inner,
        adv_runner=adv_runner,
        adv_rim_bottom=int((adv & rim_bottom).sum()),
        adv_rim_sides=int((adv & rim_sides).sum()),
        adv_rim_top=int((adv & rim_top).sum()),
        adv_y_range=[float(YY[adv_rows[0], 0] - dx / 2), float(YY[adv_rows[-1], 0] + dx / 2)] if adv_rows.size else None,
        adv_inner_absx_min=ai_absx_min,
        adv_inner_y_bands=bands(ai.any(axis=1)),
        adv_frame_y_bands=bands(af.any(axis=1)),
        adv_inner_x_bands=[[float(XX[0, a] - dx / 2), float(XX[0, b] + dx / 2)] for a, b in []],
        inner_x_half=cfg.plate_w_mm / 2 - cfg.frame_w_mm,
    ),
    frame_thk=cfg.frame_thk_mm,
    inner_thk=cfg.inner_thk_mm,
    plate=[cfg.plate_w_mm, cfg.plate_h_mm],
    frame_w=cfg.frame_w_mm,
    apex_depth=cfg.apex_depth_mm,
)
(out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False))
