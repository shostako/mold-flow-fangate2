"""充填（energy）の切替の時点の温度を保圧へ渡し、渡さない場合と縮みを比べる。

使い方: python handoff.py --cell 4 --out DIR [--vp 0.99]
保圧は 3 通り: series（従来）、layers（温度は到達時刻から作る）、handoff（充填の温度を渡す）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry  # noqa: E402
from core.materials import MaterialDB  # noqa: E402
from core.multilayer_solver import MultilayerHeleShawSolver  # noqa: E402
from core.packing_solve import solve_packing  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vp", type=float, default=0.99)
ap.add_argument("--hold-mpa", type=float, default=90.0)
ap.add_argument("--hold-s", type=float, default=20.0)
ap.add_argument("--cases", default="series,layers,handoff")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
Q = 39.27
T_MELT, T_MOLD = 260 + 273.15, 60 + 273.15
cfg = FanRunnerPlateConfig(cell_size_mm=a.cell)
geom = build_fan_runner_plate_geometry(cfg)
mat = MaterialDB()["PMMA"]
t0 = time.time()
fill = MultilayerHeleShawSolver(
    geometry=geom,
    material=mat,
    melt_temperature_K=T_MELT,
    mold_temperature_K=T_MOLD,
    injection_velocity_mms=130.0,
    injection_volume_flow_cm3s=Q,
    num_layers=7,
    thermal_coupling=True,
    solidification_temperature_fraction=0.3,
    shear_heating_enabled=True,
    fill_method="march",
    thermal_model="energy",
    thermal_snapshot_fractions=(a.vp,),
).solve(num_frames=8)
t_fill = time.time() - t0
t_arr = np.asarray(fill.fill_time_s, dtype=float)
snap_t, snap = fill.thermal_snapshots[a.vp]
print(f"fill {t_fill:.0f}s, snapshot at {snap_t:.3f}s", flush=True)
region = geom.mask if geom.product_mask is None else geom.mask & geom.product_mask
thk = geom.thickness_mm
rows = {}
for case in a.cases.split(","):
    kw: dict = {}
    if case != "series":
        kw["thermal"] = "layers"
    if case == "handoff":
        kw["initial_theta"] = snap
    t1 = time.time()
    r = solve_packing(
        geom.mask,
        thk,
        t_arr,
        geom.gates,
        a.cell,
        mat,
        T_MELT,
        T_MOLD,
        a.hold_mpa * 1e6,
        a.hold_s,
        solid_fraction=0.3,
        switch_fill_fraction=a.vp,
        max_inflow_m3_s=Q * 1e-6,
        progress=lambda t, end: print(
            f"  {case} t={t:.4f}/{end:.2f} {time.time() - t1:.0f}s", flush=True
        ),
        **kw,
    )
    row = {"elapsed_s": time.time() - t1, "switch_time_s": float(r.switch_time_s)}
    row["unfilled_volume_percent"] = 100.0 * float(r.unfilled_volume_fraction)
    row["unconverged_steps"] = int(r.unconverged_steps)
    for name, sel in (("body_t4", region & (thk > 3.5)), ("frame_t1", region & (thk < 1.5))):
        s = r.shrinkage[sel]
        vol = thk[sel]
        ok = np.isfinite(s)
        row[f"{name}_shrink_percent"] = 100.0 * float((s[ok] * vol[ok]).sum() / vol[ok].sum())
        row[f"{name}_never_full_percent"] = 100.0 * float(vol[~ok].sum() / vol.sum())
    rows[case] = row
    np.savez_compressed(
        out / f"hold_{case}_{a.cell:g}.npz", shrinkage=r.shrinkage, mask=geom.mask, thk=thk
    )
    print(case, json.dumps(row, ensure_ascii=False), flush=True)
(out / f"handoff_{a.cell:g}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
