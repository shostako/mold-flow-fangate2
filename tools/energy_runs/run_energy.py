"""energy の充填を 12.3 型プレート Gate 1 既定で回し、状態を npz に残す。
使い方: python run_energy.py --cell 4 --out DIR [--model energy|arrival]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry
from core.materials import MaterialDB
from core.multilayer_solver import MultilayerHeleShawSolver

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="energy")
ap.add_argument("--q", type=float, default=39.27)
ap.add_argument("--v", type=float, default=130.0)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"{a.model}_{a.cell:g}"
cfg = FanRunnerPlateConfig(cell_size_mm=a.cell)
geom = build_fan_runner_plate_geometry(cfg)
mat = MaterialDB()["PMMA"]
s = MultilayerHeleShawSolver(
    geometry=geom,
    material=mat,
    melt_temperature_K=260 + 273.15,
    mold_temperature_K=60 + 273.15,
    injection_velocity_mms=a.v,
    injection_volume_flow_cm3s=a.q,
    num_layers=7,
    layer_distribution="wall_refined",
    thermal_coupling=True,
    max_iterations=12,
    convergence_tol=1e-3,
    solidification_temperature_fraction=0.3,
    shear_heating_enabled=True,
    fill_method="march",
    thermal_model=a.model,
)
t0 = time.time()
r = s.solve(num_frames=24)
md = r.metadata
info = {
    k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
    for k, v in md.items()
    if not isinstance(v, (np.ndarray, dict, list))
}
info["elapsed_s"] = time.time() - t0
info["cell"] = a.cell
(out / f"{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
save = {
    "mask": geom.mask,
    "thk": geom.thickness_mm,
    "t_arr": np.asarray(r.fill_time_s, float),
    "axis_x": cfg.axis_x_mm,
    "y_edge": cfg.y_edge_mm,
}
for k in ("thermal_theta_end", "short_shot_mask", "pressure_norm"):
    v = getattr(r, k, None)
    if v is not None:
        save[k] = np.asarray(v)
np.savez_compressed(out / f"{tag}.npz", **save)
print(
    tag,
    f"{info['elapsed_s']:.0f}s",
    {
        k: info.get(k)
        for k in (
            "march_stalled",
            "march_gate_pressure_max_MPa",
            "march_volume_fraction",
            "total_fill_time_s",
        )
    },
)
