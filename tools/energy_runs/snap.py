"""4 mm 格子で、充填の途中のゲート側の帯（Y=-3〜-10）の中心温度の横の分布がいつ崩れるかを見る。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry
from core.materials import MaterialDB
from core.multilayer_solver import MultilayerHeleShawSolver

cell = float(sys.argv[1])
heating = sys.argv[2] == "1" if len(sys.argv) > 2 else True
cfg = FanRunnerPlateConfig(cell_size_mm=cell)
geom = build_fan_runner_plate_geometry(cfg)
fr = tuple(np.round(np.arange(0.02, 1.0, 0.04), 3))
s = MultilayerHeleShawSolver(
    geometry=geom,
    material=MaterialDB()["PMMA"],
    melt_temperature_K=533.15,
    mold_temperature_K=333.15,
    injection_velocity_mms=130.0,
    injection_volume_flow_cm3s=39.27,
    num_layers=7,
    thermal_coupling=True,
    solidification_temperature_fraction=0.3,
    shear_heating_enabled=heating,
    fill_method="march",
    thermal_model="energy",
    thermal_snapshot_fractions=fr,
)
r = s.solve(num_frames=8)
ny, nx = geom.mask.shape
y = (np.arange(ny) + 0.5) * cell - cfg.y_edge_mm
x = (np.arange(nx) + 0.5) * cell - cfg.axis_x_mm
i = int(np.argmin(np.abs(y + 5)))
row = geom.mask[i]
print("heating", heating, "stalled", r.metadata.get("march_stalled"), "Y", y[i], "cells", row.sum())
for f in sorted(r.thermal_snapshots):
    t, st = r.thermal_snapshots[f]
    c = st[i, row, 0]
    live = (c > 0.3).sum()
    xc = x[row]
    mid = np.abs(xc) < cell
    print(
        f"f={f:.2f} t={t:5.2f}s centre θ: axis {c[mid].mean():.2f} x=±50 {c[np.argmin(np.abs(xc - 50))]:.2f} "
        f"x=±100 {c[np.argmin(np.abs(xc - 100))]:.2f} melt {live * cell:5.0f} mm"
    )
