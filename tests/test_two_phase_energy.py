"""Two-phase short shot on a fill that carried the temperature and stalled."""

from __future__ import annotations

import numpy as np

from core.geometry import Geometry
from core.materials import MaterialDB
from core.multilayer_solver import MultilayerHeleShawSolver
from core.two_phase import solve_two_phase_short_shot


def test_cells_a_stalled_fill_never_reached_stay_out_of_the_shot():
    # a 1 mm strip, 300 mm long: PMMA reaches about 150 mm before the
    # injection pressure limit stops it (CHANGELOG 0.15.0)
    cell = 4.0
    ny, nx = 5, 75
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), 1.0)
    geom = Geometry(
        mask=mask, thickness_mm=thk, cell_size_mm=cell, gates=[(iy, 0) for iy in range(ny)]
    )
    solver = MultilayerHeleShawSolver(
        geometry=geom,
        material=MaterialDB()["PMMA"],
        melt_temperature_K=533.15,
        mold_temperature_K=333.15,
        injection_velocity_mms=60.0,
        injection_volume_flow_cm3s=1.2,
        num_layers=7,
        thermal_coupling=True,
        solidification_temperature_fraction=0.3,
        shear_heating_enabled=True,
        fill_method="march",
        thermal_model="energy",
    )
    main = solver.solve(num_frames=4)
    assert main.metadata["march_stalled"]
    unreached = main.unreached_mask
    assert unreached.any() and not unreached.all()
    full_shot = float(thk.sum()) * cell * cell / 1000.0  # the whole cavity, cm³
    tp = solve_two_phase_short_shot(solver, full_shot)
    assert not (tp.injection_mask & unreached).any()
    assert tp.metadata["injection_stalled_cells"] == int(unreached.sum())
