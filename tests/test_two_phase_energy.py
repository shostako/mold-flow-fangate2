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
    # the clock is the march's: every pool cell has arrived by the injection's end
    t_pool = tp.injection_fill_time_s[tp.injection_mask]
    assert np.nanmax(t_pool) == tp.injection_time_s
    assert np.allclose(t_pool, np.asarray(main.fill_time_s)[tp.injection_mask], rtol=1e-6)
    # no compression: what the injection left empty stays empty in the final state
    assert np.array_equal(tp.final_mask, tp.injection_mask)


def test_a_stalled_fill_is_not_shown_as_full():
    from core.visualizer import fill_frame_fractions, filled_at

    cell = 4.0
    mask = np.ones((5, 75), dtype=bool)
    geom = Geometry(
        mask=mask, thickness_mm=np.full(mask.shape, 1.0), cell_size_mm=cell,
        gates=[(iy, 0) for iy in range(5)],
    )  # fmt: skip
    r = MultilayerHeleShawSolver(
        geometry=geom, material=MaterialDB()["PMMA"], melt_temperature_K=533.15,
        mold_temperature_K=333.15, injection_velocity_mms=60.0, injection_volume_flow_cm3s=1.2,
        num_layers=7, thermal_coupling=True, fill_method="march", thermal_model="energy",
    ).solve(num_frames=4)  # fmt: skip
    assert r.unreached_mask.any()
    assert np.array_equal(r.unfillable_mask, r.unreached_mask)
    assert not (filled_at(r, np.inf) & r.unreached_mask).any()
    assert np.isclose(fill_frame_fractions(r, 6)[-1], 1.0 - r.unreached_mask.mean())


def test_a_pressure_limited_fill_that_completes_keeps_the_march_s_clock():
    cell = 4.0
    mask = np.ones((5, 30), dtype=bool)
    geom = Geometry(
        mask=mask, thickness_mm=np.full(mask.shape, 2.0), cell_size_mm=cell,
        gates=[(iy, 0) for iy in range(5)],
    )  # fmt: skip
    solver = MultilayerHeleShawSolver(
        geometry=geom, material=MaterialDB()["PMMA"], melt_temperature_K=533.15,
        mold_temperature_K=333.15, injection_velocity_mms=60.0, injection_volume_flow_cm3s=2.4,
        num_layers=7, thermal_coupling=True, fill_method="march", thermal_model="energy",
        injection_pressure_limit_Pa=20e6,
    )  # fmt: skip
    main = solver.solve(num_frames=4)
    md = main.metadata
    assert md["march_pressure_limited_steps"] > 0 and not md["march_stalled"]
    nominal = float(mask.sum()) * cell * cell * 2.0 / 1000.0 / 2.4  # V / Q
    assert main.total_fill_time_s > nominal * (1 + 1e-4)  # the limit slowed it
    tp = solve_two_phase_short_shot(solver, float(mask.sum()) * cell * cell * 2.0 / 1000.0)
    assert tp.injection_time_s == main.total_fill_time_s
