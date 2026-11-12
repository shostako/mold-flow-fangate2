"""The contact between the melt and the mould in a fill that carries the temperature."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import Geometry
from core.materials import MaterialDB
from core.melt_state import MeltState
from core.multilayer_solver import MultilayerHeleShawSolver

PMMA = MaterialDB()["PMMA"]


def _strip(htc):
    # a 1 mm strip, 300 mm long: PMMA stops at about 150 mm with the wall at
    # the mold temperature (tests/test_two_phase_energy.py)
    cell, ny, nx = 4.0, 5, 75
    geom = Geometry(
        mask=np.ones((ny, nx), dtype=bool),
        thickness_mm=np.full((ny, nx), 1.0),
        cell_size_mm=cell,
        gates=[(iy, 0) for iy in range(ny)],
    )
    return MultilayerHeleShawSolver(
        geometry=geom,
        material=PMMA,
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
        mold_htc_fill_W_m2K=htc,
    ).solve(num_frames=2)


def test_the_melt_runs_further_behind_a_contact_resistance():
    ideal = _strip(None)
    contact = _strip(5000.0)
    assert ideal.metadata["mold_htc_fill_W_m2K"] is None
    assert contact.metadata["mold_htc_fill_W_m2K"] == 5000.0
    assert ideal.metadata["march_stalled"]
    # the pressure climbs to the limit as the strip freezes: the limit stops it
    assert ideal.metadata["march_stall_cause"] == "pressure_limit"
    assert ideal.metadata["march_gate_pressure_free_max_MPa"] > 200.0
    # Bi = 5000·1 mm / k ≈ 25: the skin grows slower, the strip passes more
    assert (
        contact.metadata["march_volume_fraction"] > 1.05 * ideal.metadata["march_volume_fraction"]
    )
    # a contact that passes everything is the wall at the mold temperature
    perfect = _strip(1.0e10)
    assert perfect.metadata["march_volume_fraction"] == pytest.approx(
        ideal.metadata["march_volume_fraction"], rel=1e-3
    )


def test_the_surface_sits_between_the_melt_and_the_mould():
    h = np.array([1.0e-3, 4.0e-3])
    state = MeltState(h, 2.0e-3, PMMA, 533.15, 333.15, wall_htc_W_m2K=5000.0)
    assert np.allclose(state.biot, 5000.0 * h / PMMA.thermal_conductivity_W_mK)
    surf = state.surface_theta()  # all melt: the contact holds most of the drop at first
    assert np.all((surf > 0.0) & (surf < 1.0)) and surf[0] > surf[1]
    assert np.all(MeltState(h, 2.0e-3, PMMA, 533.15, 333.15).surface_theta() == 0.0)
    with pytest.raises(ValueError, match="wall_htc_W_m2K must be > 0"):
        MeltState(h, 2.0e-3, PMMA, 533.15, 333.15, wall_htc_W_m2K=0.0)
    with pytest.raises(ValueError, match="mold_htc_fill_W_m2K must be > 0"):
        _strip(-1.0)
