"""The contact between the melt and the mould in a fill that carries the temperature."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import Geometry
from core.materials import MaterialDB
from core.melt_state import MeltState
from core.multilayer_solver import MultilayerHeleShawSolver

PMMA = MaterialDB()["PMMA"]


def _strip(htc, **kw):
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
        **kw,
    ).solve(num_frames=2)


def test_a_limit_that_passes_nothing_from_the_start_is_not_called_a_frozen_path():
    # 1 kPa moves next to nothing through molten PMMA: the march stalls in its
    # first steps with melt on both sides of the front's faces. Nothing froze, and
    # the limit never ran slowed either: the record cannot say which (Codex P2 on PR #25)
    md = _strip(None, injection_pressure_limit_Pa=1e3).metadata
    assert md["march_stalled"] and md["march_volume_fraction"] < 0.05
    assert md["march_stall_cause"] == "undetermined"
    assert md["injection_pressure_limit_MPa"] == pytest.approx(1e-3)


def test_inflow_conductance_reads_the_feeders_melt_at_the_cells_own_thickness():
    # cells: 0 a 1 mm cell solid through, 1 a 4 mm cell of fresh melt, 2 a 1 mm cell of fresh melt
    st = MeltState(np.array([1e-3, 4e-3, 1e-3]), 2e-3, PMMA, 533.15, 333.15)
    st.theta[0] = 0.0
    st.flux_m2_s[:] = 1e-4
    own = st.conductance()
    assert own[0] == 0.0 and own[1] > 10.0 * own[2] > 0.0
    # fresh melt from the thick cell into the solid thin one: what a thin cell of
    # fresh melt passes, not what the thick feeder does
    got = st.inflow_conductance(np.array([0, 0]), np.array([1, 2]))
    assert got == pytest.approx([own[2], own[2]], rel=1e-9)
    # a feeder solid through passes nothing on
    assert st.inflow_conductance(np.array([2]), np.array([0]))[0] == 0.0
    assert st.inflow_conductance(np.array([], dtype=int), np.array([], dtype=int)).size == 0


def _framed_plate(late: bool):
    # a 4 mm plate with a 1 mm rim 8 mm wide all round, gated on the rim: the melt
    # crosses the rim into the plate and stops in the rim to either side; the
    # plate fills, and the melt comes back into the rim from the plate's edge
    ny, nx, rim = 24, 30, 4
    thk = np.full((ny, nx), 1.0)
    thk[rim : ny - rim, rim : nx - rim] = 4.0
    geom = Geometry(
        mask=np.ones((ny, nx), dtype=bool),
        thickness_mm=thk,
        cell_size_mm=2.0,
        gates=[(0, nx // 2 - 1), (0, nx // 2)],
    )
    return geom, MultilayerHeleShawSolver(
        geometry=geom,
        material=PMMA,
        melt_temperature_K=533.15,
        mold_temperature_K=333.15,
        injection_velocity_mms=60.0,
        injection_volume_flow_cm3s=1.0,
        num_layers=7,
        thermal_coupling=True,
        solidification_temperature_fraction=0.3,
        shear_heating_enabled=True,
        fill_method="march",
        thermal_model="energy",
        late_front_inflow=late,
    ).solve(num_frames=2)


def test_a_later_front_fills_the_seam_a_frozen_front_left_in_the_rim():
    geom, old = _framed_plate(False)
    md = old.metadata
    # v0.16: the cells where the front that froze in the rim meets the melt coming
    # back from the plate stay unfilled, far below the pressure limit
    assert md["march_stalled"] and md["march_stall_cause"] == "frozen_path"
    assert md["late_front_inflow"] is False
    left = old.unreached_mask
    assert left.any() and np.all(geom.thickness_mm[left] == 1.0)
    assert md["march_gate_pressure_free_max_MPa"] < 100.0
    _, new = _framed_plate(True)
    md = new.metadata
    assert md["late_front_inflow"] is True
    assert not md["march_stalled"] and md["march_unreached_cells"] == 0
    assert md["march_volume_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert md["march_pressure_limited_steps"] == 0
    # the rest of the fill is the same fill: the cells both reached, at about the same time
    both = geom.mask & ~old.unreached_mask
    shift = np.abs(new.fill_time_s - old.fill_time_s)[both]
    assert np.median(shift) < 0.01 * old.total_fill_time_s


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
