"""Holding-phase solve (core/packing_solve.py).

The reference values are computed here from the slab series and the Tait
model written out again, apart from the module under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from core.materials import MaterialDB
from core.packing_solve import (
    _cross_fluidity_factor,
    coarsening_factor,
    slab_theta,
    solve_packing,
    solve_packing_coarsened,
    summarize_solve_by_thickness,
)

DB = MaterialDB()
PMMA = DB["PMMA"]
T_MELT, T_MOLD, T_ROOM = 533.15, 333.15, 296.15
ALPHA = PMMA.thermal_diffusivity_m2_s


def _ref_volume(T, p):
    """Sharp two-domain Tait for PMMA, constants typed in again."""
    dT = T - 383.0
    melt = T > 383.0 + 2.0e-7 * p
    v0 = np.where(melt, 0.85982e-3 + 5.697e-7 * dT, 0.860702e-3 + 1.995e-7 * dT)
    B = np.where(melt, 2.09e8 * np.exp(-4.9083e-3 * dT), 2.73e8 * np.exp(-3.394e-3 * dT))
    return v0 * (1.0 - 0.0894 * np.log1p(p / B))


def _ref_mean_density(h_mm, age_s, p):
    """Thickness-mean density of a slab: series temperature, trapezoid rule."""
    fo = ALPHA * age_s / (h_mm * 1e-3) ** 2
    xi = np.linspace(0.0, 0.5, 4001)
    k = 2 * np.arange(400) + 1
    sign = (-1.0) ** np.arange(400)
    theta = (4 / np.pi) * (
        np.cos(np.outer(xi, k) * np.pi) * (sign / k * np.exp(-(k**2) * np.pi**2 * fo))
    ).sum(axis=1)
    theta = np.clip(theta, 0.0, 1.0) if fo > 1e-6 else np.ones_like(xi)
    rho = 1.0 / _ref_volume(T_MOLD + theta * (T_MELT - T_MOLD), p)
    return float(np.sum(0.5 * (rho[1:] + rho[:-1]) * np.diff(xi)) / 0.5)


def _ref_shrink(h_mm, age_s, p):
    return 1.0 - _ref_mean_density(h_mm, age_s, p) * float(_ref_volume(T_ROOM, 0.0))


def _plate(h_mm, age_s, ny=4, nx=12):
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), float(h_mm))
    t_arr = np.full((ny, nx), float(age_s))
    gates = [(iy, 0) for iy in range(ny)]
    return mask, thk, t_arr, gates


def _solve(mask, thk, t_arr, gates, p, t_hold, **kw):
    return solve_packing(mask, thk, t_arr, gates, 2.0, PMMA, T_MELT, T_MOLD, p, t_hold, **kw)


def test_cross_fluidity_factor_inverts_the_cross_model():
    n = 0.25
    x = np.array([0.0, 1e-6, 1e-2, 1.0, 30.0, 1e4, 1e8])
    stress = x / (1.0 + x ** (1.0 - n))
    assert np.allclose(_cross_fluidity_factor(stress, n), 1.0 + x ** (1.0 - n), rtol=1e-8)


def test_slab_theta_matches_the_two_wall_layers_at_short_times():
    """At small Fourier numbers the slab is two half-spaces that have not met."""
    n_z = 16
    x, _ = np.polynomial.legendre.leggauss(n_z)
    xi = 0.25 * (x + 1.0)  # distance from the centre, in thicknesses
    for fo in (1e-4, 1e-3, 1e-2):
        want = erf((0.5 - xi) / (2 * math.sqrt(fo))) + erf((0.5 + xi) / (2 * math.sqrt(fo))) - 1.0
        assert np.allclose(slab_theta(np.array([fo]), n_z)[0], want, atol=2e-4)
    assert np.all(slab_theta(np.array([0.0]), n_z) == 1.0)
    late = slab_theta(np.array([0.5]), n_z)[0]
    assert np.allclose(
        late, 4 / np.pi * math.exp(-(np.pi**2) * 0.5) * np.cos(np.pi * xi), atol=2e-4
    )


def test_sealed_cavity_keeps_its_mass_and_never_pulls_on_the_wall():
    mask, thk, t_arr, gates = _plate(4.0, 2.9)
    r = _solve(mask, thk, t_arr, gates, 0.0, 0.0)
    assert np.allclose(r.total_mass_kg, r.initial_mass_kg, rtol=1e-9)
    assert r.added_mass_kg[-1] == 0.0
    # the mass of a just-filled cavity at the start pressure, shrunk to room temperature
    assert r.shrinkage[2, 5] == pytest.approx(_ref_shrink(4.0, 2.9, 1.0e6), abs=1e-4)
    assert np.all(r.end_pressure_Pa == 0.0)
    assert np.all(np.diff(r.clamp_force_N) <= 1e-9)
    assert r.unconverged_steps == 0


@pytest.mark.parametrize(
    ("h_mm", "age_s", "p_hold", "t_hold"),
    [(4.0, 2.9, 90e6, 1.0), (4.0, 2.9, 90e6, 5.0), (2.0, 0.5, 60e6, 0.5)],
)
def test_easy_flowing_plate_packs_like_a_single_slab(h_mm, age_s, p_hold, t_hold):
    """A small plate, gated along one edge, while its core is still hot: the
    whole plate sits at the holding pressure until the gate closes, so every
    cell carries the mass of a slab at that pressure and age."""
    mask, thk, t_arr, gates = _plate(h_mm, age_s)
    r = _solve(mask, thk, t_arr, gates, p_hold, t_hold)
    want = _ref_shrink(h_mm, age_s + t_hold, p_hold)
    assert np.allclose(r.shrinkage, want, atol=1.5e-4)
    assert np.allclose(r.peak_pressure_Pa, p_hold, rtol=2e-3)
    assert r.unconverged_steps == 0


def test_mass_that_came_in_is_the_mass_that_is_there():
    mask, thk, t_arr, gates = _plate(3.0, 1.0, ny=5, nx=20)
    t_arr = np.tile(np.linspace(0.0, 1.5, 20), (5, 1))
    r = _solve(mask, thk, t_arr, gates, 60e6, 4.0)
    gained = r.total_mass_kg - r.initial_mass_kg
    assert np.allclose(gained, r.added_mass_kg, atol=1e-6 * r.initial_mass_kg)
    assert r.added_mass_fraction > 0.02
    # nothing comes in once the gate has closed
    closed = r.times_s > 4.0 + 1e-9
    assert np.allclose(r.added_mass_kg[closed], r.added_mass_kg[-1])


def test_pressure_stays_between_zero_and_the_holding_pressure():
    mask, thk, t_arr, gates = _plate(3.0, 1.0, ny=5, nx=20)
    r = _solve(mask, thk, t_arr, gates, 60e6, 4.0)
    assert np.all(r.peak_pressure_Pa <= 60e6 * (1 + 1e-6))
    assert np.nanmin(r.frame_pressure_Pa) >= 0.0
    assert np.all(r.end_pressure_Pa >= 0.0)
    assert np.all(np.isfinite(r.shrinkage))


def test_thin_neck_starves_the_pocket_behind_it():
    ny, nx = 3, 30
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), 4.0)
    thk[:, 10:15] = 1.0
    t_arr = np.tile(np.linspace(0.0, 1.5, nx), (ny, 1))
    gates = [(iy, 0) for iy in range(ny)]
    r = _solve(mask, thk, t_arr, gates, 60e6, 15.0)
    near, far = r.shrinkage[:, 2:9], r.shrinkage[:, 16:]
    assert far.mean() > near.mean() + 0.01
    # the far pocket was pressurised at first and lost it when the neck froze
    assert r.peak_pressure_Pa[1, 25] > 20e6
    assert r.freeze_pressure_Pa[1, 25] < r.freeze_pressure_Pa[1, 5]
    # the same plate without the neck feeds the far end
    r2 = _solve(mask, np.full((ny, nx), 4.0), t_arr, gates, 60e6, 15.0)
    assert r2.shrinkage[:, 16:].mean() < far.mean() - 0.01


def test_more_pressure_and_more_time_leave_less_shrinkage():
    mask, thk, t_arr, gates = _plate(3.0, 1.0, ny=3, nx=16)
    base = _solve(mask, thk, t_arr, gates, 30e6, 2.0).shrinkage.mean()
    assert _solve(mask, thk, t_arr, gates, 60e6, 2.0).shrinkage.mean() < base - 0.005
    assert _solve(mask, thk, t_arr, gates, 30e6, 5.0).shrinkage.mean() < base - 0.005


def test_bad_input():
    mask, thk, t_arr, gates = _plate(3.0, 1.0)
    with pytest.raises(ValueError, match="pvT"):
        solve_packing(mask, thk, t_arr, gates, 2.0, DB["PP"], T_MELT, T_MOLD, 60e6, 1.0)
    with pytest.raises(ValueError, match="gate"):
        _solve(mask, thk, t_arr, [], 60e6, 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        _solve(mask, thk, t_arr, gates, -1.0, 1.0)
    unreached = t_arr.copy()
    unreached[0, -1] = np.nan
    with pytest.raises(ValueError, match="did not reach"):
        _solve(mask, thk, unreached, gates, 60e6, 1.0)
    with pytest.raises(ValueError):
        _solve(mask, thk[:, :3], t_arr, gates, 60e6, 1.0)


def test_summary_rows_by_thickness():
    ny, nx = 3, 20
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), 4.0)
    thk[:, 12:] = 2.0
    t_arr = np.tile(np.linspace(0.0, 1.0, nx), (ny, 1))
    r = _solve(mask, thk, t_arr, [(iy, 0) for iy in range(ny)], 60e6, 3.0)
    rows = summarize_solve_by_thickness(r, thk, mask, cell_size_mm=2.0)
    assert [row["thickness_mm"] for row in rows] == [4.0, 2.0]
    assert rows[0]["area_cm2"] == pytest.approx(3 * 12 * 4 / 100.0)
    for row, sel in zip(rows, (thk == 4.0, thk == 2.0), strict=True):
        assert row["shrinkage_mean"] == pytest.approx(r.shrinkage[sel].mean())
        assert row["shrinkage_min"] <= row["shrinkage_mean"] <= row["shrinkage_max"]
        assert row["peak_pressure_Pa"] == pytest.approx(r.peak_pressure_Pa[sel].mean())


def test_coarsening_factor():
    assert coarsening_factor(18_000, 20_000) == 1
    assert coarsening_factor(20_001, 20_000) == 2
    assert coarsening_factor(72_000, 20_000) == 2
    assert coarsening_factor(200_000, 20_000) == 4


def test_coarse_solve_comes_back_on_the_fine_grid_and_agrees_with_the_fine_one():
    ny, nx = 11, 41  # odd on purpose: the last blocks are half outside the grid
    mask = np.ones((ny, nx), dtype=bool)
    mask[:3, 30:] = False  # a notch, so some blocks are partly cavity
    thk = np.full((ny, nx), 3.0)
    t_arr = np.tile(np.linspace(0.0, 1.5, nx), (ny, 1))
    gates = [(iy, 0) for iy in range(ny)]
    args = (mask, thk, t_arr, gates, 1.0, PMMA, T_MELT, T_MOLD, 60e6, 4.0)
    fine = solve_packing(*args)
    coarse = solve_packing_coarsened(*args, factor=2)
    for r in (fine, coarse):
        assert r.shrinkage.shape == mask.shape
        assert np.all(np.isfinite(r.shrinkage[mask])) and np.all(np.isnan(r.shrinkage[~mask]))
        assert r.frame_pressure_Pa.shape[1:] == mask.shape
    assert coarse.shrinkage[mask].mean() == pytest.approx(fine.shrinkage[mask].mean(), abs=1.5e-3)
    assert coarse.added_mass_fraction == pytest.approx(fine.added_mass_fraction, rel=0.08)
    assert fine.cell_size_mm == 1.0 and coarse.cell_size_mm == 2.0
    # the coarse cells are 2 × 2 blocks
    assert coarse.shrinkage[4, 10] == coarse.shrinkage[5, 11]


def test_coarsening_backs_off_where_it_would_cut_the_cavity_from_its_gate():
    ny, nx = 9, 27
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, 9:12] = False
    mask[4, 9:12] = True  # a one-cell neck: a third of its 3 × 3 block, so the block is dropped
    thk = np.full((ny, nx), 3.0)
    t_arr = np.tile(np.linspace(0.0, 1.0, nx), (ny, 1))
    gates = [(iy, 0) for iy in range(ny)]
    args = (mask, thk, t_arr, gates, 1.0, PMMA, T_MELT, T_MOLD, 60e6, 3.0)
    by_two = solve_packing_coarsened(*args, factor=2)
    backed_off = solve_packing_coarsened(*args, factor=3)
    assert np.array_equal(backed_off.shrinkage, by_two.shrinkage, equal_nan=True)
    assert backed_off.cell_size_mm == 2.0
    # behind the neck the pressure did arrive
    assert backed_off.peak_pressure_Pa[4, 20] > 30e6
    assert np.all(np.isfinite(backed_off.shrinkage[mask]))


# --- V/P switch-over: the holding pressure fills what the fill left empty ----


def _channel(nx=30, ny=4, h_mm=3.0, t_end=1.5):
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), float(h_mm))
    t_arr = np.tile(np.linspace(0.0, t_end, nx), (ny, 1))
    gates = [(iy, 0) for iy in range(ny)]
    # the rate that fills the channel in t_end: its volume over the fill time
    rate = nx * ny * 4e-6 * h_mm * 1e-3 / t_end
    return mask, thk, t_arr, gates, rate


def test_switch_at_full_cavity_is_the_packing_only_solve():
    mask, thk, t_arr, gates, _ = _channel()
    a = _solve(mask, thk, t_arr, gates, 60e6, 3.0)
    b = _solve(mask, thk, t_arr, gates, 60e6, 3.0, switch_fill_fraction=1.0)
    assert np.array_equal(a.shrinkage, b.shrinkage)
    assert a.switch_time_s == pytest.approx(1.5)
    assert np.all(a.late_fill_time_s == 0.0)
    assert a.unfilled_volume_fraction == 0.0


def test_holding_pressure_fills_the_rest_at_the_machine_rate():
    mask, thk, t_arr, gates, rate = _channel()
    r = _solve(mask, thk, t_arr, gates, 60e6, 3.0, switch_fill_fraction=0.8, max_inflow_m3_s=rate)
    # 80 % of the volume was in by the time the fill reached column 23 of 30
    assert r.switch_time_s == pytest.approx(t_arr[0, 23])
    assert np.all(r.late_fill_time_s[:, :24] == 0.0)
    late = r.late_fill_time_s[1, 24:]
    assert np.all(late > 0.0) and np.all(np.diff(late) >= 0.0)  # the front moves on
    assert r.unfilled_volume_fraction == 0.0
    # 20 % of the volume at the machine's rate takes 20 % of the fill time
    # (the rate is held within 10 %, and the melt also compresses on the way)
    assert late[-1] == pytest.approx(0.2 * 1.5, rel=0.25)
    # the gate stays far below the holding pressure while the rate is capped ...
    early = r.gate_pressure_Pa[(r.times_s > 0.02) & (r.times_s < 0.5 * late[-1])]
    assert early.size > 0 and early.max() < 0.3 * 60e6
    # ... and reaches it once the cavity is full
    assert r.gate_pressure_Pa[np.searchsorted(r.times_s, 2.0)] == pytest.approx(60e6)
    gained = r.total_mass_kg[-1] - r.initial_mass_kg
    assert gained == pytest.approx(r.added_mass_kg[-1], rel=1e-5)
    # in the end the plate is packed like the one that was full at the switch
    full = _solve(mask, thk, t_arr, gates, 60e6, 3.0, max_inflow_m3_s=rate)
    assert np.allclose(r.shrinkage[:, :20], full.shrinkage[:, :20], atol=1e-3)
    assert r.unconverged_steps == 0


def test_no_holding_pressure_leaves_the_rest_empty():
    mask, thk, t_arr, gates, rate = _channel()
    r = _solve(mask, thk, t_arr, gates, 0.0, 3.0, switch_fill_fraction=0.8, max_inflow_m3_s=rate)
    assert r.unfilled_volume_fraction == pytest.approx(6 / 30)
    assert np.all(np.isnan(r.late_fill_time_s[:, 24:]))
    assert np.all(np.isnan(r.shrinkage[:, 24:]))
    assert np.all(np.isfinite(r.shrinkage[:, :24]))


def test_melt_does_not_pass_a_neck_that_has_frozen():
    mask, thk, t_arr, gates, rate = _channel()
    thk[:, 18:22] = 0.6  # frozen through long before the switch
    r = _solve(mask, thk, t_arr, gates, 60e6, 3.0, switch_fill_fraction=0.5, max_inflow_m3_s=rate)
    # the switch came before the neck; the melt fills up to it and stops
    assert np.all(np.isfinite(r.late_fill_time_s[:, :18]))
    assert np.all(np.isnan(r.late_fill_time_s[:, 22:]))
    assert np.all(r.peak_pressure_Pa[:, 22:] == 0.0)
    assert r.unfilled_volume_fraction > 0.2
    gained = r.total_mass_kg[-1] - r.initial_mass_kg
    assert gained == pytest.approx(r.added_mass_kg[-1], rel=1e-5)
    assert r.unconverged_steps == 0


def test_switch_needs_a_rate_and_a_sane_fraction():
    mask, thk, t_arr, gates, rate = _channel()
    with pytest.raises(ValueError, match="max_inflow"):
        _solve(mask, thk, t_arr, gates, 60e6, 3.0, switch_fill_fraction=0.8)
    for bad in (0.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="switch_fill_fraction"):
            _solve(
                mask, thk, t_arr, gates, 60e6, 3.0, switch_fill_fraction=bad, max_inflow_m3_s=rate
            )


def test_rate_cap_slows_the_pressure_rise_of_a_full_cavity():
    mask, thk, t_arr, gates, rate = _channel()
    free = _solve(mask, thk, t_arr, gates, 60e6, 3.0)
    capped = _solve(mask, thk, t_arr, gates, 60e6, 3.0, max_inflow_m3_s=rate)
    assert np.all(free.gate_pressure_Pa[free.times_s < 3.0] == 60e6)
    assert capped.gate_pressure_Pa[1] < 10e6
    assert capped.gate_pressure_Pa[np.searchsorted(capped.times_s, 1.0)] == pytest.approx(60e6)
    # the same plate in the end
    assert np.allclose(capped.shrinkage, free.shrinkage, atol=5e-4)


def test_end_time_before_the_gate_closes_is_not_overrun():
    mask, thk, t_arr, gates = _plate(3.0, 1.0, ny=4, nx=12)
    seen = []
    r = _solve(
        mask,
        thk,
        t_arr,
        gates,
        60e6,
        4.0,
        end_time_s=0.001,
        dt_first_s=0.002,
        progress=lambda t, end: seen.append((t, end)),
    )
    assert r.times_s[-1] <= 0.001 + 1e-12
    assert all(t <= end + 1e-12 for t, end in seen)
