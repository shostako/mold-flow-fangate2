"""Holding-pressure reach (core/packing.py): the slab's freeze time and the
bottleneck path from the gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.packing import (
    freeze_age_s,
    freeze_fourier_number,
    packing_reach,
    slab_centre_temperature,
    summarize_by_thickness,
)

ALPHA = 1.1e-7


def test_slab_centre_temperature_limits_and_first_mode():
    assert slab_centre_temperature(0.0) == 1.0
    # still at the initial temperature while the two wall layers have not met
    assert slab_centre_temperature(0.002) == pytest.approx(1.0, abs=1e-9)
    # short times: two half-space solutions, 1 − 2·erfc(1 / (4·√Fo))
    assert slab_centre_temperature(0.01) == pytest.approx(1.0 - 2.0 * math.erfc(2.5), abs=1e-9)
    # late times: the first eigenmode alone, (4/π)·exp(−π²·Fo)
    assert slab_centre_temperature(0.5) == pytest.approx(
        4 / math.pi * math.exp(-(math.pi**2) * 0.5), rel=1e-9
    )


def test_slab_centre_temperature_matches_finite_difference():
    """An explicit 1-D conduction march, written apart from the series."""
    n = 201
    theta = np.ones(n)
    theta[0] = theta[-1] = 0.0
    dx = 1.0 / (n - 1)
    dfo = 0.2 * dx**2
    fo = 0.0
    while fo < 0.15 - 1e-12:
        theta[1:-1] += dfo / dx**2 * (theta[2:] - 2 * theta[1:-1] + theta[:-2])
        fo += dfo
    assert slab_centre_temperature(fo) == pytest.approx(theta[n // 2], abs=2e-4)


def test_freeze_fourier_number_is_the_root_and_orders_with_fraction():
    fo = freeze_fourier_number(0.3)
    assert slab_centre_temperature(fo) == pytest.approx(0.3, abs=1e-9)
    # a hotter solidification temperature is reached sooner
    assert freeze_fourier_number(0.5) < fo < freeze_fourier_number(0.1)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            freeze_fourier_number(bad)


def test_freeze_age_scales_with_thickness_squared():
    t = freeze_age_s(np.array([1.0, 2.0, 4.0]), ALPHA, 0.3)
    assert t[1] / t[0] == pytest.approx(4.0)
    assert t[2] / t[0] == pytest.approx(16.0)
    assert t[0] == pytest.approx(freeze_fourier_number(0.3) * 1e-6 / ALPHA)
    with pytest.raises(ValueError):
        freeze_age_s(np.array([1.0]), 0.0, 0.3)


def _channel(thicknesses, width=3):
    """A straight channel, one column per entry, gate on the left end."""
    n = len(thicknesses)
    mask = np.ones((width, n), dtype=bool)
    thk = np.tile(np.asarray(thicknesses, dtype=float), (width, 1))
    t_arr = np.zeros((width, n))
    gates = [(iy, 0) for iy in range(width)]
    return mask, thk, t_arr, gates


def test_thin_neck_limits_everything_behind_it():
    mask, thk, t_arr, gates = _channel([4, 4, 1, 4, 4])
    r = packing_reach(mask, thk, t_arr, gates, ALPHA, 0.3)
    t1, t4 = freeze_age_s(np.array([1.0, 4.0]), ALPHA, 0.3)
    # before the neck: fed for as long as the cell itself is molten
    assert r.reach_s[1, 1] == pytest.approx(t4)
    assert r.ratio[1, 1] == pytest.approx(1.0)
    # the neck and everything behind it: cut off when the neck freezes
    assert np.allclose(r.reach_s[:, 2:], t1)
    assert r.solidify_s[1, 4] == pytest.approx(t4)
    assert r.ratio[1, 4] == pytest.approx(1.0 / 16.0)
    assert r.ratio[1, 2] == pytest.approx(1.0)  # the neck: fed while it lasts


def test_best_of_two_paths_is_taken():
    """Gate at the left; the top row is a 1 mm strip, the bottom a 2 mm one;
    they rejoin at a 4 mm pocket on the right. The pocket is fed through the
    2 mm strip."""
    mask = np.ones((3, 6), dtype=bool)
    mask[1, 1:5] = False  # a wall between the two strips
    thk = np.full((3, 6), 4.0)
    thk[0, 1:5] = 1.0
    thk[2, 1:5] = 2.0
    t_arr = np.zeros((3, 6))
    r = packing_reach(mask, thk, t_arr, [(1, 0)], ALPHA, 0.3)
    t1, t2 = freeze_age_s(np.array([1.0, 2.0]), ALPHA, 0.3)
    assert r.reach_s[1, 5] == pytest.approx(t2)
    assert r.reach_s[0, 2] == pytest.approx(t1)
    assert np.isnan(r.reach_s[1, 2])  # outside the cavity


def test_arrival_time_eats_into_the_solidification_time():
    mask, thk, t_arr, gates = _channel([2, 2, 2, 2])
    t2 = float(freeze_age_s(np.array([2.0]), ALPHA, 0.3)[0])
    t_arr[:, 1] = 0.25 * t2
    t_arr[:, 2] = 2.0 * t2  # arrives older than the freeze age: frozen at the end of the fill
    t_arr[:, 3] = 0.5 * t2
    r = packing_reach(mask, thk, t_arr, gates, ALPHA, 0.3)
    assert r.solidify_s[1, 1] == pytest.approx(0.75 * t2)
    assert r.solidify_s[1, 2] == 0.0
    assert np.isnan(r.ratio[1, 2])
    # behind the frozen column nothing is fed, though the cell is still molten
    assert r.solidify_s[1, 3] == pytest.approx(0.5 * t2)
    assert r.reach_s[1, 3] == 0.0
    assert r.ratio[1, 3] == 0.0


def test_unreached_cells_and_bad_input():
    mask, thk, t_arr, gates = _channel([4, 4, 4])
    t_arr[:, 2] = np.nan  # the melt never arrived
    r = packing_reach(mask, thk, t_arr, gates, ALPHA, 0.3)
    assert np.all(r.solidify_s[:, 2] == 0.0)
    assert np.all(r.reach_s[:, 2] == 0.0)
    with pytest.raises(ValueError):
        packing_reach(mask, thk, t_arr, [], ALPHA, 0.3)
    with pytest.raises(ValueError):
        packing_reach(mask, thk[:, :2], t_arr, gates, ALPHA, 0.3)


def test_reach_never_exceeds_own_solidification_and_gate_is_fed():
    rng = np.random.default_rng(0)
    mask = np.ones((12, 15), dtype=bool)
    thk = rng.choice([1.0, 2.0, 3.0, 4.0], size=mask.shape)
    t_arr = rng.uniform(0.0, 3.0, size=mask.shape)
    r = packing_reach(mask, thk, t_arr, [(6, 0)], ALPHA, 0.3)
    assert np.all(r.reach_s <= r.solidify_s + 1e-12)
    assert r.reach_s[6, 0] == pytest.approx(r.solidify_s[6, 0])
    ok = r.solidify_s > 0
    assert np.all((r.ratio[ok] >= 0.0) & (r.ratio[ok] <= 1.0))
    # brute force on the same grid: relax until nothing changes
    s = np.where(np.isfinite(r.solidify_s), r.solidify_s, 0.0)
    ref = np.full(mask.shape, -1.0)
    ref[6, 0] = np.inf
    changed = True
    while changed:
        changed = False
        for iy in range(mask.shape[0]):
            for ix in range(mask.shape[1]):
                for jy, jx in ((iy + 1, ix), (iy - 1, ix), (iy, ix + 1), (iy, ix - 1)):
                    if 0 <= jy < mask.shape[0] and 0 <= jx < mask.shape[1]:
                        cand = min(ref[jy, jx], s[iy, ix])
                        if cand > ref[iy, ix]:
                            ref[iy, ix] = cand
                            changed = True
    ref[6, 0] = s[6, 0]
    assert np.allclose(r.reach_s, ref)


def test_summarize_by_thickness_rows():
    mask, thk, t_arr, gates = _channel([4, 4, 1, 4, 4])
    r = packing_reach(mask, thk, t_arr, gates, ALPHA, 0.3)
    rows = summarize_by_thickness(r, thk, mask, cell_size_mm=1.0)
    assert [row["thickness_mm"] for row in rows] == [4.0, 1.0]
    thick, thin = rows
    assert thick["area_cm2"] == pytest.approx(12 / 100.0)
    assert thick["frozen_at_fill_end"] == 0.0
    # 2 of the 4 thick columns sit behind the neck at ratio 1/16
    assert thick["ratio"] == pytest.approx((1.0 + 1.0 / 16.0) / 2.0)
    assert thin["ratio"] == pytest.approx(1.0)
    # a thickness frozen through everywhere has no molten cell to average
    t_arr2 = np.full(thk.shape, 1e6)
    r2 = packing_reach(mask, thk, t_arr2, gates, ALPHA, 0.3)
    rows2 = summarize_by_thickness(r2, thk, mask, cell_size_mm=1.0)
    assert all(row["frozen_at_fill_end"] == 1.0 and row["ratio"] is None for row in rows2)
