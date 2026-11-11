"""Two-domain Tait pvT (core/pvt.py) and its place in the material database."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.materials import MaterialDB
from core.pvt import TAIT_C, TaitPVT

PMMA = MaterialDB()["PMMA"].pvt


def test_material_database_carries_pvt_only_where_given():
    db = MaterialDB()
    assert isinstance(db["PMMA"].pvt, TaitPVT)
    assert db["PP"].pvt is None


def test_specific_volume_against_hand_values():
    """Each value written out from the constants, apart from the module."""
    # melt, 260 °C, no pressure: v0 alone
    assert PMMA.specific_volume(533.15, 0.0) == pytest.approx(
        0.85982e-3 + 5.697e-7 * (533.15 - 383.0), rel=1e-12
    )
    # solid, 23 °C, no pressure: density about 1186 kg/m³
    v_room = 0.860702e-3 + 1.995e-7 * (296.15 - 383.0)
    assert PMMA.specific_volume(296.15, 0.0) == pytest.approx(v_room, rel=1e-12)
    assert 1.0 / v_room == pytest.approx(1185.7, abs=0.1)
    # melt, 260 °C, 90 MPa
    B = 2.09e8 * math.exp(-4.9083e-3 * 150.15)
    want = (0.85982e-3 + 5.697e-7 * 150.15) * (1.0 - TAIT_C * math.log(1.0 + 90e6 / B))
    assert PMMA.specific_volume(533.15, 90e6) == pytest.approx(want, rel=1e-12)
    # solid, 60 °C, 50 MPa
    B = 2.73e8 * math.exp(-3.394e-3 * (333.15 - 383.0))
    want = (0.860702e-3 + 1.995e-7 * (333.15 - 383.0)) * (1.0 - TAIT_C * math.log(1.0 + 50e6 / B))
    assert PMMA.specific_volume(333.15, 50e6) == pytest.approx(want, rel=1e-12)


def test_transition_moves_with_pressure():
    # 120 °C: melt at no pressure (T_t = 110 °C), solid at 100 MPa (T_t = 130 °C)
    T = 393.15
    assert PMMA.specific_volume(T, 0.0) == pytest.approx(0.85982e-3 + 5.697e-7 * 10.15)
    B = 2.73e8 * math.exp(-3.394e-3 * 10.15)
    want = (0.860702e-3 + 1.995e-7 * 10.15) * (1.0 - TAIT_C * math.log(1.0 + 100e6 / B))
    assert PMMA.specific_volume(T, 100e6) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("band", [0.0, 2.0])
def test_slope_matches_a_finite_difference(band):
    T = np.array([320.0, 380.0, 395.0, 450.0, 530.0])
    for p in (0.0, 2.0e7, 9.0e7):
        # away from the transition line of each temperature
        ok = np.abs(T - (PMMA.b5 + PMMA.b6 * p)) > band + 0.5
        rho, slope = PMMA.density_and_slope(T, p, band)
        dp = 1.0e3
        up, _ = PMMA.density_and_slope(T, p + dp, band)
        dn, _ = PMMA.density_and_slope(T, max(p - dp, 0.0), band)
        fd = (up - dn) / (p + dp - max(p - dp, 0.0))
        assert np.allclose(slope[ok], fd[ok], rtol=1e-5)
        assert np.all(slope > 0.0)


def test_blend_removes_the_jump_and_keeps_the_slope_consistent():
    T = 393.15  # transition at (T − b5) / b6 = 50.75 MPa
    p = np.linspace(40e6, 62e6, 4401)
    sharp, _ = PMMA.density_and_slope(T, p)
    blend, slope = PMMA.density_and_slope(T, p, 2.0)
    assert np.max(np.abs(np.diff(sharp))) > 50 * np.max(np.abs(np.diff(blend)))
    assert np.all(np.diff(blend) > 0.0)  # denser with pressure all the way through
    # the slope integrates to the density difference across the band
    integral = np.sum(0.5 * (slope[1:] + slope[:-1]) * np.diff(p))
    assert integral == pytest.approx(blend[-1] - blend[0], rel=1e-4)
    # outside the band the blend is the model itself
    far = np.abs(T - (PMMA.b5 + PMMA.b6 * p)) > 2.0
    assert np.allclose(blend[far], sharp[far], rtol=1e-12)
