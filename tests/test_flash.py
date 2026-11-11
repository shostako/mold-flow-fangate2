"""Flash risk on the parting line (core.flash)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.flash import flash_risk, outline_cells
from core.geometry import Geometry
from core.materials import MaterialDB
from core.packing_solve import solve_packing

PMMA = MaterialDB()["PMMA"]
T_MELT, T_MOLD = 533.15, 333.15


def test_outline_is_the_ring_of_cells_next_to_the_outside():
    mask = np.zeros((6, 7), dtype=bool)
    mask[1:5, 1:6] = True
    edge = outline_cells(mask)
    assert edge.sum() == 2 * 5 + 2 * 2  # a 4 × 5 block: everything but its 2 × 3 inside
    assert not edge[2:4, 2:5].any()
    # the grid's own border counts as outside
    assert outline_cells(np.ones((3, 3), dtype=bool)).sum() == 8


def _geom(mask, thk, product=None):
    return Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=2.0,
        gates=[(0, 0)],
        product_mask=product,
    )


def _fake_solve(p_mpa, t_s=None):
    p = np.asarray(p_mpa, dtype=float) * 1e6
    return SimpleNamespace(
        molten_peak_pressure_Pa=p,
        molten_peak_time_s=np.zeros_like(p) if t_s is None else np.asarray(t_s, dtype=float),
        clamp_force_N=np.array([0.0, 5.0e5, 2.0e5]),
        times_s=np.array([0.0, 1.0, 2.0]),
    )


def test_spots_are_ranked_apart_on_the_product_and_the_runner():
    mask = np.ones((3, 12), dtype=bool)
    thk = np.full(mask.shape, 2.0)
    product = np.zeros(mask.shape, dtype=bool)
    product[:, 6:] = True
    p = np.zeros(mask.shape)
    p[0, :6] = 80.0  # the runner side: about the holding pressure
    p[0, 6:] = [30.0, 31.0, 32.0, 33.0, 34.0, 50.0]
    risk = flash_risk(_geom(mask, thk, product), _fake_solve(p), n_spots=3, min_gap_mm=0.0)
    assert [round(s.pressure_Pa / 1e6) for s in risk.spots] == [50, 34, 33]
    assert all(s.in_product for s in risk.spots)
    assert all(not s.in_product for s in risk.runner_spots)
    assert risk.runner_spots[0].pressure_Pa == pytest.approx(80e6)
    assert risk.clamp_peak_N == pytest.approx(5.0e5)
    assert risk.clamp_peak_time_s == pytest.approx(1.0)


def test_spots_keep_their_distance():
    mask = np.ones((3, 20), dtype=bool)
    thk = np.full(mask.shape, 2.0)
    p = np.zeros(mask.shape)
    p[0, :] = np.linspace(40.0, 20.0, 20)  # one long stretch, highest at the left
    risk = flash_risk(_geom(mask, thk), _fake_solve(p), n_spots=5, min_gap_mm=10.0)
    xs = [s.x_mm for s in risk.spots]
    assert all(abs(a - b) >= 10.0 for i, a in enumerate(xs) for b in xs[:i])
    assert risk.spots[0].pressure_Pa == pytest.approx(40e6)


def test_inside_cells_and_unsolved_cells_are_not_on_the_outline():
    mask = np.ones((5, 5), dtype=bool)
    thk = np.full(mask.shape, 2.0)
    p = np.full(mask.shape, 10.0)
    p[2, 2] = 99.0  # inside: not on the parting line
    p[0, 0] = np.nan  # not solved
    risk = flash_risk(_geom(mask, thk), _fake_solve(p))
    assert not risk.edge[2, 2] and not risk.edge[0, 0]
    assert all(s.pressure_Pa == pytest.approx(10e6) for s in risk.spots)


def test_bad_input():
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        flash_risk(_geom(mask, np.ones((3, 3))), _fake_solve(np.zeros((2, 2))))
    with pytest.raises(ValueError):
        flash_risk(_geom(mask, np.ones((3, 3))), _fake_solve(np.zeros((3, 3))), n_spots=-1)


def test_pressure_after_the_centre_froze_does_not_count():
    # a thin strip that freezes early behind a thick one: it is still squeezed
    # by the holding pressure after its centre is solid, but that is not flash
    ny, nx = 4, 30
    mask = np.ones((ny, nx), dtype=bool)
    thk = np.full((ny, nx), 3.0)
    thk[:, 20:] = 1.0
    t_arr = np.tile(np.linspace(0.0, 1.5, nx), (ny, 1))
    gates = [(iy, 0) for iy in range(ny)]
    r = solve_packing(mask, thk, t_arr, gates, 2.0, PMMA, T_MELT, T_MOLD, 60e6, 3.0)
    thin = thk < 2.0
    thick = ~thin
    assert np.all(r.molten_peak_pressure_Pa <= r.peak_pressure_Pa + 1e-6)
    # the thick part stays molten under the holding pressure
    assert np.nanmax(r.molten_peak_pressure_Pa[thick]) == pytest.approx(60e6, rel=1e-3)
    # the far end of the 1 mm part is solid before the pressure gets there
    # (age = t_arr + t): it was squeezed, but not while it could flash
    gap = r.peak_pressure_Pa[thin] - r.molten_peak_pressure_Pa[thin]
    assert np.nanmax(gap) > 0.5e6
    # its time is after the switch, within the march
    assert np.all(r.molten_peak_time_s[mask] >= 0.0)
    assert np.all(r.molten_peak_time_s[mask] <= r.times_s[-1] + 1e-12)


def test_the_map_is_drawn(tmp_path):
    from core.visualizer import render_flash_risk_map

    mask = np.ones((6, 10), dtype=bool)
    thk = np.full(mask.shape, 2.0)
    product = np.zeros(mask.shape, dtype=bool)
    product[2:, :] = True
    p = np.zeros(mask.shape)
    p[0, :] = 80.0
    p[-1, :] = np.linspace(20.0, 40.0, 10)
    geom = _geom(mask, thk, product)
    risk = flash_risk(geom, _fake_solve(p), n_spots=2, min_gap_mm=4.0)
    out = render_flash_risk_map(SimpleNamespace(geometry=geom), risk, tmp_path / "flash.png")
    assert out.exists() and out.stat().st_size > 1000
