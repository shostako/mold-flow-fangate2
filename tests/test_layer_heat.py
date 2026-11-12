"""The layer-by-layer temperature state (``core.layer_heat``)."""

from __future__ import annotations

import numpy as np
import pytest

from core.layer_heat import (
    advect,
    conduct,
    face_dissipation,
    flow_shares,
    layer_grid,
    shear_heating,
)


def _slab_series(fourier: np.ndarray, xi: np.ndarray, terms: int = 4000) -> np.ndarray:
    """A slab at θ = 1 with both faces put to θ = 0: the textbook series."""
    k = 2 * np.arange(terms) + 1
    sign = (-1.0) ** np.arange(terms)
    decay = np.exp(-np.outer(fourier, k**2) * np.pi**2)
    return (4.0 / np.pi) * (decay * (sign / k)) @ np.cos(np.outer(xi, k) * np.pi).T


def test_the_layers_cover_the_half_thickness_centre_to_wall():
    g = layer_grid(16)
    assert g.n_z == 16
    assert np.all(np.diff(g.xi) > 0) and 0.0 < g.xi[0] and g.xi[-1] < 0.5
    assert g.wt.sum() == pytest.approx(0.5)
    assert np.all(g.decay < 0.0)
    assert np.allclose(g.modes @ g.modes_inv, np.eye(16), atol=1e-10)


@pytest.mark.parametrize("fourier", [1e-3, 1e-2, 0.05, 0.2])
def test_conduction_matches_the_slab_series(fourier):
    g = layer_grid(16)
    fo = np.array([fourier])
    got = conduct(np.ones((1, 16)), fo, g)[0]
    want = _slab_series(fo, g.xi)[0]
    assert np.abs(got - want).max() < 0.012
    # the thickness mean, which is what the density sees: to 0.1 % of the span
    fine_x, fine_w = np.polynomial.legendre.leggauss(200)
    mean_exact = float(_slab_series(fo, 0.25 * (fine_x + 1.0))[0] @ (0.25 * fine_w) / 0.5)
    assert float(got @ g.wt / 0.5) == pytest.approx(mean_exact, abs=1e-3)


def test_conduction_is_exact_in_time_whatever_the_step():
    g = layer_grid(16)
    start = np.ones((1, 16))
    once = conduct(start, np.array([0.08]), g)
    many = start
    for _ in range(80):
        many = conduct(many, np.array([0.001]), g)
    assert np.allclose(once, many, atol=1e-10)
    # each cell at its own Fourier step
    both = conduct(np.ones((2, 16)), np.array([0.0, 0.08]), g)
    assert np.allclose(both[0], 1.0) and np.allclose(both[1], once[0])


def test_flow_shares_follow_the_velocity_profile():
    g = layer_grid(16)
    # one viscosity: the parabola u ∝ ¼ − ξ²
    got = flow_shares(np.ones((1, 16)), g)[0]
    want = g.wt * (0.25 - g.xi**2)
    assert np.abs(got - want / want.sum()).max() < 5e-4
    assert got.sum() == pytest.approx(1.0)
    # the four layers at the wall solid: they stand still, the core carries everything
    inv = np.ones((1, 16))
    inv[0, 12:] = 0.0
    part = flow_shares(inv, g)[0]
    assert np.all(part[12:] == 0.0) and part.sum() == pytest.approx(1.0)
    # the layers still molten slip on a thinner channel: the centre's share goes up
    assert part[0] > got[0]
    # solid through: nothing flows
    assert np.all(flow_shares(np.zeros((1, 16)), g) == 0.0)


def _channel(n: int):
    a = np.arange(n - 1)
    return a, a + 1


def test_flow_keeps_a_channel_warmer_than_standing_melt():
    g = layer_grid(16)
    n, dt, alpha = 100, 0.01, 1.1e-7
    h = np.full(n, 2e-3)
    a, b = _channel(n)
    held = np.ones(n)
    source = np.zeros(n)
    source[0] = 0.5  # half a cell of fresh melt per step
    moved = np.full(n - 1, 0.5)
    filling = np.zeros(n, dtype=bool)
    shares = flow_shares(np.ones((n, 16)), g)
    theta = np.ones((n, 16))
    for _ in range(400):  # 4 s
        theta = conduct(theta, alpha * dt / h**2, g)
        theta = advect(theta, shares, held, a, b, moved, filling, g, source=source)
        assert theta.min() >= 0.0 and theta.max() <= 1.0 + 1e-12
    still = conduct(np.ones((1, 16)), np.array([alpha * 4.0 / h[0] ** 2]), g)[0]
    centre = theta[:, 0]
    assert np.all(np.diff(centre) <= 1e-12)  # colder away from the inlet
    assert centre[0] > 0.999  # the inlet stays at the melt temperature at its core
    assert centre[-1] > still[0] + 0.1  # 100 mm on, still well above the standing slab
    mean = theta @ g.wt / 0.5
    assert np.all(mean > float(still @ g.wt / 0.5))


def test_advection_mixes_and_never_leaves_the_range_it_mixes():
    g = layer_grid(8)
    rng = np.random.default_rng(3)
    n = 30
    a, b = _channel(n)
    theta = rng.uniform(0.2, 0.9, size=(n, 8))
    shares = flow_shares(rng.uniform(0.1, 1.0, size=(n, 8)), g)
    held = rng.uniform(0.5, 1.5, size=n)
    moved = rng.uniform(-50.0, 50.0, size=n - 1)  # far beyond one cell per step
    out = advect(theta, shares, held, a, b, moved, np.zeros(n, dtype=bool), g)
    assert out.min() >= theta.min() - 1e-12 and out.max() <= theta.max() + 1e-12
    # no flow: nothing changes
    same = advect(theta, shares, held, a, b, np.zeros(n - 1), np.zeros(n, dtype=bool), g)
    assert np.array_equal(same, theta)


def test_a_filling_cell_takes_the_flow_weighted_mean_on_every_layer():
    g = layer_grid(16)
    theta = np.zeros((3, 16))
    theta[0] = np.linspace(1.0, 0.2, 16)  # hot core, cold skin
    shares = flow_shares(np.ones((3, 16)), g)
    held = np.array([1.0, 0.0, 0.0])
    filling = np.array([False, True, True])
    out = advect(
        theta, shares, held, np.array([0, 1]), np.array([1, 2]), np.array([0.3, 0.0]), filling, g
    )
    bulk = float(shares[0] @ theta[0])
    assert bulk > float(theta[0] @ g.wt / 0.5)  # the flow favours the hot core
    assert np.allclose(out[1], bulk)
    assert np.array_equal(out[0], theta[0]) and np.array_equal(out[2], theta[2])
    # a cell that already holds melt mixes it with what comes in
    theta[1] = 0.4
    held[1] = 0.3
    out = advect(
        theta, shares, held, np.array([0, 1]), np.array([1, 2]), np.array([0.3, 0.0]), filling, g
    )
    assert np.allclose(out[1], (0.3 * 0.4 + 0.3 * bulk) / 0.6)


def test_a_solid_layer_of_a_full_cell_is_not_warmed_by_the_flow():
    g = layer_grid(16)
    theta = np.ones((2, 16))
    theta[1, 10:] = 0.1  # the receiver's skin
    inv = np.ones((2, 16))
    inv[1, 10:] = 0.0
    shares = flow_shares(inv, g)
    out = advect(
        theta,
        shares,
        np.ones(2),
        np.array([0]),
        np.array([1]),
        np.array([5.0]),
        np.zeros(2, dtype=bool),
        g,
    )
    assert np.allclose(out[1, 10:], 0.1)
    assert np.allclose(out[1, :10], 1.0)


def test_dissipation_is_the_work_of_the_pressure_split_by_resistance():
    # three cells in a row; the middle one is ten times stiffer than its neighbours
    a, b = np.array([0, 1]), np.array([1, 2])
    cond = np.array([1.0e-9, 1.0e-10, 1.0e-9])
    drop = np.array([-2.0e6, -1.0e6])
    heat = face_dissipation(a, b, cond[a], cond[b], drop, 0.01, 3)
    s_face = 2.0 * cond[a] * cond[b] / (cond[a] + cond[b])
    assert heat.sum() == pytest.approx(float((s_face * drop**2 * 0.01).sum()))
    # of each face's heat the stiff cell takes 10/11
    assert heat[0] == pytest.approx(s_face[0] * 4.0e12 * 0.01 / 11.0)
    assert heat[1] == pytest.approx((s_face * drop**2 * 0.01).sum() * 10.0 / 11.0)
    # a solid cell carries nothing and is not heated
    cond[1] = 0.0
    assert np.all(face_dissipation(a, b, cond[a], cond[b], drop, 0.01, 3) == 0.0)


def test_shear_heating_puts_the_heat_where_the_melt_is_sheared():
    g = layer_grid(16)
    inv = np.full((1, 16), 1.0 / 500.0)
    inv[0, 13:] = 0.0  # solid skin
    volume, capacity, span = np.array([4.0e-8]), 2.0e6, 200.0
    heat = np.array([0.4])  # J
    rise = shear_heating(heat, inv, volume, capacity, span, g)[0]
    # all of it arrives: Σ rise · layer volume · ρcp · ΔT
    assert float(rise @ (g.wt / 0.5)) * volume[0] * capacity * span == pytest.approx(0.4)
    assert np.all(rise[13:] == 0.0) and rise[0] < 1e-3 * rise[12]
    assert np.allclose(rise[:13] / rise[12], (g.xi[:13] / g.xi[12]) ** 2)
    # capped per step
    capped = shear_heating(heat * 1e4, inv, volume, capacity, span, g)
    assert capped.max() == pytest.approx(0.25)
    # nothing flows: nothing is heated
    assert np.all(shear_heating(heat, np.zeros((1, 16)), volume, capacity, span, g) == 0.0)


def test_carried_heat_does_not_depend_on_the_step():
    # a 1 mm channel of full cells with a steady flow, at about the rates of a
    # fill: one step passes 1.5 cells' worth of melt (2.25 on the centre layer).
    # Without the substeps the centre came out 0.54 at 20 mm on this step and
    # 0.90 on a short one.
    from core.layer_heat import carry

    g = layer_grid(16)
    alpha, h, dx, w, q = 1.1e-7, 1e-3, 1e-3, 20e-3, 1.2e-6
    n = 41
    held = np.full(n, dx * w * h)
    held[-1] = 1e3  # a sink
    fa = np.arange(n - 1)
    fb = fa + 1

    def steady(dt):
        th = np.ones((n, 16))
        t = 0.0
        while t < 2.0:
            sh = flow_shares(np.where(th > 0.3, 1.0, 0.0), g)
            src = np.zeros(n)
            src[0] = q * dt
            th, _ = carry(
                th, sh, held, fa, fb, np.full(n - 1, q * dt), np.zeros(n, bool),
                np.full(n, alpha * dt / h**2), g, source=src,
            )  # fmt: skip
            t += dt
        return th[[5, 10, 20, 35], 0]

    coarse = steady(0.025)
    fine = steady(0.0005)
    assert np.allclose(coarse, fine, atol=0.02)
    # the Graetz entry: the centre cools slowly over a few cm
    assert fine[0] > 0.98 and 0.85 < fine[2] < 0.95


def test_a_step_with_no_flow_but_the_gate_keeps_float_profiles():
    g = layer_grid(16)
    th = np.full((3, 16), 0.5)
    empty = np.zeros(0, dtype=np.intp)
    out = advect(
        th, flow_shares(np.ones((3, 16)), g), np.ones(3), empty, empty, np.zeros(0),
        np.zeros(3, bool), g, source=np.array([1.0, 0.0, 0.0]),
    )  # fmt: skip
    assert out.dtype == float
    assert out[0, 0] > 0.5 and np.allclose(out[1:], 0.5)
