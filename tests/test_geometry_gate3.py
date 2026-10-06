"""Gate 3 (Gate 1 with a thick arm along each flank) of the fan-runner
builder (``docs/spec.md``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import GATE3_DEFAULTS, FanRunnerPlateConfig, build_fan_runner_plate_geometry

DX = 0.5


def _build(**kw):
    return build_fan_runner_plate_geometry(FanRunnerPlateConfig(cell_size_mm=DX, **kw))


def _grid(g):
    x0, y0 = g.display_origin_mm()
    ny, nx = g.shape
    x = (np.arange(nx) + 0.5) * g.cell_size_mm - x0  # from the sprue axis
    d = y0 - (np.arange(ny) + 0.5) * g.cell_size_mm  # depth below the product edge
    return np.meshgrid(x, d)


def _dist_to_flank(cfg: FanRunnerPlateConfig, x, d):
    """Independent oracle: distance from (|x|, d) to the flank line through
    the edge corner (W/2, 0) and the apex (0, apex_depth), by the 2-D cross
    product (not the builder's sin/cos form)."""
    p = np.array([0.5 * cfg.runner_w_mm, 0.0])
    q = np.array([0.0, 0.5 * cfg.runner_w_mm * math.tan(math.radians(cfg.fan_flank_deg))])
    u = q - p
    rx, rd = np.abs(x) - p[0], d - p[1]
    return np.abs(u[0] * rd - u[1] * rx) / np.hypot(*u)


def _runner(g):
    return g.mask & ~g.product_mask


def test_gate3_keeps_gate1_silhouette_and_product():
    g1, g3 = _build(), _build(**GATE3_DEFAULTS)
    assert np.array_equal(g1.mask, g3.mask)
    assert np.array_equal(g1.product_mask, g3.product_mask)
    assert np.array_equal(g1.compression_mask, g3.compression_mask)
    assert g1.gates == g3.gates
    # only runner cells may change, and only deeper
    diff = g3.thickness_mm - g1.thickness_mm
    assert np.all(diff[~_runner(g1)] == 0.0)
    assert diff.min() >= 0.0 and diff.max() > 0.0
    assert g3.label == "arms_runner_plate"


def test_arm_fields_do_nothing_on_gate1_and_gate2():
    for shape in (
        {},
        {"runner_shape": "pentagon", "runner_ramp_end_mm": None, "runner_end_thk_mm": 2.5},
    ):
        a = _build(**shape)
        b = _build(**shape, arm_w_mm=20.0, arm_thk_mm=5.0)
        assert np.array_equal(a.thickness_mm, b.thickness_mm)


@pytest.mark.parametrize(
    "arm_w, arm_thk, flank",
    # the 30° case tells "square to the flank" from "measured vertically":
    # 10 / cos 30° − 10 = 1.5 mm, well past the one-cell margin below
    [(9.0, 3.5, 14.0), (8.0, 3.5, 14.0), (10.0, 4.0, 14.0), (4.0, 2.0, 14.0), (10.0, 3.5, 30.0)],
)
def test_arm_thickness_matches_an_independent_strip(arm_w, arm_thk, flank):
    cfg = FanRunnerPlateConfig(
        **GATE3_DEFAULTS, cell_size_mm=DX, arm_w_mm=arm_w, arm_thk_mm=arm_thk, fan_flank_deg=flank
    )
    g1 = _build(fan_flank_deg=flank)
    g3 = build_fan_runner_plate_geometry(cfg)
    x, d = _grid(g3)
    runner = _runner(g3)
    in_end = np.hypot(x, d - cfg.runner_len_mm) <= cfg.runner_end_d_mm / 2
    dist = _dist_to_flank(cfg, x, d)
    band = d <= cfg.runner_edge_flat_mm
    # a cell-wide margin around the strip's inner border and the band's end
    clear_in = runner & ~in_end & (d > cfg.runner_edge_flat_mm + DX) & (dist < arm_w - DX)
    clear_out = runner & ~in_end & (dist > arm_w + DX)
    assert clear_in.sum() > 100
    np.testing.assert_allclose(
        g3.thickness_mm[clear_in], np.maximum(g1.thickness_mm[clear_in], arm_thk)
    )
    np.testing.assert_array_equal(g3.thickness_mm[clear_out], g1.thickness_mm[clear_out])
    # the edge band (the gate's cut face) and the round end keep their thickness
    np.testing.assert_array_equal(g3.thickness_mm[runner & band], g1.thickness_mm[runner & band])
    np.testing.assert_array_equal(
        g3.thickness_mm[runner & in_end], g1.thickness_mm[runner & in_end]
    )


def test_arms_are_mirror_symmetric():
    g3 = _build(**GATE3_DEFAULTS)
    x, d = _grid(g3)
    runner = _runner(g3)
    rows = np.where(runner.any(axis=1))[0]
    t = np.where(runner, g3.thickness_mm, 0.0)[rows]
    assert np.array_equal(t, t[:, ::-1])


def test_wider_or_thicker_arms_hold_more_runner_volume():
    def vol(**kw):
        g = _build(**GATE3_DEFAULTS, **kw)
        return float(g.thickness_mm[_runner(g)].sum()) * DX * DX

    v1 = vol()
    assert vol(arm_w_mm=10.0) > v1 > vol(arm_w_mm=8.0)
    assert vol(arm_thk_mm=4.0) > v1 > vol(arm_thk_mm=3.0)
    g1 = _build()
    assert v1 > float(g1.thickness_mm[_runner(g1)].sum()) * DX * DX


def test_balancer_still_cuts_through_gate3():
    plain = _build(**GATE3_DEFAULTS)
    cut = _build(**GATE3_DEFAULTS, balancer_on=True)
    assert np.array_equal(plain.mask, cut.mask)
    assert cut.thickness_mm.sum() < plain.thickness_mm.sum()


def test_balancer_cuts_the_arms_where_they_overlap():
    """The balancer comes after the arms: where the two overlap the cut wins
    (arms applied after it would fill the cut back in; @claude on PR #14).
    A full-width balancer (w 300, h 15) reaches the arms near the edge."""
    kw = dict(GATE3_DEFAULTS, balancer_on=True, balancer_w_mm=300.0, balancer_h_mm=15.0)
    cfg = FanRunnerPlateConfig(**kw, cell_size_mm=DX)
    g = build_fan_runner_plate_geometry(cfg)
    x, d = _grid(g)
    runner = _runner(g)
    dist = _dist_to_flank(cfg, x, d)
    in_arm = runner & (d > cfg.runner_edge_flat_mm + DX) & (dist < cfg.arm_w_mm - DX)
    in_bal = (d < cfg.balancer_h_mm - DX) & (
        np.abs(x) < 0.5 * cfg.balancer_w_mm * (1 - d / cfg.balancer_h_mm) - DX
    )
    both = in_arm & in_bal
    assert both.sum() > 50
    np.testing.assert_array_equal(g.thickness_mm[both], cfg.balancer_thk_mm)


def test_arms_start_right_below_the_edge_when_there_is_no_band():
    cfg = FanRunnerPlateConfig(
        **GATE3_DEFAULTS, cell_size_mm=DX, runner_edge_flat_mm=0.0, runner_ramp_end_mm=20.0
    )
    g = build_fan_runner_plate_geometry(cfg)
    x, d = _grid(g)
    dist = _dist_to_flank(cfg, x, d)
    first_row = _runner(g) & (d > 0) & (d < DX) & (dist < cfg.arm_w_mm - DX)
    assert first_row.sum() > 10
    np.testing.assert_array_equal(g.thickness_mm[first_row], cfg.arm_thk_mm)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"arm_w_mm": 0.0}, "arm_w_mm must be positive"),
        ({"arm_thk_mm": -1.0}, "arm_thk_mm must be positive"),
        ({"arm_w_mm": 36.3}, "sin"),
        ({"arm_thk_mm": 1.0}, "arm_thk_mm"),
        ({"arm_thk_mm": 0.8}, "only deepens"),
    ],
)
def test_validate_rejects_bad_arms(overrides, match):
    with pytest.raises(ValueError, match=match):
        FanRunnerPlateConfig(**GATE3_DEFAULTS, **overrides).validate()


def test_arm_width_bound_follows_the_flank():
    cfg = FanRunnerPlateConfig(**GATE3_DEFAULTS)
    assert cfg.arm_w_sup_mm == pytest.approx(150.0 * math.sin(math.radians(14.0)))
    FanRunnerPlateConfig(**GATE3_DEFAULTS, arm_w_mm=36.0).validate()
    steep = FanRunnerPlateConfig(**GATE3_DEFAULTS, fan_flank_deg=30.0, arm_w_mm=60.0)
    steep.validate()  # 150 · sin 30° = 75


def test_gate3_still_needs_the_triangle_to_reach_the_round_end():
    with pytest.raises(ValueError, match="triangle must reach the disc"):
        FanRunnerPlateConfig(**GATE3_DEFAULTS, fan_flank_deg=5.0).validate()
