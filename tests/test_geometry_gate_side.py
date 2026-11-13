"""The shapes that also decide the gate-side band of the rim: centre tab,
overlap tab, thick tab and direct gates (``core/fan_runner.py``, "The gate
side"). The product proper (body + rim on three sides) is the same for all."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core import (
    DIRECT_DEFAULTS,
    OVERLAP_TAB_DEFAULTS,
    RUNNER_SHAPES,
    TAB_DEFAULTS,
    THICK_TAB_DEFAULTS,
    FanRunnerPlateConfig,
    build_fan_runner_plate_geometry,
)

DX = 0.5
W, H, FRAME = 302.26, 220.6, 20.0
INNER_HALF = W / 2 - FRAME


def _build(**kw):
    return build_fan_runner_plate_geometry(FanRunnerPlateConfig(cell_size_mm=DX, **kw))


def _xy(g):
    """Cell centres: ``x`` from the sprue axis, ``y`` above the product's gate-side edge."""
    x0, y0 = g.display_origin_mm()
    ny, nx = g.shape
    x = (np.arange(nx) + 0.5) * g.cell_size_mm - x0
    y = (np.arange(ny) + 0.5) * g.cell_size_mm - y0
    return np.meshgrid(x, y)


G1 = _build()
X, Y = _xy(G1)
BAND = (Y > 0) & (Y < FRAME) & (np.abs(X) < W / 2)  # the gate-side band of the rim
THREE_SIDES = G1.product_mask & ~BAND  # body + rim on the other three sides


@pytest.mark.parametrize(
    "kw",
    [
        TAB_DEFAULTS,
        OVERLAP_TAB_DEFAULTS,
        THICK_TAB_DEFAULTS,
        DIRECT_DEFAULTS,
        {"runner_shape": "direct", "direct_layout": "corners4"},
        {"runner_shape": "direct", "direct_layout": "diagonal2"},
    ],
)
def test_the_product_proper_and_the_compression_zone_do_not_depend_on_the_gate_side(kw):
    g = _build(**kw)
    assert g.shape == G1.shape
    assert g.display_origin_mm() == G1.display_origin_mm()
    assert np.array_equal(g.mask & THREE_SIDES, THREE_SIDES)
    assert np.array_equal(g.thickness_mm[THREE_SIDES], G1.thickness_mm[THREE_SIDES])
    # only the body is squeezed, whatever the gate does to the band
    assert np.array_equal(g.compression_mask, G1.compression_mask)
    assert not (g.compression_mask & BAND).any()


def test_the_centre_tab_leaves_the_band_a_rim_and_ramps_from_the_edge_to_the_gate():
    g = _build(**TAB_DEFAULTS)
    assert g.label == "tab_runner_plate"
    assert np.array_equal(g.mask[BAND], G1.mask[BAND])
    assert np.all(g.thickness_mm[BAND] == 1.0)
    runner = g.mask & (Y < 0)
    d = -Y
    # 30 wide on the edge line, the round end's 24 on the axis line (30 below)
    top = runner & (d < DX)
    assert np.abs(X[top]).max() == pytest.approx(15.0, abs=DX)
    at_axis = runner & (np.abs(d - 30.0) < DX)
    assert np.abs(X[at_axis]).max() == pytest.approx(12.0, abs=DX)
    # t1.0 at the edge, t2.0 from 18 below it on, a straight ramp between
    ramp = runner & (d < 18.0)
    assert g.thickness_mm[ramp] == pytest.approx(1.0 + d[ramp] / 18.0)
    assert np.all(g.thickness_mm[runner & (d >= 18.0)] == 2.0)
    # the gate is the orifice on the axis, in the round end
    gy, gx = np.array(g.gates).T
    assert np.hypot(X[gy, gx], Y[gy, gx] + 30.0).max() <= 1.5


def test_the_overlap_tab_thickens_the_band_over_its_own_width_only():
    g = _build(**OVERLAP_TAB_DEFAULTS)
    assert g.label == "overlap_tab_plate"
    assert np.array_equal(g.mask[BAND], G1.mask[BAND])
    lap = BAND & (np.abs(X) < 15.0)
    assert np.all(g.thickness_mm[lap] == 3.5)
    assert np.all(g.thickness_mm[BAND & ~lap] == 1.0)
    runner = g.mask & (Y < 0)
    assert np.all(g.thickness_mm[runner] == 3.5)
    # the 2026-11-11 first version: t2.0 outside the product, 1.0 + 2.0 under the rim
    g2 = _build(runner_shape="overlap_tab", tab_thk_mm=2.0, lap_thk_mm=3.0)
    assert np.all(g2.thickness_mm[lap] == 3.0) and np.all(g2.thickness_mm[runner] == 2.0)
    assert g2.volume_cm3() - _build(
        runner_shape="overlap_tab", tab_thk_mm=2.0, lap_thk_mm=1.0
    ).volume_cm3() == (pytest.approx(30.0 * 20.0 * 2.0 / 1000.0, rel=0.02))


def test_the_thick_tab_cuts_the_band_between_the_tab_and_the_corners():
    g = _build(**THICK_TAB_DEFAULTS)
    assert g.label == "thick_tab_plate"
    ax = np.abs(X)
    tab = BAND & (ax < 50.0)
    cut = BAND & (ax >= 50.0) & (ax < INNER_HALF - 30.0)
    corner = BAND & (ax >= INNER_HALF - 30.0)
    assert np.all(g.mask[tab]) and np.all(g.thickness_mm[tab] == 4.0)
    assert not g.mask[cut].any() and np.all(g.thickness_mm[cut] == 0.0)
    assert np.all(g.mask[corner]) and np.all(g.thickness_mm[corner] == 1.0)
    # the cut cells are no longer product, and the display origin stays on the edge
    assert not (g.product_mask & cut).any()
    # the runner: a triangle 100 wide on the edge, apex on the axis, t4.0, and the disc
    runner = g.mask & (Y < 0)
    assert np.all(g.thickness_mm[runner] == 4.0)
    assert np.abs(X[runner & (-Y < DX)]).max() == pytest.approx(50.0, abs=DX)
    tri = runner & (np.hypot(X, Y + 30.0) > 12.0)
    assert np.all(np.abs(X[tri]) <= 50.0 * (1.0 + Y[tri] / 30.0) + 1e-9)


def test_a_thick_tab_as_wide_as_the_gap_cuts_nothing():
    g = _build(runner_shape="thick_tab", thick_tab_w_mm=220.0, corner_keep_mm=30.0)
    assert np.array_equal(g.mask[BAND], G1.mask[BAND])
    assert set(np.unique(g.thickness_mm[BAND])) == {1.0, 4.0}


@pytest.mark.parametrize(
    ("layout", "points"),
    [
        ("center", [(0.0, H / 2)]),
        ("corners4", [(-1, 35.0), (1, 35.0), (-1, H - 35.0), (1, H - 35.0)]),
        ("diagonal2", [(-1, 35.0), (1, H - 35.0)]),
    ],
)
def test_direct_gates_sit_on_the_body_and_there_is_no_runner(layout, points):
    g = _build(runner_shape="direct", direct_layout=layout)
    assert g.label == "direct_gate_plate"
    assert np.array_equal(g.mask, G1.product_mask)
    assert np.array_equal(g.thickness_mm[g.mask], G1.thickness_mm[g.mask])
    pts = [(sx * (INNER_HALF - 15.0) if layout != "center" else sx, y) for sx, y in points]
    gy, gx = np.array(g.gates).T
    assert np.all(g.compression_mask[gy, gx])  # every gate cell is on the body
    # each gate cell lies in one point's Φ3 disc, and every point has cells
    dist = np.array([np.hypot(X[gy, gx] - px, Y[gy, gx] - py) for px, py in pts])
    assert np.all(dist.min(axis=0) <= 1.5)
    assert set(dist.argmin(axis=0)) == set(range(len(pts)))
    # one true-scale marker for one gate; several gates draw their cells
    if layout == "center":
        x0, y0 = g.display_origin_mm()
        mx, my, mr = g.valve_marker_mm
        assert (mx - x0, my - y0, mr) == pytest.approx((0.0, H / 2, 1.5))
    else:
        assert g.valve_marker_mm is None


def test_a_corner_gate_as_wide_as_its_inset_allows_stays_on_the_body():
    g = _build(
        runner_shape="direct", direct_layout="corners4", gate_d_mm=30.0, direct_inset_mm=15.0
    )
    gy, gx = np.array(g.gates).T
    assert np.all(g.compression_mask[gy, gx])
    assert np.all(g.thickness_mm[gy, gx] == 4.0)


def test_a_body_thinner_than_the_mesh_is_a_mesh_error_not_a_gate_outside_the_cavity():
    # 1 mm of body between the rims: no 4 mm cell centre lands on it (Codex P2 on PR #27)
    cfg = FanRunnerPlateConfig(
        runner_shape="direct", plate_h_mm=41.0, frame_w_mm=20.0, gate_d_mm=0.5, cell_size_mm=4.0
    )
    with pytest.raises(ValueError, match="rasterises the body away"):
        build_fan_runner_plate_geometry(cfg)
    # the same plate on a mesh that resolves the body is fine
    g = build_fan_runner_plate_geometry(dataclasses.replace(cfg, cell_size_mm=0.5))
    gy, gx = np.array(g.gates).T
    assert np.all(g.mask[gy, gx] & g.compression_mask[gy, gx])


def test_a_mesh_coarser_than_the_orifice_still_gives_every_direct_gate_a_cell():
    g = build_fan_runner_plate_geometry(
        FanRunnerPlateConfig(cell_size_mm=4.0, runner_shape="direct", direct_layout="corners4")
    )
    assert len(g.gates) == 4 and len(set(g.gates)) == 4
    assert g.valve_marker_mm is None


@pytest.mark.parametrize(
    "shape", [s for s in RUNNER_SHAPES if s not in ("triangle", "pentagon", "triangle_arms")]
)
def test_the_fan_runners_fields_are_not_validated_for_the_other_shapes(shape):
    # a runner 60 long is too long for the default fan (its flanks stop short of
    # the round end), which is none of a tab's business
    with pytest.raises(ValueError, match="flanks"):
        _build(runner_len_mm=60.0)
    _build(runner_shape=shape, runner_len_mm=60.0, runner_ramp_end_mm=500.0, fan_flank_deg=95.0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"runner_shape": "tab", "balancer_on": True}, "balancer_on"),
        ({"runner_shape": "tab", "tab_w_mm": 0.0}, "tab_w_mm"),
        ({"runner_shape": "tab", "tab_end_thk_mm": 0.0}, "tab_end_thk_mm"),
        ({"runner_shape": "tab", "tab_ramp_len_mm": 19.0}, "tab_ramp_len_mm"),
        ({"runner_shape": "tab", "tab_w_mm": 400.0}, "tab_w_mm"),
        ({"runner_shape": "overlap_tab", "lap_thk_mm": 0.5}, "lap_thk_mm"),
        ({"runner_shape": "overlap_tab", "tab_thk_mm": -1.0}, "tab_thk_mm"),
        ({"runner_shape": "thick_tab", "thick_tab_w_mm": 0.0}, "thick_tab_w_mm"),
        ({"runner_shape": "thick_tab", "thick_tab_thk_mm": 0.0}, "thick_tab_thk_mm"),
        ({"runner_shape": "thick_tab", "corner_keep_mm": -1.0}, "corner_keep_mm"),
        ({"runner_shape": "thick_tab", "gate_d_mm": 30.0}, "gate_d_mm"),
        ({"runner_shape": "direct", "direct_layout": "ring"}, "direct_layout"),
        (
            {"runner_shape": "direct", "direct_layout": "corners4", "direct_inset_mm": 95.0},
            "direct_inset_mm",
        ),
        (
            {"runner_shape": "direct", "direct_layout": "corners4", "direct_inset_mm": -1.0},
            "direct_inset_mm",
        ),
        ({"runner_shape": "direct", "gate_d_mm": 200.0}, "gate_d_mm"),
        # a corner gate wider than its inset would reach onto the rim (Codex P2 on PR #27)
        (
            {"runner_shape": "direct", "direct_layout": "diagonal2", "gate_d_mm": 50.0},
            "direct_inset_mm",
        ),
    ],
)
def test_bad_gate_side_parameters_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        _build(**kw)


def test_the_overlap_tab_ignores_the_centre_tabs_ramp_fields():
    a = _build(**OVERLAP_TAB_DEFAULTS)
    b = _build(**OVERLAP_TAB_DEFAULTS, tab_end_thk_mm=0.3, tab_ramp_len_mm=5.0)
    assert np.array_equal(a.thickness_mm, b.thickness_mm)
