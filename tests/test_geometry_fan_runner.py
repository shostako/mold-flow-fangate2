"""Tests for the fan-runner plate builder (``core/fan_runner.py``).

The drawing (``docs/spec.md``): rim 20 × t1.0 around a 262.26 × 180.6 × t4.0
body, a 300-wide fan runner with 14° flanks ending in an R12 round end
30 mm below the product edge, thickness 1.0 → 2.5 by depth, a φ3 gate on
the axis. Only the body compresses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry

D = FanRunnerPlateConfig()


def _row_at_depth(cfg: FanRunnerPlateConfig, depth_mm: float) -> int:
    """Grid row whose cell centre is closest to ``depth_mm`` below the edge."""
    return int(round((cfg.y_edge_mm - depth_mm) / cfg.cell_size_mm - 0.5))


def _col_at_x(cfg: FanRunnerPlateConfig, x_mm: float) -> int:
    """Grid column whose cell centre is closest to ``x_mm`` off the axis."""
    return int(round((cfg.axis_x_mm + x_mm) / cfg.cell_size_mm - 0.5))


# ---------------------------------------------------------------------------
# defaults = the drawing
# ---------------------------------------------------------------------------


def test_defaults_are_the_drawing():
    assert (D.plate_w_mm, D.plate_h_mm, D.frame_w_mm) == (302.26, 220.6, 20.0)
    assert (D.frame_thk_mm, D.inner_thk_mm) == (1.0, 4.0)
    assert (D.runner_w_mm, D.runner_len_mm, D.runner_end_d_mm) == (300.0, 30.0, 24.0)
    assert (D.runner_edge_thk_mm, D.runner_edge_flat_mm) == (1.0, 1.0)
    assert (D.runner_ramp_end_mm, D.runner_thk_mm) == (20.0, 2.5)
    assert D.gate_d_mm == 3.0
    assert not D.balancer_on


def test_the_flanks_cut_the_round_end_like_the_drawing():
    """On the drawing the flanks meet the R12 circle at ±11.0 / depth 34.6 and
    the extended apex (37.4) lies inside the disc, so triangle ∪ disc is the
    whole silhouette without clipping."""
    assert D.apex_depth_mm == pytest.approx(37.4, abs=0.05)
    # the flank at depth 34.64 is 11.04 off the axis (drawing: 11.0)
    half_w = 0.5 * D.runner_w_mm - 34.64 / math.tan(math.radians(D.fan_flank_deg))
    assert half_w == pytest.approx(11.04, abs=0.05)
    # and that point sits on the R12 circle
    r = math.hypot(half_w, 34.64 - D.runner_len_mm)
    assert r == pytest.approx(D.runner_end_d_mm / 2.0, abs=0.05)
    # the apex is inside the disc
    assert D.apex_depth_mm - D.runner_len_mm < D.runner_end_d_mm / 2.0


def test_product_and_body_volumes_match_the_drawing():
    """Rim 19 314.4 mm³ + body 189 456.6 mm³ = 208.77 cm³ (spec). The plate
    is not a whole number of cells (302.26 × 220.6), so the 1 mm raster is
    only expected within 0.5 %."""
    g = build_fan_runner_plate_geometry(D)
    v_product = float(g.thickness_mm[g.product_mask].sum()) * g.cell_size_mm**2 / 1000.0
    v_body = float(g.thickness_mm[g.compression_mask].sum()) * g.cell_size_mm**2 / 1000.0
    assert v_product == pytest.approx(208.771, rel=5e-3)
    assert v_body == pytest.approx(189.457, rel=5e-3)
    # finer cells converge toward the analytic value
    g2 = build_fan_runner_plate_geometry(FanRunnerPlateConfig(cell_size_mm=0.5))
    v2 = float(g2.thickness_mm[g2.product_mask].sum()) * g2.cell_size_mm**2 / 1000.0
    assert abs(v2 - 208.771) < abs(v_product - 208.771)


# ---------------------------------------------------------------------------
# silhouette
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth_mm, expected_w_mm",
    [
        (0.5, 296.0),  # 300 − 2 · 0.5 / tan 14°
        (10.5, 215.8),
        (20.5, 135.6),  # the ramp-end line: 139.6 wide at depth 20 on the drawing
        (29.5, 63.3),  # triangle still wider than the disc at the axis line (24)
        (36.5, 20.1),  # inside the disc: 2 · sqrt(12² − 6.5²)
        (41.5, 6.9),
    ],
)
def test_runner_width_by_depth(depth_mm, expected_w_mm):
    g = build_fan_runner_plate_geometry(D)
    row = _row_at_depth(D, depth_mm)
    assert g.mask[row].sum() == pytest.approx(expected_w_mm, abs=1.5)


def test_runner_is_centred_on_the_axis_and_bounded_by_the_round_end():
    g = build_fan_runner_plate_geometry(D)
    runner = g.mask & ~g.product_mask
    ys, xs = np.where(runner)
    xc = (xs + 0.5) * D.cell_size_mm
    yc = (ys + 0.5) * D.cell_size_mm
    assert xc.mean() == pytest.approx(D.axis_x_mm, abs=0.01)
    # lowest runner cell is the bottom of the R12 disc
    assert D.y_edge_mm - yc.min() == pytest.approx(
        D.runner_len_mm + D.runner_end_d_mm / 2.0, abs=0.6
    )
    # nothing above the product edge is runner
    assert yc.max() < D.y_edge_mm


def test_plate_is_a_full_rectangle_above_the_edge():
    g = build_fan_runner_plate_geometry(D)
    rows = np.where(g.product_mask.any(axis=1))[0]
    assert rows.min() * D.cell_size_mm == pytest.approx(D.y_edge_mm, abs=D.cell_size_mm)
    assert (rows.max() + 1) * D.cell_size_mm == pytest.approx(D.y_plate_top_mm, abs=D.cell_size_mm)
    widths = g.product_mask[rows].sum(axis=1)
    assert widths.min() == widths.max()
    assert widths[0] * D.cell_size_mm == pytest.approx(D.plate_w_mm, abs=D.cell_size_mm)


# ---------------------------------------------------------------------------
# thickness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth_mm, expected_thk",
    [
        (0.5, 1.0),  # edge band = rim thickness
        (1.5, 1.0 + 1.5 * 0.5 / 19.0),
        (10.5, 1.75),
        (19.5, 1.0 + 1.5 * 18.5 / 19.0),
        (20.5, 2.5),
        (29.5, 2.5),
        (41.5, 2.5),  # round end, no well pocket
    ],
)
def test_runner_thickness_is_a_function_of_depth_only(depth_mm, expected_thk):
    g = build_fan_runner_plate_geometry(D)
    row = _row_at_depth(D, depth_mm)
    cells = g.thickness_mm[row][g.mask[row]]
    assert cells.size > 0
    np.testing.assert_allclose(cells, expected_thk, atol=1e-9)


def test_plate_thickness_rim_vs_body():
    g = build_fan_runner_plate_geometry(D)
    row_mid = _row_at_depth(D, -D.plate_h_mm / 2.0)
    # rim cells (within 20 of the outer edge) are t1, body cells t4
    assert g.thickness_mm[row_mid, _col_at_x(D, -D.plate_w_mm / 2.0 + 5.0)] == 1.0
    assert g.thickness_mm[row_mid, _col_at_x(D, 0.0)] == 4.0
    assert g.thickness_mm[row_mid, _col_at_x(D, D.plate_w_mm / 2.0 - 5.0)] == 1.0
    assert g.thickness_mm[_row_at_depth(D, -5.0), _col_at_x(D, 0.0)] == 1.0
    assert g.thickness_mm[_row_at_depth(D, -D.plate_h_mm + 5.0), _col_at_x(D, 0.0)] == 1.0
    assert set(np.unique(g.thickness_mm[g.product_mask])) == {1.0, 4.0}


def test_body_rectangle_dimensions():
    g = build_fan_runner_plate_geometry(D)
    rows = np.where(g.compression_mask.any(axis=1))[0]
    cols = np.where(g.compression_mask.any(axis=0))[0]
    assert (rows.max() - rows.min() + 1) * D.cell_size_mm == pytest.approx(180.6, abs=1.0)
    assert (cols.max() - cols.min() + 1) * D.cell_size_mm == pytest.approx(262.26, abs=1.0)


# ---------------------------------------------------------------------------
# compression zone: the body only
# ---------------------------------------------------------------------------


def test_compression_mask_is_the_t4_body_only():
    g = build_fan_runner_plate_geometry(D)
    cm = g.compression_mask
    assert cm is not None
    assert cm.any()
    assert (g.thickness_mm[cm] == D.inner_thk_mm).all()
    # every t4 product cell compresses, no t1 cell does, no runner cell does
    assert (cm == (g.product_mask & (g.thickness_mm == D.inner_thk_mm))).all()
    assert not (cm & ~g.product_mask).any()
    assert g.compression_area_mm2() == pytest.approx(262.26 * 180.6, rel=5e-3)


def test_product_mask_is_rim_plus_body_and_sets_the_display_origin():
    g = build_fan_runner_plate_geometry(D)
    assert (g.product_mask & ~g.mask).sum() == 0
    assert (g.compression_mask & ~g.product_mask).sum() == 0
    assert g.product_mask.sum() > g.compression_mask.sum()
    x0, y0 = g.display_origin_mm()
    assert x0 == pytest.approx(D.axis_x_mm)
    assert y0 == pytest.approx(D.y_edge_mm)


# ---------------------------------------------------------------------------
# injection point
# ---------------------------------------------------------------------------


def test_gate_cells_fill_the_orifice_disc_on_the_axis():
    g = build_fan_runner_plate_geometry(D)
    assert g.gates
    for iy, ix in g.gates:
        x = (ix + 0.5) * D.cell_size_mm - D.axis_x_mm
        y = (iy + 0.5) * D.cell_size_mm - D.y_axis_mm
        assert math.hypot(x, y) <= D.gate_d_mm / 2.0 + 1e-9
        assert g.mask[iy, ix]
    assert g.valve_marker_mm == pytest.approx((D.axis_x_mm, D.y_axis_mm, D.gate_d_mm / 2.0))
    assert g.valve_axis_x_mm == pytest.approx(D.axis_x_mm)


def test_coarse_mesh_snaps_the_gate_to_the_nearest_cavity_cell():
    cfg = FanRunnerPlateConfig(cell_size_mm=4.0)
    g = build_fan_runner_plate_geometry(cfg)
    assert len(g.gates) >= 1
    for iy, ix in g.gates:
        assert g.mask[iy, ix]
    if g.valve_marker_mm is None:
        # snapped: the single gate cell is the cavity cell nearest the axis
        (iy, ix), *rest = g.gates
        assert not rest
        assert abs((ix + 0.5) * 4.0 - cfg.axis_x_mm) <= 4.0
        assert abs((iy + 0.5) * 4.0 - cfg.y_axis_mm) <= 4.0


# ---------------------------------------------------------------------------
# grid alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad_mm", [5.0, 5.3, 4.75])
@pytest.mark.parametrize("cell", [1.0, 0.5])
def test_product_edge_sits_on_a_cell_edge(pad_mm, cell):
    cfg = FanRunnerPlateConfig(pad_mm=pad_mm, cell_size_mm=cell)
    g = build_fan_runner_plate_geometry(cfg)
    assert cfg.y_edge_mm / cell == pytest.approx(round(cfg.y_edge_mm / cell), abs=1e-9)
    first_product_row = int(np.where(g.product_mask.any(axis=1))[0].min())
    assert first_product_row * cell == pytest.approx(cfg.y_edge_mm, abs=1e-9)
    assert 0.0 <= cfg.grid_shift_mm < cell
    assert 0.0 <= cfg.grid_shift_x_mm < cell
    # the product keeps round(plate_w / cell) columns whatever the pad
    cols = np.where(g.product_mask.any(axis=0))[0]
    assert cols.size == round(cfg.plate_w_mm / cell)


@pytest.mark.parametrize("pad_mm", [5.0, 5.3])
@pytest.mark.parametrize("plate_w_mm", [302.26, 315.0, 300.0])
def test_raster_is_mirror_symmetric_about_the_axis(pad_mm, plate_w_mm):
    """302.26 wide puts the axis 0.13 mm off the cell grid; the x lift puts
    it back on a cell edge (even column count) or centre (odd) so every row
    of the mask and thickness map mirrors about the axis."""
    cfg = FanRunnerPlateConfig(pad_mm=pad_mm, plate_w_mm=plate_w_mm, balancer_on=True)
    g = build_fan_runner_plate_geometry(cfg)
    js = np.arange(g.nx)
    jm = np.rint(2.0 * cfg.axis_x_mm / cfg.cell_size_mm - 1.0 - js).astype(int)
    ok = (jm >= 0) & (jm < g.nx)
    # columns without a mirror partner must be empty padding
    assert not g.mask[:, ~ok].any()
    assert (g.mask[:, js[ok]] == g.mask[:, jm[ok]]).all()
    assert (g.thickness_mm[:, js[ok]] == g.thickness_mm[:, jm[ok]]).all()
    assert (g.compression_mask[:, js[ok]] == g.compression_mask[:, jm[ok]]).all()


def test_rendered_plate_is_independent_of_the_round_end_diameter():
    """The lift ``grid_shift_mm`` exists so an odd R does not move the plate
    by half a cell (fangate PR #11)."""
    a = build_fan_runner_plate_geometry(FanRunnerPlateConfig(runner_end_d_mm=24.0))
    b = build_fan_runner_plate_geometry(FanRunnerPlateConfig(runner_end_d_mm=23.0))
    ra = np.where(a.product_mask.any(axis=1))[0]
    rb = np.where(b.product_mask.any(axis=1))[0]
    assert ra.size == rb.size
    assert a.product_mask[ra].sum() == b.product_mask[rb].sum()
    assert a.compression_mask.sum() == b.compression_mask.sum()


# ---------------------------------------------------------------------------
# balancer
# ---------------------------------------------------------------------------


def test_balancer_thins_the_runner_inside_the_triangle_only():
    base = build_fan_runner_plate_geometry(D)
    cfg = FanRunnerPlateConfig(
        balancer_on=True, balancer_w_mm=100.0, balancer_h_mm=15.0, balancer_thk_mm=1.2
    )
    g = build_fan_runner_plate_geometry(cfg)
    assert (g.mask == base.mask).all()
    assert (g.compression_mask == base.compression_mask).all()
    changed = g.thickness_mm != base.thickness_mm
    assert changed.any()
    # thinned cells are runner cells with thickness exactly the balancer value
    assert not (changed & base.product_mask).any()
    np.testing.assert_allclose(g.thickness_mm[changed], 1.2)
    # inside the ▽ the runner is at most the balancer thickness, never raised
    row_edge = _row_at_depth(cfg, 0.5)  # edge band t1.0 < 1.2 → untouched
    assert g.thickness_mm[row_edge, _col_at_x(cfg, 0.0)] == 1.0
    row_deep = _row_at_depth(cfg, 10.5)  # ramp 1.75 → cut to 1.2 at the centre
    assert g.thickness_mm[row_deep, _col_at_x(cfg, 0.0)] == 1.2
    # base half-width 50 on the edge line, apex 15 deep: at depth 10.5 the
    # half-width is 50 · (15 − 10.5) / 15 = 15
    assert g.thickness_mm[row_deep, _col_at_x(cfg, 14.0)] == 1.2
    assert g.thickness_mm[row_deep, _col_at_x(cfg, 16.0)] == pytest.approx(1.75)
    # below the apex nothing changes
    row_below = _row_at_depth(cfg, 16.5)
    assert (g.thickness_mm[row_below] == base.thickness_mm[row_below]).all()
    assert (g.thickness_mm <= base.thickness_mm).all()


def test_balancer_off_leaves_the_geometry_untouched_by_its_parameters():
    a = build_fan_runner_plate_geometry(FanRunnerPlateConfig(balancer_on=False, balancer_w_mm=50.0))
    b = build_fan_runner_plate_geometry(
        FanRunnerPlateConfig(balancer_on=False, balancer_w_mm=200.0)
    )
    assert (a.thickness_mm == b.thickness_mm).all()


def test_balancer_limits_are_the_validate_bounds():
    w_max, h_max, thk_sup = D.balancer_limits_mm
    assert w_max == D.runner_w_mm
    assert h_max == D.runner_len_mm - D.runner_end_d_mm / 2.0
    assert thk_sup == D.runner_thk_mm
    FanRunnerPlateConfig(
        balancer_on=True, balancer_w_mm=w_max, balancer_h_mm=h_max, balancer_thk_mm=2.4
    ).validate()
    with pytest.raises(ValueError, match="balancer_w_mm"):
        FanRunnerPlateConfig(balancer_on=True, balancer_w_mm=w_max + 1).validate()
    with pytest.raises(ValueError, match="balancer_h_mm"):
        FanRunnerPlateConfig(balancer_on=True, balancer_h_mm=h_max + 1).validate()
    with pytest.raises(ValueError, match="balancer_thk_mm"):
        FanRunnerPlateConfig(balancer_on=True, balancer_thk_mm=thk_sup).validate()
    with pytest.raises(ValueError, match="balancer_h_mm"):
        FanRunnerPlateConfig(balancer_on=True, balancer_h_mm=0.0).validate()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"plate_w_mm": 0.0}, "plate_w_mm"),
        ({"frame_w_mm": -1.0}, "frame_w_mm"),
        ({"frame_w_mm": 111.0}, "frame_w_mm"),
        ({"fan_flank_deg": 0.0}, "fan_flank_deg"),
        ({"fan_flank_deg": 90.0}, "fan_flank_deg"),
        ({"runner_w_mm": 310.0}, "runner_w_mm"),
        ({"runner_edge_flat_mm": 25.0}, "runner_edge_flat_mm"),
        ({"runner_end_d_mm": 70.0}, "runner_end_d_mm"),
        ({"fan_flank_deg": 2.0}, "flanks meet"),  # apex 5.2 above the disc top (18)
        ({"gate_d_mm": 30.0}, "gate_d_mm"),
        ({"cell_size_mm": 30.0}, "cell_size_mm"),
    ],
)
def test_validate_rejects(overrides, match):
    with pytest.raises(ValueError, match=match):
        FanRunnerPlateConfig(**overrides).validate()


def test_validate_accepts_the_flank_that_just_reaches_the_disc_top():
    # apex exactly at the disc top: 150 · tan θ = 18 → θ = 6.84°
    theta = math.degrees(math.atan(18.0 / 150.0))
    FanRunnerPlateConfig(fan_flank_deg=theta + 1e-6).validate()
    with pytest.raises(ValueError, match="flanks meet"):
        FanRunnerPlateConfig(fan_flank_deg=theta - 1e-3).validate()


def test_builder_runs_validate():
    with pytest.raises(ValueError, match="runner_w_mm"):
        build_fan_runner_plate_geometry(FanRunnerPlateConfig(runner_w_mm=400.0))
