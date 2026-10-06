"""Gate 2 (pentagon with straight sides) of the fan-runner builder, and the
thickness options it shares with Gate 1 (``docs/spec.md``)."""

from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np
import pytest

from core import GATE2_DEFAULTS, FanRunnerPlateConfig, build_fan_runner_plate_geometry

G2 = FanRunnerPlateConfig(**GATE2_DEFAULTS)


def _fingerprint(cfg: FanRunnerPlateConfig) -> str:
    g = build_fan_runner_plate_geometry(cfg)
    m = hashlib.sha256()
    for a in (
        g.mask.astype(np.uint8),
        np.round(g.thickness_mm, 9),
        g.compression_mask.astype(np.uint8),
        g.product_mask.astype(np.uint8),
        np.array(g.gates),
    ):
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()[:16]


@pytest.mark.parametrize(
    "overrides, digest",
    [
        ({}, "99bde9b89e4d1684"),
        ({"cell_size_mm": 0.5}, "83d2aa90ff6f35c6"),
        ({"balancer_on": True}, "4c5ccfdb59cdbc83"),
        ({"cell_size_mm": 4.0}, "5642b9a7f43e86b6"),
        ({"runner_edge_flat_mm": 0.0, "runner_ramp_end_mm": 0.0}, "24efbb60e3ff5683"),
    ],
)
def test_gate1_rasterises_exactly_as_before_gate2_existed(overrides, digest):
    """Fingerprints taken from v0.5.0 before the Gate 2 fields were added:
    Gate 1 with the new fields at their defaults is the same raster."""
    assert _fingerprint(FanRunnerPlateConfig(**overrides)) == digest


def _grid(g):
    x0, y0 = g.display_origin_mm()
    dx = g.cell_size_mm
    ny, nx = g.shape
    x = (np.arange(nx) + 0.5) * dx - x0  # from the sprue axis
    d = y0 - (np.arange(ny) + 0.5) * dx  # depth below the product edge
    return x, d


def _tangent_by_bisection(cfg: FanRunnerPlateConfig) -> tuple[float, float]:
    """Independent oracle for the right tangent point: on the circle,
    (T − O) · (T − P) = 0, found by bisection on the lower-right quarter."""
    r = cfg.runner_end_d_mm / 2
    px, pd = cfg.runner_w_mm / 2, cfg.side_len_mm
    od = cfg.runner_len_mm

    def f(t: float) -> float:
        tx, td = r * math.cos(t), od + r * math.sin(t)  # depth grows downward
        return tx * (tx - px) + (td - od) * (td - pd)

    lo, hi = 0.0, math.pi / 2  # angle measured downward from +x
    assert f(lo) * f(hi) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    t = 0.5 * (lo + hi)
    return r * math.cos(t), od + r * math.sin(t)


def test_gate2_defaults_are_the_sketch():
    assert G2.runner_shape == "pentagon"
    assert G2.runner_w_mm == 300.0 and G2.side_len_mm == 28.0
    assert G2.runner_len_mm == 30.0 and G2.runner_end_d_mm == 24.0
    assert G2.ramp_end_depth_mm == 18.0  # top of R12
    assert G2.runner_end_thk_mm == 2.5 and G2.runner_thk_mm == 2.5
    assert G2.runner_edge_thk_mm == 1.0 and G2.runner_edge_flat_mm == 1.0
    assert G2.runner_ramp_on is True
    assert G2.frame_w_mm + G2.side_len_mm == 48.0  # rim-inclusive side h
    g = build_fan_runner_plate_geometry(G2)
    assert g.label == "pentagon_runner_plate"


@pytest.mark.parametrize("side", [0.0, 10.0, 28.0, 30.0, 35.0, 41.0])
def test_tangent_point_matches_an_independent_solution(side):
    cfg = dataclasses.replace(G2, side_len_mm=side)
    xt, dt = cfg.pentagon_tangent_mm
    ox, od = _tangent_by_bisection(cfg)
    assert xt == pytest.approx(ox, abs=1e-9)
    assert dt == pytest.approx(od, abs=1e-9)
    # it is on the circle, and the slant from the corner touches it there
    r = cfg.runner_end_d_mm / 2
    assert math.hypot(xt, dt - cfg.runner_len_mm) == pytest.approx(r, abs=1e-9)
    assert cfg.pentagon_slant_deg == pytest.approx(
        math.degrees(math.atan2(dt - side, cfg.runner_w_mm / 2 - xt)), abs=1e-9
    )


def test_default_slant_and_tangent_point():
    xt, dt = G2.pentagon_tangent_mm
    assert xt == pytest.approx(1.119, abs=1e-3)
    assert dt == pytest.approx(41.948, abs=1e-3)
    assert G2.pentagon_slant_deg == pytest.approx(5.352, abs=1e-3)


def _exact_runner_area(cfg: FanRunnerPlateConfig) -> float:
    """Shoelace of the hexagon through the tangent points plus the circular
    segment under the chord between them."""
    w2, s, r, ln = cfg.runner_w_mm / 2, cfg.side_len_mm, cfg.runner_end_d_mm / 2, cfg.runner_len_mm
    xt, dt = _tangent_by_bisection(cfg)
    poly = [(-w2, 0.0), (w2, 0.0), (w2, -s), (xt, -dt), (-xt, -dt), (-w2, -s)]
    area = 0.5 * abs(
        sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))
    )
    half = math.atan2(xt, dt - ln)  # half the arc's central angle, from straight down
    alpha = 2 * half
    return area + 0.5 * r * r * (alpha - math.sin(alpha))


@pytest.mark.parametrize("side", [10.0, 28.0, 40.0])
def test_runner_area_matches_the_exact_hull(side):
    cfg = dataclasses.replace(G2, side_len_mm=side, cell_size_mm=0.25)
    g = build_fan_runner_plate_geometry(cfg)
    _, d = _grid(g)
    runner = g.mask & (d[:, None] > 0)
    area = runner.sum() * g.cell_size_mm**2
    assert area == pytest.approx(_exact_runner_area(cfg), rel=2e-3)


def test_sides_drop_straight_and_the_slant_meets_the_round_end_bottom():
    cfg = dataclasses.replace(G2, cell_size_mm=0.25)
    g = build_fan_runner_plate_geometry(cfg)
    x, d = _grid(g)
    dx = g.cell_size_mm
    runner = g.mask & (d[:, None] > 0)

    def deepest(col: int) -> float:
        rows = np.nonzero(runner[:, col])[0]
        return float(d[rows].max())

    # just inside the side: down to side_len
    col = int(np.argmin(np.abs(x - (cfg.runner_w_mm / 2 - dx / 2))))
    assert deepest(col) == pytest.approx(cfg.side_len_mm, abs=dx)
    # nothing outside the runner width below the product edge
    assert not runner[:, np.abs(x) > cfg.runner_w_mm / 2].any()
    # on the axis: down to the bottom of the round end
    col = int(np.argmin(np.abs(x)))
    assert deepest(col) == pytest.approx(cfg.runner_len_mm + cfg.runner_end_d_mm / 2, abs=dx)
    # along the slant: the straight line from the corner to the tangent point
    xt, dt = cfg.pentagon_tangent_mm
    w2, s = cfg.runner_w_mm / 2, cfg.side_len_mm
    for xq in (20.0, 60.0, 100.0, 140.0):
        col = int(np.argmin(np.abs(x - xq)))
        line = s + (w2 - x[col]) * (dt - s) / (w2 - xt)
        assert deepest(col) == pytest.approx(line, abs=dx)


def test_side_len_at_the_round_end_bottom_is_a_rectangle():
    cfg = dataclasses.replace(G2, side_len_mm=42.0, cell_size_mm=0.5)
    cfg.validate()
    xt, dt = cfg.pentagon_tangent_mm
    assert xt == pytest.approx(0.0, abs=1e-6) and dt == pytest.approx(42.0, abs=1e-9)
    g = build_fan_runner_plate_geometry(cfg)
    _, d = _grid(g)
    area = (g.mask & (d[:, None] > 0)).sum() * g.cell_size_mm**2
    assert area == pytest.approx(300.0 * 42.0, rel=2e-3)


@pytest.mark.parametrize("cell", [1.0, 0.5, 4.0])
def test_gate2_raster_is_mirror_symmetric_about_the_axis(cell):
    """Same check as Gate 1's: every column mirrors about the axis."""
    cfg = dataclasses.replace(G2, cell_size_mm=cell, balancer_on=True)
    g = build_fan_runner_plate_geometry(cfg)
    js = np.arange(g.nx)
    jm = np.rint(2.0 * cfg.axis_x_mm / cfg.cell_size_mm - 1.0 - js).astype(int)
    ok = (jm >= 0) & (jm < g.nx)
    assert not g.mask[:, ~ok].any()
    assert (g.mask[:, js[ok]] == g.mask[:, jm[ok]]).all()
    assert (g.thickness_mm[:, js[ok]] == g.thickness_mm[:, jm[ok]]).all()


def _thk_at(g, xq: float, dq: float) -> float:
    x, d = _grid(g)
    return float(g.thickness_mm[int(np.argmin(np.abs(d - dq))), int(np.argmin(np.abs(x - xq)))])


def test_gate2_default_thickness_ramps_from_the_band_to_the_round_end_top():
    g = build_fan_runner_plate_geometry(dataclasses.replace(G2, cell_size_mm=0.25))
    assert _thk_at(g, 100.0, 0.375) == pytest.approx(1.0)  # edge band
    assert _thk_at(g, 100.0, 9.875) == pytest.approx(1.0 + 1.5 * (9.875 - 1.0) / 17.0)
    assert _thk_at(g, 100.0, 18.125) == pytest.approx(2.5)  # past the ramp: t_o
    assert _thk_at(g, 140.0, 27.875) == pytest.approx(2.5)  # down in the side corner
    assert _thk_at(g, 0.125, 30.125) == pytest.approx(2.5)  # in the disc
    # the ramp is a function of depth only: same value across the width
    assert _thk_at(g, -120.0, 9.875) == pytest.approx(_thk_at(g, 30.0, 9.875))


def test_round_end_depth_overrides_the_profile_inside_the_disc_only():
    g = build_fan_runner_plate_geometry(
        dataclasses.replace(G2, runner_end_thk_mm=3.5, cell_size_mm=0.25)
    )
    assert _thk_at(g, 0.125, 30.125) == pytest.approx(3.5)
    assert _thk_at(g, 0.125, 19.0) == pytest.approx(3.5)  # disc top, d 18..
    assert _thk_at(g, 13.0, 30.125) == pytest.approx(2.5)  # just outside R12
    assert _thk_at(g, 0.125, 17.0) < 2.5  # still on the ramp above the disc


def test_ramp_off_steps_from_the_band_to_t_o():
    cfg = dataclasses.replace(G2, runner_ramp_on=False, cell_size_mm=0.25)
    g = build_fan_runner_plate_geometry(cfg)
    assert _thk_at(g, 100.0, 0.625) == pytest.approx(1.0)
    assert _thk_at(g, 100.0, 1.125) == pytest.approx(2.5)
    assert _thk_at(g, 100.0, 9.875) == pytest.approx(2.5)
    # t_o = 1.0: only the disc is deep
    g = build_fan_runner_plate_geometry(dataclasses.replace(cfg, runner_thk_mm=1.0))
    assert _thk_at(g, 100.0, 9.875) == pytest.approx(1.0)
    assert _thk_at(g, 140.0, 27.875) == pytest.approx(1.0)
    assert _thk_at(g, 0.125, 30.125) == pytest.approx(2.5)


@pytest.mark.parametrize("ramp_on", [True, False])
def test_band_width_zero_means_no_band(ramp_on):
    cfg = dataclasses.replace(
        G2, runner_edge_flat_mm=0.0, runner_edge_thk_mm=0.6, runner_ramp_on=ramp_on
    )
    if ramp_on:
        # the ramp starts at the product edge from the band's thickness
        assert cfg.runner_thickness_at_depth(0.0) == pytest.approx(0.6)
        assert cfg.runner_thickness_at_depth(9.0) == pytest.approx(0.6 + 1.9 * 9.0 / 18.0)
    else:
        assert cfg.runner_thickness_at_depth(0.01) == pytest.approx(2.5)


def test_the_ramp_starts_from_the_band_thickness():
    cfg = dataclasses.replace(G2, runner_edge_thk_mm=0.6, runner_edge_flat_mm=3.0)
    assert cfg.runner_thickness_at_depth(2.9) == pytest.approx(0.6)
    assert cfg.runner_thickness_at_depth(3.0 + 7.5) == pytest.approx(0.6 + 1.9 * 0.5)
    assert cfg.runner_thickness_at_depth(18.0) == pytest.approx(2.5)


def test_ramp_end_none_follows_the_round_end_top():
    cfg = dataclasses.replace(G2, runner_end_d_mm=30.0)
    assert cfg.ramp_end_depth_mm == 15.0
    assert cfg.runner_thickness_at_depth(15.0) == pytest.approx(2.5)
    assert cfg.runner_thickness_at_depth(8.0) == pytest.approx(1.0 + 1.5 * 7.0 / 14.0)
    # an explicit value wins
    assert dataclasses.replace(G2, runner_ramp_end_mm=25.0).ramp_end_depth_mm == 25.0


def test_gate1_takes_the_shared_thickness_options():
    base = FanRunnerPlateConfig(cell_size_mm=0.25)
    g = build_fan_runner_plate_geometry(dataclasses.replace(base, runner_end_thk_mm=3.5))
    assert _thk_at(g, 0.125, 30.125) == pytest.approx(3.5)
    assert _thk_at(g, 0.125, 18.5) == pytest.approx(3.5)  # was on the ramp (ends at 20)
    off = dataclasses.replace(base, runner_ramp_on=False)
    assert off.runner_thickness_at_depth(5.0) == pytest.approx(2.5)
    assert dataclasses.replace(base, runner_ramp_end_mm=None).ramp_end_depth_mm == 18.0


def test_balancer_works_on_gate2():
    cfg = dataclasses.replace(G2, balancer_on=True, cell_size_mm=0.25)
    g = build_fan_runner_plate_geometry(cfg)
    assert _thk_at(g, 0.125, 5.125) == pytest.approx(1.0)  # inside ▽ (w100, h15)
    assert _thk_at(g, 60.0, 5.125) > 1.0  # outside it
    plain = build_fan_runner_plate_geometry(dataclasses.replace(cfg, balancer_on=False))
    assert (g.mask == plain.mask).all()
    assert g.thickness_mm.sum() < plain.thickness_mm.sum()


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"runner_shape": "hexagon"}, "runner_shape"),
        ({"side_len_mm": -1.0}, "side_len_mm must be ≥ 0"),
        ({"side_len_mm": 42.5}, "side_len_mm"),
        ({"runner_end_thk_mm": 0.0}, "runner_end_thk_mm must be positive"),
        ({"runner_ramp_end_mm": -1.0}, "runner_ramp_end_mm must be ≥ 0"),
        # corners 12 off the axis at the axis depth: on the R12 circle
        ({"runner_w_mm": 24.0, "side_len_mm": 30.0}, "outside the round end"),
        # the band ends below the ramp's deep end (18)
        ({"runner_edge_flat_mm": 19.0}, "runner_edge_flat_mm"),
    ],
)
def test_gate2_validate_rejects(overrides, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(G2, **overrides).validate()


def test_a_band_deeper_than_the_ramp_is_fine_without_the_ramp():
    dataclasses.replace(G2, runner_edge_flat_mm=19.0, runner_ramp_on=False).validate()


def test_gate2_flank_angle_is_not_checked_against_the_round_end():
    """The flank angle belongs to Gate 1; a value that would not reach the
    disc there does not stop Gate 2."""
    dataclasses.replace(G2, fan_flank_deg=2.0).validate()
    with pytest.raises(ValueError, match="flanks meet"):
        FanRunnerPlateConfig(fan_flank_deg=2.0).validate()


def test_gate2_on_a_coarse_mesh_is_one_cavity_with_a_gate():
    g = build_fan_runner_plate_geometry(dataclasses.replace(G2, cell_size_mm=4.0))
    from scipy import ndimage

    assert ndimage.label(g.mask)[1] == 1
    assert g.gates
