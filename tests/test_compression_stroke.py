"""Tests for the absolute-stroke compression model.

The legacy compression model multiplies the cavity thickness by a constant
``compression_factor``. For stepped plates (e.g. lower-zone t=0.35 mm,
upper-zone t=0.50 mm) this is physically inaccurate: the mold shim adds a
fixed absolute distance to every target cell, so both zones should grow by
the same stroke and the step thickness must be preserved.

This module exercises:

* Backward compatibility: ``compression_stroke_mm=None`` is byte-for-byte
  equivalent to the legacy factor model.
* Additivity: in stroke mode every target cell grows by exactly the same
  stroke, preserving the step thickness on a stepped plate.
* ``stroke=0`` collapses to compression OFF.
* The ``effective_factor`` used in T_fill shortening is consistent between
  the two modes on a uniform plate (so a factor matching the stroke gives
  the same T_fill).
* Metadata exposes ``compression_stroke_mm`` / ``compression_mode``.
"""

from __future__ import annotations

import numpy as np

from core import FanRunnerPlateConfig, HeleShawSolver, MaterialDB, build_fan_runner_plate_geometry


def _plate_cfg(**overrides) -> FanRunnerPlateConfig:
    """Small frame plate: rim t=0.35 all around / body t=0.50. Only the body
    is the compression zone; the rim and the runner (0.3 → 2.0 ramp, round
    end, gate) are fixed."""
    base = dict(
        plate_w_mm=120.0,
        plate_h_mm=80.0,
        frame_w_mm=15.0,
        frame_thk_mm=0.35,
        inner_thk_mm=0.50,
        runner_w_mm=80.0,
        fan_flank_deg=25.0,
        runner_len_mm=20.0,
        runner_end_d_mm=12.0,
        runner_thk_mm=2.0,
        runner_edge_thk_mm=0.3,
        runner_edge_flat_mm=2.0,
        runner_ramp_end_mm=6.0,
        gate_d_mm=4.0,
        cell_size_mm=1.0,
        pad_mm=5.0,
    )
    base.update(overrides)
    return FanRunnerPlateConfig(**base)


def _stepped_geometry():
    """The plate with a step carved into the compression zone: the lower
    half of the body rows are set to t=0.35, the upper half stay t=0.50.
    The builder's own body is uniform (t4 on the drawing), so the step the
    stroke model must preserve is written into the thickness map here."""
    g = build_fan_runner_plate_geometry(_plate_cfg())
    cm = g.compression_mask
    assert cm is not None
    rows = np.where(cm.any(axis=1))[0]
    mid = rows[len(rows) // 2]
    lower = cm & (np.arange(g.ny)[:, None] < mid)
    g.thickness_mm[lower] = 0.35
    return g


def _solver(g, **kwargs) -> HeleShawSolver:
    db = MaterialDB()
    base = dict(
        geometry=g,
        material=db["PP"],
        melt_temperature_K=503.15,
        mold_temperature_K=313.15,
        injection_velocity_mms=100.0,
        injection_volume_flow_cm3s=20.0,
    )
    base.update(kwargs)
    return HeleShawSolver(**base)


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_stroke_none_matches_legacy_factor_model() -> None:
    """When ``compression_stroke_mm`` is left at the default ``None``,
    ``_open_thickness_field`` must reproduce the exact factor-model output.
    """
    g = _stepped_geometry()

    legacy = _solver(g, compression_molding=True, compression_factor=1.8)
    new_default = _solver(
        g,
        compression_molding=True,
        compression_factor=1.8,
        compression_stroke_mm=None,
    )

    np.testing.assert_allclose(
        legacy._open_thickness_field(),
        new_default._open_thickness_field(),
        rtol=1e-12,
    )


def test_stroke_none_metadata_reports_factor_mode() -> None:
    g = _stepped_geometry()
    solver = _solver(g, compression_molding=True, compression_factor=1.8)
    result = solver.solve(num_frames=4)
    assert result.metadata["compression_mode"] == "factor"
    assert result.metadata["compression_stroke_mm"] is None


# --------------------------------------------------------------------------
# Stroke additivity / step preservation
# --------------------------------------------------------------------------


def test_stroke_preserves_step_thickness_on_stepped_plate() -> None:
    """On a t=0.35 / t=0.50 stepped plate, applying stroke=0.70 mm must
    grow the thin zone to 1.05 mm and the thick zone to 1.20 mm — the
    0.15 mm step is preserved (factor model would not preserve it).
    """
    g = _stepped_geometry()
    solver = _solver(g, compression_molding=True, compression_stroke_mm=0.70)

    h_open = solver._open_thickness_field()
    cm = g.compression_mask
    assert cm is not None

    # Identify lower (t=0.35) and upper (t=0.50) bands by reading
    # geometry.thickness_mm at compression-mask cells.
    h_cast = g.thickness_mm
    lower = cm & g.mask & (np.isclose(h_cast, 0.35))
    upper = cm & g.mask & (np.isclose(h_cast, 0.50))

    assert lower.any(), "stepped geometry must produce some t=0.35 cells"
    assert upper.any(), "stepped geometry must produce some t=0.50 cells"

    np.testing.assert_allclose(h_open[lower], 1.05, rtol=1e-9)
    np.testing.assert_allclose(h_open[upper], 1.20, rtol=1e-9)


def test_stroke_uniform_addition_on_compression_cells() -> None:
    """Every compression-target cell grows by *exactly* the same stroke,
    regardless of its as-cast thickness."""
    g = _stepped_geometry()
    solver_off = _solver(g)
    solver_on = _solver(g, compression_molding=True, compression_stroke_mm=0.70)

    h_off = solver_off._open_thickness_field()
    h_on = solver_on._open_thickness_field()

    cm = g.compression_mask
    assert cm is not None
    target = cm & g.mask
    np.testing.assert_allclose(h_on[target] - h_off[target], 0.70, rtol=1e-9)

    # Rim and runner cells (edge band, ramp, round end, gate) must stay untouched.
    other = g.mask & ~cm
    np.testing.assert_allclose(h_on[other], h_off[other], rtol=1e-9)


def test_stroke_zero_matches_compression_off() -> None:
    g = _stepped_geometry()
    solver_off = _solver(g)
    solver_zero = _solver(g, compression_molding=True, compression_stroke_mm=0.0)

    np.testing.assert_allclose(
        solver_off._open_thickness_field(),
        solver_zero._open_thickness_field(),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------
# effective_factor consistency on a uniform plate
# --------------------------------------------------------------------------


def test_uniform_plate_stroke_factor_equivalence_for_T_fill() -> None:
    """On a uniform plate (lower = upper = plate_thk), choosing
    ``factor = (h + stroke) / h`` should yield the same T_fill in both
    modes. The ``effective_factor`` derivations are different but must
    coincide here, which guards against arithmetic drift between the two
    code paths.
    """
    # Uniform compression zone: the body is 0.50 mm (the rim is not compressed)
    g = build_fan_runner_plate_geometry(_plate_cfg(inner_thk_mm=0.50))

    # All compression cells (the body) are 0.50 mm. stroke=0.70 -> factor=2.4.
    stroke = 0.70
    h_plate = 0.50
    factor = (h_plate + stroke) / h_plate  # = 2.4

    solver_stroke = _solver(
        g,
        compression_molding=True,
        compression_stroke_mm=stroke,
        compression_fraction=0.7,
    )
    solver_factor = _solver(
        g,
        compression_molding=True,
        compression_factor=factor,
        compression_fraction=0.7,
    )

    r_stroke = solver_stroke.solve(num_frames=4)
    r_factor = solver_factor.solve(num_frames=4)

    # T_fill is the absolute time scaling; both modes must agree.
    np.testing.assert_allclose(
        r_stroke.total_fill_time_s,
        r_factor.total_fill_time_s,
        rtol=1e-6,
    )
    # tau field shape must also coincide (same conductance).
    np.testing.assert_allclose(
        np.nan_to_num(r_stroke.tau, nan=0.0),
        np.nan_to_num(r_factor.tau, nan=0.0),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Metadata surfacing
# --------------------------------------------------------------------------


def test_stroke_mode_metadata() -> None:
    g = _stepped_geometry()
    solver = _solver(g, compression_molding=True, compression_stroke_mm=0.70)
    result = solver.solve(num_frames=4)
    assert result.metadata["compression_mode"] == "stroke"
    assert result.metadata["compression_stroke_mm"] == 0.70


# --------------------------------------------------------------------------
# Geometry helper
# --------------------------------------------------------------------------


def test_compression_area_mm2_matches_the_body() -> None:
    """``Geometry.compression_area_mm2`` should report the planar area of
    the compression target zone (cell count × cell_area_mm2) — here the
    t=0.50 body alone, not the rim."""
    g = _stepped_geometry()
    cm = g.compression_mask
    assert cm is not None
    expected = float(np.sum(cm & g.mask)) * g.cell_size_mm**2
    assert g.compression_area_mm2() == expected


def test_compression_area_mm2_full_when_mask_none() -> None:
    """When ``compression_mask`` is ``None`` (legacy whole-cavity mode),
    the helper must return the full cavity area."""
    from core import Geometry

    mask = np.ones((4, 6), dtype=bool)
    thk = np.full(mask.shape, 1.0)
    g = Geometry(mask=mask, thickness_mm=thk, cell_size_mm=2.0)
    # 24 cells × 4 mm² each = 96 mm²
    assert g.compression_area_mm2() == 96.0
