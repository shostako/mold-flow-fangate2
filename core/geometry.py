"""Cavity geometry definition.

The simulation domain is a 2D structured grid where each cell is either
inside the cavity (mask=True) or outside (mask=False). Each in-cavity
cell carries a thickness h [mm] (gap between mold halves). Gates are
point-like Dirichlet boundaries at tau=0.

This module provides:
- Geometry: container of mask, thickness map, gates, and cell size.
- build_demo_geometry: synthetic cavity (rectangular plate + runner + sprue).

The fan-runner plate builder (frame plate with body-only compression +
fan runner + hot-runner gate, see ``docs/spec.md``) is added on top of this
in ``core/fan_runner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Geometry:
    mask: np.ndarray  # bool [Ny, Nx]; True=in cavity
    thickness_mm: np.ndarray  # float [Ny, Nx]; mm; valid only where mask
    cell_size_mm: float  # square cell, mm
    gates: list[tuple[int, int]] = field(default_factory=list)  # [(iy, ix), ...]
    label: str = "cavity"
    # Cells that are inflated by ``compression_factor`` while the compression
    # phase is open. ``None`` keeps the legacy behaviour where the whole
    # cavity expands (used by ``build_demo_geometry``). Builders that
    # distinguish a product body from runners/sprues set this to a bool array
    # (only the product body is True). Cells outside ``mask`` are ignored
    # regardless of the value here.
    compression_mask: np.ndarray | None = None
    # Product body proper, when it differs from ``compression_mask`` (this
    # repo's plate compresses only its t4 body while the display origin must
    # sit on the t1 rim's edge; fangate compressed product + gate land).
    # ``None`` means "the compression zone is the product" (sim's builders).
    product_mask: np.ndarray | None = None
    # Nominal valve-axis x [mm, grid frame] for the display origin. Set by
    # the parametric builders from the configured valve position; when None
    # the display falls back to the rasterized gate-cell centroid (which an
    # orifice clipped by a one-sided pocket shifts mesh-dependently).
    valve_axis_x_mm: float | None = None
    # Nominal injection orifice ``(x_mm, y_mm, radius_mm)`` in the grid frame,
    # recorded by the builder from the configured geometry (fan-runner plate:
    # the hot-runner gate orifice at the sprue axis). The gate marker draws this one true-scale circle
    # instead of one symbol per rasterized Dirichlet cell (sim v0.38.1);
    # ``None`` falls back to the 4-connected components of the gate cells.
    valve_marker_mm: tuple[float, float, float] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def ny(self) -> int:
        return self.mask.shape[0]

    @property
    def nx(self) -> int:
        return self.mask.shape[1]

    def volume_cm3(self) -> float:
        cell_area_mm2 = self.cell_size_mm**2
        vol_mm3 = float(np.sum(self.thickness_mm[self.mask]) * cell_area_mm2)
        return vol_mm3 / 1000.0

    def compression_volume_fraction(self) -> float:
        """Fraction of the cavity volume that participates in compression.

        Returns ``1.0`` when ``compression_mask`` is ``None`` (legacy mode
        where the whole cavity expands). Otherwise returns the volume of the
        ``compression_mask & mask`` cells divided by the total cavity volume.
        """
        cm = self.compression_mask
        if cm is None:
            return 1.0
        denom = float(np.sum(self.thickness_mm[self.mask]))
        if denom <= 0:
            return 0.0
        return float(np.sum(self.thickness_mm[self.mask & cm]) / denom)

    def compression_area_mm2(self) -> float:
        """Planar area of the compression target zone in mm^2.

        Returns the cavity area when ``compression_mask`` is ``None``
        (legacy whole-cavity inflation). Otherwise returns the area of the
        ``compression_mask & mask`` cells. Used by the stroke-mode
        compression model where ``ΔV = stroke * A`` independent of the
        local thickness.
        """
        cm = self.compression_mask
        target = self.mask if cm is None else (self.mask & cm)
        return float(np.sum(target) * self.cell_size_mm**2)

    def add_gate(self, iy: int, ix: int) -> None:
        if not self.mask[iy, ix]:
            raise ValueError(f"gate ({iy},{ix}) is outside the cavity mask")
        self.gates.append((iy, ix))

    def display_origin_mm(self) -> tuple[float, float]:
        """Return ``(x0, y0)`` in mm — the display origin shared by the
        preview and every result-time map.

        ``x0`` is the nominal valve axis when the builder recorded it
        (``valve_axis_x_mm``), else the gate-cell centroid. ``y0`` is the **bottom edge of
        the product zone** (``product_mask`` when the builder set one, else
        the ``compression_mask`` cells, which sim's builders set to the
        product plate body), so the product's gate-side edge reads ``y = 0``:
        a film gate's gate block / runner sits at ``y < 0`` and a direct
        gate's gate lands inside the product at ``y > 0`` — one
        product-referenced convention for both. Falls back to the gate
        centroid ``y`` when there is no product marker (legacy demo
        geometry), and to the grid center / bottom (0, 0) corner when the
        geometry has no gates.
        """
        if not self.gates:
            return float(self.nx * self.cell_size_mm) / 2.0, 0.0
        gate_iys = np.fromiter((gy for gy, _ in self.gates), dtype=float)
        gate_ixs = np.fromiter((gx for _, gx in self.gates), dtype=float)
        if self.valve_axis_x_mm is not None:
            # The nominal axis, not the gate-cell centroid: an orifice
            # clipped by a one-sided pocket (asymmetric profile gate) only
            # keeps cells on one side, and the centroid then drifts off the
            # valve axis by a mesh-dependent amount (Codex P2, PR #76).
            x0 = float(self.valve_axis_x_mm)
        elif self.valve_marker_mm is not None:
            # Same record, other field: a copy path that carried only the
            # marker must not put x = 0 off the disk it draws.
            x0 = float(self.valve_marker_mm[0])
        else:
            x0 = float((float(gate_ixs.mean()) + 0.5) * self.cell_size_mm)
        y0 = float((float(gate_iys.mean()) + 0.5) * self.cell_size_mm)
        marker = self.product_mask if self.product_mask is not None else self.compression_mask
        product = None if marker is None else (self.mask & marker)
        if product is not None and product.any():
            y0 = float(np.where(product)[0].min()) * self.cell_size_mm
        return x0, y0


def build_demo_geometry(
    plate_w_mm: float = 120.0,
    plate_h_mm: float = 80.0,
    plate_thk_mm: float = 2.0,
    runner_thk_mm: float = 4.0,
    sprue_thk_mm: float = 6.0,
    cell_size_mm: float = 1.0,
    gate_count: int = 1,
) -> Geometry:
    """Build a flat plate + central runner + sprue. The product part is
    the rectangular plate; the runner is a thin horizontal strip below
    feeding into one or more film gates; the sprue is a small square at
    the runner inlet.
    """
    pad = 10.0
    runner_h_mm = 6.0
    sprue_size_mm = 8.0

    total_w = plate_w_mm + 2 * pad
    total_h = plate_h_mm + runner_h_mm + sprue_size_mm + 2 * pad

    nx = int(round(total_w / cell_size_mm))
    ny = int(round(total_h / cell_size_mm))

    mask = np.zeros((ny, nx), dtype=bool)
    thk = np.zeros((ny, nx), dtype=float)

    # plate (product)
    py0 = int(round(pad / cell_size_mm))
    py1 = py0 + int(round(plate_h_mm / cell_size_mm))
    px0 = int(round(pad / cell_size_mm))
    px1 = px0 + int(round(plate_w_mm / cell_size_mm))
    mask[py0:py1, px0:px1] = True
    thk[py0:py1, px0:px1] = plate_thk_mm

    # runner (just below the plate, full plate width)
    ry0 = py1
    ry1 = ry0 + int(round(runner_h_mm / cell_size_mm))
    mask[ry0:ry1, px0:px1] = True
    thk[ry0:ry1, px0:px1] = runner_thk_mm

    # sprue (square, centered on runner)
    sy0 = ry1
    sy1 = sy0 + int(round(sprue_size_mm / cell_size_mm))
    cx_mm = pad + plate_w_mm / 2.0
    sx0 = int(round((cx_mm - sprue_size_mm / 2.0) / cell_size_mm))
    sx1 = sx0 + int(round(sprue_size_mm / cell_size_mm))
    mask[sy0:sy1, sx0:sx1] = True
    thk[sy0:sy1, sx0:sx1] = sprue_thk_mm

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=cell_size_mm,
        label="demo_plate",
    )

    # gate(s): inject at sprue base (center bottom of sprue)
    sprue_center_iy = sy1 - 1
    sprue_center_ix = (sx0 + sx1) // 2
    if gate_count <= 1:
        geom.add_gate(sprue_center_iy, sprue_center_ix)
    else:
        # multiple gates spread along the runner-plate interface (film gating)
        gate_y = ry0 - 1  # last row of plate adjacent to runner
        # but we need gate inside cavity; ry0-1 is plate, fine.
        positions = np.linspace(px0 + 4, px1 - 5, gate_count, dtype=int)
        for gx in positions:
            geom.add_gate(int(gate_y), int(gx))

    return geom
