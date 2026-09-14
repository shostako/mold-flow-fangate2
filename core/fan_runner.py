"""12.3-inch frame-thickness plate fed by a fan-shaped cold runner from a
hot-runner gate (``docs/spec.md``).

Top-down layout, grid frame (mm, y up, x right; the runner sits below the
product like fangate's gate block)::

    y_plate_top = y_edge + plate_h
    y_edge      = y_axis + runner_len        product long edge (display y = 0)
    y_axis      = pad + grid_shift + R_end   sprue axis = round-end centre
    y = pad + grid_shift                     round-end bottom

``grid_shift`` (< one cell, see ``grid_shift_mm``) pads the bottom so the
product edge sits on a cell edge at the chosen resolution; ``grid_shift_x``
pads the left so the sprue axis sits on a cell edge (or centre) and the
raster is mirror-symmetric about it (the plate is 302.26 wide, so without
it the axis lands 0.13 mm off the cell grid and the runner rasterises
lopsided).

Runner silhouette (one shape; the drawing has no gate variants):

- a triangle hanging from the product edge, ``runner_w`` wide on the edge
  line, both flanks straight at ``fan_flank_deg`` to the edge line, so the
  half-width at depth ``d`` below the edge is ``runner_w/2 − d / tan(flank)``
  and the flanks would meet on the axis at ``apex_depth_mm``;
- a disc of diameter ``runner_end_d`` (the R12 round end) centred on the
  sprue axis, ``runner_len`` below the edge.

On the drawing the flanks cut the circle (they are not tangent to it) and
the extended apex lies inside the disc, so the silhouette is exactly the
union of the two; ``validate`` only requires the triangle to reach the top
of the disc so the union stays connected.

Runner thickness is a function of the depth ``d`` alone (the section is
taken on the axis, the plan view has no other thickness lines):
``runner_edge_thk`` for ``d ≤ runner_edge_flat`` (the rim thickness carries
1 mm past the edge), a linear ramp to ``runner_thk`` at
``d = runner_ramp_end``, then ``runner_thk`` to the round end. No well
pocket, no cold slug.

Gate: the hot-runner orifice ``gate_d`` on the sprue axis. The Hele-Shaw
model has no vertical channel, so that disc is the Dirichlet τ=0 injection
point (a mesh too coarse to put a cell centre inside it snaps to the
nearest cavity cell, like fangate's sprue foot).

Balancer (``balancer_on``, fangate's ▽ 肉盗み): an inverted isosceles
triangle carved into the runner, centred on the axis, base ``balancer_w``
on the product edge line, apex ``balancer_h`` toward the sprue. Inside it
the thickness is ``min(runner, balancer_thk)`` — a cut never adds
material, so on the 1 mm rim-thickness band next to the edge it is a no-op
by construction. ``validate`` keeps the apex out of the round-end disc and
requires ``balancer_thk`` below ``runner_thk`` so the cut is real somewhere.

Plate: ``frame_thk`` on the ``frame_w`` border, ``inner_thk`` inside.

Compression (ICM): **only the inner body** inflates (``compression_mask``);
the rim and the whole runner are fixed. ``product_mask`` is rim + body so
the display origin stays on the product edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry import Geometry


@dataclass(frozen=True)
class FanRunnerPlateConfig:
    """Parameters for :func:`build_fan_runner_plate_geometry`. Defaults are
    the drawing (``docs/spec.md``)."""

    # product
    plate_w_mm: float = 302.26
    plate_h_mm: float = 220.6
    frame_w_mm: float = 20.0
    frame_thk_mm: float = 1.0
    inner_thk_mm: float = 4.0
    # runner silhouette
    runner_w_mm: float = 300.0  # width on the product edge line
    fan_flank_deg: float = 14.0  # flank angle to the product edge line
    runner_len_mm: float = 30.0  # product edge → sprue axis
    runner_end_d_mm: float = 24.0  # round end (R12) centred on the axis
    # runner thickness profile by depth below the product edge
    runner_thk_mm: float = 2.5  # full thickness (ramp end → round end)
    runner_edge_thk_mm: float = 1.0  # on the edge band (= rim thickness)
    runner_edge_flat_mm: float = 1.0  # edge band length at runner_edge_thk
    runner_ramp_end_mm: float = 20.0  # depth where the ramp reaches runner_thk
    # hot-runner gate orifice on the axis (the injection point)
    gate_d_mm: float = 3.0
    # balancer: inverted triangle thinning, base on the product edge line
    balancer_on: bool = False
    balancer_w_mm: float = 100.0  # base width on the edge line
    balancer_h_mm: float = 15.0  # edge → apex (≤ runner_len − runner_end_d/2 = 18)
    balancer_thk_mm: float = 1.0  # thickness inside the triangle
    # discretisation
    cell_size_mm: float = 1.0
    pad_mm: float = 5.0

    def validate(self) -> None:
        eps = 1e-6
        positives = (
            ("plate_w_mm", self.plate_w_mm),
            ("plate_h_mm", self.plate_h_mm),
            ("frame_thk_mm", self.frame_thk_mm),
            ("inner_thk_mm", self.inner_thk_mm),
            ("runner_w_mm", self.runner_w_mm),
            ("fan_flank_deg", self.fan_flank_deg),
            ("runner_len_mm", self.runner_len_mm),
            ("runner_end_d_mm", self.runner_end_d_mm),
            ("runner_thk_mm", self.runner_thk_mm),
            ("runner_edge_thk_mm", self.runner_edge_thk_mm),
            ("gate_d_mm", self.gate_d_mm),
            ("cell_size_mm", self.cell_size_mm),
        )
        for name, val in positives:
            if val <= 0:
                raise ValueError(f"{name} must be positive (got {val})")
        for name, val in (
            ("frame_w_mm", self.frame_w_mm),
            ("runner_edge_flat_mm", self.runner_edge_flat_mm),
            ("runner_ramp_end_mm", self.runner_ramp_end_mm),
            ("pad_mm", self.pad_mm),
        ):
            if val < 0:
                raise ValueError(f"{name} must be ≥ 0 (got {val})")
        if 2 * self.frame_w_mm >= min(self.plate_w_mm, self.plate_h_mm) - eps:
            raise ValueError(
                f"frame_w_mm ({self.frame_w_mm}) must be < half of the smaller plate side"
            )
        if self.fan_flank_deg >= 90.0 - eps:
            raise ValueError(
                f"fan_flank_deg ({self.fan_flank_deg}) must be < 90 (the flanks converge)"
            )
        if self.runner_w_mm > self.plate_w_mm + eps:
            raise ValueError(
                f"runner_w_mm ({self.runner_w_mm}) must be ≤ plate_w_mm ({self.plate_w_mm})"
            )
        if self.runner_edge_flat_mm > self.runner_ramp_end_mm + eps:
            raise ValueError(
                f"runner_edge_flat_mm ({self.runner_edge_flat_mm}) must be ≤ "
                f"runner_ramp_end_mm ({self.runner_ramp_end_mm})"
            )
        r_end = self.runner_end_d_mm / 2.0
        if r_end > self.runner_len_mm + eps:
            raise ValueError(
                f"runner_end_d_mm / 2 ({r_end}) must be ≤ runner_len_mm ({self.runner_len_mm}); "
                f"the round end must not reach into the product"
            )
        if self.apex_depth_mm < self.runner_len_mm - r_end - eps:
            raise ValueError(
                f"the flanks meet at depth {self.apex_depth_mm:.2f} (runner_w_mm / 2 · "
                f"tan(fan_flank_deg)) which is above the round end's top "
                f"({self.runner_len_mm - r_end}); the triangle must reach the disc"
            )
        if self.runner_end_d_mm > self.runner_w_mm + eps:
            raise ValueError(
                f"runner_end_d_mm ({self.runner_end_d_mm}) must be ≤ runner_w_mm "
                f"({self.runner_w_mm}); the round end is not wider than the runner's edge "
                f"width, so the disc stays inside the plate's x extent (Codex P2 on PR #4)"
            )
        if self.gate_d_mm > self.runner_end_d_mm + eps:
            raise ValueError(
                f"gate_d_mm ({self.gate_d_mm}) must be ≤ runner_end_d_mm ({self.runner_end_d_mm})"
            )
        if self.cell_size_mm > self.runner_end_d_mm + eps:
            raise ValueError(
                f"cell_size_mm ({self.cell_size_mm}) must be ≤ runner_end_d_mm "
                f"({self.runner_end_d_mm}); a mesh coarser than the round end cannot resolve "
                f"the runner"
            )
        if self.balancer_on:
            for name, val in (
                ("balancer_w_mm", self.balancer_w_mm),
                ("balancer_h_mm", self.balancer_h_mm),
                ("balancer_thk_mm", self.balancer_thk_mm),
            ):
                if val <= 0:
                    raise ValueError(f"{name} must be positive when balancer_on (got {val})")
            w_max, h_max, thk_sup = self.balancer_limits_mm
            if self.balancer_thk_mm >= thk_sup - eps:
                raise ValueError(
                    f"balancer_thk_mm ({self.balancer_thk_mm}) must be < runner_thk_mm "
                    f"({thk_sup}); a balancer is a cut, it cannot add material"
                )
            if self.balancer_w_mm > w_max + eps:
                raise ValueError(
                    f"balancer_w_mm ({self.balancer_w_mm}) must be ≤ runner_w_mm ({w_max})"
                )
            if self.balancer_h_mm > h_max + eps:
                raise ValueError(
                    f"balancer_h_mm ({self.balancer_h_mm}) must be ≤ runner_len_mm − "
                    f"runner_end_d_mm / 2 ({h_max}) so the apex stays outside the round end"
                )

    # ----- derived quantities -----
    @property
    def apex_depth_mm(self) -> float:
        """Depth below the product edge where the extended flanks meet on the axis."""
        return 0.5 * self.runner_w_mm * math.tan(math.radians(self.fan_flank_deg))

    @property
    def balancer_limits_mm(self) -> tuple[float, float, float]:
        """``(w_max, h_max, thk_sup)`` for the balancer: base ≤ the runner width
        on the edge line, height ≤ ``runner_len − runner_end_d / 2`` (apex
        outside the round end), thickness < ``runner_thk`` (the cut has to
        remove material somewhere; on the edge band it is a no-op anyway).
        The single source for :meth:`validate` and the sidebar bounds."""
        return self.runner_w_mm, self.runner_len_mm - self.runner_end_d_mm / 2.0, self.runner_thk_mm

    @property
    def grid_shift_mm(self) -> float:
        """Bottom-pad extension (< one cell) aligning the product edge to the
        cell grid (fangate's fix for a half-cell-off plate: an odd
        ``runner_end_d / 2`` would land the edge on a cell centre and shift
        the rendered product with an unrelated runner change)."""
        y_edge_raw = self.pad_mm + self.runner_end_d_mm / 2.0 + self.runner_len_mm
        shift = (-y_edge_raw) % self.cell_size_mm
        return 0.0 if shift > self.cell_size_mm - 1e-9 else shift

    @property
    def grid_shift_x_mm(self) -> float:
        """Left-pad extension (< one cell) putting the sprue axis on a cell
        edge when the plate is an even number of cells wide, on a cell
        centre when odd — either way every runner / plate row rasterises
        mirror-symmetric about the axis, and the product keeps
        ``round(plate_w / cell)`` columns."""
        axis_raw = self.pad_mm + self.plate_w_mm / 2.0
        n_cols = int(round(self.plate_w_mm / self.cell_size_mm))
        target = 0.0 if n_cols % 2 == 0 else 0.5 * self.cell_size_mm
        shift = (target - axis_raw) % self.cell_size_mm
        return 0.0 if shift > self.cell_size_mm - 1e-9 else shift

    @property
    def y_axis_mm(self) -> float:
        return self.pad_mm + self.grid_shift_mm + self.runner_end_d_mm / 2.0

    @property
    def y_edge_mm(self) -> float:
        """Product long edge (display ``y = 0``) = compression-free boundary
        between the runner and the rim."""
        return self.y_axis_mm + self.runner_len_mm

    @property
    def y_plate_top_mm(self) -> float:
        return self.y_edge_mm + self.plate_h_mm

    @property
    def x_plate_left_mm(self) -> float:
        """Left edge of the product (includes ``grid_shift_x_mm`` in the pad)."""
        return self.pad_mm + self.grid_shift_x_mm

    @property
    def axis_x_mm(self) -> float:
        return self.x_plate_left_mm + self.plate_w_mm / 2.0

    def runner_thickness_at_depth(self, depth_mm: np.ndarray | float) -> np.ndarray | float:
        """Runner thickness profile: ``runner_edge_thk`` on the edge band, a
        linear ramp to ``runner_thk`` at ``runner_ramp_end``, then constant."""
        ramp_len = self.runner_ramp_end_mm - self.runner_edge_flat_mm
        if ramp_len > 1e-12:
            t = np.clip(
                (np.asarray(depth_mm, dtype=float) - self.runner_edge_flat_mm) / ramp_len, 0.0, 1.0
            )
        else:
            t = (np.asarray(depth_mm, dtype=float) > self.runner_edge_flat_mm).astype(float)
        return self.runner_edge_thk_mm + (self.runner_thk_mm - self.runner_edge_thk_mm) * t


def build_fan_runner_plate_geometry(cfg: FanRunnerPlateConfig) -> Geometry:
    """Rasterise :class:`FanRunnerPlateConfig` onto a square-cell grid."""
    cfg.validate()

    pad = cfg.pad_mm
    dx = cfg.cell_size_mm
    cx = cfg.axis_x_mm
    y_axis = cfg.y_axis_mm
    y_edge = cfg.y_edge_mm
    y_top = cfg.y_plate_top_mm
    r_end = cfg.runner_end_d_mm / 2.0
    apex = cfg.apex_depth_mm

    x_left = cfg.x_plate_left_mm
    total_w = x_left + cfg.plate_w_mm + pad  # includes grid_shift_x_mm in the left pad
    total_h = y_top + pad  # includes grid_shift_mm in the bottom pad
    nx = int(round(total_w / dx))
    ny = int(round(total_h / dx))

    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    yy = (iy_idx + 0.5) * dx
    xx = (ix_idx + 0.5) * dx
    ax = np.abs(xx - cx)
    r2_axis = (xx - cx) ** 2 + (yy - y_axis) ** 2
    dd = y_edge - yy  # depth below the product edge

    # --- silhouette: triangle from the edge ∪ round-end disc ---
    half_w_at_depth = 0.5 * cfg.runner_w_mm * (1.0 - dd / apex)
    in_tri = (dd >= 0.0) & (dd <= apex) & (ax <= half_w_at_depth)
    in_end = r2_axis <= r_end**2
    in_runner = in_tri | in_end
    in_x_plate = (xx >= x_left) & (xx <= x_left + cfg.plate_w_mm)
    in_plate = (yy > y_edge) & (yy <= y_top) & in_x_plate
    mask = in_runner | in_plate
    if not mask.any():
        raise ValueError(
            f"cell_size_mm ({dx}) rasterises the whole cavity away ({ny}x{nx} grid, no cavity cell)"
        )
    # A triangle that only just reaches the disc (validate's lower bound) can
    # rasterise into two islands on a coarse mesh: the gate then feeds the
    # round end and the plate is orphaned, which the solver rejects later in
    # check_gate_reachability with a far less useful message (Codex P2 on PR #4)
    _, n_parts = ndimage.label(mask)
    if n_parts != 1:
        raise ValueError(
            f"cell_size_mm ({dx}) rasterises the cavity into {n_parts} disconnected parts "
            f"(the runner's flanks barely reach the round end); refine the mesh or steepen "
            f"fan_flank_deg"
        )

    # --- thickness ---
    thk = np.zeros_like(xx, dtype=float)
    runner_thk = np.asarray(cfg.runner_thickness_at_depth(dd), dtype=float)
    thk[in_runner] = runner_thk[in_runner]

    # balancer: base on the product edge line, apex balancer_h toward the
    # sprue; half-width grows linearly apex → base; a cut never adds material
    if cfg.balancer_on:
        y_apex = y_edge - cfg.balancer_h_mm
        t_bal = np.clip((yy - y_apex) / max(cfg.balancer_h_mm, 1e-12), 0.0, 1.0)
        in_balancer = (
            in_runner & (yy >= y_apex) & (yy <= y_edge) & (ax <= 0.5 * cfg.balancer_w_mm * t_bal)
        )
        thk[in_balancer] = np.minimum(thk[in_balancer], cfg.balancer_thk_mm)

    # plate: frame border vs inner body
    in_inner = (
        in_plate
        & (xx > x_left + cfg.frame_w_mm)
        & (xx < x_left + cfg.plate_w_mm - cfg.frame_w_mm)
        & (yy > y_edge + cfg.frame_w_mm)
        & (yy < y_top - cfg.frame_w_mm)
    )
    thk[in_plate] = cfg.frame_thk_mm
    thk[in_inner] = cfg.inner_thk_mm

    thk[~mask] = 0.0

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label="fan_runner_plate",
        # ICM squeezes only the t4 body; the rim and the runner are fixed
        compression_mask=in_inner & mask,
        product_mask=in_plate & mask,
        valve_axis_x_mm=cx,
        # nominal injection orifice = the hot-runner gate disc on the axis;
        # the gate marker draws this one true-scale circle
        valve_marker_mm=(float(cx), float(y_axis), cfg.gate_d_mm / 2.0),
    )

    # --- injection point: gate disc on the axis ---
    in_gate = r2_axis <= (cfg.gate_d_mm / 2.0) ** 2
    gate_iys, gate_ixs = np.where(in_gate & mask)
    if gate_iys.size == 0:
        # Mesh coarser than the orifice: no cell centre lands inside the
        # disc. Snap to the cavity cell nearest the axis so every built
        # geometry has an injection boundary.
        d2 = np.where(mask, r2_axis, np.inf)
        ic_y, ic_x = np.unravel_index(int(np.argmin(d2)), d2.shape)
        geom.gates.append((int(ic_y), int(ic_x)))
        # the snapped cell is where the solver actually injects; drawing the
        # nominal disc there would lie about the injection point
        geom.valve_marker_mm = None
    else:
        for iy, ix in zip(gate_iys, gate_ixs, strict=True):
            geom.gates.append((int(iy), int(ix)))
    return geom
