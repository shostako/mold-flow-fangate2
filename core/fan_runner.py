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

Runner silhouette, ``runner_shape``. **Gate 1** (``"triangle"``, the
customer drawing):

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

**Gate 2** (``"pentagon"``, the 2026-10-06 sketch): straight sides drop
``side_len`` from both ends of the ``runner_w`` edge line (measured below
the product edge, i.e. below the 20 mm rim — the rim-inclusive side is
``frame_w + side_len``), then a slant runs from each corner to the round end
and is **tangent** to it on its lower side; the bottom between the two
tangent points is the disc's arc. The silhouette is the convex hull of the
``runner_w × side_len`` rectangle and the disc. ``validate`` keeps each
corner outside the disc and no deeper than the disc's bottom
(``side_len ≤ runner_len + runner_end_d/2``; at equality the slants are
level and the runner is a plain rectangle).

**Gate 3** (``"triangle_arms"``, the 2026-10-06 proposal): Gate 1's
silhouette and thickness, plus an **arm** along each flank — a groove
``arm_w`` wide (measured square to the flank) cut to ``arm_thk``, from the end
of the edge band to the round end. Inside the arm the thickness is
``max(profile, arm_thk)`` (a groove only removes steel); the edge band and the
round end keep their own thickness, so the gate's cut face at the product
edge is unchanged. ``validate`` keeps the arms off the axis on the edge line
(``arm_w < runner_w/2 · sin(flank)``) and requires ``arm_thk`` above the
edge band so the groove is real.

Runner thickness is a function of the depth ``d`` alone (the section is
taken on the axis, the plan view has no other thickness lines), for every
shape: ``runner_edge_thk`` on the edge band ``d ≤ runner_edge_flat`` (the
band's width; 0 = no band), a linear ramp to ``runner_thk`` at
``d = runner_ramp_end`` (``None`` = the top of the round end), then
``runner_thk`` to the round end. ``runner_ramp_on = False`` drops the ramp:
the band steps straight to ``runner_thk``. ``runner_end_thk`` (``None`` =
follow the profile) overrides the thickness inside the round-end disc. No
well pocket, no cold slug.

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

**The gate side.** The product proper is the body and the rim on three
sides; the rim's fourth side (the band ``frame_w`` deep along the gate-side
edge) belongs to the gate, and ``runner_shape`` decides both it and the
runner. The three fan runners above leave the band a plain rim. The others:

- **Centre tab** (``"tab"``, the 2026-11-10 trial): a trapezoid ``tab_w`` wide
  on the edge line narrowing (or widening) to the round end's diameter on the
  axis line, ∪ the disc. Thickness ``tab_end_thk`` at the edge, a linear ramp
  to ``tab_thk`` at ``tab_ramp_len``, ``tab_thk`` on to and inside the disc.
  The band stays a plain rim.
- **Overlap tab** (``"overlap_tab"``): the same trapezoid, ``tab_thk``
  throughout, and it runs on under the rim: the band is ``lap_thk`` over the
  tab's width (rim + tab), a plain rim elsewhere.
- **Thick tab** (``"thick_tab"``): a triangle ``thick_tab_w`` wide on the edge
  line with its apex on the sprue axis, ∪ the disc, ``thick_tab_thk``
  throughout; the band is ``thick_tab_thk`` over the tab's width, **cut away**
  (no cavity) from there out to ``corner_keep`` short of the body's side
  edge, and a plain rim beyond (the corners).
- **Direct gates** (``"direct"``): no runner. Valve gates ``gate_d`` across
  open straight onto the body: ``direct_layout`` ``"center"`` (one, at the
  body's centre), ``"corners4"`` (one ``direct_inset`` in from each corner of
  the body) or ``"diagonal2"`` (two of those, on a diagonal). They open
  together at one pressure. The band stays a plain rim.

The runner fields that a shape does not read (the fan's width, flank, side,
profile, arms and balancer on the four shapes above) are not validated for it.

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
from .transient_fill import DelayedGate

#: ``runner_shape`` values: Gate 1 = the customer drawing, Gate 2 = the
#: pentagon with straight sides (2026-10-06), Gate 3 = Gate 1 with a thick
#: arm along each flank (2026-10-06); then the shapes that also decide the
#: gate-side band of the rim (module docstring, "The gate side")
FAN_SHAPES = ("triangle", "pentagon", "triangle_arms")
TAB_SHAPES = ("tab", "overlap_tab")
RUNNER_SHAPES = (*FAN_SHAPES, *TAB_SHAPES, "thick_tab", "direct")

#: ``direct_layout`` values
DIRECT_LAYOUTS = ("center", "corners4", "diagonal2")
#: ``direct_open_mode`` values: every valve opens with the injection; the gates
#: after the first open a set time after the start of the injection; or each
#: opens once the melt of the gates before it has reached it
DIRECT_OPEN_MODES = ("together", "delay", "arrival")

#: Gate 2 as drawn (``docs/spec.md``): on top of the shared defaults, the ramp
#: ends at the top of the round end and the disc is 2.5 thick.
GATE2_DEFAULTS: dict = {
    "runner_shape": "pentagon",
    "runner_ramp_end_mm": None,
    "runner_end_thk_mm": 2.5,
}

#: Gate 3 as proposed: Gate 1 (the drawing) plus the arms at their field
#: defaults (``arm_w_mm`` / ``arm_thk_mm``).
GATE3_DEFAULTS: dict = {
    "runner_shape": "triangle_arms",
}

#: Centre tab as tried on 2026-11-10: 30 wide, t1.0 at the edge → t2.0.
TAB_DEFAULTS: dict = {
    "runner_shape": "tab",
}

#: Overlap tab as drawn on 2026-11-11 (the t3.5 revision): 30 wide, t3.5, and
#: t3.5 where it laps the rim.
OVERLAP_TAB_DEFAULTS: dict = {
    "runner_shape": "overlap_tab",
    "tab_thk_mm": 3.5,
}

#: Thick tab: 100 wide at the body's thickness, the corners keep 30 of rim.
THICK_TAB_DEFAULTS: dict = {
    "runner_shape": "thick_tab",
}

#: One valve gate at the centre of the body.
DIRECT_DEFAULTS: dict = {
    "runner_shape": "direct",
}


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
    runner_shape: str = "triangle"  # one of RUNNER_SHAPES
    side_len_mm: float = 28.0  # Gate 2: side length below the product edge (rim excluded)
    # Gate 3: a groove along each flank (width square to the flank, thickness inside it)
    arm_w_mm: float = 9.0
    arm_thk_mm: float = 3.5
    # centre tab / overlap tab: tab_w on the edge line → the round end's diameter
    tab_w_mm: float = 30.0
    tab_thk_mm: float = 2.0  # near the gate and inside the round end (overlap tab: throughout)
    tab_end_thk_mm: float = 1.0  # centre tab: at the product edge
    tab_ramp_len_mm: float = 18.0  # centre tab: edge → where tab_thk is reached
    lap_thk_mm: float = 3.5  # overlap tab: the gate-side band over the tab's width
    # thick tab
    thick_tab_w_mm: float = 100.0
    thick_tab_thk_mm: float = 4.0
    corner_keep_mm: float = 30.0  # rim kept inward of the body's side edge, each corner
    # direct gates on the body
    direct_layout: str = "center"
    direct_inset_mm: float = 15.0  # corners: gate centre in from the body's corner, x and y
    # when the gates after the first (``direct_points_mm`` order) open: DIRECT_OPEN_MODES
    direct_open_mode: str = "together"
    # "delay": seconds after the start of the injection; "arrival": seconds after
    # the melt has reached the gate
    direct_delay_s: float = 0.0
    # runner thickness profile by depth below the product edge
    runner_thk_mm: float = 2.5  # t_o: outside the round end (ramp end → bottom)
    runner_edge_thk_mm: float = 1.0  # on the edge band (= rim thickness)
    runner_edge_flat_mm: float = 1.0  # edge band width at runner_edge_thk (0 = no band)
    runner_ramp_on: bool = True  # False: the band steps straight to runner_thk
    runner_ramp_end_mm: float | None = 20.0  # ramp's deep end; None = top of the round end
    runner_end_thk_mm: float | None = None  # inside the round-end disc; None = the profile
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
        if self.runner_end_thk_mm is not None:
            positives += (("runner_end_thk_mm", self.runner_end_thk_mm),)
        for name, val in positives:
            if val <= 0:
                raise ValueError(f"{name} must be positive (got {val})")
        for name, val in (
            ("frame_w_mm", self.frame_w_mm),
            ("side_len_mm", self.side_len_mm),
            ("runner_edge_flat_mm", self.runner_edge_flat_mm),
            ("runner_ramp_end_mm", self.runner_ramp_end_mm),
            ("pad_mm", self.pad_mm),
        ):
            if val is not None and val < 0:
                raise ValueError(f"{name} must be ≥ 0 (got {val})")
        if self.runner_shape not in RUNNER_SHAPES:
            raise ValueError(f"runner_shape ({self.runner_shape!r}) must be one of {RUNNER_SHAPES}")
        if 2 * self.frame_w_mm >= min(self.plate_w_mm, self.plate_h_mm) - eps:
            raise ValueError(
                f"frame_w_mm ({self.frame_w_mm}) must be < half of the smaller plate side"
            )
        fan = self.runner_shape in FAN_SHAPES
        if fan and self.fan_flank_deg >= 90.0 - eps:
            raise ValueError(
                f"fan_flank_deg ({self.fan_flank_deg}) must be < 90 (the flanks converge)"
            )
        if fan and self.runner_w_mm > self.plate_w_mm + eps:
            raise ValueError(
                f"runner_w_mm ({self.runner_w_mm}) must be ≤ plate_w_mm ({self.plate_w_mm})"
            )
        r_end = self.runner_end_d_mm / 2.0
        if r_end > self.runner_len_mm + eps:
            raise ValueError(
                f"runner_end_d_mm / 2 ({r_end}) must be ≤ runner_len_mm ({self.runner_len_mm}); "
                f"the round end must not reach into the product"
            )
        if not fan:
            self._validate_gate_side(eps)
            return
        if self.runner_ramp_on and self.runner_edge_flat_mm > self.ramp_end_depth_mm + eps:
            raise ValueError(
                f"runner_edge_flat_mm ({self.runner_edge_flat_mm}) must be ≤ the ramp's deep end "
                f"({self.ramp_end_depth_mm:g}, runner_ramp_end_mm = {self.runner_ramp_end_mm})"
            )
        if self.runner_shape == "pentagon":
            if self.side_len_mm > self.runner_len_mm + r_end + eps:
                raise ValueError(
                    f"side_len_mm ({self.side_len_mm}) must be ≤ runner_len_mm + runner_end_d_mm / 2 "
                    f"({self.runner_len_mm + r_end:g}); a corner below the round end's bottom "
                    f"would turn the slants upward"
                )
            corner_dist = math.hypot(0.5 * self.runner_w_mm, self.runner_len_mm - self.side_len_mm)
            if corner_dist <= r_end + eps:
                raise ValueError(
                    f"the corners (runner_w_mm / 2 = {0.5 * self.runner_w_mm:g} off the axis) must "
                    f"lie outside the round end (R {r_end:g}) for the slants to be tangent to it"
                )
        elif self.apex_depth_mm < self.runner_len_mm - r_end - eps:
            raise ValueError(
                f"the flanks meet at depth {self.apex_depth_mm:.2f} (runner_w_mm / 2 · "
                f"tan(fan_flank_deg)) which is above the round end's top "
                f"({self.runner_len_mm - r_end}); the triangle must reach the disc"
            )
        if self.runner_shape == "triangle_arms":
            for name, val in (("arm_w_mm", self.arm_w_mm), ("arm_thk_mm", self.arm_thk_mm)):
                if val <= 0:
                    raise ValueError(f"{name} must be positive for Gate 3 (got {val})")
            if self.arm_w_mm >= self.arm_w_sup_mm - eps:
                raise ValueError(
                    f"arm_w_mm ({self.arm_w_mm}) must be < runner_w_mm / 2 · sin(fan_flank_deg) "
                    f"({self.arm_w_sup_mm:.2f}); wider arms would meet on the axis at the edge line"
                )
            if self.arm_thk_mm <= self.runner_edge_thk_mm + eps:
                raise ValueError(
                    f"arm_thk_mm ({self.arm_thk_mm}) must be > runner_edge_thk_mm "
                    f"({self.runner_edge_thk_mm}); an arm is a groove, it only deepens the runner"
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

    def _validate_gate_side(self, eps: float) -> None:
        """The shapes that are not a fan runner: only what each one reads."""
        shape = self.runner_shape
        if self.balancer_on:
            raise ValueError(
                f"balancer_on applies to the fan runners {FAN_SHAPES} only "
                f"(runner_shape = {shape!r}); it cuts the fan's thickness profile"
            )
        if shape != "direct":
            if self.runner_end_d_mm > self.plate_w_mm + eps:
                raise ValueError(
                    f"runner_end_d_mm ({self.runner_end_d_mm}) must be ≤ plate_w_mm "
                    f"({self.plate_w_mm}); the round end stays inside the plate's x extent"
                )
            if self.gate_d_mm > self.runner_end_d_mm + eps:
                raise ValueError(
                    f"gate_d_mm ({self.gate_d_mm}) must be ≤ runner_end_d_mm "
                    f"({self.runner_end_d_mm})"
                )
            if self.cell_size_mm > self.runner_end_d_mm + eps:
                raise ValueError(
                    f"cell_size_mm ({self.cell_size_mm}) must be ≤ runner_end_d_mm "
                    f"({self.runner_end_d_mm}); a mesh coarser than the round end cannot "
                    f"resolve the runner"
                )
        if shape in TAB_SHAPES:
            checks = [("tab_w_mm", self.tab_w_mm), ("tab_thk_mm", self.tab_thk_mm)]
            if shape == "tab":
                checks.append(("tab_end_thk_mm", self.tab_end_thk_mm))
            else:
                checks.append(("lap_thk_mm", self.lap_thk_mm))
            for name, val in checks:
                if val <= 0:
                    raise ValueError(f"{name} must be positive (got {val})")
            if self.tab_w_mm > self.plate_w_mm + eps:
                raise ValueError(
                    f"tab_w_mm ({self.tab_w_mm}) must be ≤ plate_w_mm ({self.plate_w_mm})"
                )
            if shape == "tab":
                if self.tab_ramp_len_mm < 0:
                    raise ValueError(f"tab_ramp_len_mm must be ≥ 0 (got {self.tab_ramp_len_mm})")
                ramp_max = self.runner_len_mm - self.runner_end_d_mm / 2.0
                if self.tab_ramp_len_mm > ramp_max + eps:
                    raise ValueError(
                        f"tab_ramp_len_mm ({self.tab_ramp_len_mm}) must be ≤ runner_len_mm − "
                        f"runner_end_d_mm / 2 ({ramp_max:g}); the ramp stays outside the round end"
                    )
            elif self.lap_thk_mm < self.frame_thk_mm - eps:
                raise ValueError(
                    f"lap_thk_mm ({self.lap_thk_mm}) must be ≥ frame_thk_mm ({self.frame_thk_mm}); "
                    f"the tab laps under the rim, it only adds to it"
                )
        elif shape == "thick_tab":
            for name, val in (
                ("thick_tab_w_mm", self.thick_tab_w_mm),
                ("thick_tab_thk_mm", self.thick_tab_thk_mm),
            ):
                if val <= 0:
                    raise ValueError(f"{name} must be positive (got {val})")
            if self.corner_keep_mm < 0:
                raise ValueError(f"corner_keep_mm must be ≥ 0 (got {self.corner_keep_mm})")
            if self.thick_tab_w_mm > self.plate_w_mm + eps:
                raise ValueError(
                    f"thick_tab_w_mm ({self.thick_tab_w_mm}) must be ≤ plate_w_mm "
                    f"({self.plate_w_mm})"
                )
        else:  # direct
            if self.gate_d_mm > min(self.plate_w_mm, self.plate_h_mm) - 2 * self.frame_w_mm + eps:
                raise ValueError(
                    f"gate_d_mm ({self.gate_d_mm}) must fit on the body "
                    f"({self.plate_w_mm - 2 * self.frame_w_mm:g} × "
                    f"{self.plate_h_mm - 2 * self.frame_w_mm:g})"
                )
            if self.direct_layout not in DIRECT_LAYOUTS:
                raise ValueError(
                    f"direct_layout ({self.direct_layout!r}) must be one of {DIRECT_LAYOUTS}"
                )
            if self.direct_open_mode not in DIRECT_OPEN_MODES:
                raise ValueError(
                    f"direct_open_mode ({self.direct_open_mode!r}) must be one of "
                    f"{DIRECT_OPEN_MODES}"
                )
            if not 0.0 <= self.direct_delay_s < float("inf"):
                raise ValueError(f"direct_delay_s ({self.direct_delay_s}) must be ≥ 0")
            if self.direct_layout != "center":
                inset_sup = 0.5 * (min(self.plate_w_mm, self.plate_h_mm) - 2 * self.frame_w_mm)
                r_gate = self.gate_d_mm / 2.0
                if not r_gate - eps <= self.direct_inset_mm < inset_sup - eps:
                    raise ValueError(
                        f"direct_inset_mm ({self.direct_inset_mm}) must be in [{r_gate:g}, "
                        f"{inset_sup:g}): at least the gate's radius (gate_d_mm / 2), so the "
                        f"orifice stays on the body, and under half of the body's smaller side"
                    )

    # ----- derived quantities -----
    @property
    def direct_points_mm(self) -> list[tuple[float, float]]:
        """Direct gates: ``(x, y)`` of each gate centre, ``x`` from the sprue
        axis, ``y`` above the product's gate-side edge."""
        yc = 0.5 * self.plate_h_mm
        if self.direct_layout == "center":
            return [(0.0, yc)]
        xk = 0.5 * self.plate_w_mm - self.frame_w_mm - self.direct_inset_mm
        y0 = self.frame_w_mm + self.direct_inset_mm
        y1 = self.plate_h_mm - y0
        if self.direct_layout == "diagonal2":
            return [(-xk, y0), (xk, y1)]
        return [(-xk, y0), (xk, y0), (-xk, y1), (xk, y1)]

    @property
    def ramp_end_depth_mm(self) -> float:
        """The ramp's deep end below the product edge: ``runner_ramp_end_mm``,
        or the top of the round end (``runner_len − runner_end_d / 2``) when
        that is ``None``."""
        if self.runner_ramp_end_mm is None:
            return self.runner_len_mm - self.runner_end_d_mm / 2.0
        return self.runner_ramp_end_mm

    @property
    def pentagon_tangent_mm(self) -> tuple[float, float]:
        """Gate 2: ``(x, depth)`` of the right slant's tangent point on the
        round end, ``x`` from the axis, ``depth`` below the product edge (the
        left one is the mirror image). The lower of the two tangents from the
        corner ``(runner_w/2, side_len)``."""
        r_end = self.runner_end_d_mm / 2.0
        vx = 0.5 * self.runner_w_mm
        vy = self.runner_len_mm - self.side_len_mm  # corner above the axis: > 0
        phi = math.atan2(vy, vx)
        theta = phi - math.acos(min(1.0, r_end / math.hypot(vx, vy)))
        x_t = max(0.0, r_end * math.cos(theta))
        return x_t, self.runner_len_mm - r_end * math.sin(theta)

    @property
    def pentagon_slant_deg(self) -> float:
        """Gate 2: slant angle to the product edge line (follows from
        ``side_len`` and the round end; not an input)."""
        x_t, d_t = self.pentagon_tangent_mm
        return math.degrees(math.atan2(d_t - self.side_len_mm, 0.5 * self.runner_w_mm - x_t))

    @property
    def apex_depth_mm(self) -> float:
        """Depth below the product edge where the extended flanks meet on the axis."""
        return 0.5 * self.runner_w_mm * math.tan(math.radians(self.fan_flank_deg))

    @property
    def arm_w_sup_mm(self) -> float:
        """Gate 3: supremum of ``arm_w_mm`` — the distance from a flank to the
        axis on the edge line, ``runner_w/2 · sin(flank)``. The single source
        for :meth:`validate` and the sidebar bound."""
        return 0.5 * self.runner_w_mm * math.sin(math.radians(self.fan_flank_deg))

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
        linear ramp to ``runner_thk`` at the ramp's deep end, then constant;
        with the ramp off, a step at the band's end. The round-end override
        (``runner_end_thk``) is the builder's, not part of this profile."""
        ramp_end = self.ramp_end_depth_mm if self.runner_ramp_on else self.runner_edge_flat_mm
        ramp_len = ramp_end - self.runner_edge_flat_mm
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

    shape = cfg.runner_shape
    in_end = r2_axis <= r_end**2
    if shape == "direct":
        # no runner at all: the gates sit on the body
        in_end = np.zeros_like(in_end)
        in_runner = in_end
    elif shape in TAB_SHAPES:
        # --- silhouette: trapezoid tab_w on the edge → the disc's diameter on
        # the axis line, ∪ the disc ---
        half_tab = 0.5 * cfg.tab_w_mm + (r_end - 0.5 * cfg.tab_w_mm) * dd / cfg.runner_len_mm
        in_tab = (dd >= 0.0) & (dd <= cfg.runner_len_mm) & (ax <= half_tab)
        in_runner = in_tab | in_end
    elif shape == "thick_tab":
        # --- silhouette: triangle thick_tab_w on the edge, apex on the axis, ∪ the disc ---
        half_tab = 0.5 * cfg.thick_tab_w_mm * (1.0 - dd / cfg.runner_len_mm)
        in_runner = ((dd >= 0.0) & (dd <= cfg.runner_len_mm) & (ax <= half_tab)) | in_end
    elif shape == "pentagon":
        # --- silhouette: convex hull of the side rectangle and the disc ---
        half_w = 0.5 * cfg.runner_w_mm
        s = cfg.side_len_mm
        x_t, d_t = cfg.pentagon_tangent_mm
        # slant from the corner (half_w, s) to the tangent point (x_t, d_t)
        run = max(half_w - x_t, 1e-12)
        d_slant = s + (half_w - ax) * (d_t - s) / run
        in_side = (ax >= x_t) & (ax <= half_w) & (dd >= 0.0) & (dd <= d_slant)
        # between the tangent points the bottom is the disc's arc
        arc = cfg.runner_len_mm + np.sqrt(np.clip(r_end**2 - ax**2, 0.0, None))
        in_mid = (ax < x_t) & (dd >= 0.0) & (dd <= arc)
        in_runner = in_side | in_mid | in_end
    else:
        # --- silhouette: triangle from the edge ∪ round-end disc ---
        half_w_at_depth = 0.5 * cfg.runner_w_mm * (1.0 - dd / apex)
        in_tri = (dd >= 0.0) & (dd <= apex) & (ax <= half_w_at_depth)
        in_runner = in_tri | in_end
    in_x_plate = (xx >= x_left) & (xx <= x_left + cfg.plate_w_mm)
    in_plate = (yy > y_edge) & (yy <= y_top) & in_x_plate
    # the gate-side band of the rim: the gate's to shape (module docstring)
    in_band = in_plate & (yy < y_edge + cfg.frame_w_mm)
    inner_half = 0.5 * cfg.plate_w_mm - cfg.frame_w_mm  # the body's side edge, from the axis
    in_cut = np.zeros_like(in_band)
    if shape == "thick_tab":
        in_cut = in_band & (ax >= 0.5 * cfg.thick_tab_w_mm) & (ax < inner_half - cfg.corner_keep_mm)
    mask = (in_runner | in_plate) & ~in_cut
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
        if shape not in FAN_SHAPES:
            raise ValueError(
                f"cell_size_mm ({dx}) rasterises the cavity into {n_parts} disconnected parts "
                f"(runner_shape = {shape!r}); refine the mesh or widen the tab"
            )
        raise ValueError(
            f"cell_size_mm ({dx}) rasterises the cavity into {n_parts} disconnected parts "
            f"(the runner's flanks barely reach the round end); refine the mesh or steepen "
            f"fan_flank_deg"
        )

    # --- thickness ---
    thk = np.zeros_like(xx, dtype=float)
    runner_thk = np.asarray(cfg.runner_thickness_at_depth(dd), dtype=float)
    thk[in_runner] = runner_thk[in_runner]
    if shape == "tab":
        # tab_end_thk at the edge → tab_thk at tab_ramp_len, on to and inside the disc
        if cfg.tab_ramp_len_mm > 1e-12:
            t_tab = np.clip(dd / cfg.tab_ramp_len_mm, 0.0, 1.0)
        else:
            t_tab = np.ones_like(dd)
        tab_thk = cfg.tab_end_thk_mm + (cfg.tab_thk_mm - cfg.tab_end_thk_mm) * t_tab
        thk[in_runner] = tab_thk[in_runner]
        thk[in_end] = cfg.tab_thk_mm
    elif shape == "overlap_tab":
        thk[in_runner] = cfg.tab_thk_mm
    elif shape == "thick_tab":
        thk[in_runner] = cfg.thick_tab_thk_mm
    elif cfg.runner_end_thk_mm is not None:
        thk[in_end] = cfg.runner_end_thk_mm

    # Gate 3 arms: a strip arm_w wide (square to the flank) inside the
    # triangle, past the edge band, outside the round end; a groove only
    # deepens, so the thickness there is max(profile, arm_thk)
    if shape == "triangle_arms":
        th = math.radians(cfg.fan_flank_deg)
        dist_flank = (0.5 * cfg.runner_w_mm - ax) * math.sin(th) - dd * math.cos(th)
        in_arm = in_tri & ~in_end & (dd > cfg.runner_edge_flat_mm) & (dist_flank <= cfg.arm_w_mm)
        thk[in_arm] = np.maximum(thk[in_arm], cfg.arm_thk_mm)

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
    # the gate-side band where the gate thickens it
    if shape == "overlap_tab":
        thk[in_band & (ax < 0.5 * cfg.tab_w_mm)] = cfg.lap_thk_mm
    elif shape == "thick_tab":
        thk[in_band & (ax < 0.5 * cfg.thick_tab_w_mm)] = cfg.thick_tab_thk_mm

    thk[~mask] = 0.0

    geom = Geometry(
        mask=mask,
        thickness_mm=thk,
        cell_size_mm=dx,
        label={
            "triangle": "fan_runner_plate",
            "pentagon": "pentagon_runner_plate",
            "triangle_arms": "arms_runner_plate",
            "tab": "tab_runner_plate",
            "overlap_tab": "overlap_tab_plate",
            "thick_tab": "thick_tab_plate",
            "direct": "direct_gate_plate",
        }[shape],
        # ICM squeezes only the t4 body; the rim and the runner are fixed
        compression_mask=in_inner & mask,
        product_mask=in_plate & mask,
        valve_axis_x_mm=cx,
        # nominal injection orifice = the hot-runner gate disc on the axis;
        # the gate marker draws this one true-scale circle
        valve_marker_mm=(float(cx), float(y_axis), cfg.gate_d_mm / 2.0),
    )

    if shape == "direct":
        # --- injection points: a gate disc at each point on the body ---
        pts = cfg.direct_points_mm
        # on the body only: a gate cell on the thin rim would feed the rim
        # past the step that the melt has to cross to reach it
        body = in_inner & mask
        if not body.any():
            raise ValueError(
                f"cell_size_mm ({dx}) rasterises the body away (it is "
                f"{cfg.plate_w_mm - 2 * cfg.frame_w_mm:g} × {cfg.plate_h_mm - 2 * cfg.frame_w_mm:g}"
                f" mm); a direct gate needs a body cell to sit on. Refine the mesh"
            )
        for k, (px, py) in enumerate(pts):
            r2 = (xx - cx - px) ** 2 + (yy - y_edge - py) ** 2
            sel = (r2 <= (cfg.gate_d_mm / 2.0) ** 2) & body
            if not sel.any():
                sel[np.unravel_index(int(np.argmin(np.where(body, r2, np.inf))), r2.shape)] = True
                geom.valve_marker_mm = None
            own = []
            for iy, ix in zip(*np.where(sel), strict=True):
                if (int(iy), int(ix)) not in geom.gates:
                    geom.gates.append((int(iy), int(ix)))
                    own.append((int(iy), int(ix)))
            # the gates after the first open later, when asked to
            if k > 0 and own and cfg.direct_open_mode != "together":
                geom.delayed_gates.append(
                    DelayedGate(
                        cells=tuple(own),
                        delay_s=float(cfg.direct_delay_s),
                        on_arrival=cfg.direct_open_mode == "arrival",
                    )
                )
        if len(pts) == 1 and geom.valve_marker_mm is not None:
            geom.valve_marker_mm = (
                float(cx + pts[0][0]),
                float(y_edge + pts[0][1]),
                cfg.gate_d_mm / 2.0,
            )
        else:
            # one marker cannot stand for several gates: draw the gate cells
            geom.valve_marker_mm = None
        return geom

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
