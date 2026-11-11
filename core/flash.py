"""Where the parting line is likely to flash.

The cavity's outline lies on the parting line (PL): the plate and the fan
runner are cut in the parting plane, so wherever the outline meets the
mould there is a seal between the two halves. Melt that is still molten and
under pressure there pushes into the gap and leaves a flash. This module
reads the holding solve (:mod:`core.packing_solve`) on those outline cells:
the highest pressure each one saw while its centre was still molten. The
pressure after the centre has solidified does not count: solid melt does
not enter the gap.

What is not here: the gap itself (vent depth, wear, the mould's deflection
under the clamp), so the map ranks the places and gives their pressure; a
pressure at which a given mould flashes is the user's to judge. The clamp
check compares the force that opens the mould (∫ p dA, from the solve) with
the machine's clamp force.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry


@dataclass(frozen=True)
class FlashSpot:
    """One place on the outline, at the cell with the highest pressure there."""

    x_mm: float  # display frame (valve axis = 0, product's gate-side edge = 0)
    y_mm: float
    pressure_Pa: float  # highest pressure while the centre was molten
    time_s: float  # when, after the V/P switch
    thickness_mm: float
    in_product: bool  # False: on the runner


@dataclass(frozen=True)
class FlashRisk:
    edge: np.ndarray  # bool map: the cavity cells on the outline (the PL)
    pressure_Pa: np.ndarray  # map: the pressure above on the outline, NaN elsewhere
    spots: list[FlashSpot] = field(default_factory=list)  # on the product, highest first
    # on the runner, highest first: next to the gate the outline holds about the
    # holding pressure whatever the shape, so it is ranked apart from the product
    runner_spots: list[FlashSpot] = field(default_factory=list)
    clamp_peak_N: float = 0.0  # the most the cavity pushed the mould open
    clamp_peak_time_s: float = 0.0
    machine_clamp_N: float | None = None


def outline_cells(mask: np.ndarray) -> np.ndarray:
    """Cavity cells that touch a cell outside the cavity (or the grid's edge)
    across a face."""
    mask = np.asarray(mask, dtype=bool)
    pad = np.pad(mask, 1, constant_values=False)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~inner


def flash_risk(
    geometry: Geometry,
    solve,
    *,
    n_spots: int = 5,
    min_gap_mm: float = 20.0,
    machine_clamp_N: float | None = None,
) -> FlashRisk:
    """Rank the outline by the pressure it held while still molten.

    ``solve`` is a :class:`core.packing_solve.PackingSolve` on the
    geometry's grid. Spots are taken greedily from the highest pressure, each
    at least ``min_gap_mm`` from the ones before, so that one long stretch of
    high pressure does not fill the list. The product's outline and the
    runner's (``Geometry.product_mask``) are ranked apart, ``n_spots`` each.
    """
    if n_spots < 0 or min_gap_mm < 0.0:
        raise ValueError("n_spots and min_gap_mm must be >= 0")
    mask = np.asarray(geometry.mask, dtype=bool)
    p_map = np.asarray(solve.molten_peak_pressure_Pa, dtype=float)
    t_map = np.asarray(solve.molten_peak_time_s, dtype=float)
    if p_map.shape != mask.shape:
        raise ValueError(f"the solve is on {p_map.shape}, the geometry on {mask.shape}")
    edge = outline_cells(mask) & np.isfinite(p_map)
    on_edge = np.where(edge, p_map, np.nan)
    product = (
        np.ones_like(mask) if geometry.product_mask is None else np.asarray(geometry.product_mask)
    )

    dx = float(geometry.cell_size_mm)
    x0, y0 = geometry.display_origin_mm()

    def rank(where: np.ndarray, in_product: bool) -> list[FlashSpot]:
        iy, ix = np.nonzero(edge & where & (p_map > 0.0))
        order = np.argsort(-p_map[iy, ix], kind="stable")
        spots: list[FlashSpot] = []
        for k in order:
            if len(spots) >= n_spots:
                break
            y, x = int(iy[k]), int(ix[k])
            xm, ym = (x + 0.5) * dx - x0, (y + 0.5) * dx - y0
            if any((xm - s.x_mm) ** 2 + (ym - s.y_mm) ** 2 < min_gap_mm**2 for s in spots):
                continue
            spots.append(
                FlashSpot(
                    x_mm=float(xm),
                    y_mm=float(ym),
                    pressure_Pa=float(p_map[y, x]),
                    time_s=float(t_map[y, x]),
                    thickness_mm=float(geometry.thickness_mm[y, x]),
                    in_product=in_product,
                )
            )
        return spots

    clamp = np.asarray(solve.clamp_force_N, dtype=float)
    times = np.asarray(solve.times_s, dtype=float)
    j = int(np.argmax(clamp)) if clamp.size else 0
    return FlashRisk(
        edge=edge,
        pressure_Pa=on_edge,
        spots=rank(product, True),
        runner_spots=rank(~product, False),
        clamp_peak_N=float(clamp[j]) if clamp.size else 0.0,
        clamp_peak_time_s=float(times[j]) if times.size else 0.0,
        machine_clamp_N=None if machine_clamp_N is None else float(machine_clamp_N),
    )
