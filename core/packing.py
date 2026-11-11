"""How long holding pressure reaches each cell after the fill (stage 1).

A reading of the fill result, not a packing solve: no melt moves, nothing is
compressed. It answers one question -- for how much of a cell's own
solidification time is there still a molten path from the gate to it?

Model
-----

The temperature treatment is the fill solver's own
(:mod:`core.multilayer_thermal`): the melt that sits in a cell at the end of
the fill has been between two walls at the mold temperature for as long as it
took to get there, its arrival time ``t_arr``. After the fill the flow stops
and the slab goes on cooling by conduction alone. The centre of a slab of
thickness ``h`` reaches the solidification temperature

    T_solid = T_mold + f · (T_melt − T_mold)

at the melt age

    t_f(h) = Fo*(f) · h² / α

with ``Fo*`` the Fourier number at which the exact series for a slab cooled
on both faces,

    θ_c(Fo) = (4/π) · Σ_n (−1)ⁿ / (2n+1) · exp(−(2n+1)² π² Fo),

falls to ``f``. Counted from the end of the fill, a cell therefore stays
molten at its centre for

    s = max(t_f(h) − t_arr, 0)

(``s = 0``: frozen through by the time the fill ends).

Holding pressure reaches a cell for as long as some path from a gate to it is
molten along its whole length. A path lasts as long as its first cell to
freeze; the best path is the one whose first freeze comes latest:

    R(x) = max over paths gate→x of  min over cells c on the path of  s(c)

-- a widest-path (bottleneck) problem on the 4-connected cavity grid, solved
with a best-first search. The gates themselves are taken as never freezing
(a hot runner with the valve held open). The ratio ``R / s`` is the share of
the cell's solidification time during which it can still be fed.

Not in the model
----------------

* the heat the holding flow carries into the path (lengthens the reach),
* the frozen layer that grew on the walls *during* the fill near the gate,
  where the melt is counted as fresh (shortens the reach),
* the narrowing of the path by that layer, and the melt's compressibility
  (pvT): no pressure history, no shrinkage, no sink depth.

The first two pull in opposite directions, so the seconds are a guide. The
ranking between shapes and the ratios follow the thinnest section on the
path, which is what the reading is for.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

#: Terms kept in the slab series. At the smallest Fourier number the root
#: search visits (``_FO_MIN``) the n-th term is ~exp(−(2n+1)²·π²·1e-4); 200
#: terms leave a tail below 1e-60.
_SERIES_TERMS = 200
_FO_MIN, _FO_MAX = 1e-4, 50.0


def slab_centre_temperature(fourier_number: float) -> float:
    """Dimensionless centre temperature of a slab cooled on both faces.

    1 = the initial (melt) temperature, 0 = the wall temperature.
    ``fourier_number = α·t / h²`` on the full thickness ``h``.
    """
    fo = float(fourier_number)
    if fo <= 0.0:
        return 1.0
    n = np.arange(_SERIES_TERMS)
    k = 2 * n + 1
    return float((4.0 / np.pi) * np.sum((-1.0) ** n / k * np.exp(-(k**2) * np.pi**2 * fo)))


def freeze_fourier_number(solid_fraction: float) -> float:
    """Fourier number at which the slab's centre has cooled to ``solid_fraction``.

    ``solid_fraction`` is the solver's ``solidification_temperature_fraction``:
    0 = the mold temperature, 1 = the melt temperature.
    """
    f = float(solid_fraction)
    if not 0.0 < f < 1.0:
        raise ValueError(f"solid_fraction must be in (0, 1), got {solid_fraction}")
    # θ_c is 1 at _FO_MIN (to rounding) and ~0 at _FO_MAX, and falls in between
    return float(brentq(lambda fo: slab_centre_temperature(fo) - f, _FO_MIN, _FO_MAX, xtol=1e-12))


def freeze_age_s(
    thickness_mm: np.ndarray, thermal_diffusivity_m2_s: float, solid_fraction: float
) -> np.ndarray:
    """Melt age [s] at which the centre of each cell's slab solidifies."""
    alpha = float(thermal_diffusivity_m2_s)
    if alpha <= 0.0:
        raise ValueError(f"thermal_diffusivity_m2_s must be > 0, got {alpha}")
    h_m = np.asarray(thickness_mm, dtype=float) * 1e-3
    return freeze_fourier_number(solid_fraction) * h_m**2 / alpha


@dataclass(frozen=True)
class PackingReach:
    """Per-cell reading; every array is NaN outside the cavity."""

    solidify_s: np.ndarray  # time after the fill until the centre freezes (0 = frozen at the end)
    reach_s: np.ndarray  # time after the fill during which a molten path from a gate remains
    ratio: np.ndarray  # reach / solidify in [0, 1]; NaN where solidify is 0
    freeze_age_s: np.ndarray  # t_f(h): melt age at which the centre freezes
    fourier_number: float  # Fo*(solid_fraction)
    solid_fraction: float


def packing_reach(
    mask: np.ndarray,
    thickness_mm: np.ndarray,
    arrival_time_s: np.ndarray,
    gates: list[tuple[int, int]],
    thermal_diffusivity_m2_s: float,
    solid_fraction: float = 0.3,
) -> PackingReach:
    """Solidification time and holding-pressure reach of every cavity cell.

    ``arrival_time_s`` is the fill result's ``fill_time_s`` (NaN where the
    melt never arrived: such a cell is not part of any path and gets
    ``solidify = reach = 0``).
    """
    mask = np.asarray(mask, dtype=bool)
    thk = np.asarray(thickness_mm, dtype=float)
    t_arr = np.asarray(arrival_time_s, dtype=float)
    if not (mask.shape == thk.shape == t_arr.shape):
        raise ValueError(
            f"shape mismatch: mask {mask.shape}, thickness {thk.shape}, arrival {t_arr.shape}"
        )
    gate_cells = [(int(iy), int(ix)) for iy, ix in gates if mask[int(iy), int(ix)]]
    if not gate_cells:
        raise ValueError("no gate cell inside the cavity")
    t_f = freeze_age_s(np.where(mask, thk, 0.0), thermal_diffusivity_m2_s, solid_fraction)
    arrived = mask & np.isfinite(t_arr)
    s = np.where(arrived, np.maximum(t_f - np.where(arrived, t_arr, 0.0), 0.0), 0.0)

    ny, nx = mask.shape
    reach = np.full(mask.shape, -1.0)
    heap: list[tuple[float, int, int]] = []
    for iy, ix in gate_cells:
        reach[iy, ix] = np.inf
        heap.append((-np.inf, iy, ix))
    heapq.heapify(heap)
    while heap:
        neg, iy, ix = heapq.heappop(heap)
        r = -neg
        if r < reach[iy, ix]:
            continue
        for jy, jx in ((iy + 1, ix), (iy - 1, ix), (iy, ix + 1), (iy, ix - 1)):
            if not (0 <= jy < ny and 0 <= jx < nx) or not mask[jy, jx]:
                continue
            cand = min(r, s[jy, jx])
            if cand > reach[jy, jx]:
                reach[jy, jx] = cand
                heapq.heappush(heap, (-cand, jy, jx))
    # a gate cell is fed for as long as it is molten itself; a cell no gate
    # connects to (cannot happen in a connected cavity) is never fed
    reach = np.where(np.isinf(reach), s, np.maximum(reach, 0.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(s > 0.0, np.minimum(reach / s, 1.0), np.nan)
    nan = np.nan
    return PackingReach(
        solidify_s=np.where(mask, s, nan),
        reach_s=np.where(mask, reach, nan),
        ratio=np.where(mask, ratio, nan),
        freeze_age_s=np.where(mask, t_f, nan),
        fourier_number=freeze_fourier_number(solid_fraction),
        solid_fraction=float(solid_fraction),
    )


def summarize_by_thickness(
    reach: PackingReach,
    thickness_mm: np.ndarray,
    region: np.ndarray,
    cell_size_mm: float,
    decimals: int = 1,
) -> list[dict]:
    """One row per wall thickness inside ``region`` (thickest first).

    Each row: the thickness, its area, the share already frozen through at
    the end of the fill, and -- over the cells still molten -- the mean
    solidification time, the mean reach and the mean ratio.
    """
    region = np.asarray(region, dtype=bool)
    thk = np.round(np.asarray(thickness_mm, dtype=float), decimals)
    rows: list[dict] = []
    cell_cm2 = float(cell_size_mm) ** 2 / 100.0
    for h in sorted({float(v) for v in thk[region]}, reverse=True):
        sel = region & (thk == h)
        s = reach.solidify_s[sel]
        molten = s > 0.0
        row = {
            "thickness_mm": h,
            "area_cm2": float(sel.sum()) * cell_cm2,
            "frozen_at_fill_end": float((~molten).mean()),
            "solidify_s": None,
            "reach_s": None,
            "ratio": None,
        }
        if molten.any():
            row["solidify_s"] = float(s[molten].mean())
            row["reach_s"] = float(reach.reach_s[sel][molten].mean())
            row["ratio"] = float(reach.ratio[sel][molten].mean())
        rows.append(row)
    return rows
