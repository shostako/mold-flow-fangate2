"""Temperature across the thickness, carried layer by layer and advanced in time.

The closed forms used so far (:func:`core.multilayer_thermal.neumann_layer_temperatures`
in the fill, the slab series in :mod:`core.packing_solve`) read a cell's
temperature profile off one number, the time its melt has been cooling in a
slab of the cell's own thickness. That holds on a plate of one thickness. It
fails where the melt changes thickness on its way: melt that crossed a 4 mm
plate and enters a 1 mm rim late is taken to have cooled in 1 mm all along, so
the rim is solid before the melt gets there.

Here the profile is a state: ``theta[cell, layer]``, the temperature as a
fraction of the way from the wall to the melt (``θ = (T − T_mold) / (T_melt −
T_mold)``), on the half thickness (the profile is symmetric). Per time step

* :func:`conduct` cools it to the walls -- finite volumes across the thickness,
  exact in time (the operator is diagonalised once, so a step of any length is
  one matrix product per cell and never unstable);
* :func:`advect` lets the in-plane flow carry it from cell to cell, layer by
  layer, upwind. A cell that is still filling takes the flow-weighted mean of
  what comes in, on all its layers: the fountain flow at the front lays core
  melt on the wall;
* :func:`shear_heating` adds the viscous dissipation.

The layers are the Gauss nodes of :mod:`core.packing_solve`'s quadrature, with
the weights as their thicknesses; layer ``k`` of one cell is taken to feed
layer ``k`` of the next whatever the two thicknesses (the Hele-Shaw mid-plane
view: the same relative depth).

Not in the model: conduction in the plane, the heat of compression, the latent
heat (PMMA has none), a contact resistance at the wall (the wall is at the mold
temperature).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class LayerGrid:
    """Layers over the half thickness, centre → wall, in units of the thickness."""

    xi: np.ndarray  # (n_z,) layer centres, in (0, ½)
    wt: np.ndarray  # (n_z,) layer thicknesses, sum ½
    decay: np.ndarray  # (n_z,) eigenvalues of the conduction operator per unit Fourier number (≤ 0)
    modes: np.ndarray  # (n_z, n_z) columns: eigenvectors
    modes_inv: np.ndarray  # (n_z, n_z)

    @property
    def n_z(self) -> int:
        return int(self.xi.size)


@lru_cache(maxsize=8)
def layer_grid(n_z: int = 16) -> LayerGrid:
    """Gauss layers and their conduction operator.

    ``dθ_k/dFo = (1/w_k)·[(θ_{k+1} − θ_k)/(ξ_{k+1} − ξ_k) − (θ_k − θ_{k−1})/(ξ_k − ξ_{k−1})]``
    with no flux at the centre and ``θ = 0`` at the wall (``ξ = ½``);
    ``Fo = α·t/h²``. The operator is similar to a symmetric matrix, so its
    eigenvalues are real and negative.
    """
    x, w = np.polynomial.legendre.leggauss(int(n_z))
    xi = 0.25 * (x + 1.0)
    wt = 0.25 * w
    stiff = np.zeros((n_z, n_z))
    for k in range(n_z - 1):
        c = 1.0 / (xi[k + 1] - xi[k])
        stiff[k, k] -= c
        stiff[k + 1, k + 1] -= c
        stiff[k, k + 1] += c
        stiff[k + 1, k] += c
    stiff[-1, -1] -= 1.0 / (0.5 - xi[-1])
    r = 1.0 / np.sqrt(wt)
    lam, q = np.linalg.eigh(stiff * r[:, None] * r[None, :])
    for a in (xi, wt, lam):
        a.setflags(write=False)
    return LayerGrid(xi=xi, wt=wt, decay=lam, modes=q * r[:, None], modes_inv=q.T / r[None, :])


def conduct(theta: np.ndarray, fourier_step: np.ndarray, grid: LayerGrid) -> np.ndarray:
    """``theta`` after conduction to the walls over ``fourier_step = α·dt/h²``.

    ``theta`` is ``(n, n_z)``, ``fourier_step`` ``(n,)``. Exact in time for the
    layered slab, whatever the step.
    """
    coeff = theta @ grid.modes_inv.T
    coeff *= np.exp(np.outer(np.asarray(fourier_step, dtype=float), grid.decay))
    out = coeff @ grid.modes.T
    # the exact operator keeps 0 ≤ θ ≤ 1 up to round-off
    return np.clip(out, 0.0, None)


def flow_shares(inverse_viscosity: np.ndarray, grid: LayerGrid) -> np.ndarray:
    """Share of a cell's in-plane flow that each layer carries, ``(n, n_z)``.

    With no slip at the wall the speed at depth ``ξ`` is
    ``u(ξ) ∝ ∫_ξ^½ ξ'/η dξ'``: a layer moves with everything between it and
    the wall. ``inverse_viscosity`` is ``1/η`` at the layers, zero where the
    melt is solid, so a solid layer and all layers outside it stand still.
    Rows where nothing flows are all zero.
    """
    f = inverse_viscosity * (grid.xi * grid.wt)[None, :]
    outside = np.cumsum(f[:, ::-1], axis=1)[:, ::-1]
    carried = (outside - 0.5 * f) * grid.wt[None, :]
    total = carried.sum(axis=1, keepdims=True)
    ok = total > 0.0
    return np.where(ok, carried / np.where(ok, total, 1.0), 0.0)


def advect(
    theta: np.ndarray,
    shares: np.ndarray,
    held: np.ndarray,
    face_a: np.ndarray,
    face_b: np.ndarray,
    moved_a_to_b: np.ndarray,
    filling: np.ndarray,
    grid: LayerGrid,
    *,
    source: np.ndarray | None = None,
    source_theta: float = 1.0,
) -> np.ndarray:
    """``theta`` after the flow of one step has carried it across the faces.

    Parameters
    ----------
    theta, shares
        ``(n, n_z)``: the profiles before the step, and :func:`flow_shares`.
    held
        ``(n,)`` amount of melt in each cell before the step (mass or volume;
        the same unit as ``moved_a_to_b``).
    face_a, face_b, moved_a_to_b
        The faces as pairs of cell indices and the amount that crossed each
        over the step, positive from ``a`` to ``b``.
    filling
        ``(n,)`` bool: cells that are not full. They take the flow-weighted
        mean of what comes in on all layers (the fountain flow), mixed with
        what they hold.
    source
        ``(n,)`` amount that entered each cell from outside the cavity over
        the step (the gate), at ``source_theta``.

    A full cell's layer ``k`` takes ``share_k`` of the cell's inflow, at the
    donor's layer ``k`` (upwind, implicit in the receiver):
    ``θ_k ← (m_k·θ_k + Σ in_k·θ_donor,k) / (m_k + Σ in_k)``. That is a mean
    with positive weights, so the result stays between the values it mixes
    whatever the step. A full cell with no flowing layer (solid through)
    keeps its profile.
    """
    n, n_z = theta.shape
    moved = np.asarray(moved_a_to_b, dtype=float)
    fwd = moved > 0.0
    rev = moved < 0.0
    donor = np.concatenate([face_a[fwd], face_b[rev]])
    taker = np.concatenate([face_b[fwd], face_a[rev]])
    amount = np.concatenate([moved[fwd], -moved[rev]])

    inflow = np.bincount(taker, amount, minlength=n)
    # full cells: layer by layer, at the donor's same layer
    carried = np.empty((n, n_z))
    for k in range(n_z):
        carried[:, k] = np.bincount(taker, amount * theta[donor, k], minlength=n)
    # filling cells: the donor's flow-weighted mean, on every layer
    donor_bulk = np.einsum("ij,ij->i", shares, theta)
    stagnant = shares.sum(axis=1) <= 0.0
    if stagnant.any():  # a donor with no flowing layer passes its mean (it passes next to nothing)
        donor_bulk = np.where(stagnant, theta @ grid.wt / 0.5, donor_bulk)
    bulk_in = np.bincount(taker, amount * donor_bulk[donor], minlength=n)
    if source is not None:
        src = np.asarray(source, dtype=float)
        inflow = inflow + src
        carried += (src * source_theta)[:, None]
        bulk_in = bulk_in + src * source_theta

    layer_held = held[:, None] * (grid.wt / 0.5)[None, :]
    layer_in = inflow[:, None] * shares
    denom = layer_held + layer_in
    with np.errstate(invalid="ignore", divide="ignore"):
        full_new = np.where(denom > 0.0, (layer_held * theta + shares * carried) / denom, theta)
        tot = held + inflow
        fill_new = np.where(
            tot[:, None] > 0.0,
            (held[:, None] * theta + bulk_in[:, None]) / np.maximum(tot, 1e-300)[:, None],
            theta,
        )
    out = np.where(filling[:, None], fill_new, full_new)
    # a cell nothing flowed into keeps its profile to the last bit
    return np.where(inflow[:, None] > 0.0, out, theta)


def face_dissipation(
    face_a: np.ndarray,
    face_b: np.ndarray,
    conductance_a: np.ndarray,
    conductance_b: np.ndarray,
    pressure_drop_Pa: np.ndarray,
    dt_s: float,
    n: int,
) -> np.ndarray:
    """Heat [J] the flow dissipated in each cell over a step, ``(n,)``.

    A face between cells of conductance ``S_a`` and ``S_b`` [m³/(Pa·s)]
    carries ``S_f·Δp`` with ``S_f`` their harmonic mean, and turns
    ``S_f·Δp²`` into heat. The two cells are resistances in series, so each
    takes the share of the drop that falls across it: cell ``a`` gets
    ``S_b/(S_a + S_b)``. Summed over the faces this is the work the pressure
    does on the melt, to round-off -- which a gradient taken at the cell
    centres is not: next to a nearly solid cell the whole drop sits on the face,
    and reading it as the open cell's own gradient heats that cell with work
    nothing did.
    """
    total = conductance_a + conductance_b
    ok = (conductance_a > 0.0) & (conductance_b > 0.0)
    safe = np.where(ok, total, 1.0)
    s_face = np.where(ok, 2.0 * conductance_a * conductance_b / safe, 0.0)
    heat = s_face * np.asarray(pressure_drop_Pa, dtype=float) ** 2 * dt_s
    share_a = np.where(ok, conductance_b / safe, 0.0)
    return np.bincount(face_a, heat * share_a, minlength=n) + np.bincount(
        face_b, heat * (1.0 - share_a) * ok, minlength=n
    )


def shear_heating(
    dissipated_J: np.ndarray,
    inverse_viscosity: np.ndarray,
    cell_volume_m3: np.ndarray,
    volumetric_heat_capacity_J_m3K: float,
    delta_T_K: float,
    grid: LayerGrid,
    *,
    max_rise: float = 0.25,
) -> np.ndarray:
    """Rise of ``theta`` from the heat a cell's flow dissipated, ``(n, n_z)``.

    The shear stress at depth ``z`` is ``|∇p|·z`` and the dissipation
    ``τ²/η``, so a cell's heat (:func:`face_dissipation`) goes to its layers
    in proportion to ``ξ²/η``: most where the melt is sheared hardest and
    still flows, none at the centre and none in a solid layer.
    ``inverse_viscosity`` is ``1/η`` at the layers (zero where solid). The
    rise per step is capped at ``max_rise`` of the melt-to-mold span.
    """
    weight = inverse_viscosity * (grid.xi**2)[None, :]
    norm = weight @ grid.wt  # ∫ ξ²/η dξ over the half thickness
    ok = norm > 0.0
    per_layer = np.where(ok[:, None], weight / np.where(ok, norm, 1.0)[:, None], 0.0)
    # a layer of thickness w_k (of ½) holds w_k/½ of the cell's volume
    rise = (
        np.asarray(dissipated_J, dtype=float)[:, None]
        * per_layer
        * 0.5
        / (cell_volume_m3[:, None] * volumetric_heat_capacity_J_m3K * delta_T_K)
    )
    return np.minimum(rise, max_rise)
