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

The wall is at the mold temperature unless a contact is given
(:func:`wall_contact`): then the melt's surface and the mould are joined by a
heat transfer coefficient ``h_c`` [W/(m²·K)], the flux at the wall is
``h_c·(T_surface − T_mold)``, and the slab cools as its Biot number
``Bi = h_c·h/k`` allows (``Bi → ∞`` is the wall at the mold temperature).

Not in the model: conduction in the plane, the heat of compression, the latent
heat (PMMA has none).
"""

from __future__ import annotations

import math
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
    # the two transposed, contiguous (a matrix product with a transposed view
    # is several times slower)
    modes_t: np.ndarray
    modes_inv_t: np.ndarray

    @property
    def n_z(self) -> int:
        return int(self.xi.size)


@lru_cache(maxsize=2048)
def layer_grid(n_z: int = 16, biot: float = math.inf) -> LayerGrid:
    """Gauss layers and their conduction operator.

    ``dθ_k/dFo = (1/w_k)·[(θ_{k+1} − θ_k)/(ξ_{k+1} − ξ_k) − (θ_k − θ_{k−1})/(ξ_k − ξ_{k−1})]``
    with no flux at the centre and ``θ = 0`` at the wall (``ξ = ½``);
    ``Fo = α·t/h²``. With a finite ``biot = h_c·h/k`` the outermost layer
    reaches the mould through its own half and the contact in series:
    ``−θ_last / (½ − ξ_last + 1/Bi)`` (``biot = 0``: an insulated wall). The
    operator is similar to a symmetric matrix, so its eigenvalues are real and
    not positive. ``xi`` and ``wt`` do not depend on ``biot``.
    """
    if not biot >= 0.0:
        raise ValueError(f"biot must be >= 0 (got {biot})")
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
    if biot > 0.0:
        stiff[-1, -1] -= 1.0 / ((0.5 - xi[-1]) + (0.0 if math.isinf(biot) else 1.0 / biot))
    r = 1.0 / np.sqrt(wt)
    lam, q = np.linalg.eigh(stiff * r[:, None] * r[None, :])
    lam = np.minimum(lam, 0.0)  # (an insulated wall has one mode at zero, to round-off)
    modes, modes_inv = q * r[:, None], q.T / r[None, :]
    for a in (xi, wt, lam):
        a.setflags(write=False)
    return LayerGrid(
        xi=xi,
        wt=wt,
        decay=lam,
        modes=modes,
        modes_inv=modes_inv,
        modes_t=np.ascontiguousarray(modes.T),
        modes_inv_t=np.ascontiguousarray(modes_inv.T),
    )


#: Biot numbers are grouped in bins this wide (in ``ln Bi``): cells of one bin
#: share one diagonalised operator. The bin's centre is within 1 % of every
#: Biot number in it, which moves a slab's cooling time by under 1 %.
_BIOT_BIN = math.log1p(0.02)


@dataclass(frozen=True)
class WallContact:
    """The cells' contact with the mould, grouped by Biot number.

    ``groups`` is ``(cells, grid)`` pairs: the rows that share the conduction
    operator ``grid`` (:func:`layer_grid` at the bin's Biot number). Every cell
    is in exactly one group.
    """

    n: int
    groups: tuple[tuple[np.ndarray, LayerGrid], ...]


def wall_contact(biot: np.ndarray, n_z: int = 16) -> WallContact:
    """Group the cells by their Biot number ``h_c·h/k`` (``(n,)``, ≥ 0;
    ``inf``: the wall at the mold temperature, 0: insulated)."""
    bi = np.asarray(biot, dtype=float).ravel()
    if not np.all(bi >= 0.0):
        raise ValueError("biot must be >= 0")
    key = np.full(bi.shape, np.iinfo(np.int64).max, dtype=np.int64)  # inf
    finite = np.isfinite(bi) & (bi > 0.0)
    key[finite] = np.round(np.log(bi[finite]) / _BIOT_BIN).astype(np.int64)
    key[bi == 0.0] = np.iinfo(np.int64).min
    groups = []
    for k in np.unique(key):
        if k == np.iinfo(np.int64).max:
            value = math.inf
        elif k == np.iinfo(np.int64).min:
            value = 0.0
        else:
            value = math.exp(float(k) * _BIOT_BIN)
        groups.append((np.flatnonzero(key == k), layer_grid(int(n_z), value)))
    return WallContact(n=int(bi.size), groups=tuple(groups))


def conduct(
    theta: np.ndarray,
    fourier_step: np.ndarray,
    grid: LayerGrid,
    wall: WallContact | None = None,
) -> np.ndarray:
    """``theta`` after conduction to the walls over ``fourier_step = α·dt/h²``.

    ``theta`` is ``(n, n_z)``, ``fourier_step`` ``(n,)``. Exact in time for the
    layered slab, whatever the step. ``wall`` (:func:`wall_contact`) puts a
    contact resistance between the melt and the mould; without it the wall is
    at the mold temperature.
    """
    return _conduct_parts(theta, _decay_parts(fourier_step, grid, wall))


def freeze_fourier(
    wall: WallContact, grid: LayerGrid, solid_fraction: float, *, tol: float = 1e-4
) -> np.ndarray:
    """Fourier number ``α·t/h²`` at which the centre of a slab that starts as
    all melt (``θ = 1``) and only conducts falls to ``solid_fraction``, per
    cell of ``wall``. ``inf`` behind an insulated wall."""
    if not 0.0 < solid_fraction < 1.0:
        raise ValueError(f"solid_fraction must be in (0, 1) (got {solid_fraction})")
    out = np.empty(wall.n)
    one = np.ones((1, grid.n_z))
    for cells, g in wall.groups:
        if g.decay.max() >= -1e-8:  # a mode that does not decay: insulated
            out[cells] = np.inf
            continue

        def centre(fo: float, g=g) -> float:
            return float(conduct_by(one, decay_factors(np.array([fo]), g), g)[0, 0])

        lo, hi = 0.0, 0.05
        while centre(hi) > solid_fraction:
            lo, hi = hi, 2.0 * hi
        while hi - lo > tol * hi:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if centre(mid) > solid_fraction else (lo, mid)
        out[cells] = hi
    return out


def _decay_parts(fourier_step: np.ndarray, grid: LayerGrid, wall: WallContact | None) -> list:
    """``(cells, grid, factors)`` per group of cells that share an operator
    (``cells`` ``None``: all of them)."""
    fo = np.asarray(fourier_step, dtype=float)
    if wall is None:
        return [(None, grid, decay_factors(fo, grid))]
    if fo.shape != (wall.n,):
        raise ValueError(f"fourier_step must be ({wall.n},) for this wall, got {fo.shape}")
    if len(wall.groups) == 1:
        return [(None, wall.groups[0][1], decay_factors(fo, wall.groups[0][1]))]
    return [(cells, g, decay_factors(fo[cells], g)) for cells, g in wall.groups]


def _conduct_parts(theta: np.ndarray, parts: list) -> np.ndarray:
    if len(parts) == 1 and parts[0][0] is None:
        return conduct_by(theta, parts[0][2], parts[0][1])
    out = np.empty_like(theta)
    for cells, g, factors in parts:
        out[cells] = conduct_by(theta[cells], factors, g)
    return out


def decay_factors(fourier_step: np.ndarray, grid: LayerGrid) -> np.ndarray:
    """``exp(Fo·λ_j)`` per cell and mode, ``(n, n_z)``: what :func:`conduct_by`
    multiplies the modes by (computed once for repeated equal steps)."""
    return np.exp(np.outer(np.asarray(fourier_step, dtype=float), grid.decay))


def conduct_by(theta: np.ndarray, factors: np.ndarray, grid: LayerGrid) -> np.ndarray:
    """:func:`conduct` with the factors of :func:`decay_factors` given."""
    out = ((theta @ grid.modes_inv_t) * factors) @ grid.modes_t
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
    carried = (
        np.bincount(
            (taker[:, None] * n_z + np.arange(n_z)[None, :]).ravel(),
            (amount[:, None] * theta[donor]).ravel(),
            minlength=n * n_z,
        )
        .reshape(n, n_z)
        .astype(float, copy=False)
    )  # (an empty bincount comes back as int)
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


def carry(
    theta: np.ndarray,
    shares: np.ndarray,
    held: np.ndarray,
    face_a: np.ndarray,
    face_b: np.ndarray,
    moved_a_to_b: np.ndarray,
    filling: np.ndarray,
    fourier_step: np.ndarray,
    grid: LayerGrid,
    *,
    source: np.ndarray | None = None,
    max_courant: float = 0.1,
    max_fourier: float = 5e-5,
    max_substeps: int = 400,
    wall: WallContact | None = None,
) -> tuple[np.ndarray, int]:
    """Conduction (:func:`conduct`) and the flow's transport (:func:`advect`)
    over one step, split into substeps; returns ``(theta, substeps)``.

    :func:`advect` treats each layer of a cell as a mixed tank. When a step
    passes more than a tank's content through it, the melt still spends the
    whole step's conduction in the cell, so it reaches the next cell colder
    than it would: the error grows with the Courant number of the layer (what
    flows into it over the step ÷ what it holds). In a 1 mm channel at the
    rates of a fill this cooled the centre by 0.3--0.4 in 20 mm where it
    cools by 0.1. The flow of the step is spread evenly over substeps of
    Courant number at most ``max_courant`` (0.1: within about 0.01 of a short
    step) -- except where a cell hardly cools over the step (``α·dt/h²`` per
    substep below ``max_fourier``): a parcel that crosses it is not cooled
    by the error that counts on every full cell's layers
    (a filling cell mixes what comes in anyway). ``fourier_step`` is
    ``α·dt/h²`` per cell for the whole step; ``held`` is what the cells hold
    before it, and grows by the step's net inflow as the substeps go.
    Conduction acts where a cell holds melt; ``wall`` is its contact with the
    mould (:func:`conduct`).
    """
    n = theta.shape[0]
    moved = np.asarray(moved_a_to_b, dtype=float)
    src = np.zeros(n) if source is None else np.asarray(source, dtype=float)
    held = np.asarray(held, dtype=float)
    inflow = (
        np.bincount(face_b[moved > 0.0], moved[moved > 0.0], minlength=n)
        + np.bincount(face_a[moved < 0.0], -moved[moved < 0.0], minlength=n)
        + src
    )
    net = np.bincount(face_b, moved, minlength=n) - np.bincount(face_a, moved, minlength=n) + src
    full = (~np.asarray(filling, dtype=bool)) & (held > 0.0)
    layer_held = held[:, None] * (grid.wt / 0.5)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        courant = np.where(
            full[:, None] & (layer_held > 0.0), inflow[:, None] * shares / layer_held, 0.0
        )
    fo_step = np.broadcast_to(np.asarray(fourier_step, dtype=float), (n,))
    c_cell = courant.max(axis=1) if courant.size else np.zeros(n)
    # a substep for each max_courant of Courant number, but no more than keeps
    # the conduction a parcel picks up per cell it crosses below max_fourier:
    # a thick cell near the gate is crossed fast but hardly cools in a step
    need = np.where(
        c_cell > 0.5 * max_courant, np.minimum(c_cell / max_courant, fo_step / max_fourier), 1.0
    )
    worst = float(np.max(need)) if need.size else 1.0
    m = int(min(max(math.ceil(worst), 1), max_substeps))
    parts = _decay_parts(fo_step / m, grid, wall)
    th = theta
    h_now = held.copy()
    for _ in range(m):
        th = np.where((h_now > 0.0)[:, None], _conduct_parts(th, parts), th)
        th = advect(th, shares, h_now, face_a, face_b, moved / m, filling, grid, source=src / m)
        h_now = np.maximum(h_now + net / m, 0.0)
    return th, m


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
