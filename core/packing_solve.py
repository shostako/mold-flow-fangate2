"""Holding (packing) phase after the fill: a compressible Hele-Shaw solve (stage 2).

Where :mod:`core.packing` (stage 1) only reads the fill result, this module
moves melt: the gate is held at the holding pressure, the melt is compressible
(two-domain Tait pvT), and the pressure field, the mass that still flows in and
the volumetric shrinkage that is left come out of a time march.

Model
-----

*Temperature.* Not solved: the same closed form as stage 1. A cell's slab of
thickness ``h`` has been cooling between two walls at the mold temperature
for its melt age ``t_arr + t`` (``t`` counted from the end of the fill), and
its profile across the thickness is the exact series for a slab cooled on
both faces. The heat that the holding flow carries in is not counted. With
a V/P switch before the cavity is full the age is ``min(t_arr, t_switch) + t``
(``t`` from the switch), also for the melt that enters an empty cell later:
a thin section that is empty at the switch is as good as frozen from the
start, whatever came through the thick part to reach it. Whether holding
fills such a section is therefore not something this model can tell.

*Mass balance* of a cell of plan area ``A = dx²`` and volume ``V = A·h``:

    d(ρ̄·V)/dt = Σ_faces ρ_f · S_f · (p_nb − p)

``ρ̄(p)`` is the density averaged over the thickness at the cell's current
temperature profile (Tait), and ``S = ∫ z²/η dz`` the fluidity of the slab:
Cross-WLF viscosity at each depth, shear-thinned by the local shear stress
``τ = |∇p|·z``, and zero in the layers colder than the solidification
temperature ``T_mold + f·(T_melt − T_mold)``. ``S_f`` is the harmonic mean of
the two cells, both taken at the face's own pressure gradient, so a face
next to a frozen cell carries nothing. The shear thinning is solved together
with the pressure (its derivative is in the Newton matrix): evaluated at the
previous pressure field instead, the pressure would enter a thin neck by one
cell per update.

*Gate.* The gate cells are held at the holding pressure for the holding time,
then closed (a valve gate: no flow back).

*Leaving the wall.* The melt cannot pull on the wall. A cell whose mass no
longer fills its volume at zero pressure has left the wall, and what it lacks
is a mass deficit, not a tension. Both states share one unknown ``u`` per cell:
for ``u ≥ 0`` it is the pressure and the mass follows the pvT; for ``u < 0``
the cell is soft, its relative mass deficit ``1e-6·(−u)`` (an empty cell is
at ``u = −1 MPa``). What drives the flow is a potential that follows ``u``
near zero and levels off 2 kPa below it (``_drive``): a cell that is not full
takes melt from a full neighbour like a free surface, and passes next to none
on. The reported pressure is ``max(u, 0)``. The same branch carries the cells
that are still empty at the V/P switch.

*Shrinkage.* When nothing flows any more, the mass ``M`` of a cell is final,
and its volume at room temperature and no pressure is ``M·v(T_room, 0)``:

    S_v = 1 − M·v(T_room, 0) / V

Negative ``S_v`` is overpacking: the part would be larger than the cavity.

Not in the model: the heat of the holding flow, the deflection of the mold,
the pressure loss in the nozzle and the hot runner (the holding pressure is
taken at the gate), the pressure dependence of the viscosity, and the fill's
own pressure field: the march starts from a just-filled cavity at a uniform
1 MPa (``start_pressure_Pa``). Starting at exactly zero would put every cell
off the wall after the first instant of cooling, to be picked up again by
the pressure front a moment later; real cavities end the fill under pressure.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .layer_heat import carry, conduct, flow_shares, layer_grid, shear_heating
from .materials import Material
from .packing import freeze_age_s

# Half-width over which the two Tait branches are blended at the transition.
# The sharp model makes ρ(p) jump there, and a pressure solve chatters across
# the jump; measured pvT curves round the transition over a few kelvin anyway.
_TRANSITION_BAND_K = 2.0
# A cell that is not full (empty, part-filled, or off the wall) carries its
# relative mass deficit on a straight soft branch of ``u < 0``:
# ``deficit = _SOFT_PER_PA · (−u)``, so an empty cell is at ``u = −1 MPa``.
_SOFT_PER_PA = 1e-6
_EMPTY_U = -1.0 / _SOFT_PER_PA
# What drives the flow is not that ``u`` but a potential that follows it only
# near zero and levels off at −2 kPa: ``φ = u / (1 − u/_DRAW_PA)`` for
# ``u < 0`` (``φ = u`` for a full cell). It joins the pressure with a
# continuous slope, so a cell that has only just left the wall stays in the
# pressure solve smoothly; melt is drawn into a not-full cell by 2 kPa at
# most, and next to nothing moves between two of them.
_DRAW_PA = 2.0e3


def _drive(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flow potential ``φ(u)`` and its slope."""
    scale = 1.0 - np.minimum(u, 0.0) / _DRAW_PA
    return u / scale, 1.0 / scale**2


_FO_GRID = np.logspace(-5.0, 1.0, 601)
_SERIES_TERMS = 500
_theta_tables: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _slab_nodes(n_z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss nodes ``ξ ∈ (0, ½)`` (centre → wall), their weights (sum ½) and
    the table ``θ(Fo, ξ)`` of the slab series on ``_FO_GRID``."""
    if n_z not in _theta_tables:
        x, w = np.polynomial.legendre.leggauss(n_z)
        xi = 0.25 * (x + 1.0)
        wt = 0.25 * w
        k = 2 * np.arange(_SERIES_TERMS) + 1
        sign = (-1.0) ** np.arange(_SERIES_TERMS)
        decay = np.exp(-np.outer(_FO_GRID, k**2) * np.pi**2)  # (nFo, K)
        cosine = np.cos(np.outer(xi, k) * np.pi)  # (nz, K)
        table = (4.0 / np.pi) * (decay * (sign / k)) @ cosine.T  # (nFo, nz)
        _theta_tables[n_z] = (xi, wt, np.clip(table, 0.0, 1.0))
    return _theta_tables[n_z]


def slab_theta(fourier_number: np.ndarray, n_z: int = 16) -> np.ndarray:
    """Dimensionless temperature (1 = melt, 0 = wall) at the Gauss nodes.

    ``fourier_number`` has shape ``(n,)``; the result ``(n, n_z)``.
    """
    _, _, table = _slab_nodes(n_z)
    fo = np.asarray(fourier_number, dtype=float)
    ln = np.log(np.clip(fo, _FO_GRID[0], _FO_GRID[-1]))
    grid = np.log(_FO_GRID)
    out = np.empty((fo.size, n_z))
    for j in range(n_z):
        out[:, j] = np.interp(ln, grid, table[:, j])
    out[fo < _FO_GRID[0]] = 1.0
    return out


def _cross_fluidity_factor(stress_ratio: np.ndarray, n: float) -> np.ndarray:
    """``η0/η`` of the Cross model at a given shear stress ``τ/τ*``.

    With ``x = η0·γ̇/τ*`` the model reads ``τ/τ* = x / (1 + x^(1−n))``;
    ``η0/η = 1 + x^(1−n)``. Solved for ``ln x`` by Newton (the function is
    monotone with slope in ``[n, 1]``).
    """
    s = np.asarray(stress_ratio, dtype=float)
    out = np.ones_like(s)
    pos = s > 1e-12
    if not pos.any():
        return out
    ln_s = np.log(s[pos])
    u = np.maximum(ln_s, ln_s / n)
    for _ in range(60):
        e = np.logaddexp(0.0, (1.0 - n) * u)
        f = u - e - ln_s
        slope = 1.0 - (1.0 - n) * np.exp((1.0 - n) * u - e)
        step = f / slope
        u = u - step
        if np.max(np.abs(step)) < 1e-12:
            break
    out[pos] = 1.0 + np.exp(np.minimum((1.0 - n) * u, 700.0))
    return out


_LN_S_GRID = np.linspace(np.log(1e-8), np.log(1e8), 1201)
_cross_tables: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _cross_table(n: float) -> tuple[np.ndarray, np.ndarray]:
    """``ln(η0/η)`` and ``d ln(η0/η) / d ln τ`` on ``_LN_S_GRID`` for exponent ``n``."""
    if n not in _cross_tables:
        f = _cross_fluidity_factor(np.exp(_LN_S_GRID), n)
        a = (1.0 - n) * (f - 1.0) / f  # d ln f / d ln x
        _cross_tables[n] = (np.log(f), a / (1.0 - a))
    return _cross_tables[n]


@dataclass(frozen=True)
class PackingSolve:
    """Result of :func:`solve_packing`. Maps are NaN outside the solved cells."""

    shrinkage: np.ndarray  # final volumetric shrinkage (negative = overpacked)
    peak_pressure_Pa: np.ndarray  # highest pressure the cell saw
    freeze_pressure_Pa: np.ndarray  # pressure when the cell's centre solidified
    end_pressure_Pa: np.ndarray  # pressure at the end of the march
    hold_pressure_Pa: float
    hold_time_s: float
    solid_fraction: float
    times_s: np.ndarray  # one entry per time step
    clamp_force_N: np.ndarray  # ∫ p dA at each step
    added_mass_kg: np.ndarray  # mass that came in through the gates so far
    total_mass_kg: np.ndarray  # mass in the cavity at each step
    initial_mass_kg: float
    unconverged_steps: int = 0  # steps accepted at the smallest time step without converging
    # highest pressure while the cell's centre was molten, and its time after the switch
    molten_peak_pressure_Pa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    molten_peak_time_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # steps the gate took in more than the machine rate (by over 10 %): the
    # pressure that holds the rate down could not be found in the tries allowed
    rate_cap_exceeded_steps: int = 0
    newton_iterations: int = 0  # linear solves over the whole march
    switch_fill_fraction: float = 1.0  # share of the volume filled before the V/P switch
    switch_time_s: float = 0.0  # fill time at the switch (the march's t = 0)
    # time after the switch at which a cell became full: 0 = full at the switch, NaN = never
    late_fill_time_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    unfilled_volume_fraction: float = 0.0  # share of the volume in cells that never filled
    # pressure on the gate at each step (below the holding pressure while the rate is capped)
    gate_pressure_Pa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cell_size_mm: float = (
        0.0  # cell size the march ran on (coarser than the maps' grid if coarsened)
    )
    thermal: str = "series"  # temperature model: "series" (melt age) or "layers" (carried state)
    frame_times_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frame_pressure_Pa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    @property
    def added_mass_fraction(self) -> float:
        """Mass packed in after the fill, relative to the mass at the end of the fill."""
        return float(self.added_mass_kg[-1] / self.initial_mass_kg)


def solve_packing(
    mask: np.ndarray,
    thickness_mm: np.ndarray,
    arrival_time_s: np.ndarray,
    gates: list[tuple[int, int]],
    cell_size_mm: float,
    material: Material,
    T_melt_K: float,
    T_mold_K: float,
    hold_pressure_Pa: float,
    hold_time_s: float,
    *,
    solid_fraction: float = 0.3,
    T_room_K: float = 296.15,
    end_time_s: float | None = None,
    n_z: int = 16,
    dt_first_s: float = 2e-3,
    dt_max_s: float | None = None,
    dt_growth: float = 1.3,
    n_frames: int | None = None,
    start_pressure_Pa: float = 1.0e6,
    switch_fill_fraction: float = 1.0,
    max_inflow_m3_s: float | None = None,
    thermal: str = "series",
    initial_theta: np.ndarray | None = None,
    shear_heating_enabled: bool = True,
    progress: Callable[[float, float], None] | None = None,
) -> PackingSolve:
    """March from the V/P switch-over until nothing flows.

    ``arrival_time_s`` is the fill result's ``fill_time_s`` (the fill run at
    the machine's rate all the way; every cavity cell must have been reached).
    ``switch_fill_fraction`` is the share of the cavity volume the machine
    fills at that rate before it switches to the holding pressure: the cells
    reached by then are full, the rest are empty and have to be filled by the
    holding pressure. 1.0 takes the cavity as full and only packs it.
    ``max_inflow_m3_s`` is the most the machine delivers (its injection rate):
    the gate pressure stays below the holding pressure while that would draw
    more. ``None`` puts the full holding pressure on the gate at once.
    ``thermal`` picks the temperature model: ``"series"`` reads each cell's
    profile off its melt age (the slab series), ``"layers"`` carries the
    profile as a state (:mod:`core.layer_heat`): conduction to the walls, the
    heat the flow carries from cell to cell, viscous heating; what comes in
    through the gate is at the melt temperature. ``initial_theta``
    ``(ny, nx, n_z)`` is that state at the switch (from a fill that carried
    it); without it the state starts from the slab series at the arrival time.
    With ``"layers"`` the heat the flow dissipates goes into the state
    (``shear_heating_enabled``; the pressure's work on each face).
    ``progress(t, end_time)`` is called after every time step.
    ``end_time_s`` defaults to the later of the holding time and the moment
    the last cell's centre solidifies. The pressure field is kept at every
    time step (the first steps are milliseconds apart, so the rise shows);
    ``n_frames`` thins that to evenly spaced times, 0 keeps none.
    """
    pvt = material.pvt
    if pvt is None:
        raise ValueError(f"material {material.key} has no pvT data")
    if hold_pressure_Pa < 0.0 or hold_time_s < 0.0:
        raise ValueError("hold pressure and time must be >= 0")
    if thermal not in ("series", "layers"):
        raise ValueError(f"thermal must be 'series' or 'layers' (got {thermal!r})")
    layered = thermal == "layers"
    if initial_theta is not None and not layered:
        raise ValueError("initial_theta needs thermal='layers'")
    mask = np.asarray(mask, dtype=bool)
    thk = np.asarray(thickness_mm, dtype=float)
    t_arr2d = np.asarray(arrival_time_s, dtype=float)
    if not (mask.shape == thk.shape == t_arr2d.shape):
        raise ValueError(
            f"shape mismatch: mask {mask.shape}, thickness {thk.shape}, arrival {t_arr2d.shape}"
        )
    if np.any(mask & ~np.isfinite(t_arr2d)):
        raise ValueError("the fill did not reach every cavity cell")
    gate_cells = [(int(iy), int(ix)) for iy, ix in gates if mask[int(iy), int(ix)]]
    if not gate_cells:
        raise ValueError("no gate cell inside the cavity")

    ny, nx = mask.shape
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(int(mask.sum()))
    n = int(mask.sum())
    h = thk[mask] * 1e-3
    t_arr = t_arr2d[mask]
    dx = float(cell_size_mm) * 1e-3
    vol = h * dx * dx
    alpha = float(material.thermal_diffusivity_m2_s)
    xi, wt, _ = _slab_nodes(n_z)

    # V/P switch-over: the cells the fill reached by the time the machine had
    # delivered the switch volume are full, the others empty. The melt that
    # fills an empty cell later has been in the cavity since the start of the
    # injection, so its age counts from the switch time like the full cells'.
    if not 0.0 < switch_fill_fraction <= 1.0:
        raise ValueError(f"switch_fill_fraction must be in (0, 1], got {switch_fill_fraction}")
    if switch_fill_fraction >= 1.0:
        t_switch = float(t_arr.max())
    else:
        order = np.argsort(t_arr, kind="stable")
        filled = np.cumsum(vol[order])
        # (a hair under the target: a sum that lands on it exactly must count as reached)
        k = int(np.searchsorted(filled, switch_fill_fraction * filled[-1] * (1.0 - 1e-9)))
        t_switch = float(t_arr[order[min(k, n - 1)]])
    full_at_switch = t_arr <= t_switch
    if not full_at_switch.all() and max_inflow_m3_s is None:
        # the full holding pressure on a part-empty cavity fills it in no time
        raise ValueError("a switch before the cavity is full needs max_inflow_m3_s")
    t_arr = np.minimum(t_arr, t_switch)
    dT_melt = float(T_melt_K) - float(T_mold_K)
    heat_capacity = float(material.density_melt_kgm3) * float(material.specific_heat_J_kgK)

    # faces between neighbouring cavity cells: (+x) and (+y)
    fx_i, fx_j = index[:, :-1].ravel(), index[:, 1:].ravel()
    keep = (fx_i >= 0) & (fx_j >= 0)
    fx_i, fx_j = fx_i[keep], fx_j[keep]
    fy_i, fy_j = index[:-1, :].ravel(), index[1:, :].ravel()
    keep = (fy_i >= 0) & (fy_j >= 0)
    fy_i, fy_j = fy_i[keep], fy_j[keep]
    f_i = np.concatenate([fx_i, fy_i])
    f_j = np.concatenate([fx_j, fy_j])
    cnt_x = np.bincount(fx_i, minlength=n) + np.bincount(fx_j, minlength=n)
    cnt_y = np.bincount(fy_i, minlength=n) + np.bincount(fy_j, minlength=n)

    gate_idx = np.array(sorted({int(index[iy, ix]) for iy, ix in gate_cells}), dtype=int)
    is_gate = np.zeros(n, dtype=bool)
    is_gate[gate_idx] = True

    grid = layer_grid(n_z)
    fourier_rate = alpha / h**2
    if initial_theta is not None:
        start = np.asarray(initial_theta, dtype=float)
        if start.shape != (*mask.shape, n_z):
            raise ValueError(f"initial_theta must be {(*mask.shape, n_z)}, got {start.shape}")
        theta_state = np.clip(start[mask], 0.0, None)
        # The state is kept by the volume delivered, the switch is found from the
        # arrival times: a few cells that the melt reached right at the switch
        # can be on either side. They hold melt that has only just arrived (θ = 1
        # below). Many of them means the state is from another moment.
        missing = full_at_switch & ~np.all(np.isfinite(theta_state), axis=1)
        if missing.sum() > max(0.01 * full_at_switch.sum(), 4):
            raise ValueError(
                f"initial_theta has no value in {int(missing.sum())} cells that are full at the switch"
            )
        theta_state = np.where(np.isfinite(theta_state), theta_state, 1.0)
    else:
        theta_state = slab_theta(alpha * t_arr / h**2, n_z)
    # a cell that holds no melt yet has no profile of its own: it stands for
    # the melt that will come in, and takes that melt's temperature when it does
    theta_state[~full_at_switch] = 1.0

    def temperatures(t: float, dt: float, held: np.ndarray):
        """Profiles at the end of a step: ``t`` after the switch, ``dt`` after
        the state. ``held`` is the mass in the cells before the step."""
        if layered:
            theta = np.where(
                (held > 0.0)[:, None], conduct(theta_state, fourier_rate * dt, grid), theta_state
            )
        else:
            theta = slab_theta(alpha * (t_arr + t) / h**2, n_z)
        return theta, float(T_mold_K) + theta * dT_melt

    def mean_density(iso, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Thickness-mean density and its pressure slope; ``iso`` is the pvT
        model prepared at the cells' node temperatures."""
        rho, slope = iso.density_and_slope(p[:, None])
        return rho @ wt / 0.5, slope @ wt / 0.5

    ln_d1 = np.log(material.D1)
    rows = np.concatenate([np.arange(n), f_i, f_j])
    cols = np.concatenate([np.arange(n), f_j, f_i])

    ln_f_tab, dln_f_tab = _cross_table(float(material.n))
    arm = xi[None, :] * h[:, None] / material.tau_star  # τ/τ* per unit pressure gradient
    quad = 2.0 * h[:, None] ** 3 * (wt * xi**2)[None, :]

    def inverse_viscosity(theta: np.ndarray, T: np.ndarray) -> np.ndarray:
        """``1/η0`` at each node, zero in the solidified layers."""
        dT = T - material.D2
        denom = np.maximum(material.A2_tilde + dT, 1e-6)
        out = np.exp(-np.clip(ln_d1 - material.A1 * dT / denom, -700.0, 700.0))
        out[theta <= solid_fraction] = 0.0
        return out

    ln_arm = np.log(arm)
    ln_s0 = float(_LN_S_GRID[0])
    inv_step = 1.0 / float(_LN_S_GRID[1] - _LN_S_GRID[0])
    last_bin = _LN_S_GRID.size - 2

    def fluidity(cells: np.ndarray, weight: np.ndarray, ln_grad: np.ndarray):
        """``S = ∫ z²/η dz`` of ``cells`` at the pressure gradient ``exp(ln_grad)``,
        and ``grad·dS/dgrad``. ``weight`` is ``quad·(1/η0)``, shape ``(n, n_z)``."""
        x = np.clip((ln_grad[:, None] + ln_arm[cells] - ln_s0) * inv_step, 0.0, last_bin + 1.0)
        k = np.minimum(x.astype(np.intp), last_bin)
        w = x - k
        base = weight[cells] * np.exp(ln_f_tab[k] * (1.0 - w) + ln_f_tab[k + 1] * w)
        slope = dln_f_tab[k] * (1.0 - w) + dln_f_tab[k + 1] * w
        return base.sum(axis=1), (base * slope).sum(axis=1)

    n_fx = fx_i.size

    def face_flow(weight: np.ndarray, rho: np.ndarray, u_now: np.ndarray):
        """Mass conductance of every face at the pressure field ``u_now`` and
        its Newton coefficient ``d(flow)/d(Δφ)``, with ``Δφ`` and ``dφ/du``.
        The flow follows the potential ``φ`` (``_drive``): the pressure where a
        cell is full, at most 2 kPa below zero where it is not, so a not-full
        cell takes melt from a full neighbour and passes next to none on.

        The shear stress at a face follows the full pressure gradient: the
        component along the face normal from the two cells' difference, the
        one across it from the neighbouring differences (averaged).
        """
        phi, slope_phi = _drive(u_now)
        du = phi[f_j] - phi[f_i]
        gx = (
            np.bincount(fx_i, du[:n_fx], minlength=n) + np.bincount(fx_j, du[:n_fx], minlength=n)
        ) / (np.maximum(cnt_x, 1) * dx)
        gy = (
            np.bincount(fy_i, du[n_fx:], minlength=n) + np.bincount(fy_j, du[n_fx:], minlength=n)
        ) / (np.maximum(cnt_y, 1) * dx)
        across = np.concatenate([0.5 * (gy[fx_i] + gy[fx_j]), 0.5 * (gx[fy_i] + gx[fy_j])])
        along = du / dx
        grad = np.hypot(along, across)
        ln_grad = np.log(np.maximum(grad, 1e-300))
        s_i, d_i = fluidity(f_i, weight, ln_grad)
        s_j, d_j = fluidity(f_j, weight, ln_grad)
        both = (s_i > 0.0) & (s_j > 0.0)
        total = np.where(both, s_i + s_j, 1.0)
        r_i, r_j = s_i / total, s_j / total  # ratios: S itself can be 1e-200 near freezing
        s_face = np.where(both, 2.0 * s_i * r_j, 0.0)
        # grad·dS_face/dgrad of the harmonic mean, then only its part along the normal
        d_face = np.where(both, 2.0 * (r_j**2 * d_i + r_i**2 * d_j), 0.0)
        share = np.where(grad > 0.0, (along / np.where(grad > 0.0, grad, 1.0)) ** 2, 1.0)
        rho_f = 0.5 * (rho[f_i] + rho[f_j])
        return (
            rho_f * s_face,
            rho_f * (s_face + d_face * share),
            du,
            slope_phi,
            np.hypot(gx, gy),
            np.where(both, r_i, 0.0),
        )

    v_room = float(pvt.specific_volume(T_room_K, 0.0))
    t_freeze = freeze_age_s(thk[mask], alpha, solid_fraction) - t_arr  # after the switch
    # With the layered state the flow can keep a centre molten past the slab's
    # own time: an end the caller did not set is put off while one still is.
    open_end = end_time_s is None and layered
    # how long a slab of the cell's thickness takes from all melt: the carried
    # state is no hotter than that, and the slab's own estimate (t_freeze, by
    # the melt's age) says nothing about a state handed over from the fill
    fresh_freeze = float(np.max(freeze_age_s(thk[mask], alpha, solid_fraction)))
    if end_time_s is None:
        longest = fresh_freeze if layered else float(np.max(t_freeze))
        end_time_s = max(float(hold_time_s), longest) * 1.02 + 1e-3
    # the flow can hold a centre molten while the gate is open; once it has
    # closed, nothing stays molten longer than a fresh slab
    end_limit = max(3.0 * end_time_s, float(hold_time_s) + 3.0 * fresh_freeze)

    if dt_max_s is None:
        dt_max_s = max(end_time_s / 60.0, 0.1)

    _, T = temperatures(0.0, 0.0, np.zeros(n))
    u = np.where(full_at_switch, float(start_pressure_Pa), _EMPTY_U)
    rho0, _ = mean_density(pvt.at_temperature(T, _TRANSITION_BAND_K), np.maximum(u, 0.0))
    mass = np.where(full_at_switch, rho0 * vol, 0.0)
    mass_initial = float(mass.sum())
    late_fill = np.where(full_at_switch, 0.0, np.nan)  # time after the switch a cell became full

    peak = np.maximum(u, 0.0)
    p_freeze = np.full(n, np.nan)
    if layered:
        solid_now = full_at_switch & (theta_state[:, 0] <= solid_fraction)
    else:
        solid_now = t_freeze <= 0.0
    p_freeze[solid_now] = peak[solid_now]
    # the highest pressure while the centre was still molten, and when: melt
    # that can still flow is what a parting line lets out as flash
    molten_peak = np.where(~solid_now, peak, 0.0)
    molten_peak_t = np.zeros(n)
    times = [0.0]
    clamp = [float(peak.sum()) * dx * dx]
    added = [0.0]
    total = [mass_initial]
    # pressure frames: every step (None), or thinned to about n_frames evenly spaced times
    every_step = n_frames is None
    frame_at = np.zeros(0) if every_step else np.linspace(0.0, end_time_s, max(int(n_frames), 0))
    keep_frames = every_step or frame_at.size > 0
    frames = [peak.copy()] if keep_frames else []
    frame_times = [0.0] if keep_frames else []
    unconverged = 0
    over_rate = 0
    counter = [0]

    def step(u0, mass_old, t_new, dt, p_gate):
        """One implicit step with the gate cells held at ``p_gate`` (``None``:
        gate closed); returns ``(u, mass, gate_inflow_kg, converged, iterations)``."""
        gate_open = p_gate is not None
        theta_n, T_n = temperatures(t_new, dt, mass_old)
        iso = pvt.at_temperature(T_n, _TRANSITION_BAND_K)
        rho_z, slope_z = mean_density(iso, np.zeros(n))
        weight = quad * inverse_viscosity(theta_n, T_n)
        full_mass = rho_z * vol  # of a cell just full at no pressure
        soft = full_mass * _SOFT_PER_PA
        gate_scale = float(np.mean(slope_z * vol))  # keeps the gate rows at the mass rows' size
        u_it = u0.copy()
        # No cell can end the step above the highest pressure around: the gate's
        # while it is open, otherwise the highest one the step started from (the
        # cavity only cools). Newton on a shear-thinning flow (flow ∝ Δp^(1/n))
        # overshoots this by orders of magnitude from a poor start; the bound
        # brings each iterate back.
        ceiling = max(float(u0.max()), p_gate if gate_open else 0.0)
        # ... nor below the lowest value a cell reaches on its own: the deficit
        # its old mass leaves at the new temperature when nothing flows in.
        floor = min(float(np.min((mass_old - full_mass) / soft)), 0.0)
        converged = False
        lu = None
        last = np.inf
        used = 0
        for _ in range(150):
            used += 1
            on = u_it >= 0.0
            rho, slope = mean_density(iso, np.maximum(u_it, 0.0))
            m_fun = np.where(on, rho * vol, full_mass + soft * u_it)
            dm = np.where(on, slope * vol, soft)
            g_face, k_face, dp, slope_phi, _, _ = face_flow(weight, rho, u_it)
            flow = g_face * dp  # mass rate i ← j
            net = np.bincount(f_i, flow, minlength=n) - np.bincount(f_j, flow, minlength=n)
            res = m_fun - mass_old - dt * net
            if gate_open:
                res[gate_idx] = gate_scale * (u_it[gate_idx] - p_gate)
            if lu is None:
                diag = dm + dt * slope_phi * (
                    np.bincount(f_i, k_face, minlength=n) + np.bincount(f_j, k_face, minlength=n)
                )
                off_ij = -dt * k_face * slope_phi[f_j]  # d res_i / d u_j
                off_ji = -dt * k_face * slope_phi[f_i]
                if gate_open:
                    diag[gate_idx] = gate_scale
                    off_ij = np.where(is_gate[f_i], 0.0, off_ij)
                    off_ji = np.where(is_gate[f_j], 0.0, off_ji)
                lu = splu(
                    csc_matrix(
                        (np.concatenate([diag, off_ij, off_ji]), (rows, cols)), shape=(n, n)
                    ),
                    permc_spec="MMD_AT_PLUS_A",
                    options={"SymmetricMode": True},
                )
            u_next = np.clip(u_it + lu.solve(-res), floor, ceiling)
            # A cell that crosses the kink at u = 0 (just full: below it the mass
            # follows u about a thousand times faster than above) stops on it, and
            # the next iteration takes the slope of the side it goes on to. Without
            # this, Newton throws such a cell from the floor to the ceiling and back
            # and never converges.
            cross = (u_it != 0.0) & ((u_it < 0.0) != (u_next < 0.0))
            u_next = np.where(cross, 0.0, u_next)
            # a step in ``u`` counts as the pressure step that moves the same mass
            size = float(np.max(np.abs(u_next - u_it)))
            counter[0] += 1
            u_it = u_next
            # (a cell just stopped on the kink has not been tried on its other side)
            if size < 200.0 and not cross.any():  # Pa
                converged = True
                break
            # the factorisation is kept while it still converges fast, renewed when not
            if size > 0.25 * last:
                lu = None
            last = size
        on = u_it >= 0.0
        rho, _ = mean_density(iso, np.maximum(u_it, 0.0))
        mass_new = np.where(on, rho * vol, full_mass + soft * u_it)
        inflow = 0.0
        theta_new = None
        from_gate = np.zeros(n)
        if gate_open or layered:
            g_face, _, dp, _, grad, r_i = face_flow(weight, rho, u_it)
            flow = g_face * dp
        if gate_open:
            net = np.bincount(f_i, flow, minlength=n) - np.bincount(f_j, flow, minlength=n)
            # what left the gate cells into the cavity, plus what the gate cells took up
            from_gate[gate_idx] = -dt * net[gate_idx] + (mass_new[gate_idx] - mass_old[gate_idx])
            inflow = float(from_gate.sum())
        if layered:
            # the heat the step's flow carried, layer by layer, and what it dissipated
            x = np.clip(
                (np.log(np.maximum(grad, 1e-300))[:, None] + ln_arm - ln_s0) * inv_step,
                0.0,
                last_bin + 1.0,
            )
            k = np.minimum(x.astype(np.intp), last_bin)
            w = x - k
            inv_eta = inverse_viscosity(theta_n, T_n) * np.exp(
                ln_f_tab[k] * (1.0 - w) + ln_f_tab[k + 1] * w
            )
            # (the melt that comes in through the gate is at the melt temperature).
            # Conduction and transport from the step's start, in substeps short
            # enough for the transport; theta_n above (conduction only) is what
            # the step's densities and viscosities were read from.
            theta_new, _ = carry(
                theta_state,
                flow_shares(inv_eta, grid),
                mass_old,
                f_j,
                f_i,
                flow * dt,
                u0 < 0.0,
                fourier_rate * dt,
                grid,
                source=np.maximum(from_gate, 0.0),
            )
            # the heat the flow dissipated: S_f·Δφ² per face (the pressure's work),
            # shared between the two cells as resistances in series (core.layer_heat)
            heat_face = (g_face / np.maximum(0.5 * (rho[f_i] + rho[f_j]), 1e-12)) * dp**2 * dt
            heat = np.bincount(f_i, heat_face * (1.0 - r_i), minlength=n) + np.bincount(
                f_j, heat_face * r_i, minlength=n
            )
            heat = np.where(mass_new > 0.0, heat, 0.0)
            if shear_heating_enabled and np.any(heat > 0.0):
                theta_new = theta_new + shear_heating(
                    heat, inv_eta, vol, heat_capacity, dT_melt, grid
                )
        return u_it, mass_new, inflow, converged, used, theta_new

    # The machine cannot deliver melt faster than ``max_inflow_m3_s``: while
    # the holding pressure would draw more than that, the gate pressure is
    # held below it (the machine is still limited by its speed, as in the
    # fill), and it climbs to the holding pressure as the cavity fills up.
    rate_cap = None if max_inflow_m3_s is None else float(max_inflow_m3_s)
    rho_gate = float(rho0[gate_idx].mean())
    p_gate_now = float(start_pressure_Pa) if rate_cap is not None else float(hold_pressure_Pa)
    gate_hist = [p_gate_now if hold_time_s > 0.0 else 0.0]

    t = 0.0
    dt = float(dt_first_s)
    next_frame = 1
    while t < end_time_s - 1e-12:
        gate_open = t < hold_time_s - 1e-12 and hold_time_s > 0.0
        limit = min(hold_time_s, end_time_s) if gate_open else end_time_s
        dt_try = min(dt, limit - t)
        if gate_open and rate_cap is not None:
            # try to come up towards the holding pressure, a factor per step
            p_try = min(float(hold_pressure_Pa), max(p_gate_now * 1.5, p_gate_now + 1.0e5))
        else:
            p_try = float(hold_pressure_Pa)
        tries = 0
        while True:
            u_new, mass_new, inflow, ok, used, theta_new = step(
                u, mass, t + dt_try, dt_try, p_try if gate_open else None
            )
            if not ok and dt_try > 1e-5:
                dt_try *= 0.5
                continue
            rate = inflow / (rho_gate * dt_try)
            if gate_open and rate_cap is not None and rate > 1.1 * rate_cap and tries < 6:
                # flow ∝ Δp^(1/n) at worst: bring the pressure down by the rate ratio to the n
                p_try = max(p_try * (rate_cap / rate) ** max(float(material.n), 0.2), 1.0e2)
                tries += 1
                continue
            break
        unconverged += 0 if ok else 1
        if gate_open and rate_cap is not None and rate > 1.1 * rate_cap:
            # accepted as it is, but counted: the answer took in melt faster than the machine
            over_rate += 1
        if gate_open:
            p_gate_now = p_try
        was_open_until = gate_open and abs(t + dt_try - hold_time_s) < 1e-12
        t += dt_try
        u, mass = u_new, mass_new
        p = np.maximum(u, 0.0)
        peak = np.maximum(peak, p)
        late_fill[np.isnan(late_fill) & (u >= 0.0)] = t
        if layered:
            theta_state = theta_new
            solid_now = ~np.isnan(late_fill) & (theta_state[:, 0] <= solid_fraction)
            if (
                open_end
                and t >= end_time_s - 1e-12
                and not solid_now[~np.isnan(late_fill)].all()
                and end_time_s < end_limit
            ):
                end_time_s = min(end_time_s + dt_max_s, end_limit)  # a centre is still molten
        else:
            solid_now = t_freeze <= t
        just_frozen = np.isnan(p_freeze) & solid_now
        rise = (np.isnan(p_freeze) | just_frozen) & (p > molten_peak)
        molten_peak[rise] = p[rise]
        molten_peak_t[rise] = t
        p_freeze[just_frozen] = p[just_frozen]
        times.append(t)
        gate_hist.append(p_try if gate_open else 0.0)
        clamp.append(float(p.sum()) * dx * dx)
        added.append(added[-1] + inflow)
        total.append(float(mass.sum()))
        if progress is not None:
            progress(t, float(end_time_s))
        if every_step:
            frames.append(p.copy())
            frame_times.append(t)
        while next_frame < len(frame_at) and frame_at[next_frame] <= t + 1e-12:
            frames.append(p.copy())
            frame_times.append(t)
            next_frame += 1
        # The next step. (Not sized by the iteration count: each iteration lets
        # the pressure into one more row of not-full cells whatever the step,
        # so right after the switch a shorter step needs just as many.)
        if not ok:
            dt = float(dt_first_s)
        elif was_open_until:
            dt = float(dt_first_s) * 5.0  # the gate closing is a jump
        else:
            dt = min(dt_try * dt_growth, dt_max_s)

    p = np.maximum(u, 0.0)
    p_freeze[np.isnan(p_freeze)] = p[np.isnan(p_freeze)]
    never_full = np.isnan(late_fill)

    def to_map(values: np.ndarray) -> np.ndarray:
        out = np.full(mask.shape, np.nan)
        out[mask] = values
        return out

    frame_stack = np.full((len(frames), ny, nx), np.nan, dtype=np.float32)
    for k, fr in enumerate(frames):
        frame_stack[k][mask] = fr
    return PackingSolve(
        shrinkage=to_map(np.where(never_full, np.nan, 1.0 - mass * v_room / vol)),
        switch_fill_fraction=float(switch_fill_fraction),
        switch_time_s=t_switch,
        late_fill_time_s=to_map(late_fill),
        unfilled_volume_fraction=float(vol[never_full].sum() / vol.sum()),
        peak_pressure_Pa=to_map(peak),
        molten_peak_pressure_Pa=to_map(molten_peak),
        molten_peak_time_s=to_map(molten_peak_t),
        freeze_pressure_Pa=to_map(p_freeze),
        end_pressure_Pa=to_map(p),
        hold_pressure_Pa=float(hold_pressure_Pa),
        hold_time_s=float(hold_time_s),
        solid_fraction=float(solid_fraction),
        times_s=np.array(times),
        clamp_force_N=np.array(clamp),
        gate_pressure_Pa=np.array(gate_hist),
        added_mass_kg=np.array(added),
        total_mass_kg=np.array(total),
        initial_mass_kg=mass_initial,
        unconverged_steps=unconverged,
        rate_cap_exceeded_steps=over_rate,
        cell_size_mm=float(cell_size_mm),
        thermal=thermal,
        newton_iterations=counter[0],
        frame_times_s=np.array(frame_times),
        frame_pressure_Pa=frame_stack,
    )


def coarsening_factor(n_cells: int, max_cells: int = 20_000) -> int:
    """Smallest block size (cells per side) that brings ``n_cells`` under ``max_cells``."""
    return max(1, math.ceil(math.sqrt(n_cells / max(int(max_cells), 1))))


def _coarsen(mask, thk, t_arr, gates, f):
    """Blocks of ``f × f`` cells → one cell where at least half the block is
    cavity; thickness and arrival time are the means over its cavity cells."""
    ny, nx = mask.shape
    NY, NX = math.ceil(ny / f), math.ceil(nx / f)
    pad = ((0, NY * f - ny), (0, NX * f - nx))
    m = np.pad(mask, pad).reshape(NY, f, NX, f)
    count = m.sum(axis=(1, 3))
    c_mask = 2 * count >= f * f

    def mean(a):
        a = np.pad(np.where(mask, a, 0.0), pad).reshape(NY, f, NX, f)
        return np.where(c_mask, a.sum(axis=(1, 3)) / np.maximum(count, 1), 0.0)

    c_gates = sorted({(iy // f, ix // f) for iy, ix in gates if c_mask[iy // f, ix // f]})
    return c_mask, mean(thk), mean(t_arr), c_gates


def _coarsen_theta(mask, thk, theta, f, c_mask):
    """The layered temperature state ``(ny, nx, n_z)`` on the ``f × f`` blocks:
    the thickness-weighted mean of the cells that hold one (a block with none
    stays NaN, as a cell that held no melt at the switch)."""
    ny, nx, n_z = theta.shape
    NY, NX = c_mask.shape
    pad = ((0, NY * f - ny), (0, NX * f - nx), (0, 0))
    have = mask[..., None] & np.isfinite(theta)
    w = np.where(have, thk[..., None], 0.0)
    num = np.pad(np.where(have, theta, 0.0) * w, pad).reshape(NY, f, NX, f, n_z).sum(axis=(1, 3))
    den = np.pad(w, pad).reshape(NY, f, NX, f, n_z).sum(axis=(1, 3))
    out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)
    return np.where(c_mask[..., None], out, np.nan)


def _gate_fed_everywhere(mask, gates) -> bool:
    lab, _ = ndi_label(mask)
    fed = {int(lab[iy, ix]) for iy, ix in gates}
    return bool(gates) and bool(np.all(np.isin(lab[mask], list(fed))))


def solve_packing_coarsened(
    mask: np.ndarray,
    thickness_mm: np.ndarray,
    arrival_time_s: np.ndarray,
    gates: list[tuple[int, int]],
    cell_size_mm: float,
    material: Material,
    T_melt_K: float,
    T_mold_K: float,
    hold_pressure_Pa: float,
    hold_time_s: float,
    *,
    factor: int = 1,
    **kwargs,
) -> PackingSolve:
    """:func:`solve_packing` on a grid ``factor`` times coarser, with the maps
    brought back to the given grid (each cell takes its block's value).

    The cost of the march is set by the number of cells, and the holding phase
    has no moving front to resolve, so a fine fill grid is solved coarser here.
    If a block size would cut the cavity off from its gates (a feature narrower
    than the block), the next smaller one is used.
    """
    mask = np.asarray(mask, dtype=bool)
    thk = np.asarray(thickness_mm, dtype=float)
    t_arr = np.asarray(arrival_time_s, dtype=float)
    if not (mask.shape == thk.shape == t_arr.shape):
        raise ValueError(
            f"shape mismatch: mask {mask.shape}, thickness {thk.shape}, arrival {t_arr.shape}"
        )
    if np.any(mask & ~np.isfinite(t_arr)):
        raise ValueError("the fill did not reach every cavity cell")
    gate_cells = [(int(iy), int(ix)) for iy, ix in gates if mask[int(iy), int(ix)]]
    f = max(int(factor), 1)
    while f > 1:
        c_mask, c_thk, c_arr, c_gates = _coarsen(mask, thk, t_arr, gate_cells, f)
        if _gate_fed_everywhere(c_mask, c_gates):
            break
        f -= 1
    if f == 1:
        return solve_packing(
            mask, thk, t_arr, gate_cells, cell_size_mm, material, T_melt_K, T_mold_K,
            hold_pressure_Pa, hold_time_s, **kwargs,
        )  # fmt: skip
    if kwargs.get("initial_theta") is not None:
        theta = np.asarray(kwargs["initial_theta"], dtype=float)
        if theta.shape[:2] != mask.shape:
            raise ValueError(f"initial_theta must be {mask.shape} + (n_z,), got {theta.shape}")
        kwargs = {**kwargs, "initial_theta": _coarsen_theta(mask, thk, theta, f, c_mask)}
    r = solve_packing(
        c_mask, c_thk, c_arr, c_gates, float(cell_size_mm) * f, material, T_melt_K, T_mold_K,
        hold_pressure_Pa, hold_time_s, **kwargs,
    )  # fmt: skip
    ny, nx = mask.shape
    # a cavity cell whose block is not a coarse cell takes the nearest solved block
    _, (jy, jx) = distance_transform_edt(~c_mask, return_indices=True)

    def fine(a: np.ndarray) -> np.ndarray:
        a = a[..., jy, jx]
        a = np.repeat(np.repeat(a, f, axis=-2), f, axis=-1)[..., :ny, :nx]
        return np.where(mask, a, np.nan).astype(a.dtype)

    return replace(
        r,
        shrinkage=fine(r.shrinkage),
        peak_pressure_Pa=fine(r.peak_pressure_Pa),
        molten_peak_pressure_Pa=fine(r.molten_peak_pressure_Pa),
        molten_peak_time_s=fine(r.molten_peak_time_s),
        freeze_pressure_Pa=fine(r.freeze_pressure_Pa),
        end_pressure_Pa=fine(r.end_pressure_Pa),
        frame_pressure_Pa=fine(r.frame_pressure_Pa),
        late_fill_time_s=fine(r.late_fill_time_s),
    )


def summarize_solve_by_thickness(
    solve: PackingSolve,
    thickness_mm: np.ndarray,
    region: np.ndarray,
    cell_size_mm: float,
    decimals: int = 1,
) -> list[dict]:
    """One row per wall thickness inside ``region`` (thickest first): area,
    mean and largest shrinkage, and the mean of the peak and of the
    at-solidification pressure."""
    region = np.asarray(region, dtype=bool) & np.isfinite(solve.shrinkage)
    thk = np.round(np.asarray(thickness_mm, dtype=float), decimals)
    cell_cm2 = float(cell_size_mm) ** 2 / 100.0
    rows: list[dict] = []
    for hh in sorted({float(v) for v in thk[region]}, reverse=True):
        sel = region & (thk == hh)
        rows.append(
            {
                "thickness_mm": hh,
                "area_cm2": float(sel.sum()) * cell_cm2,
                "shrinkage_mean": float(solve.shrinkage[sel].mean()),
                "shrinkage_max": float(solve.shrinkage[sel].max()),
                "shrinkage_min": float(solve.shrinkage[sel].min()),
                "peak_pressure_Pa": float(solve.peak_pressure_Pa[sel].mean()),
                "freeze_pressure_Pa": float(solve.freeze_pressure_Pa[sel].mean()),
            }
        )
    return rows
