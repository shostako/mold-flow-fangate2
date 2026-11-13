"""Time-marching fill: the melt front advanced in time (v0.62.0).

The τ solvers order the fill with a single elliptic solve, ``-∇·(S∇τ) = 1``.
Every cell is a unit sink in that problem, so the deep parts of a gate block --
where most of the sink volume sits behind a small resistance -- come out
"filled first", and the thin product only starts once the block is nearly full.
The FG9 VP-series photos contradict that: at 5.8 g the block is 64 % full and
the product centre is already ahead (2026-10-07). Advancing the front in time
gets the block's partial fill and the early centre lead right.

This module is the marching core, kept free of the material model: the caller
hands in the conductance of a cell as a function of the time the melt reached
it, and the volume the machine has injected as a function of time.

Scheme (IMPES: pressure implicit, fill explicit), per step of length ``dt``:

* full cells (``V >= C``) carry the unknown pressure; every cell that is not
  full but touches a full cell (or already holds melt) is the front, ``p = 0``;
* the gate cells share one pressure ``p_g`` (the τ solvers hold them all at
  ``τ = 0``), chosen so that the melt leaving them over the step is what the
  machine delivered. The system is linear in ``p_g``, so one factorization with
  two right-hand sides gives it;
* volume balance of a full cell ``i``::

      Σ_j S_ij (p_i - p_j) = (V_i - C_i) / dt + q_i

  where ``S_ij`` is the harmonic mean of the two cell conductances (the same
  face rule as the τ system), ``q_i`` the gate inflow over the step and
  ``(V_i - C_i)/dt`` pushes out what a cell received beyond its capacity in the
  previous step -- the volume is conserved to round-off;
* front cells take the inflow explicitly. A cell that fills during the step is
  stamped with the time its capacity was reached, interpolated linearly in the
  step, and its conductance is evaluated once at that time and kept;
* the next step is ``cfl`` times the time the fastest front cell needs to take
  one cell of melt, capped by ``dt_max_s``.

The conductance of a front cell is evaluated at the current time (the melt
arriving now). Units are SI: volumes m³, conductance m³/(Pa·s), pressure Pa.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: A cell counts as full once it holds this fraction of its capacity. The step is
#: sized to fill a cell exactly, and the inflow comes out of a linear solve, so
#: "exactly" lands a few parts in 10^9 either side; a cell left that far short
#: took a second cell of melt in the next step (a strip was stamped 1.5 % late).
FULL_REL = 1.0 - 1e-6
#: Room left in the cavity (net of surplus) below this fraction of its capacity
#: counts as none: the march stops delivering and only moves the surplus on.
ROOM_TOL = 1e-9


@dataclass
class MarchFillResult:
    """Outcome of :func:`march_fill`.

    ``t_arr_s`` is the time each cell became full (NaN outside the cavity and
    for cells still unfilled when the march stopped). ``pressure_end_Pa`` is
    the pressure the injection needed in the last step that delivered melt --
    the field at the end of the fill, without the surplus push (zero at the
    front and outside the cavity).
    """

    t_arr_s: np.ndarray
    pressure_end_Pa: np.ndarray
    n_steps: int
    t_end_s: float
    complete: bool
    volume_injected_m3: float
    volume_held_m3: float
    stalled: bool = False  # stopped at the pressure limit with the rate fallen away
    pressure_limited_steps: int = 0  # steps the gate spent at the pressure limit
    gate_pressure_max_Pa: float = 0.0  # highest gate pressure the injection needed
    #: highest gate pressure of the steps the limit did not hold back: what the
    #: fill needed before it met the limit
    gate_pressure_free_max_Pa: float = 0.0
    #: steps at the limit that still delivered more than the stall fraction of the
    #: programmed rate. A stalled march with none of these never ran slowed at the
    #: limit: either the way to the cells left closed outright, or the limit passed
    #: next to nothing from its first step (the rate history cannot tell which).
    pressure_slowed_steps: int = 0
    #: one entry per delayed gate that opened, in the order they opened:
    #: ``{"gate": index into delayed_gates, "t_s": when it opened,
    #: "melt_already_there": whether its cells were full by then}``
    gate_events: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class DelayedGate:
    """A gate (a valve) that opens after the injection has begun.

    ``cells`` are its gate cells ``(iy, ix)``. Until it opens they are ordinary
    cavity cells: the melt of the other gates fills them like any other.
    ``on_arrival=False``: it opens ``delay_s`` after the start of the injection
    (``t = 0`` on the march's clock). ``on_arrival=True``: it opens ``delay_s``
    after the melt of the other gates has filled all its cells -- and not at
    all if that never happens.
    """

    cells: tuple[tuple[int, int], ...]
    delay_s: float = 0.0
    on_arrival: bool = False


@dataclass
class MarchStep:
    """What one step of :func:`march_fill` did, handed to ``on_step``.

    Arrays are flat over the whole grid (``ny·nx``); faces are pairs of flat
    cell indices.
    """

    t_s: float  # time at the start of the step
    dt_s: float
    face_a: np.ndarray
    face_b: np.ndarray
    moved_m3: np.ndarray  # melt that crossed each face over the step, positive a → b
    held_m3: np.ndarray  # melt in each cell before the step
    full: np.ndarray  # cells that were full before the step
    newly: np.ndarray  # flat indices of the cells that became full in the step
    gate_in_m3: np.ndarray  # melt the machine delivered into each gate cell over the step
    # the pressure the delivery alone needed in the step, without the push that
    # moves a cell's surplus on (that one scales as 1/dt); zero outside the full cells
    pressure_Pa: np.ndarray
    volume_injected_m3: float  # delivered by the end of the step
    #: faces where a full cell fed a front cell that had begun to fill before it
    #: (``late_inflow``): their indices into ``face_a``/``face_b``, whether the
    #: front cell is the ``b`` side, and the conductance the front cell was given
    late_faces: np.ndarray | None = None
    late_front_is_b: np.ndarray | None = None
    late_front_conductance: np.ndarray | None = None


def _faces(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-neighbour faces inside ``mask`` as flat index pairs ``(a, b)``."""
    ny, nx = mask.shape
    ids = np.arange(ny * nx).reshape(ny, nx)
    fa, fb = [], []
    for dy, dx in ((1, 0), (0, 1)):
        both = mask[: ny - dy, : nx - dx] & mask[dy:, dx:]
        fa.append(ids[: ny - dy, : nx - dx][both])
        fb.append(ids[dy:, dx:][both])
    return np.concatenate(fa), np.concatenate(fb)


def march_fill(
    mask: np.ndarray,
    capacity_m3: np.ndarray,
    gates: list[tuple[int, int]],
    injected_volume_m3: Callable[[float], float],
    conductance: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    t_start_s: float,
    dt_max_s: float,
    cfl: float = 1.0,
    breakpoints_s: tuple[float, ...] = (),
    late_inflow: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    max_steps: int = 200_000,
    on_step: Callable[[MarchStep], np.ndarray | None] | None = None,
    pressure_limit_Pa: float | None = None,
    stall_rate_fraction: float = 0.01,
    stall_steps: int = 5,
    delayed_gates: Sequence[DelayedGate] = (),
) -> MarchFillResult:
    """Advance the melt front from the gates until the cavity is full.

    Parameters
    ----------
    mask
        ``(ny, nx)`` cavity cells.
    capacity_m3
        ``(ny, nx)`` melt volume each cell holds when full.
    gates
        Gate cells ``(iy, ix)``; they start full at ``t_start_s``.
    injected_volume_m3
        Total volume the machine has delivered by time ``t``. It must reach the
        gate cells' capacity at ``t_start_s``; the march takes in the increments
        over each step (a staged profile is followed exactly).
    conductance
        ``conductance(flat_index, t_arr)`` → conductance [m³/(Pa·s)] of those
        cells for melt that reached them at ``t_arr``.
    breakpoints_s
        Times where the delivery rate changes (a staged profile's switches). A
        step never crosses one, and the step after it is sized on the new rate.
    on_step
        Called after every step with a :class:`MarchStep`. It may return the
        conductance of every cell (flat, ``ny·nx``) as it is now: the full
        cells take it, instead of keeping the value they got when the melt
        reached them. This is how a caller that carries the temperature as a
        state lets the cells behind the front go on cooling.
    late_inflow
        ``late_inflow(front_idx, feeder_idx)`` → the conductance a front cell
        offers to the melt of that feeder. A front cell's own conductance is
        that of the melt already in it: right for the neighbour that melt came
        from, which has to push through it. A neighbour that became full only
        *after* the front cell had begun to fill is another front, arriving at
        the cell's empty side: it meets the void, not the melt already there.
        On those faces the front cell takes the larger of its own conductance
        and this one. Without it, a front that stopped and froze shuts its
        cell to a front that arrives later, and a line of cells is left
        unfilled between the two. ``None``: the cell's own conductance on
        every face.
    pressure_limit_Pa
        The most the machine puts on the gate. While the programmed rate would
        need more, the gate is held at the limit and the melt comes in at
        whatever rate that gives (the screw falls behind its programme, and
        does not catch up). ``None``: no limit, the volume is delivered
        whatever it takes.
    stall_rate_fraction
        With a pressure limit, the march stops as stalled once the rate at the
        limit has stayed below this fraction of the programmed rate, without
        growing by half, for ``stall_steps`` steps in a row; the cells still not full
        keep ``t_arr = NaN`` and ``complete`` is ``False``. (A conductance that
        ``on_step`` derives from the flow starts from no flow at all -- the
        zero-shear value -- and grows step by step towards the flow it allows:
        a slow first step is not a stall.)
    delayed_gates
        Gates that open later (:class:`DelayedGate`), besides ``gates``. Once
        open, a gate shares the one gate pressure and the delivery with the
        others (the manifold's own loss is not modelled). A gate that opens on
        cells the melt has not filled yet fills them at once; that melt counts
        as delivered on top of the programme (``volume_injected_m3``). A gate
        whose cell holds a surplus from the last step opens one step later,
        once the surplus has moved on. While a
        gate with a set opening time is still closed, the march is not stopped
        as stalled: the gate may open the way.
    """
    if not 0.0 < cfl <= 1.0:
        raise ValueError(f"cfl must be in (0, 1] (got {cfl})")
    if dt_max_s <= 0:
        raise ValueError(f"dt_max_s must be positive (got {dt_max_s})")
    ny, nx = mask.shape
    n = ny * nx
    m = mask.ravel().astype(bool)
    C = np.where(m, np.asarray(capacity_m3, float).ravel(), 0.0)
    if np.any(C[m] <= 0):
        raise ValueError("every cavity cell needs a positive capacity")
    g = np.array([iy * nx + ix for iy, ix in gates], dtype=np.int64)
    g = g[m[g]]
    if g.size == 0:
        raise ValueError("no gate cell lies in the cavity")
    fa, fb = _faces(mask)
    is_gate = np.zeros(n, dtype=bool)
    is_gate[g] = True
    # the gates that open later: their cells, and when they open (None: not known yet)
    d_cells: list[np.ndarray] = []
    for dg in delayed_gates:
        if dg.delay_s < 0:
            raise ValueError(f"a delayed gate's delay_s must be >= 0 (got {dg.delay_s})")
        c = np.array([iy * nx + ix for iy, ix in dg.cells], dtype=np.int64)
        c = c[m[c]] if c.size else c
        if c.size == 0:
            raise ValueError("a delayed gate has no cell in the cavity")
        if is_gate[c].any():
            raise ValueError("a delayed gate shares a cell with a gate that is open from the start")
        d_cells.append(c)
    d_open_at: list[float | None] = [
        None if dg.on_arrival else float(dg.delay_s) for dg in delayed_gates
    ]
    d_opened = [False] * len(d_cells)
    gate_events: list[dict] = []
    v_extra = 0.0  # melt that opening gates put into cells not yet full
    t_tol = 1e-9 * max(float(dt_max_s), 1.0)

    V = np.zeros(n)
    V[g] = C[g]
    t = float(t_start_s)
    t_arr = np.full(n, np.nan)
    t_arr[g] = t
    t_first = np.full(n, np.nan)  # the start of the step a cell took its first melt in
    S = np.zeros(n)
    S[g] = conductance(g, t_arr[g])
    p = np.zeros(n)
    p_report = np.zeros(n)
    cap_total = float(C[m].sum())
    v_prev_inj = float(injected_volume_m3(t))
    # with a pressure limit the screw can fall behind its programme: the volume
    # is read on the screw's own clock, ``t − lag_s``, which stands still for the
    # part of a step the limit held back. A staged profile switches by screw
    # position, i.e. by the volume delivered: on that clock it still does.
    lag_s = 0.0
    limited_steps = 0
    slowed_steps = 0
    p_free_max = 0.0
    stalled = False
    low_run = 0  # steps in a row at the limit with a rate below the stall fraction
    q_low_prev = 0.0
    p_report_max = 0.0
    bps = sorted(b for b in breakpoints_s if b > t)
    probe = min(float(dt_max_s), 1e-6 * max(abs(t), 1.0))

    def rate_ahead(tt: float) -> float:
        """Delivery rate just after ``tt`` (one-sided, so a switch at ``tt`` counts)."""
        return max(
            (float(injected_volume_m3(tt + probe)) - float(injected_volume_m3(tt))) / probe, 1e-300
        )

    # first step: the gate's neighbours share the injection; give the smallest of
    # them one cell of melt at most
    rate0 = rate_ahead(t)
    nb = np.zeros(n, dtype=bool)
    nb[fa[is_gate[fb]]] = True
    nb[fb[is_gate[fa]]] = True
    nb &= m & ~is_gate
    dt = float(dt_max_s)
    if nb.any():
        dt = min(dt, cfl * float(np.min(C[nb])) * int(nb.sum()) / rate0)
    steps = 0
    complete = False

    while steps < max_steps:
        # --- the valves ---
        for k, dg in enumerate(delayed_gates):
            if d_opened[k]:
                continue
            c = d_cells[k]
            if d_open_at[k] is None and np.all(V[c] >= C[c] * FULL_REL):
                d_open_at[k] = t + float(dg.delay_s)  # the melt has reached the gate
            if d_open_at[k] is None or t < d_open_at[k] - t_tol:
                continue
            if np.any(V[c] > C[c]):
                # a cell that took more than it holds in the last step passes the
                # surplus on in this one; a gate cell is held exactly full, so
                # opening now would drop that melt. Open at the next step.
                continue
            was_full = bool(np.all(V[c] >= C[c] * FULL_REL))
            new = c[np.isnan(t_arr[c])]
            v_extra += float(np.sum(C[c] - np.minimum(V[c], C[c])))
            V[c] = C[c]
            t_arr[new] = t
            S[new] = conductance(new, t_arr[new])
            is_gate[c] = True
            g = np.flatnonzero(is_gate)
            d_opened[k] = True
            gate_events.append({"gate": k, "t_s": float(t), "melt_already_there": was_full})
            # the cells next to the gate that opened take one cell of melt at most
            ig = np.zeros(n, dtype=bool)
            ig[c] = True
            nb = np.zeros(n, dtype=bool)
            nb[fa[ig[fb]]] = True
            nb[fb[ig[fa]]] = True
            nb &= m & ~is_gate & (V < C * FULL_REL)
            if nb.any():
                dt = min(dt, cfl * float(np.min(C[nb])) * int(nb.sum()) / rate_ahead(t - lag_s))
        d_next = [
            d_open_at[k]
            for k in range(len(d_cells))
            if not d_opened[k] and d_open_at[k] is not None
        ]
        d_ahead = [x for x in d_next if x > t + t_tol]  # not the ones waiting on a surplus

        full = m & (V >= C * FULL_REL)
        if full[m].all():
            complete = True
            break
        touch = np.zeros(n, dtype=bool)
        touch[fa[full[fb]]] = True
        touch[fb[full[fa]]] = True
        front = m & ~full & (touch | (V > 0))
        fr = np.flatnonzero(front)
        if fr.size == 0:
            # nothing left the front can reach: the open gates' part of the cavity
            # is full. A valve still to open at a set time may feed the rest (a
            # part of its own): the screw stands still until then.
            if d_ahead:
                wait = min(d_ahead) - t
                t += wait
                lag_s += wait
                continue
            break
        S[fr] = conductance(fr, np.full(fr.size, t))

        # a switch within round-off of now has been reached (t is a running sum)
        tp = t - lag_s  # the programme's clock
        while bps and bps[0] <= tp + 1e-9 * max(abs(tp), dt_max_s):
            bps.pop(0)
        if bps:
            dt = min(dt, bps[0] - tp)
        if d_ahead:
            dt = min(dt, min(d_ahead) - t)  # a step ends where a valve opens
        # never deliver more than the cavity still has room for, net of the surplus
        # cells hold and pass on next step: the step that would overshoot is cut to
        # deliver exactly the room left, and once the melt in the cavity fills it
        # the steps only move the surplus on (@claude review on PR #114: with the
        # room at zero the cut did not apply and a step could deliver dt_max's worth)
        room_left = float(np.sum(C[m] - V[m]))
        if room_left > ROOM_TOL * cap_total:
            dt = max(min(dt, room_left / rate_ahead(tp)), 1e-12)
            v_next = float(injected_volume_m3(tp + dt))
        else:
            v_next = v_prev_inj
        q_total = (v_next - v_prev_inj) / dt
        # unknowns: full cells that are not gates; Dirichlet: front (0) and gates (p_g)
        unk = full & ~is_gate
        u = np.flatnonzero(unk)
        uid = -np.ones(n, dtype=np.int64)
        uid[u] = np.arange(u.size)

        Sa, Sb = S[fa], S[fb]
        late_k = late_b = late_S = None
        if late_inflow is not None:
            # a full cell feeding a front cell that began to fill before it was full
            # (NaN -- a front cell with no melt yet -- compares false)
            with np.errstate(invalid="ignore"):
                a_late = full[fa] & front[fb] & (t_arr[fa] > t_first[fb])
                b_late = full[fb] & front[fa] & (t_arr[fb] > t_first[fa])
            if a_late.any() or b_late.any():
                ka, kb = np.flatnonzero(a_late), np.flatnonzero(b_late)
                late_k = np.concatenate([ka, kb])
                late_b = np.concatenate([np.ones(ka.size, bool), np.zeros(kb.size, bool)])
                fr_c = np.concatenate([fb[ka], fa[kb]])
                fd_c = np.concatenate([fa[ka], fb[kb]])
                late_S = np.maximum(S[fr_c], np.asarray(late_inflow(fr_c, fd_c), dtype=float))
                Sb[ka] = late_S[: ka.size]
                Sa[kb] = late_S[ka.size :]
        tot = Sa + Sb
        Sf = np.where(tot > 0, 2.0 * Sa * Sb / np.where(tot > 0, tot, 1.0), 0.0)
        ua, ub = unk[fa], unk[fb]
        uu = ua & ub
        diag = np.zeros(u.size)
        np.add.at(diag, uid[fa[uu]], Sf[uu])
        np.add.at(diag, uid[fb[uu]], Sf[uu])
        # faces from an unknown to a Dirichlet cell (front or gate) add to the diagonal
        ud = ua & (front[fb] | is_gate[fb])
        du = ub & (front[fa] | is_gate[fa])
        np.add.at(diag, uid[fa[ud]], Sf[ud])
        np.add.at(diag, uid[fb[du]], Sf[du])
        rhs0 = (V[u] - C[u]) / dt  # sources: push out the previous step's surplus
        rhs1 = np.zeros(u.size)  # response to a unit gate pressure
        ug = ua & is_gate[fb]
        gu = ub & is_gate[fa]
        np.add.at(rhs1, uid[fa[ug]], Sf[ug])
        np.add.at(rhs1, uid[fb[gu]], Sf[gu])
        if u.size:
            ia, ib = uid[fa[uu]], uid[fb[uu]]
            A = sp.coo_matrix(
                (
                    np.concatenate([-Sf[uu], -Sf[uu], diag]),
                    (
                        np.concatenate([ia, ib, np.arange(u.size)]),
                        np.concatenate([ib, ia, np.arange(u.size)]),
                    ),
                ),
                shape=(u.size, u.size),
            ).tocsc()
            lu = spla.splu(A)
            x0, x1 = lu.solve(rhs0), lu.solve(rhs1)
        else:
            x0 = x1 = np.zeros(0)
        # gate balance: melt leaving the gate cells over the step = what was delivered
        # (the gate cells stay exactly full, so they carry no surplus of their own)
        p0 = np.zeros(n)
        p1 = np.zeros(n)
        p0[u], p1[u] = x0, x1
        p1[is_gate] = 1.0
        gface_a = is_gate[fa] & ~is_gate[fb] & (unk[fb] | front[fb])
        gface_b = is_gate[fb] & ~is_gate[fa] & (unk[fa] | front[fa])
        out1 = float(
            np.sum(Sf[gface_a] * (1.0 - p1[fb[gface_a]]))
            + np.sum(Sf[gface_b] * (1.0 - p1[fa[gface_b]]))
        )
        out0 = float(
            np.sum(Sf[gface_a] * (0.0 - p0[fb[gface_a]]))
            + np.sum(Sf[gface_b] * (0.0 - p0[fa[gface_b]]))
        )
        if out1 <= 0:
            raise RuntimeError("the gate cells have no open face to the melt")
        p_g = (q_total - out0) / out1
        was_limited = False
        if pressure_limit_Pa is not None and p_g > pressure_limit_Pa and q_total > 0:
            was_limited = True
            # the machine is at its limit: the gate stays there, the rate follows
            p_g = float(pressure_limit_Pa)
            q_limited = max(out0 + p_g * out1, 0.0)
            # the screw got as far as the programme does at tp + dt_done
            lo, hi = 0.0, dt
            target = v_prev_inj + q_limited * dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if float(injected_volume_m3(tp + mid)) < target:
                    lo = mid
                else:
                    hi = mid
            lag_s += dt - 0.5 * (lo + hi)
            v_next = v_prev_inj + q_limited * dt
            limited_steps += 1
            if q_limited < stall_rate_fraction * q_total:
                # a rate still growing (the flow-dependent conductance settling) starts over
                # (growing = half as much again as the most this run has seen: a rate
                # that only wobbles cannot keep that up below the stall fraction)
                if low_run and q_limited <= 1.5 * q_low_prev:
                    low_run += 1
                else:
                    low_run = 1
                q_low_prev = max(q_limited, q_low_prev if low_run > 1 else 0.0)
                # a valve still to open at a set time may open the way: keep going
                stalled = low_run >= max(int(stall_steps), 1) and not d_next
            else:
                low_run = 0
                slowed_steps += 1
            q_total = q_limited
        p = p0 + p_g * p1
        if q_total > 0:
            # the pressure the injection alone needs to reach the front, without the
            # artificial push of the surplus terms (which scale as 1/dt and dominate
            # a short final step): what the end-of-fill pressure map reports
            p_report = (q_total / out1) * p1
            p_report_max = max(p_report_max, q_total / out1)
            if not was_limited:
                p_free_max = max(p_free_max, q_total / out1)

        flux = Sf * (p[fa] - p[fb])  # a -> b
        inflow = np.zeros(n)
        np.add.at(inflow, fb, flux)
        np.add.at(inflow, fa, -flux)

        V_old = V.copy()
        V[u] = C[u]
        V[g] = C[g]
        V[fr] = V_old[fr] + dt * inflow[fr]
        began = fr[(V_old[fr] <= 0.0) & (V[fr] > 0.0)]
        t_first[began] = t
        newly = fr[V[fr] >= C[fr] * FULL_REL]
        if newly.size:
            rate = np.maximum(inflow[newly], 1e-300)
            frac = np.clip((C[newly] - V_old[newly]) / rate, 0.0, dt)
            t_arr[newly] = t + frac
            S[newly] = conductance(newly, t_arr[newly])
        if on_step is not None:
            gate_in = np.zeros(n)
            gate_in[g] = np.maximum(-inflow[g], 0.0) * dt
            S_now = on_step(
                MarchStep(
                    t_s=t,
                    dt_s=dt,
                    face_a=fa,
                    face_b=fb,
                    moved_m3=flux * dt,
                    held_m3=V_old,
                    full=full,
                    newly=newly,
                    gate_in_m3=gate_in,
                    pressure_Pa=(max(q_total, 0.0) / out1) * p1,
                    volume_injected_m3=v_next + v_extra,
                    late_faces=late_k,
                    late_front_is_b=late_b,
                    late_front_conductance=late_S,
                )
            )
            if S_now is not None:
                behind = m & (V >= C * FULL_REL)
                S[behind] = np.asarray(S_now, dtype=float)[behind]
        t += dt
        if lag_s == 0.0 and bps and abs(bps[0] - t) <= 1e-9 * max(abs(t), dt_max_s):
            t = bps[0]  # land on the switch, so the rate read after it is the new stage's
        v_prev_inj = v_next
        steps += 1
        if stalled:
            break

        # the next step's inflow scales with the delivery rate (a switch just passed)
        r_next = rate_ahead(t - lag_s)
        if was_limited:
            # at the pressure limit the melt comes in at the rate the limit gives,
            # not the programme's: size the step by what did flow
            r_next = min(r_next, q_total) if q_total > 0 else r_next
        scale = r_next / max(q_total, 1e-300) if q_total > 0 else 1.0
        dt = _next_step(C, V, m, fa, fb, fr, newly, inflow * scale, cfl, dt_max_s, r_next)
        if was_limited:
            # A front cell that is nearly full, fed at the limit's falling rate, is
            # approached in ever shorter steps and never reached. Below this the
            # step is not cut further (the surplus moves on with the next step).
            dt = max(dt, 0.02 * float(dt_max_s))

    return MarchFillResult(
        t_arr_s=np.where(m, t_arr, np.nan).reshape(ny, nx),
        pressure_end_Pa=np.where(m, p_report, 0.0).reshape(ny, nx),
        n_steps=steps,
        t_end_s=t,
        complete=complete,
        volume_injected_m3=v_prev_inj + v_extra,
        volume_held_m3=float(V[m].sum()),
        stalled=stalled,
        pressure_limited_steps=limited_steps,
        gate_pressure_max_Pa=p_report_max,
        gate_pressure_free_max_Pa=p_free_max,
        pressure_slowed_steps=slowed_steps,
        gate_events=gate_events,
    )


def enclosed_seams(unreached: np.ndarray, cavity: np.ndarray) -> bool:
    """Whether the cells a stalled march left are lines with melt on both sides.

    True when every one of them has, along ``x`` or along ``y``, a full cavity
    cell on each side, next to it or one unreached cell away (a line at most
    two cells wide) -- or lies against the cavity wall next to one that has: the
    end of such a line, where it runs into the wall. A front that stopped and froze, met later by
    another, leaves such a line. A rim or a channel the melt did not fill has a
    wall on one side all along, and a region has cells with no full cell that
    near: both are a short shot, whatever their width.
    """
    un = np.asarray(unreached, dtype=bool)
    if not un.any():
        return False
    full = np.asarray(cavity, dtype=bool) & ~un
    ny, nx = un.shape
    f = np.pad(full, 2)
    u = np.pad(un, 2)

    def side(dy: int, dx: int) -> np.ndarray:
        def at(arr, k):
            return arr[2 + k * dy : 2 + k * dy + ny, 2 + k * dx : 2 + k * dx + nx]

        return at(f, 1) | (at(u, 1) & at(f, 2))

    both = un & ((side(0, -1) & side(0, 1)) | (side(-1, 0) & side(1, 0)))
    # the end of a line at the wall: beside a cell of the line, and against the wall
    b = np.pad(both, 1)
    beside = b[:-2, 1:-1] | b[2:, 1:-1] | b[1:-1, :-2] | b[1:-1, 2:]
    c = np.pad(np.asarray(cavity, dtype=bool), 1)
    at_wall = ~(c[:-2, 1:-1] & c[2:, 1:-1] & c[1:-1, :-2] & c[1:-1, 2:])
    return bool(np.all((both | (beside & at_wall))[un]))


def _next_step(C, V, m, fa, fb, fr, newly, inflow, cfl, dt_max_s, rate) -> float:
    """Length of the next step: no front cell of the *next* step should take more
    than ``cfl`` times its own capacity.

    A cell still at the front keeps its inflow. A cell that just joined the
    front is fed by the cells that just filled next to it; each of those passes
    its inflow on, split over the neighbours that are not yet full. Reading the
    step off the old front instead (the cells that just filled) let a thick cell
    set the step for the thin cell after it -- a ramp feeding a 0.35 mm land took
    several cells of melt in one step, and the surplus rode ahead of the front
    (a strip of uneven cells was stamped up to 7 % late).
    """
    n = C.size
    full = m & (V >= C * FULL_REL)
    est = np.zeros(n)
    keep = fr[~full[fr]]
    est[keep] = np.maximum(inflow[keep], 0.0)
    if newly.size:
        is_new = np.zeros(n, dtype=bool)
        is_new[newly] = True
        open_ = m & ~full
        # neighbours of each newly full cell that are still open
        cnt = np.zeros(n)
        np.add.at(cnt, fa[is_new[fa] & open_[fb]], 1.0)
        np.add.at(cnt, fb[is_new[fb] & open_[fa]], 1.0)
        share = np.where(cnt > 0, np.maximum(inflow, 0.0) / np.where(cnt > 0, cnt, 1.0), 0.0)
        a2b = is_new[fa] & open_[fb]
        b2a = is_new[fb] & open_[fa]
        np.add.at(est, fb[a2b], share[fa[a2b]])
        np.add.at(est, fa[b2a], share[fb[b2a]])
    fed = np.flatnonzero(m & ~full & (est > 0))
    if not fed.size:
        # no inflow to read (a step cut short at a switch): share the delivery rate
        # over the open cells next to the melt, as for the first step
        touch = np.zeros(n, dtype=bool)
        touch[fa[full[fb]]] = True
        touch[fb[full[fa]]] = True
        nxt = np.flatnonzero(m & ~full & touch)
        if not nxt.size:
            return float(dt_max_s)
        return max(min(cfl * float(np.min(C[nxt])) * nxt.size / max(rate, 1e-300), dt_max_s), 1e-12)
    # Sized on the whole capacity, not the room left. A partly filled cell may then
    # take up to a cell of melt beyond what fills it; it is stamped at the right
    # time within the step, but its surplus only moves on in the next step, so the
    # cells after it come in late by up to one cell-fill time (a strip's earliest
    # cells, where that is a large fraction of their time, by 14 % with a coarse
    # step cap). Sizing on the room instead (floored at a tenth / half a cell) cost
    # 5x / 1.6x the steps on the FG9 block, and halving every step (cfl 0.5) moved
    # the photo fronts there by under 0.02 mm: the error is grid-scale in time and
    # does not show in the fill pattern
    return max(min(cfl * float(np.min(C[fed] / est[fed])), dt_max_s), 1e-12)
