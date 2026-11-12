"""The melt in the cavity as a state the fill carries along: its temperature
across the thickness, and the flow each cell is carrying.

The τ-era reading of the layers takes both off fixed numbers: the temperature
off the time the melt reached the cell, the shear rate off the screw speed
(``6·V_screw/h``). Neither knows what the melt is doing where it is. In a 1 mm
land that passes the whole shot the melt runs several times faster than the
screw, so the viscosity is read several times too high and the pressure with
it; and a rim the melt reaches late is read as cold before anything got there.

Here a cell's conductance ``S = ∫ z²/η dz`` comes from

* the temperature profile it actually has (:mod:`core.layer_heat`: conduction
  to the walls, the heat the flow brings, viscous heating), and
* the Cross-WLF viscosity at the shear stress the cell's own flow puts on each
  layer, ``τ = |∇p|·z``. The gradient is not read off the pressure field (the
  conductance of a shear-thinning melt goes as a high power of it, and a
  lagged gradient oscillates) but solved from the flow the cell carried in
  the step before: ``S(g)·g = q``. The flow is set by the machine and the
  shape and changes slowly from step to step.
"""

from __future__ import annotations

import numpy as np

from .layer_heat import carry, face_dissipation, flow_shares, layer_grid, shear_heating
from .materials import Material
from .packing_solve import _LN_S_GRID, _cross_table


class MeltState:
    """Layer temperatures and cell flows on ``n`` cavity cells.

    ``theta`` is ``(n, n_z)``: ``(T − T_mold)/(T_melt − T_mold)`` on the half
    thickness, centre → wall. ``flux_m2_s`` is the flow each cell carries per
    unit width. Layers at or below ``solid_fraction`` do not flow.
    """

    def __init__(
        self,
        thickness_m: np.ndarray,
        cell_size_m: float,
        material: Material,
        T_melt_K: float,
        T_mold_K: float,
        *,
        solid_fraction: float = 0.3,
        n_z: int = 16,
        shear_heating_enabled: bool = True,
    ) -> None:
        self.h = np.asarray(thickness_m, dtype=float)
        self.n = int(self.h.size)
        self.dx = float(cell_size_m)
        self.material = material
        self.T_mold = float(T_mold_K)
        self.span = float(T_melt_K) - float(T_mold_K)
        self.solid_fraction = float(solid_fraction)
        self.heating = bool(shear_heating_enabled)
        self.grid = layer_grid(int(n_z))
        g = self.grid
        self.fourier_rate = float(material.thermal_diffusivity_m2_s) / self.h**2
        self.heat_capacity = float(material.density_melt_kgm3) * float(material.specific_heat_J_kgK)
        self.cell_m3 = self.h * self.dx**2
        self._quad = 2.0 * self.h[:, None] ** 3 * (g.wt * g.xi**2)[None, :]
        self._ln_arm = np.log(g.xi[None, :] * self.h[:, None] / float(material.tau_star))
        self._ln_f, self._dln_f = _cross_table(float(material.n))
        self._ln_s0 = float(_LN_S_GRID[0])
        self._inv_step = 1.0 / float(_LN_S_GRID[1] - _LN_S_GRID[0])
        self._last_bin = _LN_S_GRID.size - 2
        self.theta = np.ones((self.n, g.n_z))
        self.flux_m2_s = np.zeros(self.n)
        self.ln_grad = np.full(self.n, -np.inf)  # of the last conductance, to start the next from
        self.dissipated_J = 0.0  # running sums, for the energy balance
        self.heated_J = 0.0
        self.last_substeps = 1  # transport substeps the last step took (layer_heat.carry)
        self.max_substeps_seen = 1

    # ------------------------------------------------------------------ rheology
    def zero_shear_inverse_viscosity(self, theta: np.ndarray) -> np.ndarray:
        """``1/η0`` at the layers (WLF), zero in the solid ones."""
        m = self.material
        dT = self.T_mold + theta * self.span - m.D2
        denom = np.maximum(m.A2_tilde + dT, 1e-6)
        out = np.exp(-np.clip(np.log(m.D1) - m.A1 * dT / denom, -700.0, 700.0))
        out[theta <= self.solid_fraction] = 0.0
        return out

    def _thinning(self, rows, ln_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``ln(η0/η)`` of the Cross model at the stress ``g·z`` of each layer,
        and its slope in ``ln g``."""
        x = np.clip(
            (ln_grad[:, None] + self._ln_arm[rows] - self._ln_s0) * self._inv_step,
            0.0,
            self._last_bin + 1.0,
        )
        k = np.minimum(x.astype(np.intp), self._last_bin)
        w = x - k
        return (
            self._ln_f[k] * (1.0 - w) + self._ln_f[k + 1] * w,
            self._dln_f[k] * (1.0 - w) + self._dln_f[k + 1] * w,
        )

    def flow_state(
        self, rows=None, theta: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conductance ``S`` [m³/(Pa·s)], the gradient ``g`` [Pa/m] and ``1/η``
        at the layers, for the flow the cells carry: ``S(g)·g = q``.

        Solved for ``ln g`` by Newton from the Newtonian value (the largest
        gradient the flow could need): ``ln S + ln g − ln q`` rises with a
        slope between 1 and ``1/n``. A cell that carries nothing is Newtonian.
        """
        rows = slice(None) if rows is None else rows
        th = self.theta[rows] if theta is None else theta
        inv0 = self.zero_shear_inverse_viscosity(th)
        weight = self._quad[rows] * inv0
        s_newton = weight.sum(axis=1)
        q = self.flux_m2_s[rows]
        active = (q > 0.0) & (s_newton > 0.0)
        ln_g = np.full(q.shape, -np.inf)
        s_out = s_newton.copy()
        inv_eta = inv0.copy()
        if active.any():
            idx = np.flatnonzero(active)
            sub = (np.arange(self.n)[rows])[idx]
            w_a = weight[idx]
            ln_q = np.log(q[idx])
            x = ln_q - np.log(s_newton[idx])
            for _ in range(30):
                ln_f, dln_f = self._thinning(sub, x)
                base = w_a * np.exp(ln_f)
                s = base.sum(axis=1)
                f = np.log(s) + x - ln_q
                slope = (base * dln_f).sum(axis=1) / s + 1.0
                step = f / slope
                x = x - step
                if np.max(np.abs(step)) < 1e-8:
                    break
            ln_f, _ = self._thinning(sub, x)
            thin = np.exp(ln_f)
            s_out[idx] = (w_a * thin).sum(axis=1)
            inv_eta[idx] = inv0[idx] * thin
            ln_g[idx] = x
        return s_out, np.exp(ln_g), inv_eta

    def conductance(self, rows=None) -> np.ndarray:
        return self.flow_state(rows)[0]

    # ------------------------------------------------------------------ one step
    def advance(
        self,
        dt_s: float,
        face_a: np.ndarray,
        face_b: np.ndarray,
        moved_m3: np.ndarray,
        held_m3: np.ndarray,
        full: np.ndarray,
        gate_in_m3: np.ndarray,
        pressure_drop_Pa: np.ndarray | None,
        along_x: np.ndarray,
    ) -> None:
        """Take the state over one step of the fill.

        ``moved_m3`` is the melt that crossed each face (positive ``a → b``),
        ``held_m3`` what the cells held before the step, ``full`` the cells
        that were full before it, ``gate_in_m3`` what the machine delivered
        into each cell, ``pressure_drop_Pa`` the drop ``p_b − p_a`` across each
        face that the delivery needed (for the viscous heating; ``None``
        leaves it out), ``along_x`` which faces join neighbours in ``x``.
        """
        g = self.grid
        s_before, _, inv_eta = self.flow_state()
        # conduction and transport together, in substeps short enough for the
        # transport (core.layer_heat.carry)
        th, self.last_substeps = carry(
            self.theta,
            flow_shares(inv_eta, g),
            held_m3,
            face_a,
            face_b,
            moved_m3,
            ~full,
            self.fourier_rate * dt_s,
            g,
            source=gate_in_m3,
        )
        self.max_substeps_seen = max(self.max_substeps_seen, self.last_substeps)
        if self.heating and pressure_drop_Pa is not None:
            heat = face_dissipation(
                face_a, face_b, s_before[face_a], s_before[face_b], pressure_drop_Pa, dt_s, self.n
            )
            rise = shear_heating(heat, inv_eta, self.cell_m3, self.heat_capacity, self.span, g)
            th = th + rise
            self.dissipated_J += float(heat.sum())
            self.heated_J += float(
                (rise * (g.wt / 0.5)[None, :] * self.cell_m3[:, None]).sum()
                * self.heat_capacity
                * self.span
            )
        self.theta = th
        self._take_flux(dt_s, face_a, face_b, moved_m3, gate_in_m3, full, along_x)

    def _take_flux(self, dt_s, face_a, face_b, moved_m3, gate_in_m3, full, along_x) -> None:
        """The flow each cell carried over the step, per unit width.

        Through a cell in ``x``: the mean of what crossed its two ``x`` faces
        (a wall counts as a face nothing crossed); likewise ``y``. A gate cell
        also passes on what the machine put into it. A cell that has only begun
        to take melt is given the flow of the neighbour that feeds it, so that
        the front does not meet a Newtonian wall every time it moves on a cell.
        """
        n = self.n
        rate = moved_m3 / dt_s
        qx = 0.5 * (
            np.bincount(face_a[along_x], rate[along_x], minlength=n)
            + np.bincount(face_b[along_x], rate[along_x], minlength=n)
        )
        qy = 0.5 * (
            np.bincount(face_a[~along_x], rate[~along_x], minlength=n)
            + np.bincount(face_b[~along_x], rate[~along_x], minlength=n)
        )
        q = np.hypot(qx, qy)
        # a source spreads out: half of what it delivers passes through its own cell
        q = np.maximum(q, 0.5 * gate_in_m3 / dt_s)
        q = q / self.dx
        fed = q.copy()
        np.maximum.at(fed, face_b, q[face_a])
        np.maximum.at(fed, face_a, q[face_b])
        self.flux_m2_s = np.where(full, q, np.maximum(q, fed))
