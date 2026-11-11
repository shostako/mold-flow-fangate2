"""Two-domain Tait pvT model (specific volume of a polymer vs T and p).

    v(T, p) = v0(T) · [1 − C · ln(1 + p / B(T))]          C = 0.0894

with, above the transition temperature ``T_t(p) = b5 + b6·p`` (melt),

    v0 = b1m + b2m·(T − b5),    B = b3m · exp(−b4m·(T − b5))

and below it (solid) the same expressions with ``b1s … b4s``. The extra
crystallisation term of the full model (``b7 … b9``) is zero for an amorphous
polymer and is not carried here.

Units are SI throughout: v [m³/kg], T [K], p [Pa].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAIT_C = 0.0894


@dataclass(frozen=True)
class TaitPVT:
    b1m: float  # m³/kg
    b2m: float  # m³/(kg·K)
    b3m: float  # Pa
    b4m: float  # 1/K
    b1s: float
    b2s: float
    b3s: float
    b4s: float
    b5: float  # K
    b6: float  # K/Pa

    @classmethod
    def from_dict(cls, d: dict) -> TaitPVT:
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__})

    def at_temperature(self, T_K, band_K: float = 0.0) -> TaitIsotherm:
        """The model at fixed temperatures, ready to be evaluated at many
        pressures (the temperature-only parts are worked out once)."""
        return TaitIsotherm(self, np.asarray(T_K, dtype=float), float(band_K))

    def specific_volume(self, T_K, p_Pa, band_K: float = 0.0):
        """v(T, p) [m³/kg]; broadcasts over arrays."""
        return self.at_temperature(T_K, band_K).state(p_Pa)[0]

    def density_and_slope(self, T_K, p_Pa, band_K: float = 0.0):
        """``(ρ, ∂ρ/∂p)`` at fixed temperature [kg/m³, kg/(m³·Pa)].

        The model switches branch where ``T = b5 + b6·p``, and its two branches
        do not meet there: ρ(p) jumps. ``band_K > 0`` replaces the jump by a
        linear blend over ``±band_K``, which keeps ρ(p) continuous (a Newton
        solve on the pressure needs that) and the slope consistent with it.
        """
        return self.at_temperature(T_K, band_K).density_and_slope(p_Pa)


class TaitIsotherm:
    """:class:`TaitPVT` at fixed temperatures (see ``TaitPVT.at_temperature``)."""

    def __init__(self, pvt: TaitPVT, T_K: np.ndarray, band_K: float) -> None:
        dT = T_K - pvt.b5
        self._above0 = dT  # T − T_t at p = 0
        self._b6 = pvt.b6
        self._band = band_K
        self._v0m = pvt.b1m + pvt.b2m * dT
        self._v0s = pvt.b1s + pvt.b2s * dT
        self._Bm = pvt.b3m * np.exp(-pvt.b4m * dT)
        self._Bs = pvt.b3s * np.exp(-pvt.b4s * dT)

    def state(self, p_Pa):
        """``(v, ∂v/∂p)`` with the two branches blended over ``±band`` around
        the transition (``band = 0``: the sharp switch of the model)."""
        p = np.asarray(p_Pa, dtype=float)
        vm = self._v0m * (1.0 - TAIT_C * np.log1p(p / self._Bm))
        vs = self._v0s * (1.0 - TAIT_C * np.log1p(p / self._Bs))
        dvm = -self._v0m * TAIT_C / (self._Bm + p)
        dvs = -self._v0s * TAIT_C / (self._Bs + p)
        above = self._above0 - self._b6 * p  # T − T_t(p)
        if self._band <= 0.0:
            melt = above > 0.0
            return np.where(melt, vm, vs), np.where(melt, dvm, dvs)
        w = np.clip(0.5 + above / (2.0 * self._band), 0.0, 1.0)  # 1 = melt
        dw = np.where((w > 0.0) & (w < 1.0), -self._b6 / (2.0 * self._band), 0.0)
        return w * vm + (1.0 - w) * vs, w * dvm + (1.0 - w) * dvs + dw * (vm - vs)

    def density_and_slope(self, p_Pa):
        v, dv = self.state(p_Pa)
        rho = 1.0 / v
        return rho, -rho * rho * dv
