"""均一板（t2・t3・t4）× ゲート 5 種で、温度を到達時刻から読む方と刻みごとに解く方を比べる。

使い方: python uniform_check.py <case> <出力フォルダ> [格子 mm] [late|nolate]
5 つ目を付けると刻みごとに解く方だけを解く。nolate は、後から来た先端の流れ込みの扱い（late_front_inflow）を切る。
case は t{厚さ}_{FAN1|FILM|TAB2|TABF|DC}。形は fanuc-monitor/scripts/pack_reach_20261111/uniform_matrix.py の build。
充填（arrival / energy）と保圧（series / 充填の温度を渡す layers）を解き、板の中の縮みと圧力を JSON に書く。
保圧は 60 MPa・10 秒、VP 99 %、送る量の上限は射出率。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "ClaudeCode/fanuc-monitor/scripts/pack_reach_20261111"))
import uniform_matrix as um  # noqa: E402

from core.flash import flash_risk  # noqa: E402
from core.materials import MaterialDB  # noqa: E402
from core.multilayer_solver import MultilayerHeleShawSolver  # noqa: E402
from core.packing_solve import solve_packing  # noqa: E402

Q = math.pi / 4 * 50**2 * 20 / 1000  # cm³/s（φ50 × 20 mm/s）
VP = 0.99


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    cell = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    late = sys.argv[4] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    t, gate, neck, geom, X, Y = um.build(name, cell)
    mat = MaterialDB()["PMMA"]
    plate = geom.mask & (Y > 0) & (Y <= um.PLATE_H) & (np.abs(X) <= um.PLATE_W / 2)
    row: dict = {"case": name, "thickness_mm": t, "gate": gate, "neck_mm": neck, "cell_mm": cell}
    for model in ("arrival", "energy") if late is None else ("energy",):
        t0 = time.time()
        fill = MultilayerHeleShawSolver(
            geometry=geom, material=mat, melt_temperature_K=533.15, mold_temperature_K=333.15,
            injection_velocity_mms=130.0, injection_volume_flow_cm3s=Q, compression_molding=False,
            num_layers=7, thermal_coupling=True, solidification_temperature_fraction=um.SOLID,
            shear_heating_enabled=True, fill_method="march", thermal_model=model,
            thermal_snapshot_fractions=(VP,) if model == "energy" else (),
            late_front_inflow=late != "nolate",
        ).solve(num_frames=4)  # fmt: skip
        md = fill.metadata
        row[f"{model}_fill_s"] = time.time() - t0
        row[f"{model}_stalled"] = bool(md.get("march_stalled", False))
        row[f"{model}_gate_p_max_mpa"] = md.get("march_gate_pressure_max_MPa")
        kw = {}
        if model == "energy":
            kw = {"thermal": "layers", "initial_theta": fill.thermal_snapshots[VP][1]}
        t1 = time.time()
        r = solve_packing(
            geom.mask, geom.thickness_mm, np.asarray(fill.fill_time_s, float), geom.gates, cell, mat,
            533.15, 333.15, 60e6, 10.0, solid_fraction=um.SOLID, switch_fill_fraction=VP,
            max_inflow_m3_s=Q * 1e-6, **kw,
        )  # fmt: skip
        s = r.shrinkage[plate]
        ok = np.isfinite(s)
        fr = flash_risk(geom, r)
        edge = fr.edge & plate
        tag = "series" if model == "arrival" else "layers"
        row.update(
            {
                f"{tag}_hold_s": time.time() - t1,
                f"{tag}_unconverged": int(r.unconverged_steps),
                f"{tag}_shrink_mean": float(s[ok].mean()),
                f"{tag}_shrink_max": float(s[ok].max()),
                f"{tag}_unfilled_pct": 100.0 * float(r.unfilled_volume_fraction),
                f"{tag}_peak_mean_mpa": float(np.nanmean(r.peak_pressure_Pa[plate]) / 1e6),
                f"{tag}_added_pct": 100.0 * float(r.added_mass_fraction),
                f"{tag}_clamp_kN": fr.clamp_peak_N / 1e3,
                f"{tag}_outline_molten_max_mpa": float(np.nanmax(fr.pressure_Pa[edge]) / 1e6),
                f"{tag}_outline_molten_mean_mpa": float(np.nanmean(fr.pressure_Pa[edge]) / 1e6),
            }
        )
        print(
            name,
            model,
            json.dumps({k: v for k, v in row.items() if k.startswith(tag) or k.startswith(model)}),
            flush=True,
        )
    (out / f"{name}_{cell:g}{'' if late is None else '_' + late}.json").write_text(
        json.dumps(row, indent=1, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
