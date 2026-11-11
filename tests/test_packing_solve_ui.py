"""AppTest wiring for the holding-phase solve pane in ``app.py``."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts

OFF_NOTE = "「保圧を解く」を ON にして解析すると出ます"
ICM_NOTE = "射出圧縮成形 ON の解析では解きません"
PVT_NOTE = "pvT データが無いので解けません"
THIN_NOTE = "薄い所（1 mm の額縁など）が保圧で埋まるかの判定は当てにならない"


def _run(
    at,
    *,
    icm: bool,
    hold: bool,
    pressure: float = 60.0,
    time_s: float = 10.0,
    vp: float | None = None,
):
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(icm)
    at.checkbox(key="hold_on").set_value(hold).run()
    if hold:
        at.number_input(key="hold_p_mpa").set_value(pressure)
        if vp is not None:  # None leaves the sidebar's default
            at.number_input(key="hold_vp_pct").set_value(vp)
        at.number_input(key="hold_t_s").set_value(time_s).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_holding_is_not_solved_unless_asked():
    at = _run(_app(), icm=False, hold=False)
    assert at.session_state["mfs_hold_path"] is None
    assert at.session_state["mfs_hold_rows"] is None
    assert OFF_NOTE in texts(at)
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert "packing_solve.png" not in names and "packing_solve.json" not in names
    # the inputs are there, greyed out
    assert at.number_input(key="hold_p_mpa").disabled
    assert at.number_input(key="hold_t_s").disabled


def test_holding_solve_shows_shrinkage_pressure_and_the_table():
    at = _run(_app(), icm=False, hold=True, pressure=60.0, time_s=10.0)
    path = at.session_state["mfs_hold_path"]
    hist = at.session_state["mfs_hold_hist_path"]
    rows = at.session_state["mfs_hold_rows"]
    info = at.session_state["mfs_hold_info"]
    assert Path(path).exists() and Path(hist).exists()
    assert [r["thickness_mm"] for r in rows] == [4.0, 1.0]
    thick, thin = rows
    # holding pressure reached the 4 mm plate, and packed melt into it
    assert 1e6 < thick["peak_pressure_Pa"] <= 60e6 * 1.001
    assert -0.02 < thick["shrinkage_mean"] < 0.08
    assert info["pressure_MPa"] == 60.0 and info["time_s"] == 10.0
    # the default: switch at 99 % of the volume, solved on the fill's own grid
    assert info["vp_switch_fill_percent"] == 99.0
    assert 0.0 < info["vp_switch_time_s"] < at.session_state["mfs_result"].total_fill_time_s
    assert info["grid_mm"] == 4.0
    assert 0.0 <= info["unfilled_volume_percent"] < 100.0
    assert Path(at.session_state["mfs_hold_late_path"]).exists()
    assert info["added_mass_percent"] > 0.5
    assert info["clamp_force_peak_kN"] > 0.0
    assert info["unconverged_steps"] == 0
    assert OFF_NOTE not in texts(at)
    # what the late fill cannot tell is said next to it
    assert THIN_NOTE in texts(at)
    shown = [df for df in at.dataframe if "縮み 平均 [%]" in df.value.columns]
    assert len(shown) == 1
    assert list(shown[0].value["肉厚 [mm]"]) == [4.0, 1.0]
    labels = [str(m.label) for m in at.metric]
    assert "保圧で入った樹脂" in labels and "型を開く力の最大" in labels
    assert "埋まらなかった体積" in labels
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert names.count("packing_solve.png") == 1
        assert names.count("packing_pressure.png") == 1
        record = json.loads(zf.read("packing_solve.json"))
        assert record["pressure_MPa"] == 60.0
        assert [r["thickness_mm"] for r in record["by_thickness"]] == [4.0, 1.0]
        settings = json.loads(zf.read("settings.json"))
        assert settings["holding"] == {
            "enabled": True,
            "pressure_MPa": 60.0,
            "time_s": 10.0,
            "vp_switch_fill_percent": 99.0,
            "coarse_grid": False,
        }
        assert names.count("packing_late_fill.png") == 1


def test_changing_the_holding_pressure_marks_the_result_stale():
    at = _run(_app(), icm=False, hold=True, pressure=60.0, time_s=5.0)
    assert "入力が前回の解析から変更されています" not in texts(at)
    at.number_input(key="hold_p_mpa").set_value(90.0).run()
    assert "入力が前回の解析から変更されています" in texts(at)


def test_compression_molding_and_missing_pvt_explain_why_nothing_is_solved():
    at = _run(_app(), icm=True, hold=True)
    assert at.session_state["mfs_hold_path"] is None
    assert ICM_NOTE in texts(at)

    at = _app()
    materials = [s for s in at.selectbox if "PMMA" in list(s.options) and "PP" in list(s.options)]
    assert len(materials) == 1
    materials[0].set_value("PP").run()
    at = _run(at, icm=False, hold=True)
    assert at.session_state["mfs_result"] is not None
    assert at.session_state["mfs_hold_path"] is None
    assert PVT_NOTE in texts(at)


def test_switch_at_100_percent_packs_a_full_cavity():
    at = _run(_app(), icm=False, hold=True, pressure=60.0, time_s=5.0, vp=100.0)
    info = at.session_state["mfs_hold_info"]
    assert info["vp_switch_fill_percent"] == 100.0
    assert info["unfilled_volume_percent"] == 0.0
    assert "全部埋まったものとして、保圧だけを解いた" in texts(at)
    assert THIN_NOTE not in texts(at)
