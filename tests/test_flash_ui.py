"""AppTest wiring for the flash risk pane and the layered temperature in ``app.py``."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts

SERIES_NOTE = "製品の輪郭の圧力が実際より低く（0 に）出る"
THIN_NOTE = "薄い所（1 mm の額縁など）が保圧で埋まるかの判定は当てにならない"


def _run(at, *, wall: str, energy: bool = False, clamp_kN: float | None = None):
    at.radio(key="wall_model").set_value(wall)
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(False)
    at.checkbox(key="hold_on").set_value(True).run()
    at.number_input(key="hold_p_mpa").set_value(90.0)
    at.number_input(key="hold_t_s").set_value(10.0)
    if clamp_kN is not None:
        at.number_input(key="machine_clamp_kN").set_value(clamp_kN)
    at.run()
    if wall == "multilayer":
        at.radio(key="fill_method").set_value("march").run()
        at.radio(key="thermal_model").set_value("energy" if energy else "arrival").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_flash_pane_reads_the_holding_solve_and_says_what_the_series_model_hides():
    at = _run(_app(), wall="none", clamp_kN=1000.0)
    info = at.session_state["mfs_flash_info"]
    assert Path(at.session_state["mfs_flash_path"]).exists()
    assert info["thermal"] == "series"
    assert info["machine_clamp_kN"] == 1000.0
    assert info["clamp_peak_kN"] > 0.0
    # the runner next to the gate holds about the holding pressure
    assert info["runner_spots"] and info["runner_spots"][0]["最大圧力 [MPa]"] > 50.0
    assert SERIES_NOTE in texts(at)
    tables = [df for df in at.dataframe if "最大圧力 [MPa]" in df.value.columns]
    assert len(tables) == 2
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        assert "flash_risk.png" in zf.namelist()
        saved = json.loads(zf.read("flash_risk.json"))
        assert saved["machine_clamp_kN"] == 1000.0
        settings = json.loads(zf.read("settings.json"))
        assert settings["holding"]["machine_clamp_kN"] == 1000.0


def test_layered_temperature_goes_from_the_fill_to_the_holding_solve():
    at = _run(_app(), wall="multilayer", energy=True)
    md = at.session_state["mfs_result"].metadata
    assert md["thermal_model"] == "energy"
    hold = at.session_state["mfs_hold_info"]
    assert hold["thermal"] == "layers"
    flash = at.session_state["mfs_flash_info"]
    assert flash["thermal"] == "layers"
    # with the temperature carried, the thin rim is still molten at the switch
    assert flash["product_spots"]
    assert flash["product_spots"][0]["最大圧力 [MPa]"] > 5.0
    shown = texts(at)
    assert SERIES_NOTE not in shown and THIN_NOTE not in shown
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        settings = json.loads(zf.read("settings.json"))
        assert settings["wall_cooling"]["thermal_model"] == "energy"
        assert settings["wall_cooling"]["injection_pressure_limit_MPa"] == 200.0


def test_the_temperature_choice_needs_the_marching_fill():
    at = _app()
    at.radio(key="wall_model").set_value("multilayer").run()
    at.radio(key="fill_method").set_value("tau").run()
    assert at.radio(key="thermal_model").disabled
    assert at.number_input(key="inj_p_limit_mpa").disabled
    at.radio(key="fill_method").set_value("march").run()
    assert not at.radio(key="thermal_model").disabled
    at.radio(key="thermal_model").set_value("energy").run()
    assert not at.number_input(key="inj_p_limit_mpa").disabled


def test_a_fill_that_stalls_before_the_switch_hands_over_where_it_stopped():
    # (the fill's stall is checked first: if this mesh ever fills it all, the test says so)
    at = _app()
    at.radio(key="wall_model").set_value("multilayer")
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(False)
    at.checkbox(key="hold_on").set_value(True).run()
    at.radio(key="fill_method").set_value("march").run()
    at.radio(key="thermal_model").set_value("energy").run()
    at.number_input(key="inj_p_limit_mpa").set_value(20.0)
    # switching at 100 %: the fill stops short of it (on the 4 mm grid a few
    # cells of the thin rim are never reached), so the switch comes where it stopped
    at.number_input(key="hold_vp_pct").set_value(100.0).run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    assert md["march_stalled"] and md["march_volume_fraction"] < 1.0
    hold = at.session_state["mfs_hold_info"]
    assert hold["thermal"] == "layers"
    assert hold["vp_switch_note"] and "で止まった" in hold["vp_switch_note"]
    assert hold["vp_switch_fill_percent"] == pytest.approx(
        100.0 * md["march_volume_fraction"], abs=0.5
    )
    assert "で止まった" in texts(at)
