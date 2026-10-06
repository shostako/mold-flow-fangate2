"""AppTest wiring for Gate 3 (Gate 1 + flank arms) in the gate-shape selector."""

from __future__ import annotations

import dataclasses

from core import GATE3_DEFAULTS, FanRunnerPlateConfig
from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts


def _keys(at) -> set[str]:
    keys = {n.key for n in at.number_input} | {c.key for c in at.checkbox}
    return {k for k in keys if k}


def test_gate3_inputs_only_show_on_gate3():
    at = _app()
    assert not any(k.startswith("fg_g3_") for k in _keys(at))
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    assert not any(k.startswith("fg_g3_") for k in _keys(at))


def test_gate3_shows_the_arms_on_top_of_gate1_and_records_them():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("triangle_arms").run()
    assert not at.exception
    keys = _keys(at)
    # Gate 3 is Gate 1 plus the arms: the triangle's own inputs stay
    assert {"fg_fan_flank_deg", "fg_runner_ramp_end_mm", "fg_g1_end_thk_on"} <= keys
    assert "fg_side_len_mm" not in keys and not any(k.startswith("fg_g2_") for k in keys)
    assert at.number_input(key="fg_g3_arm_w_mm").value == 9.0
    assert at.number_input(key="fg_g3_arm_thk_mm").value == 3.5
    text = _texts(at)
    assert "36.3 mm 未満" in text
    assert "形状パラメータが不正" not in text
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]["geometry"]["config"]
    assert rec == dataclasses.asdict(FanRunnerPlateConfig(**GATE3_DEFAULTS, cell_size_mm=4.0))
    assert at.session_state["mfs_geom"].label == "arms_runner_plate"


def test_arms_add_runner_volume():
    at = _app()
    v_gate1 = at.session_state["mfs_shot_volume_auto"]
    at.radio(key="fg_runner_shape").set_value("triangle_arms").run()
    v_gate3 = at.session_state["mfs_shot_volume_auto"]
    assert v_gate3 > v_gate1
    at.number_input(key="fg_g3_arm_w_mm").set_value(14.0).run()
    assert not at.exception
    assert at.session_state["mfs_shot_volume_auto"] > v_gate3


def test_an_arm_no_thicker_than_the_edge_band_is_an_error_message():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("triangle_arms").run()
    at.number_input(key="fg_g3_arm_thk_mm").set_value(1.0).run()
    assert not at.exception
    text = _texts(at)
    assert "形状パラメータが不正" in text and "arm_thk_mm" in text


def test_arm_width_bound_follows_the_flank():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("triangle_arms").run()
    at.number_input(key="fg_fan_flank_deg").set_value(30.0).run()
    assert not at.exception
    assert "75.0 mm 未満" in _texts(at)
    assert at.number_input(key="fg_g3_arm_w_mm").max == 74.5
    at.number_input(key="fg_g3_arm_w_mm").set_value(60.0).run()
    assert not at.exception
    # back to 14°: the bound shrinks under the entered width
    at.number_input(key="fg_fan_flank_deg").set_value(14.0).run()
    assert not at.exception
    assert at.number_input(key="fg_g3_arm_w_mm").value <= 36.0
    assert "形状パラメータが不正" not in _texts(at)
