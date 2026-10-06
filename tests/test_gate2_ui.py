"""AppTest wiring for the gate-shape selector (Gate 1 / Gate 2) and the
thickness options in the sidebar."""

from __future__ import annotations

import dataclasses

from core import GATE2_DEFAULTS, FanRunnerPlateConfig
from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts


def _keys(at) -> set[str]:
    return {n.key for n in at.number_input} | {c.key for c in at.checkbox}


def test_gate1_is_the_default_and_hides_the_gate2_inputs():
    at = _app()
    assert at.radio(key="fg_runner_shape").value == "triangle"
    keys = _keys(at)
    assert "fg_fan_flank_deg" in keys and "fg_runner_ramp_end_mm" in keys
    assert "fg_side_len_mm" not in keys
    assert "fg_g2_runner_end_thk_mm" not in keys and "fg_g2_ramp_at_end_top" not in keys
    # the round end follows the profile unless asked
    assert at.checkbox(key="fg_g1_end_thk_on").value is False
    assert "fg_g1_runner_end_thk_mm" not in keys


def test_gate2_shows_its_inputs_and_records_the_sketch():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    assert not at.exception
    keys = _keys(at)
    assert "fg_fan_flank_deg" not in keys and "fg_runner_ramp_end_mm" not in keys
    assert at.number_input(key="fg_side_len_mm").value == 28.0
    assert at.number_input(key="fg_g2_runner_end_thk_mm").value == 2.5
    assert at.checkbox(key="fg_g2_ramp_at_end_top").value is True
    text = _texts(at)
    assert "額縁込みの側辺 h = 48.0 mm" in text
    assert "傾き 5.4°" in text
    assert "傾斜の開始位置（下限）= 18.0 mm" in text
    assert "形状パラメータが不正" not in text
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]["geometry"]["config"]
    assert rec == dataclasses.asdict(FanRunnerPlateConfig(**GATE2_DEFAULTS, cell_size_mm=4.0))
    assert at.session_state["mfs_geom"].label == "pentagon_runner_plate"


def test_side_length_is_measured_below_the_rim():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    at.number_input(key="fg_side_len_mm").set_value(10.0).run()
    assert not at.exception
    assert "額縁込みの側辺 h = 30.0 mm" in _texts(at)


def test_gate2_ramp_start_can_be_set_by_hand():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    at.checkbox(key="fg_g2_ramp_at_end_top").set_value(False).run()
    assert not at.exception
    assert at.number_input(key="fg_g2_runner_ramp_end_mm").value == 18.0
    v0 = at.session_state["mfs_shot_volume_auto"]
    at.number_input(key="fg_g2_runner_ramp_end_mm").set_value(25.0).run()
    assert not at.exception
    # a longer ramp is thinner on average
    assert at.session_state["mfs_shot_volume_auto"] < v0


def test_ramp_off_hides_the_ramp_start_and_steps_to_t_o():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    v_ramp = at.session_state["mfs_shot_volume_auto"]
    at.checkbox(key="fg_runner_ramp_on").set_value(False).run()
    assert not at.exception
    keys = _keys(at)
    assert "fg_g2_ramp_at_end_top" not in keys and "fg_g2_runner_ramp_end_mm" not in keys
    # no ramp: t_o right after the band, so more material than with it
    assert at.session_state["mfs_shot_volume_auto"] > v_ramp
    # t_o = 1.0 with the ramp off: only the disc is deep
    at.number_input(key="fg_runner_thk_mm").set_value(1.0).run()
    assert not at.exception
    assert "形状パラメータが不正" not in _texts(at)
    assert at.session_state["mfs_shot_volume_auto"] < v_ramp


def test_gate1_can_take_a_round_end_depth():
    at = _app()
    v0 = at.session_state["mfs_shot_volume_auto"]
    at.checkbox(key="fg_g1_end_thk_on").set_value(True).run()
    assert not at.exception
    assert at.number_input(key="fg_g1_runner_end_thk_mm").value == 2.5
    at.number_input(key="fg_g1_runner_end_thk_mm").set_value(4.0).run()
    assert not at.exception
    assert at.session_state["mfs_shot_volume_auto"] > v0


def test_a_gate2_side_below_the_round_end_is_an_error_message():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    at.number_input(key="fg_side_len_mm").set_value(50.0).run()
    assert not at.exception
    text = _texts(at)
    assert "形状パラメータが不正" in text and "side_len_mm" in text
