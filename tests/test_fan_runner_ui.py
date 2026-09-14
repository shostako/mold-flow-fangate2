"""AppTest wiring for the fan-runner geometry sidebar in ``app.py``."""

from __future__ import annotations

import dataclasses

import pytest

from core import FanRunnerPlateConfig
from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts


def test_the_defaults_are_the_drawing_and_are_recorded_in_full():
    at = _app(fast=False)
    assert not at.exception
    at.checkbox(key="two_phase_on").set_value(False)
    at.slider(key="fg_cell_size_mm").set_value(4.0).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]["geometry"]
    assert rec["input"] == "Fan runner plate (parametric)"
    expected = dataclasses.asdict(FanRunnerPlateConfig(cell_size_mm=4.0))
    assert rec["config"] == expected
    geom = at.session_state["mfs_geom"]
    assert geom.label == "fan_runner_plate"
    assert geom.product_mask is not None and geom.compression_mask is not None
    # the compression zone is the body alone
    assert geom.compression_mask.sum() < geom.product_mask.sum()
    assert (geom.thickness_mm[geom.compression_mask] == 4.0).all()


def test_the_preview_reports_the_product_volume_and_the_compression_zone():
    at = _app()
    text = _texts(at)
    assert "うち製品" in text
    assert "赤枠＝圧縮部" in text


def test_every_geometry_widget_reaches_the_builder():
    """Each sidebar field is a config field: the defaults on the widgets are
    the drawing, and a changed value lands in the recorded config."""
    at = _app()
    d = FanRunnerPlateConfig()
    keys = {n.key for n in at.number_input}
    for f in (
        "plate_w_mm",
        "plate_h_mm",
        "frame_w_mm",
        "frame_thk_mm",
        "inner_thk_mm",
        "runner_w_mm",
        "fan_flank_deg",
        "runner_len_mm",
        "runner_end_d_mm",
        "runner_edge_thk_mm",
        "runner_edge_flat_mm",
        "runner_ramp_end_mm",
        "runner_thk_mm",
        "gate_d_mm",
    ):
        assert f"fg_{f}" in keys, f
        assert at.number_input(key=f"fg_{f}").value == pytest.approx(getattr(d, f))
    v0 = at.session_state["mfs_shot_volume_auto"]
    at.number_input(key="fg_runner_thk_mm").set_value(3.5).run()
    assert not at.exception
    assert at.session_state["mfs_shot_volume_auto"] > v0
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    cfg = at.session_state["mfs_settings"]["geometry"]["config"]
    assert cfg["runner_thk_mm"] == 3.5
    assert cfg["fan_flank_deg"] == d.fan_flank_deg


def test_the_flank_caption_reports_the_apex_depth():
    at = _app()
    assert "延長が軸で交わる深さ 37.4 mm" in _texts(at)
    at.number_input(key="fg_fan_flank_deg").set_value(20.0).run()
    assert not at.exception
    assert "延長が軸で交わる深さ 54.6 mm" in _texts(at)


def test_a_bad_parameter_is_an_error_message_not_a_crash():
    at = _app()
    # frames overlap: 2 * frame_w (40) >= plate_h
    at.number_input(key="fg_plate_h_mm").set_value(30.0).run()
    assert not at.exception
    assert "形状パラメータが不正" in _texts(at)
    assert "mfs_geom" not in at.session_state
    # a flank too shallow to reach the round end is the builder's own message
    at.number_input(key="fg_plate_h_mm").set_value(220.6).run()
    at.number_input(key="fg_fan_flank_deg").set_value(2.0).run()
    assert not at.exception
    assert "形状パラメータが不正" in _texts(at) and "flanks meet" in _texts(at)


def test_a_second_run_removes_the_previous_output_directory():
    """Codex P2 on fangate PR #5: every run makes a fresh temp dir; the
    superseded one must go when the new results replace it."""
    from pathlib import Path

    at = _app()
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    first = Path(at.session_state["mfs_tmp_dir"])
    assert first.is_dir()
    at.button[0].click().run()
    assert not at.exception
    second = Path(at.session_state["mfs_tmp_dir"])
    assert second != first
    assert second.is_dir() and not first.exists()
    assert Path(at.session_state["mfs_gif_path"]).parent == second


def test_changing_an_input_after_a_run_marks_the_result_stale_until_rerun():
    """Codex P1 on fangate PR #5: the result pane must say when the sidebar
    no longer matches the run it shows, and the cached result must stay (no
    silent re-solve)."""
    at = _app()
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    result = at.session_state["mfs_result"]
    assert "入力が前回の解析から変更されています" not in _texts(at)
    at.number_input(key="fg_inner_thk_mm").set_value(3.0).run()
    assert not at.exception
    assert "入力が前回の解析から変更されています" in _texts(at)
    assert at.session_state["mfs_result"] is result
    # the weld threshold is re-thresholded live, so it must not count as stale
    at.number_input(key="fg_inner_thk_mm").set_value(4.0).run()
    assert "入力が前回の解析から変更されています" not in _texts(at)
    at.slider(key="weld_min_angle").set_value(20).run()
    assert "入力が前回の解析から変更されています" not in _texts(at)
    at.number_input(key="fg_inner_thk_mm").set_value(3.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert "入力が前回の解析から変更されています" not in _texts(at)
    assert at.session_state["mfs_result"] is not result


def test_the_balancer_toggle_thins_the_runner_and_follows_its_bounds():
    at = _app()
    assert at.checkbox(key="fg_balancer_on").value is False
    keys = {n.key for n in at.number_input}
    assert "fg_balancer_w_mm" not in keys
    v_plain = at.session_state["mfs_shot_volume_auto"]
    at.checkbox(key="fg_balancer_on").set_value(True).run()
    assert not at.exception
    keys = {n.key for n in at.number_input}
    assert {"fg_balancer_w_mm", "fg_balancer_h_mm", "fg_balancer_thk_mm"} <= keys
    d = FanRunnerPlateConfig()
    assert at.number_input(key="fg_balancer_w_mm").value == d.balancer_w_mm
    assert at.number_input(key="fg_balancer_h_mm").value == d.balancer_h_mm
    assert at.session_state["mfs_shot_volume_auto"] < v_plain
    # a narrower runner (80, flanks steepened so they still reach the round
    # end): the base-width bound shrinks to it and the default (100) is
    # clamped instead of erroring
    at.number_input(key="fg_fan_flank_deg").set_value(30.0).run()
    at.number_input(key="fg_runner_w_mm").set_value(80.0).run()
    assert not at.exception
    assert "形状パラメータが不正" not in _texts(at)
    assert at.number_input(key="fg_balancer_w_mm").value == 80.0
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    cfg = at.session_state["mfs_settings"]["geometry"]["config"]
    assert cfg["balancer_on"] is True and cfg["balancer_w_mm"] == 80.0
    # the thickness bound follows the runner thickness: 0.8 clamps the
    # default 1.0 below it instead of erroring
    at.number_input(key="fg_runner_w_mm").set_value(300.0).run()
    at.number_input(key="fg_fan_flank_deg").set_value(14.0).run()
    at.number_input(key="fg_runner_thk_mm").set_value(0.8).run()
    assert not at.exception
    assert "形状パラメータが不正" not in _texts(at)
    assert at.number_input(key="fg_balancer_thk_mm").value == pytest.approx(0.75)
    at.number_input(key="fg_runner_thk_mm").set_value(2.5).run()
    # off again: the widgets go, the defaults are recorded
    at.checkbox(key="fg_balancer_on").set_value(False).run()
    assert "fg_balancer_w_mm" not in {n.key for n in at.number_input}


def test_a_runner_too_short_for_a_balancer_warns_instead_of_an_unfixable_error():
    """With runner_len − runner_end_d/2 below the 1 mm widget minimum the
    sidebar says why and leaves the balancer off (fangate PR #9)."""
    at = _app()
    at.number_input(key="fg_runner_len_mm").set_value(12.5).run()
    assert not at.exception
    at.checkbox(key="fg_balancer_on").set_value(True).run()
    assert not at.exception
    assert "形状パラメータが不正" not in _texts(at)
    assert "肉盗みを置けない" in _texts(at)
    assert "fg_balancer_h_mm" not in {n.key for n in at.number_input}
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_settings"]["geometry"]["config"]["balancer_on"] is False
    # a long enough runner brings the widgets back
    at.number_input(key="fg_runner_len_mm").set_value(30.0).run()
    assert "fg_balancer_h_mm" in {n.key for n in at.number_input}
    assert "肉盗みを置けない" not in _texts(at)
