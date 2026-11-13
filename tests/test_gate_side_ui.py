"""AppTest wiring for the gate-side shapes beyond the fan runners: centre tab,
overlap tab, thick tab and direct gates."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core import (
    DIRECT_DEFAULTS,
    OVERLAP_TAB_DEFAULTS,
    TAB_DEFAULTS,
    THICK_TAB_DEFAULTS,
    FanRunnerPlateConfig,
)
from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts

FAN_ONLY = {"fg_runner_w_mm", "fg_fan_flank_deg", "fg_runner_edge_thk_mm", "fg_balancer_on"}
SHOWN = {
    "tab": {"fg_tab_w_mm", "fg_tab_end_thk_mm", "fg_tab_thk_mm", "fg_tab_ramp_len_mm"},
    "overlap_tab": {"fg_tab_w_mm", "fg_ov_tab_thk_mm", "fg_ov_lap_thk_mm"},
    "thick_tab": {"fg_thick_tab_w_mm", "fg_thick_tab_thk_mm", "fg_corner_keep_mm"},
    "direct": {"fg_gate_d_mm"},
}
DEFAULTS = {
    "tab": TAB_DEFAULTS,
    "overlap_tab": OVERLAP_TAB_DEFAULTS,
    "thick_tab": THICK_TAB_DEFAULTS,
    "direct": DIRECT_DEFAULTS,
}
LABELS = {
    "tab": "tab_runner_plate",
    "overlap_tab": "overlap_tab_plate",
    "thick_tab": "thick_tab_plate",
    "direct": "direct_gate_plate",
}


def _keys(at) -> set[str]:
    keys = {n.key for n in at.number_input} | {c.key for c in at.checkbox}
    return {k for k in keys if k}


def test_the_fan_runners_show_none_of_the_other_shapes_inputs():
    keys = _keys(_app())
    assert FAN_ONLY <= keys
    assert not keys & set().union(*SHOWN.values()) - {"fg_gate_d_mm"}


@pytest.mark.parametrize("shape", list(SHOWN))
def test_each_gate_side_shape_shows_its_own_inputs_runs_and_is_recorded(shape):
    at = _app()
    at.radio(key="fg_runner_shape").set_value(shape).run()
    assert not at.exception
    keys = _keys(at)
    assert SHOWN[shape] <= keys
    assert not keys & FAN_ONLY
    # the gate orifice is every shape's
    others = (
        set().union(*(v for k, v in SHOWN.items() if k != shape)) - SHOWN[shape] - {"fg_gate_d_mm"}
    )
    assert not keys & others
    # runner length and round end belong to the shapes that have a runner
    assert ("fg_runner_len_mm" in keys) == (shape != "direct")
    assert "形状パラメータが不正" not in _texts(at)
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]["geometry"]["config"]
    assert rec == dataclasses.asdict(FanRunnerPlateConfig(**DEFAULTS[shape], cell_size_mm=4.0))
    geom = at.session_state["mfs_geom"]
    assert geom.label == LABELS[shape]
    # the fill reached the whole cavity
    t = np.asarray(at.session_state["mfs_result"].fill_time_s)
    assert np.isfinite(t[geom.mask]).all()


def test_the_overlap_tab_inputs_reach_the_builder():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("overlap_tab").run()
    v0 = at.session_state["mfs_shot_volume_auto"]
    at.number_input(key="fg_ov_lap_thk_mm").set_value(2.0).run()
    assert not at.exception
    # 30 × 20 of band, 1.5 thinner
    assert v0 - at.session_state["mfs_shot_volume_auto"] == pytest.approx(0.9, rel=0.25)
    # the lap cannot go below the rim it laps
    assert at.number_input(key="fg_ov_lap_thk_mm").min == 1.0


def test_the_thick_tab_reports_the_cut_and_follows_its_inputs():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("thick_tab").run()
    assert "片側 51.1 mm" in _texts(at)  # 131.13 − 30 − 50
    v0 = at.session_state["mfs_shot_volume_auto"]
    at.number_input(key="fg_corner_keep_mm").set_value(81.13).run()
    assert not at.exception
    text = _texts(at)
    assert "片側 0.0 mm" in text and "切る所が無く" in text
    # 2 × 51.13 × 20 of band at t1.0 came back
    assert at.session_state["mfs_shot_volume_auto"] - v0 == pytest.approx(2.05, rel=0.25)


def test_the_centre_tabs_ramp_is_bounded_by_the_round_end():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("tab").run()
    ramp = at.number_input(key="fg_tab_ramp_len_mm")
    assert ramp.value == 18.0 and ramp.max == 18.0
    at.number_input(key="fg_runner_end_d_mm").set_value(30.0).run()
    assert not at.exception
    assert at.number_input(key="fg_tab_ramp_len_mm").max == 15.0
    assert "形状パラメータが不正" not in _texts(at)


def test_direct_gates_offer_three_layouts_and_the_inset_only_for_the_corners():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("direct").run()
    assert at.radio(key="fg_direct_layout").value == "center"
    assert "fg_direct_inset_mm" not in _keys(at)
    assert "同じ圧力を共有する" in _texts(at)
    # one gate: nothing to open later
    assert "fg_direct_open_mode" not in {r.key for r in at.radio}
    for layout, n_gates in (("corners4", 4), ("diagonal2", 2)):
        at.radio(key="fg_direct_layout").set_value(layout).run()
        assert not at.exception
        assert at.number_input(key="fg_direct_inset_mm").value == 15.0
        assert at.number_input(key="fg_direct_inset_mm").min == 1.5  # the gate's radius
        at.checkbox(key="two_phase_on").set_value(False).run()
        at.button[0].click().run()
        assert not at.exception
        geom = at.session_state["mfs_geom"]
        assert len(geom.gates) == n_gates  # one cell each on the 4 mm test mesh
        t = np.asarray(at.session_state["mfs_result"].fill_time_s)
        assert np.isfinite(t[geom.mask]).all()


def test_a_wide_corner_gate_pushes_the_inset_in_instead_of_an_error():
    at = _app()
    at.radio(key="fg_runner_shape").set_value("direct").run()
    at.radio(key="fg_direct_layout").set_value("corners4").run()
    at.number_input(key="fg_gate_d_mm").set_value(50.0).run()
    assert not at.exception
    inset = at.number_input(key="fg_direct_inset_mm")
    assert inset.min == 25.0 and inset.value == 25.0
    assert "形状パラメータが不正" not in _texts(at)


def test_going_back_to_a_fan_runner_restores_the_balancer():
    at = _app()
    at.checkbox(key="fg_balancer_on").set_value(True).run()
    at.radio(key="fg_runner_shape").set_value("tab").run()
    assert not at.exception and "形状パラメータが不正" not in _texts(at)
    at.radio(key="fg_runner_shape").set_value("pentagon").run()
    assert not at.exception
    assert "fg_balancer_on" in _keys(at)


@pytest.mark.parametrize("shape", ["thick_tab", "direct"])
def test_conventional_molding_with_holding_runs_on_the_new_shapes(shape):
    # the layered temperature from the fill into the holding solve and the flash pane
    at = _app()
    at.radio(key="fg_runner_shape").set_value(shape).run()
    at.radio(key="wall_model").set_value("multilayer")
    at.checkbox(key="two_phase_on").set_value(False)
    at.radio(key="molding_method").set_value("conventional")
    at.checkbox(key="hold_on").set_value(True).run()
    at.number_input(key="hold_t_s").set_value(10.0).run()
    at.radio(key="fill_method").set_value("march").run()
    at.radio(key="thermal_model").set_value("energy").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_result"].metadata["thermal_model"] == "energy"
    assert at.session_state["mfs_hold_info"]["thermal"] == "layers"
    flash = at.session_state["mfs_flash_info"]
    assert flash["clamp_peak_kN"] > 0.0 and flash["product_spots"]
    # without a runner there is no runner outline to rank
    assert bool(flash["runner_spots"]) == (shape != "direct")


def _open_direct(layout="diagonal2"):
    at = _app()
    at.radio(key="fg_runner_shape").set_value("direct").run()
    at.radio(key="fg_direct_layout").set_value(layout).run()
    at.checkbox(key="two_phase_on").set_value(False).run()
    return at


def test_the_later_direct_gates_open_together_unless_asked():
    at = _open_direct()
    mode = at.radio(key="fg_direct_open_mode")
    assert mode.value == "together"
    assert list(mode.options) == ["同時に開く（既定）", "時間で遅らせる", "樹脂が届いてから開く"]
    assert "fg_direct_delay_s" not in _keys(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_geom"].delayed_gates == []
    md = at.session_state["mfs_result"].metadata
    assert md["delayed_gates"] == [] and md["gate_events"] == []
    assert "遅らせて開くゲート" not in _texts(at)


def test_a_direct_gate_delayed_by_a_time_opens_then_and_is_reported():
    at = _open_direct()
    at.radio(key="fg_direct_open_mode").set_value("delay").run()
    # the default injection fills the plate in about 0.53 s
    at.number_input(key="fg_direct_delay_s").set_value(0.2).run()
    at.button[0].click().run()
    assert not at.exception
    geom = at.session_state["mfs_geom"]
    assert len(geom.gates) == 2 and len(geom.delayed_gates) == 1
    md = at.session_state["mfs_result"].metadata
    assert md["delayed_gates"] == [{"delay_s": 0.2, "on_arrival": False}]
    (ev,) = md["gate_events"]
    assert ev["t_s"] == pytest.approx(0.2) and ev["melt_already_there"] is False
    assert "2 つ目のゲートは 0.20 秒で開いた（樹脂が届く前）" in _texts(at)
    cfg = at.session_state["mfs_settings"]["geometry"]["config"]
    assert cfg["direct_open_mode"] == "delay" and cfg["direct_delay_s"] == 0.2
    # until it opens the first gate fills alone: the second gate's cell fills as it opens
    t = np.asarray(at.session_state["mfs_result"].fill_time_s)
    (gy, gx) = geom.delayed_gates[0].cells[0]
    assert t[gy, gx] == pytest.approx(0.2)
    assert np.isfinite(t[geom.mask]).all()
    # a time past the end of the fill: the gate stays shut, and the run says so
    at.number_input(key="fg_direct_delay_s").set_value(5.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_result"].metadata["gate_events"] == []
    assert "開く時刻 5.00 秒より先に充填が終わった" in _texts(at)


def test_a_direct_gate_opening_on_arrival_waits_for_the_melt():
    at = _open_direct()
    at.radio(key="fg_direct_open_mode").set_value("arrival").run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    (ev,) = md["gate_events"]
    assert ev["melt_already_there"] is True
    assert "（樹脂が届いたあと）" in _texts(at)
    # one front to the end: it reaches the far corner late in the fill
    t = np.asarray(at.session_state["mfs_result"].fill_time_s)
    assert ev["t_s"] > 0.5 * float(np.nanmax(t))


def test_delayed_direct_gates_need_the_time_marched_fill_and_conventional_molding():
    at = _open_direct()
    at.radio(key="fg_direct_open_mode").set_value("delay").run()
    at.number_input(key="fg_direct_delay_s").set_value(1.0).run()
    at.radio(key="fill_method").set_value("tau").run()
    at.button[0].click().run()
    assert not at.exception
    assert "「時間で追う」のときだけ解ける" in _texts(at)
    # injection-compression: the choice goes, and every gate opens with the injection
    at = _open_direct()
    at.radio(key="fg_direct_open_mode").set_value("delay").run()
    at.radio(key="molding_method").set_value("icm").run()
    assert "fg_direct_open_mode" not in {r.key for r in at.radio}
    assert "射出圧縮成形では、ゲートは全部同時に開く" in _texts(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_geom"].delayed_gates == []


def test_the_two_phase_short_shot_is_skipped_with_a_delayed_gate():
    """Its injection-phase times are rebuilt from the volume order, which knows
    nothing of a valve that opens later (Codex P2 on PR #28)."""
    at = _open_direct()
    at.radio(key="fg_direct_open_mode").set_value("delay").run()
    at.number_input(key="fg_direct_delay_s").set_value(0.2).run()
    at.checkbox(key="two_phase_on").set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is None
    assert "直接ゲートを遅らせて開く設定では解きません" in _texts(at)
    assert len(at.session_state["mfs_result"].metadata["gate_events"]) == 1
    # opening together, the short shot is solved as before
    at.radio(key="fg_direct_open_mode").set_value("together").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is not None
