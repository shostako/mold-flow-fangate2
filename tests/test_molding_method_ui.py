"""AppTest wiring for the sidebar's first choice: 通常成形 / 射出圧縮成形."""

from __future__ import annotations

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts
from tests.ui_helpers import two_phase as _two_phase


def _sliders(at) -> set[str]:
    return {str(s.label) for s in at.sidebar.slider}


def _hold_keys(at) -> set[str]:
    keys = {n.key for n in at.number_input} | {c.key for c in at.checkbox}
    return {k for k in keys if k and (k.startswith("hold_") or k in HOLD_OTHER)}


HOLD_OTHER = {"machine_clamp_kN", "flash_threshold_mpa"}
HOLD_KEYS = {"hold_on", "hold_p_mpa", "hold_t_s", "hold_vp_pct", "hold_coarse"} | HOLD_OTHER


def test_conventional_molding_is_the_default_with_the_holding_inputs_and_no_compression():
    at = _app()
    method = at.radio(key="molding_method")
    assert method.value == "conventional"
    assert list(method.options) == ["通常成形", "射出圧縮成形"]
    # the method is chosen before the geometry
    first = at.sidebar.radio[0]
    assert first.key == "molding_method"
    assert not {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} & _sliders(at)
    assert _hold_keys(at) == HOLD_KEYS
    # the two-phase short shot stands for injection-compression: not offered here
    assert "two_phase_on" not in {c.key for c in at.checkbox}
    assert "ショートショット（計量制限）" not in {str(e.label) for e in at.sidebar.expander}
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_two_phase_result"] is None
    assert at.session_state["mfs_two_phase_skip"] is None
    assert at.session_state["mfs_settings"]["two_phase_short_shot"]["enabled"] is False
    assert "二相ショートショット" not in _texts(at)


def test_injection_compression_shows_its_conditions_and_hides_the_holding_inputs():
    at = _app()
    at.radio(key="molding_method").set_value("icm").run()
    assert not at.exception
    assert {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} <= _sliders(at)
    assert _hold_keys(at) == set()
    assert at.checkbox(key="two_phase_on").value is True  # offered, and on
    assert "ショートショット（計量制限）" in {str(e.label) for e in at.sidebar.expander}
    _two_phase(at, False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]
    assert rec["compression_molding"]["enabled"] is True


def test_conventional_molding_hides_the_compression_conditions_and_solves_without_them():
    at = _app()
    at.radio(key="molding_method").set_value("conventional").run()
    assert not at.exception
    assert not {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} & _sliders(at)
    _two_phase(at, False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]
    assert rec["compression_molding"]["enabled"] is False
    assert at.session_state["mfs_result"].metadata.get("compression_molding") in (False, None)
    # back again: the conditions return at their defaults
    at.radio(key="molding_method").set_value("icm").run()
    assert {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} <= _sliders(at)
    assert "形状パラメータが不正" not in _texts(at)


def test_holding_switched_on_does_not_carry_into_injection_compression():
    """ON under conventional molding, then ICM: the inputs go, the run solves no
    holding and says why, and the record carries none."""
    at = _app()
    _two_phase(at, False)
    at.checkbox(key="hold_on").set_value(True).run()
    at.radio(key="molding_method").set_value("icm").run()
    assert _hold_keys(at) == set()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_hold_info"] is None
    assert "射出圧縮成形の解析では解きません" in at.session_state["mfs_hold_skip"]
    assert at.session_state["mfs_settings"]["holding"]["enabled"] is False
