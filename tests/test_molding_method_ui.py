"""AppTest wiring for the sidebar's first choice: 通常成形 / 射出圧縮成形."""

from __future__ import annotations

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts


def _sliders(at) -> set[str]:
    return {str(s.label) for s in at.sidebar.slider}


def test_injection_compression_is_the_default_and_shows_its_conditions():
    at = _app()
    method = at.radio(key="molding_method")
    assert method.value == "icm"
    assert list(method.options) == ["射出圧縮成形", "通常成形"]
    # the method is chosen before the geometry
    first = at.sidebar.radio[0]
    assert first.key == "molding_method"
    assert {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} <= _sliders(at)
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]
    assert rec["compression_molding"]["enabled"] is True


def test_conventional_molding_hides_the_compression_conditions_and_solves_without_them():
    at = _app()
    at.radio(key="molding_method").set_value("conventional").run()
    assert not at.exception
    assert not {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} & _sliders(at)
    at.checkbox(key="two_phase_on").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    rec = at.session_state["mfs_settings"]
    assert rec["compression_molding"]["enabled"] is False
    assert at.session_state["mfs_result"].metadata.get("compression_molding") in (False, None)
    # back again: the conditions return at their defaults
    at.radio(key="molding_method").set_value("icm").run()
    assert {"圧縮ストローク [mm]", "圧縮位相の充填占有率"} <= _sliders(at)
    assert "形状パラメータが不正" not in _texts(at)
