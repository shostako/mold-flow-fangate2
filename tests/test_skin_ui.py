"""AppTest wiring for the skin-layer iteration cap and the convergence report
(sim v0.39.0 / PR #81, ported): the sidebar cap defaults to the solver's 20,
and the result pane says how many iterations ran, whether they converged, and
warns when the fixed-point loop was cut off (a cut-off field looks converged
but under-reports seal-off)."""

from __future__ import annotations

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts as _texts


def _iter_slider(at):
    (s,) = [s for s in at.slider if str(s.label) == "fixed-point 反復上限"]
    return s


def test_the_iteration_cap_defaults_to_the_solver_default():
    at = _app()
    assert at.radio(key="wall_model").value == "skin"
    s = _iter_slider(at)
    assert s.value == 20
    assert s.max == 40


def test_the_result_pane_reports_iterations_and_warns_on_cut_off():
    at = _app()
    at.checkbox(key="two_phase_on").set_value(False)
    _iter_slider(at).set_value(1).run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    text = _texts(at)
    assert f"反復={md['skin_iterations']}, 収束={md['skin_converged']}" in text
    # one iteration of a coupled tau <-> skin loop cannot have met the tolerance
    assert md["skin_converged"] is False and not md.get("no_flow")
    assert "反復が上限で打ち切られた" in text
    # a converged run carries no warning
    _iter_slider(at).set_value(40).run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    if md["skin_converged"]:
        assert "反復が上限で打ち切られた" not in _texts(at)
