"""AppTest wiring for the skin-layer iteration cap and the convergence report
(sim v0.39.0 / PR #81, ported): the sidebar cap defaults to the solver's 20,
and the result pane says how many iterations ran, whether they converged, and
warns when the fixed-point loop was cut off (a cut-off field looks converged
but under-reports seal-off)."""

from __future__ import annotations

from tests.ui_helpers import app as _page
from tests.ui_helpers import texts as _texts


def _app():
    """The page with the skin wall model selected (the page opens on the
    layered model since v0.8.0)."""
    at = _page()
    at.radio(key="wall_model").set_value("skin").run()
    return at


def _iter_slider(at):
    (s,) = [s for s in at.slider if str(s.label) == "fixed-point 反復上限"]
    return s


def test_the_iteration_cap_defaults_to_the_solver_default():
    at = _app()
    assert at.radio(key="wall_model").value == "skin"
    s = _iter_slider(at)
    assert s.value == 20
    assert s.max == 100  # the default plate on the constant-pressure clock needs ~80


def test_the_result_pane_reports_iterations_and_warns_on_cut_off():
    at = _app()
    at.checkbox(key="two_phase_on").set_value(False)
    # pin the growth constant: the convergence test is "first iteration whose
    # relative tau change is below tol", so with c_skin = 0 (no skin, tau
    # unchanged) even one iteration would count as converged. At c_skin = 1 on
    # the default plate the first iteration moves tau far more than 1e-3.
    (c_skin,) = [s for s in at.slider if str(s.label).startswith("スキン層成長定数")]
    c_skin.set_value(1.0)
    _iter_slider(at).set_value(1).run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    text = _texts(at)
    assert f"反復={md['skin_iterations']}, 収束={md['skin_converged']}" in text
    assert md["skin_converged"] is False and not md.get("no_flow")
    assert "スキン層の fixed-point 反復が上限で打ち切られた" in text
    # a converged run carries no warning: on the constant-rate clock (the UI
    # default) the loop settles in a few iterations, so 40 must converge
    assert at.radio(key="skin_clock").value == "constant_rate"
    _iter_slider(at).set_value(40).run()
    at.button[0].click().run()
    assert not at.exception
    md = at.session_state["mfs_result"].metadata
    assert md["skin_converged"] is True
    assert "反復が上限で打ち切られた" not in _texts(at)


def test_the_two_phase_pane_reports_its_own_injection_phase_loop():
    """The two-phase injection phase runs its own skin loop under the same
    cap; a cut-off there must be reported in the (expanded) two-phase pane,
    not only for the main solve."""
    at = _app()
    assert at.checkbox(key="two_phase_on").value is True
    (c_skin,) = [s for s in at.slider if str(s.label).startswith("スキン層成長定数")]
    c_skin.set_value(1.0)
    _iter_slider(at).set_value(1).run()
    at.button[0].click().run()
    assert not at.exception
    md2 = at.session_state["mfs_two_phase_result"].metadata
    text = _texts(at)
    assert f"反復={md2['skin_iterations']}, 収束={md2['skin_converged']}" in text
    assert md2["skin_converged"] is False
    assert "二相の射出相のスキン層の fixed-point 反復が上限で打ち切られた" in text
