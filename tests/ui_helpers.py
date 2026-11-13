"""Shared AppTest helpers for the ``app.py`` wiring tests."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def app(timeout: float = 240.0, *, fast: bool = True) -> AppTest:
    """Load the app; ``fast`` coarsens the mesh (4 mm) and trims the frame
    count so a full run takes seconds instead of a minute."""
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    if fast:
        at.slider(key="fg_cell_size_mm").set_value(4.0)
        frames = [s for s in at.slider if str(s.label).startswith("アニメーションフレーム数")]
        frames[0].set_value(12)
        at.run()
    return at


def texts(at: AppTest) -> str:
    parts = []
    for group in (at.caption, at.info, at.warning, at.error, at.markdown, at.exception):
        parts.extend(str(getattr(el, "value", "")) for el in group)
    return "\n".join(parts)


def two_phase(at: AppTest, on: bool) -> AppTest:
    """Switch the two-phase short shot on or off; returns ``at`` (for ``.run()``).

    The short shot stands for injection-compression molding (injection into the
    open gap, then the closing), so the sidebar offers it with that method only.
    Under conventional molding it is not run and there is nothing to switch off;
    asking for it there is a mistake in the test."""
    boxes = [c for c in at.checkbox if c.key == "two_phase_on"]
    if boxes:
        boxes[0].set_value(on)
    elif on:
        raise AssertionError("the two-phase short shot is offered with injection-compression only")
    return at
