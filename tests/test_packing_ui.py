"""AppTest wiring for the holding-pressure reach pane in ``app.py``."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from tests.ui_helpers import app as _app
from tests.ui_helpers import texts

ICM_NOTE = "射出圧縮成形 ON の解析では出しません"
FRACTION_NOTE = "固化判定 fraction が 0 のときは出しません"


def _run(at, *, icm: bool):
    at.radio(key="wall_model").set_value("none")
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(icm).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_conventional_molding_shows_the_map_and_the_table():
    at = _run(_app(), icm=False)
    path = at.session_state["mfs_pack_path"]
    rows = at.session_state["mfs_pack_rows"]
    assert path is not None and Path(path).exists()
    # the product: the 4 mm inner plate and the 1 mm frame, thickest first
    assert [r["thickness_mm"] for r in rows] == [4.0, 1.0]
    thick, thin = rows
    assert thick["frozen_at_fill_end"] == 0.0
    # the gate reaches the inner plate only through thinner sections, so
    # holding pressure is cut off long before the plate itself freezes
    assert 0.0 < thick["reach_s"] < thick["solidify_s"]
    assert 0.0 < thick["ratio"] < 0.5
    assert ICM_NOTE not in texts(at)
    shown = [df for df in at.dataframe if "肉厚 [mm]" in df.value.columns]
    assert len(shown) == 1
    assert list(shown[0].value["肉厚 [mm]"]) == [4.0, 1.0]
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert names.count("packing_reach.png") == 1
        record = json.loads(zf.read("packing_reach.json"))
        assert record["solid_fraction"] == 0.3
        assert [r["thickness_mm"] for r in record["by_thickness"]] == [4.0, 1.0]
        assert zf.read("packing_reach.png") == Path(path).read_bytes()


def test_compression_molding_explains_why_there_is_no_map():
    at = _run(_app(), icm=True)
    assert at.session_state["mfs_pack_path"] is None
    assert at.session_state["mfs_pack_rows"] is None
    assert ICM_NOTE in texts(at)
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert "packing_reach.png" not in names and "packing_reach.json" not in names


def test_zero_solid_fraction_skips_the_estimate_and_keeps_the_fill_result():
    """The multilayer model's slider goes down to 0, where the slab's centre
    never reaches the solidification temperature: the estimate is skipped with
    its reason, and the finished fill is still shown (Codex P2 on PR #18)."""
    at = _app()
    assert at.radio(key="wall_model").value == "multilayer"
    at.checkbox(key="two_phase_on").set_value(False)
    at.checkbox(key="icm_on").set_value(False).run()
    sliders = [s for s in at.slider if str(s.label).startswith("固化判定 fraction")]
    assert len(sliders) == 1
    sliders[0].set_value(0.0).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["mfs_result"] is not None
    assert at.session_state["mfs_pack_path"] is None
    assert at.session_state["mfs_pack_rows"] is None
    assert FRACTION_NOTE in texts(at)
    assert ICM_NOTE not in texts(at)
    with zipfile.ZipFile(io.BytesIO(at.session_state["mfs_zip_bytes"])) as zf:
        names = zf.namelist()
        assert "fill.gif" in names and "packing_reach.png" not in names
