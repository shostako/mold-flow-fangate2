"""Tests for the run-settings record bundled with the results ZIP.

The point of this record is that a downloaded ZIP can be reproduced. Before
it existed, ``metadata.json`` held only what the solver computed, so the
geometry had to be recovered by measuring the rendered images and searching
for a config whose volume and tau_max matched. These tests pin the two
properties that make the record worth having: it must carry *every* config
field, and it must not carry the contents of an uploaded spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from core.fan_runner import FanRunnerPlateConfig
from core.settings_record import config_settings, file_fingerprint, settings_json

SOURCE = "Fan runner plate (parametric)"


def test_every_config_field_is_recorded():
    """A field that is not recorded is a field the reader has to guess.

    Compared against the dataclass itself rather than a hand-written list, so
    a new geometry parameter cannot be added without appearing here.
    """
    cfg = FanRunnerPlateConfig(plate_w_mm=300.0, plate_h_mm=50.0, frame_w_mm=10.0)
    rec = config_settings(SOURCE, cfg)
    assert set(rec["config"]) == set(vars(cfg))
    assert rec["config"]["plate_w_mm"] == 300.0
    assert rec["input"] == SOURCE


def test_none_fields_survive_as_null():
    """``None`` is a setting ("no taper", say), not a missing value. This
    builder has no optional field, so the case is pinned on a local config."""

    @dataclass(frozen=True)
    class _TaperConfig:
        thk_mm: float = 2.0
        thk_well_mm: float | None = None

    rec = config_settings(SOURCE, _TaperConfig(thk_well_mm=None))
    assert "thk_well_mm" in rec["config"]
    assert rec["config"]["thk_well_mm"] is None


def test_tuple_fields_become_lists_so_json_round_trips():
    """A multi-stage taper would be a tuple field; JSON has no tuple."""

    @dataclass(frozen=True)
    class _StagedConfig:
        widths_mm: tuple[float, ...] = (20.0, 40.0)
        thicknesses_mm: tuple[float, ...] = (0.2, 0.3)

    rec = config_settings("staged", _StagedConfig())
    assert rec["config"]["widths_mm"] == [20.0, 40.0]
    assert json.loads(settings_json(rec)) == rec


def test_uploaded_spec_is_fingerprinted_not_embedded():
    """The ZIP is made to be forwarded; drawing dimensions must not ride along.

    Checked by looking for the actual numbers in the serialized record, not by
    inspecting which keys were set -- a future change that starts embedding
    the spec would keep the keys and still leak.
    """
    spec_text = json.dumps(
        {"name": "customer_partno_rev3", "gate_exit_width": 299.0, "land": {"depth": 0.35}}
    )
    rec = config_settings(
        "Fan runner plate (JSONスペック)",
        FanRunnerPlateConfig(),
        spec=file_fingerprint("real_part.json", spec_text),
    )
    blob = settings_json(rec)
    assert "299.0" not in blob
    assert "gate_exit_width" not in blob
    # The spec's own ``name`` is content too, and it is the field most likely
    # to carry a part or customer identifier.
    assert "customer_partno_rev3" not in blob
    assert rec["spec"]["sha256"] == file_fingerprint("x", spec_text)["sha256"]
    assert rec["spec"]["name"] == "real_part.json"


def test_fingerprint_identifies_the_revision():
    """Same bytes -> same hash, one edited number -> different hash."""
    a = file_fingerprint("s.json", '{"depth": 0.35}')
    b = file_fingerprint("s.json", '{"depth": 0.35}')
    c = file_fingerprint("s.json", '{"depth": 0.36}')
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != c["sha256"]
    assert a["bytes"] == len('{"depth": 0.35}')


def test_fingerprint_accepts_bytes_and_text_alike():
    """Uploads arrive as bytes, local files as text; both must hash the same."""
    text = '{"depth": 0.35}'
    assert file_fingerprint("s", text)["sha256"] == file_fingerprint("s", text.encode())["sha256"]


def test_extra_fields_land_next_to_the_config():
    """Anything that is not a config field (a solver-side setting, say) rides
    next to the config rather than inside it."""
    rec = config_settings(SOURCE, FanRunnerPlateConfig(), compression_stroke_mm=0.8)
    assert rec["compression_stroke_mm"] == 0.8


def test_rejects_a_missing_config():
    """Every input must describe itself with a config dataclass.

    ``cfg`` was optional while the image importer described itself with loose
    keyword arguments instead. That input is gone, so a record without a
    ``config`` key is now a shape nothing produces -- and a settings file that
    silently omitted the geometry would be worse than one that was never
    written.
    """
    with pytest.raises(TypeError):
        config_settings("何かの入力", None, threshold=128)


def test_rejects_a_dataclass_type_passed_by_mistake():
    with pytest.raises(TypeError):
        config_settings("x", FanRunnerPlateConfig)


def test_rejects_an_empty_source():
    """Checked with a valid config, so the failure can only be the source."""
    with pytest.raises(ValueError):
        config_settings("", FanRunnerPlateConfig())


def test_only_the_fingerprint_keys_describe_an_uploaded_spec():
    """Whatever ends up under ``spec`` must be derivable without reading the file.

    ``name`` is the name the *user* chose for the upload, not a field lifted
    out of the JSON. Anything else here would be content by another route --
    which is how the spec's internal ``name`` slipped in the first time.
    """
    rec = config_settings(
        "Fan runner plate (JSONスペック)",
        FanRunnerPlateConfig(),
        spec=file_fingerprint("part.json", '{"name": "secret", "depth": 0.35}'),
    )
    assert set(rec["spec"]) == {"name", "sha256", "bytes"}
    assert rec["spec"]["name"] == "part.json"
