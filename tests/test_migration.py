"""Stored settings.json migrations: removed 4-bit values and the monolithic `variant` (API.md)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from webstudio.app import create_app

from conftest import StubEngine

PAIR = {"image_model": "official-8bit", "text_encoder": "heretic-8bit"}


def start(tmp_path, stored):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "settings.json"
    path.write_text(json.dumps(stored))
    return path, create_app(engine=StubEngine(), data_dir=data)


@pytest.fixture(autouse=True)
def _ready(ready, pe_ready):
    pass


OTHERS = {"turbo": False, "steps": 20, "resolution": 512, "low_memory": False, "vae_tiling": True,
          "pe_thinking": False, "pe_max_new_tokens": 1024, "pe_language": "auto", "pe_variant": "heretic"}


@pytest.mark.parametrize("variant", ["4bit", "8bit", "something-else"])
def test_variant_migrates_to_default_pair_and_file_is_rewritten(tmp_path, variant):
    path, app = start(tmp_path, {"variant": variant, "pe_quant": "8bit", **OTHERS})
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert "variant" not in s
    assert {k: s[k] for k in PAIR} == PAIR
    assert {k: s[k] for k in OTHERS} == OTHERS  # everything else preserved
    on_disk = json.loads(path.read_text())
    assert "variant" not in on_disk and {k: on_disk[k] for k in PAIR} == PAIR
    assert {k: on_disk[k] for k in OTHERS} == OTHERS


def test_pe_quant_4bit_still_coerced_and_variant_migrated_together(tmp_path):
    path, app = start(tmp_path, {"variant": "4bit", "pe_quant": "4bit", **OTHERS})
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert s["pe_quant"] == "8bit" and "variant" not in s and s["image_model"] == "official-8bit"
    assert {k: s[k] for k in OTHERS} == OTHERS
    on_disk = json.loads(path.read_text())
    assert on_disk["pe_quant"] == "8bit" and "variant" not in on_disk


def test_pe_quant_alone_is_coerced(tmp_path):
    path, app = start(tmp_path, {"pe_quant": "4bit", "steps": 8})
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert s["pe_quant"] == "8bit" and s["steps"] == 8 and s["turbo"] is True
    assert json.loads(path.read_text())["pe_quant"] == "8bit"


def test_migration_logs_one_line_and_runs_once(tmp_path, caplog):
    path, app = start(tmp_path, {"variant": "4bit"})
    with caplog.at_level(logging.INFO):
        with TestClient(app):
            pass
    lines = [r for r in caplog.records if "variant" in r.getMessage() and "migrat" in r.getMessage().lower()]
    assert len(lines) == 1
    # a second start on the rewritten file has nothing left to migrate
    caplog.clear()
    with caplog.at_level(logging.INFO):
        with TestClient(create_app(engine=StubEngine(), data_dir=path.parent)):
            pass
    assert not [r for r in caplog.records if "migrat" in r.getMessage().lower()]


def test_migrated_settings_are_usable(tmp_path):
    _, app = start(tmp_path, {"variant": "4bit"})
    with TestClient(app) as c:
        assert c.put("/api/settings", json={"resolution": 768}).status_code == 200
        s = c.get("/api/settings").json()
        assert s["image_model"] == "official-8bit" and s["text_encoder"] == "heretic-8bit" and s["resolution"] == 768
        pid = c.post("/api/projects", json={}).json()["id"]
        assert c.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 202


def test_valid_file_is_left_alone(tmp_path):
    stored = {**PAIR, "pe_quant": "8bit", "turbo": True, "steps": 8, "text_encoder": "official-8bit"}
    path, app = start(tmp_path, stored)
    before = path.read_text()
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert s["steps"] == 8 and s["text_encoder"] == "official-8bit"
    assert path.read_text() == before


def test_fresh_install_defaults(tmp_path):
    with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "fresh")) as c:
        s = c.get("/api/settings").json()
    assert s["image_model"] == "official-8bit" and s["text_encoder"] == "heretic-8bit"
    assert s["pe_quant"] == "8bit" and "variant" not in s


# ---- pe_t2i_variant -> pe_variant -------------------------------------------------------------
@pytest.mark.parametrize("value", ["heretic", "official"])
def test_pe_t2i_variant_becomes_pe_variant(tmp_path, value):
    path, app = start(tmp_path, {"pe_t2i_variant": value, "steps": 8})
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert s["pe_variant"] == value and "pe_t2i_variant" not in s and s["steps"] == 8
    on_disk = json.loads(path.read_text())
    assert on_disk["pe_variant"] == value and "pe_t2i_variant" not in on_disk  # rewritten, old key dropped
    assert on_disk["steps"] == 8


def test_pe_t2i_variant_migration_logs_one_line_and_runs_once(tmp_path, caplog):
    path, app = start(tmp_path, {"pe_t2i_variant": "heretic"})
    with caplog.at_level(logging.INFO):
        with TestClient(app):
            pass
    lines = [r for r in caplog.records if "pe_t2i_variant" in r.getMessage() and "migrat" in r.getMessage().lower()]
    assert len(lines) == 1
    caplog.clear()
    with caplog.at_level(logging.INFO):
        with TestClient(create_app(engine=StubEngine(), data_dir=path.parent)) as c:
            assert c.get("/api/settings").json()["pe_variant"] == "heretic"
    assert not [r for r in caplog.records if "migrat" in r.getMessage().lower()]


def test_pe_t2i_variant_migrates_alongside_variant_and_pe_quant(tmp_path):
    stored = {"variant": "4bit", "pe_quant": "4bit", "pe_t2i_variant": "heretic", **{k: v for k, v in OTHERS.items()
                                                                                       if k != "pe_variant"}}
    path, app = start(tmp_path, stored)
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert {k: s[k] for k in PAIR} == PAIR and s["pe_quant"] == "8bit" and s["pe_variant"] == "heretic"
    assert "variant" not in s and "pe_t2i_variant" not in s
    assert {k: s[k] for k in OTHERS if k != "pe_variant"} == {k: OTHERS[k] for k in OTHERS if k != "pe_variant"}
    on_disk = json.loads(path.read_text())
    assert "variant" not in on_disk and "pe_t2i_variant" not in on_disk
    assert on_disk["pe_variant"] == "heretic" and on_disk["pe_quant"] == "8bit" and {k: on_disk[k] for k in PAIR} == PAIR


def test_explicit_pe_variant_wins_over_stale_old_key(tmp_path):
    path, app = start(tmp_path, {"pe_variant": "official", "pe_t2i_variant": "heretic"})
    with TestClient(app) as c:
        s = c.get("/api/settings").json()
    assert s["pe_variant"] == "official" and "pe_t2i_variant" not in s
    assert "pe_t2i_variant" not in json.loads(path.read_text())


def test_fresh_install_has_no_old_key(tmp_path):
    with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "fresh")) as c:
        s = c.get("/api/settings").json()
    assert s["pe_variant"] == "official" and "pe_t2i_variant" not in s
