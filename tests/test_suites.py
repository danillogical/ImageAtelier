"""Whole-suite download/conversion lifecycle, gating, publication and API constraints."""
import contextlib
import io
import json
import threading
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from imageatelier import models, suites
from imageatelier.app import create_app
from conftest import StubEngine, wait_until, new_project
from test_parts import source, make_suite, publish  # noqa: F401

@pytest.fixture(autouse=True)
def isolated_jobs(monkeypatch):
    suites.start()
    monkeypatch.setattr(suites, "_jobs", {})
    monkeypatch.setattr(models, "_downloads", {})
    yield
    suites.stop()

@pytest.fixture
def suite_client(source):
    with TestClient(create_app(engine=StubEngine(), data_dir=source["image"].parent / "data")) as client:
        yield client


def test_conversion_locked_until_every_bf16_part_downloaded(source, suite_client):
    (source["t2i"] / "0.safetensors").unlink()
    rows = suite_client.get("/api/models").json()
    assert [r["name"] for r in rows] == ["bf16", "q8", "q6", "q4"]
    assert all(not r["can_convert"] for r in rows[1:])
    assert suite_client.post("/api/models/q8/convert").status_code == 409
    assert suite_client.post("/api/models/q5/convert").status_code == 404


def test_download_includes_only_three_official_sources(source, monkeypatch):
    monkeypatch.setattr(suites, "source_parts", lambda: None)
    calls = []
    monkeypatch.setattr(models, "_start", lambda key, plan, verify: calls.append((key, plan)) or True)
    assert suites.start_download()
    key, plan = calls[0]
    assert key == "suite:bf16"
    assert {row[0] for row in plan} == {v[0] for v in suites.SOURCES.values()}
    assert all("Heretic" not in row[0] and "turbo" not in row[0].lower() for row in plan)
    assert all(len(row[1]) == 40 for row in plan)


def test_complete_bf16_download_is_idempotent(source, monkeypatch):
    monkeypatch.setattr(models, "_start", lambda *a: pytest.fail("must reuse downloaded BF16 suite"))
    assert not suites.start_download()


@pytest.mark.parametrize("ram_gib,expected", [
    (None, None), (0, None), (16, None), (24, None),
    (31, None), (32, "q8"), (47, "q8"), (48, "bf16"), (64, "bf16"),
])
def test_ram_recommendation_reserves_six_gb_headroom(monkeypatch, ram_gib, expected):
    monkeypatch.setattr(suites, "physical_memory", lambda: None if ram_gib is None else ram_gib * 1024**3)
    recommended = [name for name in suites.VARIANTS if suites._base(name)["recommended"]]
    assert recommended == ([] if expected is None else [expected])


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert command[:3] == [__import__("sys").executable, "-m", "imageatelier.convert_suite"]
        self.stage = Path(command[command.index("--out") + 1])
        make_suite(self.stage)
        self.stdout = io.StringIO("SUITE_PHASE Converting I2I prompt enhancer\n")
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def wait(self): return 0
    def poll(self): return 0


@pytest.mark.parametrize("name,bits", [("q8", "8"), ("q6", "6"), ("q4", "4")])
def test_conversion_publishes_complete_suite_atomically(source, monkeypatch, name, bits):
    calls = []
    def process(command, **kwargs):
        calls.append(command)
        assert suites.parts(name) is None
        return FakeProcess(command, **kwargs)
    entered = []
    @contextlib.contextmanager
    def maintenance():
        entered.append("entered")
        yield
        entered.append("exited")
    monkeypatch.setattr(suites.subprocess, "Popen", process)
    assert suites.start_conversion(name, maintenance)
    wait_until(lambda: suites.status(name)["state"] != "converting")
    assert suites.status(name)["state"] == "ready", suites.status(name)
    assert calls[0][calls[0].index("--bits") + 1] == bits
    assert entered == ["entered", "exited"]
    assert suites.parts(name)["image"] == suites.suite_dir() / name / "image"
    assert not suites.start_conversion(name)
    assert not list(suites.suite_dir().glob("*.tmp"))


def test_failure_does_not_publish_and_can_retry(source, monkeypatch):
    class Failure(FakeProcess):
        def wait(self): return 1
    monkeypatch.setattr(suites.subprocess, "Popen", Failure)
    assert suites.start_conversion("q8")
    wait_until(lambda: suites.status("q8")["state"] != "converting")
    assert suites.parts("q8") is None
    assert "failed" in suites.status("q8")["error"].lower()
    monkeypatch.setattr(suites.subprocess, "Popen", FakeProcess)
    assert suites.start_conversion("q8")
    wait_until(lambda: suites.status("q8")["state"] == "ready")
    assert not suites.status("q8")["error"]


def test_only_one_conversion_runs_at_a_time(source, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    @contextlib.contextmanager
    def maintenance():
        entered.set()
        assert release.wait(5)
        yield
    monkeypatch.setattr(suites.subprocess, "Popen", FakeProcess)
    try:
        assert suites.start_conversion("q8", maintenance)
        assert entered.wait(5)
        with pytest.raises(suites.SuiteError, match="already running"):
            suites.start_conversion("q4")
        assert suites.parts("q8") is None
    finally:
        release.set()
    wait_until(lambda: suites.status("q8")["state"] == "ready")


def test_api_uses_one_suite_for_generation_and_enhancement(source, suite_client):
    c = suite_client
    assert c.put("/api/settings", json={"model_suite": "q6"}).status_code == 409
    assert c.put("/api/settings", json={"model_suite": "bf16"}).status_code == 200
    pid = new_project(c)["id"]
    assert c.post(f"/api/projects/{pid}/runs", json={"prompt": "cat", "turbo": True}).status_code == 400
    assert c.post(f"/api/projects/{pid}/runs", json={"prompt": "cat", "text_encoder": "heretic-8bit"}).status_code == 400
    assert c.post(f"/api/projects/{pid}/enhance", json={"prompt": "cat", "quant": "8bit"}).status_code == 400
    r = c.post(f"/api/projects/{pid}/runs", json={"prompt": "cat"})
    assert r.status_code == 202, r.text
    job = c.app.state.engine.submitted[-1]
    assert job["image_model"] == job["text_encoder"] == "official-bf16"
    assert job["turbo"] is False and job["steps"] == 40
