"""Whole-suite deletion frees only the chosen weights and refuses active work."""
import contextlib
import hashlib
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from imageatelier import models, suites
from imageatelier.app import create_app
from imageatelier.engine import Engine, EngineBusy
from conftest import StubEngine, wait_until
from test_parts import source, make_suite, publish  # noqa: F401

@pytest.mark.parametrize("kind,name", [("image", "official-8bit"), ("text_encoder", "official-8bit"), ("pe", "t2i-8bit")])
def test_independent_part_operations_retired(client, kind, name):
    url = f"/api/models/{kind}/{name}"
    assert client.post(url + "/download").status_code == 410
    assert client.delete(url).status_code == 410


@pytest.fixture(autouse=True)
def isolate_suites(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGEATELIER_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(suites, "_jobs", {})
    monkeypatch.setattr(models, "_downloads", {})
    suites.start()
    yield
    suites.stop()


@pytest.mark.parametrize("name", ["q8", "q6", "q4"])
def test_delete_whole_conversion_preserves_other_suites(source, name):
    for quant in ("q8", "q6", "q4"):
        publish(suites.suite_dir() / quant, quant)
    root = suites.suite_dir() / name
    expected = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    entered = []
    @contextlib.contextmanager
    def maintenance():
        assert root.exists()
        entered.append(True)
        yield
    assert suites.delete(name, maintenance) == expected
    assert entered == [True]
    assert not root.exists()
    assert suites.parts("bf16") == source
    assert all(suites.parts(q) for q in ("q8", "q6", "q4") if q != name)
    assert suites.delete(name) == 0


def cache_snapshot(repo, revision, source):
    """Build a real temporary Hub cache with shared blobs and snapshot symlinks."""
    root = models._storage_folder(repo)
    snapshot = root / "snapshots" / revision
    shutil.copytree(source, snapshot)
    blobs = root / "blobs"
    blobs.mkdir(exist_ok=True)
    for path in snapshot.rglob("*"):
        if path.is_file():
            payload = path.read_bytes()
            blob = blobs / hashlib.sha256(payload).hexdigest()
            blob.write_bytes(payload)
            path.unlink()
            path.symlink_to(Path("../" * (len(path.relative_to(snapshot).parts) + 1)) / "blobs" / blob.name)
    return snapshot


def test_delete_bf16_preserves_conversions_and_other_cached_revisions(tmp_path):
    original = make_suite(tmp_path / "original")
    snapshots = [cache_snapshot(repo, rev, original[kind]) for kind, (repo, rev) in suites.SOURCES.items()]
    assert suites.parts("bf16") is not None
    repo, _ = suites.SOURCES["image"]
    other = cache_snapshot(repo, "a" * 40, original["image"])
    unrelated = cache_snapshot("Someone/Other", "b" * 40, original["t2i"])
    publish(suites.suite_dir() / "q8", "q8")
    models._intent("suite:bf16", "complete")
    assert suites.delete("bf16") > 0
    assert all(not p.exists() for p in snapshots)
    assert (other / "transformer" / "0.safetensors").read_bytes() == b"weights"
    assert (unrelated / "0.safetensors").read_bytes() == b"weights"
    assert suites.parts("bf16") is None
    assert suites.parts("q8") is not None
    assert suites.status("q6")["can_convert"] is False
    assert models._intent("suite:bf16") == "deleted"


def test_busy_engine_refuses_before_any_deletion(source):
    root = suites.suite_dir() / "q8"
    publish(root, "q8")
    engine = Engine()
    # A queued job is enough to block maintenance even before starting the worker.
    engine.submit({"kind": "enhance", "prompt": "cat", "mode": "t2i", "quant": "8bit"})
    with pytest.raises(EngineBusy):
        suites.delete("q8", engine.maintenance)
    assert suites.parts("q8") is not None


def test_active_download_refuses_deletion(source):
    publish(suites.suite_dir() / "q8", "q8")
    models._intent("suite:bf16", "running")
    with pytest.raises(suites.SuiteError, match="download or conversion"):
        suites.delete("q8")
    assert suites.parts("q8") is not None


def test_conversion_refuses_deletion_until_finished(source, monkeypatch):
    from test_suites import FakeProcess
    publish(suites.suite_dir() / "q8", "q8")
    entered, release = threading.Event(), threading.Event()
    @contextlib.contextmanager
    def maintenance():
        entered.set()
        assert release.wait(5)
        yield
    monkeypatch.setattr(suites.subprocess, "Popen", FakeProcess)
    try:
        assert suites.start_conversion("q6", maintenance)
        assert entered.wait(5)
        with pytest.raises(suites.SuiteError, match="download or conversion"):
            suites.delete("bf16")
        with pytest.raises(suites.SuiteError):
            suites.delete("q8")
    finally:
        release.set()
    wait_until(lambda: suites.status("q6")["state"] == "ready")


@pytest.mark.parametrize("name,lock", [("q8", "conversion"), ("bf16", "download")])
def test_other_process_lock_refuses_deletion(source, name, lock):
    import fcntl
    publish(suites.suite_dir() / "q8", "q8")
    path = (suites.suite_dir() / ".conversion.lock" if lock == "conversion"
            else models._download_root("suite:bf16") / "worker.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(suites.SuiteError, match="Another app"):
            suites.delete(name)
    assert suites.parts("q8") is not None
    assert suites.parts("bf16") is not None


def test_symlink_directory_refused_without_touching_target(source, tmp_path):
    external = tmp_path / "external"
    publish(external, "q8")
    suites.suite_dir().mkdir(parents=True)
    (suites.suite_dir() / "q8").symlink_to(external, target_is_directory=True)
    with pytest.raises(suites.SuiteError, match="symbolic link"):
        suites.delete("q8")
    assert (external / "suite.json").is_file()


def test_api_delete_updates_readiness_without_changing_selection(source, tmp_path):
    publish(suites.suite_dir() / "q8", "q8")
    with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "data")) as c:
        assert c.put("/api/settings", json={"model_suite": "q8"}).status_code == 200
        assert c.delete("/api/models/q5").status_code == 404
        result = c.delete("/api/models/q8")
        assert result.status_code == 200, result.text
        assert result.json()["deleted"] and result.json()["freed_bytes"] > 0
        assert c.get("/api/settings").json()["model_suite"] == "q8"
        rows = c.get("/api/models").json()
        assert next(r for r in rows if r["name"] == "q8")["state"] == "not_converted"
        models._intent("suite:bf16", "running")
        assert c.delete("/api/models/bf16").status_code == 409
        models._intent("suite:bf16", "complete")
