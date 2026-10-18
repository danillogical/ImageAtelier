"""Real HTTP range transfers and subprocess lifecycle tests; never fetch large models."""
import hashlib
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import snapshot_download

from imageatelier import download_worker as worker, models, suites
from conftest import wait_until


@pytest.fixture
def server():
    state = SimpleNamespace(data=b"qwen weights\n" * 300_000, ranges=[], mode="range")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            value = self.headers.get("Range")
            state.ranges.append(value)
            start = int(value.removeprefix("bytes=").removesuffix("-")) if value else 0
            if state.mode == "ignore":
                start = 0
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(state.data) - start))
            if start:
                self.send_header("Content-Range", f"bytes {start + (1 if state.mode == 'bad' else 0)}-{len(state.data)-1}/{len(state.data)}")
            self.end_headers()
            if state.mode == "drop":
                self.wfile.write(state.data[start:start + 1_500_000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.wfile.write(state.data[start:])
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{http.server_port}/model"
    yield state
    http.shutdown()
    http.server_close()
    thread.join()


def test_disconnect_preserves_bytes_and_next_request_resumes(server, tmp_path):
    partial = tmp_path / "model.incomplete"
    with httpx.Client() as client:
        server.mode = "drop"
        with pytest.raises(httpx.RemoteProtocolError):
            worker.transfer(client, server.url, partial, len(server.data))
        saved = partial.stat().st_size
        assert 0 < saved < len(server.data)
        server.mode = "range"
        worker.transfer(client, server.url, partial, len(server.data))
    assert server.ranges == [None, f"bytes={saved}-"]
    assert partial.read_bytes() == server.data


@pytest.mark.parametrize("mode", ["range", "ignore"])
def test_resume_or_safe_restart_when_server_ignores_range(server, tmp_path, mode):
    partial = tmp_path / "model.incomplete"
    partial.write_bytes(server.data[:1024])
    server.mode = mode
    with httpx.Client() as client:
        worker.transfer(client, server.url, partial, len(server.data))
    assert partial.read_bytes() == server.data
    assert server.ranges == ["bytes=1024-"]


def test_invalid_range_does_not_corrupt_saved_bytes(server, tmp_path):
    partial = tmp_path / "model.incomplete"
    partial.write_bytes(server.data[:1024])
    server.mode = "bad"
    with httpx.Client() as client, pytest.raises(worker.DownloadError, match="invalid resume range"):
        worker.transfer(client, server.url, partial, len(server.data))
    assert partial.read_bytes() == server.data[:1024]


def test_verified_file_published_in_standard_hub_cache(server, tmp_path, monkeypatch):
    revision = "a" * 40
    digest = hashlib.sha256(server.data).hexdigest()
    monkeypatch.setattr(worker, "get_hf_file_metadata", lambda *a, **kw: SimpleNamespace(
        commit_hash=revision, size=len(server.data), etag=digest, location=server.url))
    cache = str(tmp_path / "cache")
    with httpx.Client() as client:
        worker.download_file(client, "test/model", revision, "transformer/model.safetensors", len(server.data), cache, lambda: None)
        # Completed cache files are reused without another HTTP transfer.
        worker.download_file(client, "test/model", revision, "transformer/model.safetensors", len(server.data), cache, lambda: None)
    root = Path(snapshot_download("test/model", revision=revision, cache_dir=cache, local_files_only=True))
    assert (root / "transformer/model.safetensors").read_bytes() == server.data
    assert len(server.ranges) == 1
    assert not list(Path(cache).rglob("*.incomplete"))


def test_checksum_failure_never_publishes_model(server, tmp_path, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(worker, "get_hf_file_metadata", lambda *a, **kw: SimpleNamespace(
        commit_hash=revision, size=len(server.data), etag="0" * 64, location=server.url))
    with httpx.Client() as client, pytest.raises(worker.DownloadError, match="checksum"):
        worker.download_file(client, "test/model", revision, "model.safetensors", len(server.data), str(tmp_path), lambda: None)
    assert not list(tmp_path.rglob("*.incomplete"))
    assert not (tmp_path / "models--test--model/snapshots").exists()


@pytest.fixture
def manager(monkeypatch, tmp_path):
    models.stop_downloads()
    monkeypatch.setattr(models, "_downloads", {})
    monkeypatch.setattr(models, "POLL_SECONDS", .02)
    monkeypatch.setattr(models, "STALL_SECONDS", .15)
    monkeypatch.setattr(models, "RETRY_SECONDS", .05)
    real_popen = subprocess.Popen
    procs = []
    ready = tmp_path / "ready"
    modes = []
    def launch(command, **kwargs):
        root = command[-1]
        mode = modes.pop(0) if modes else "hang"
        code = "import time; time.sleep(600)"
        if mode == "done":
            code = f"from pathlib import Path; Path({str(ready)!r}).touch()"
        elif mode in ("transient", "permanent"):
            result = {"error": "Test connection failed", "retryable": mode == "transient"}
            code = f"from pathlib import Path; Path({str(Path(root) / 'result.json')!r}).write_text({json.dumps(result)!r}); raise SystemExit(1)"
        p = real_popen([sys.executable, "-c", code], **kwargs)
        procs.append(p)
        return p
    monkeypatch.setattr(models.subprocess, "Popen", launch)
    yield SimpleNamespace(procs=procs, ready=ready, modes=modes,
                          start=lambda: models._start("suite:bf16", [], ready.exists),
                          status=lambda: models.download_status("suite:bf16"))
    models.stop_downloads()
    for p in procs:
        assert p.poll() is not None


def test_watchdog_replaces_stalled_worker_and_finishes(manager):
    manager.modes[:] = ["hang", "done"]
    assert manager.start()
    wait_until(lambda: models._intent("suite:bf16") == "complete")
    assert len(manager.procs) == 2
    assert manager.procs[0].returncode == -signal.SIGTERM


def test_manual_pause_cancels_worker_and_prevents_watchdog_resume(manager):
    manager.modes[:] = ["hang", "done"]
    manager.start()
    wait_until(lambda: manager.procs)
    assert models.pause_download("suite:bf16")
    wait_until(lambda: manager.procs[0].poll() is not None)
    time.sleep(.3)
    assert manager.status()["state"] == "paused"
    assert len(manager.procs) == 1
    assert manager.start()
    wait_until(lambda: models._intent("suite:bf16") == "complete")
    assert len(manager.procs) == 2


def test_pause_during_backoff_does_not_auto_resume(manager, monkeypatch):
    monkeypatch.setattr(models, "RETRY_SECONDS", .3)
    manager.modes[:] = ["transient", "done"]
    manager.start()
    wait_until(lambda: manager.status()["state"] == "retrying")
    models.pause_download("suite:bf16")
    time.sleep(.4)
    assert manager.status()["state"] == "paused"
    assert len(manager.procs) == 1


def test_permanent_error_stops_retrying_and_explicit_retry_works(manager):
    manager.modes[:] = ["permanent", "done"]
    manager.start()
    wait_until(lambda: manager.status()["state"] == "failed")
    wait_until(lambda: not models._downloads["suite:bf16"].thread.is_alive())
    assert manager.status()["error"] == "Test connection failed"
    assert manager.start()
    wait_until(lambda: models._intent("suite:bf16") == "complete")


def test_shutdown_preserves_running_intent_for_next_launch(manager, monkeypatch):
    manager.start()
    wait_until(lambda: manager.procs)
    models.stop_downloads()
    assert models._intent("suite:bf16") == "running"
    monkeypatch.setattr(suites, "source_parts", lambda: None)
    monkeypatch.setattr(suites, "start_download", manager.start)
    manager.modes[:] = ["done"]
    suites.restore_download()
    wait_until(lambda: models._intent("suite:bf16") == "complete")


def test_pause_survives_shutdown_and_restore(manager, monkeypatch):
    manager.start()
    wait_until(lambda: manager.procs)
    models.pause_download("suite:bf16")
    models.stop_downloads()
    monkeypatch.setattr(suites, "start_download", lambda: pytest.fail("user paused; must not restart"))
    suites.restore_download()
    assert manager.status()["state"] == "paused"


def test_pause_resume_api(client, monkeypatch):
    monkeypatch.setattr(models, "pause_download", lambda key: key == "suite:bf16")
    monkeypatch.setattr(suites, "start_download", lambda: True)
    assert client.post("/api/models/bf16/pause").json() == {"paused": True}
    assert client.post("/api/models/bf16/resume").json() == {"started": True}
