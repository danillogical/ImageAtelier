"""Shared fixtures: a stub engine (no models) and an app/client over a tmp data dir."""

from __future__ import annotations

import io
from pathlib import Path
import threading
import time
import uuid
from collections import deque

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webstudio import models
from webstudio.app import create_app


class StubModelManager:
    """Stub of engine.ModelManager: one resident key, records every load, emits model events."""

    def __init__(self, emit):
        self._emit = emit
        self.key = None
        self.loads: list[tuple] = []

    def ensure(self, key):
        if key == self.key:
            return 0.0
        if self.key is not None:
            self._emit({"type": "model", "state": "unloaded", "key": list(self.key), "load_s": None})
        self._emit({"type": "model", "state": "loading", "key": list(key), "load_s": None})
        self.loads.append(key)
        self.key = key
        self._emit({"type": "model", "state": "loaded", "key": list(key), "load_s": 0.0})
        return 0.0


class _Gate:
    """Event-like flag whose changes are made under the engine's condition lock and wake the worker at once.

    The worker tests the gate and pops the queue inside one critical section, so clear()/set() can never
    interleave with a dequeue decision (no polling window).
    """

    def __init__(self, cv):
        self._cv = cv
        self._flag = True

    def set(self):
        with self._cv:
            self._flag = True
            self._cv.notify_all()

    def clear(self):
        with self._cv:
            self._flag = False

    def is_set(self):
        with self._cv:
            return self._flag


class StubEngine:
    """Same public surface as webstudio.engine.Engine, generating a tiny RGBA PNG per job."""

    def __init__(self):
        self.submitted: list[dict] = []  # copies of every job dict passed to submit()
        self._jobs: dict[str, dict] = {}
        self._queue: deque[str] = deque()
        self._listeners = []
        self._stopping = False
        self._thread = None
        self._resident = {"key": None, "state": "unloaded"}
        self._cv = threading.Condition()
        self.gate = _Gate(self._cv)  # cleared -> worker holds jobs in "queued"
        self.running = threading.Event()  # set while a job is in "running"
        self.release = threading.Event()  # if cleared when a job starts, it waits here (stays running)
        self.release.set()
        self.outcome = deque()  # per-job outcomes consumed FIFO: "done" | "fail"; empty -> done
        self.steps_per_job = 3
        self.image_factory = None  # optional callable(job dict) -> RGBA image written as the job's output
        self.models = StubModelManager(self._emit_model)
        self.enhance_fell_back = False
        self.enhance_text = None  # optional callable(prompt) -> rewrite for enhance jobs
        self.reject_submit = None  # optional callable(job) that raises ValueError to refuse a submit

    # ---- Engine surface ---------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, timeout=None):
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        self.gate.set()
        self.release.set()
        if self._thread:
            self._thread.join(timeout or 5)

    def on_event(self, fn):
        self._listeners.append(fn)

    def submit(self, job):
        prompt = job.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        kind = job.get("kind", "generate")
        if kind == "generate" and job.get("mode") == "edit" and not job.get("ref_image_path"):
            raise ValueError("edit mode requires ref_image_path")
        if kind == "enhance" and job.get("mode") == "i2i" and not job.get("ref_image_path"):
            raise ValueError("i2i enhance requires ref_image_path")
        if self.reject_submit:
            self.reject_submit(job)
        self.submitted.append(dict(job))
        jid = uuid.uuid4().hex[:12]
        if kind == "enhance":
            j = {
                "id": jid, "kind": "enhance", "prompt": prompt, "mode": job["mode"],
                "ref_image_path": job.get("ref_image_path"), "quant": job.get("quant"),
                "thinking": job.get("thinking"), "max_new_tokens": job.get("max_new_tokens"),
                "language": job.get("language"), "phase": None, "tokens": 0, "text": "",
                "state": "queued", "error": None, "result": None, "created_at": time.time(),
            }
        else:
            j = {
                "id": jid, "kind": "generate", "prompt": prompt, "mode": job["mode"],
                "ref_image_path": job.get("ref_image_path"),
                "seed": job.get("seed"), "steps": job.get("steps"), "turbo": job.get("turbo"),
                "image_model": job.get("image_model"), "text_encoder": job.get("text_encoder"),
                "transparent": bool(job.get("transparent", False)), "output_path": job["output_path"], "state": "queued",
                "step": 0, "total_steps": 0, "error": None, "result": None, "created_at": time.time(),
            }
        with self._cv:
            self._jobs[jid] = j
            self._queue.append(jid)
            self._cv.notify_all()
        self._emit(j, "state")
        return jid

    def resident(self):
        return dict(self._resident)

    def _emit_model(self, event):
        self._resident = {"key": event["key"] if event["state"] != "unloaded" else None, "state": event["state"]}
        for fn in list(self._listeners):
            fn(dict(event))

    def emit_model(self, state, key, **extra):
        """Test hook: emit a model event as the engine would."""
        self._emit_model({"type": "model", "state": state, "key": key, "load_s": None, **extra})

    def cancel(self, job_id):
        with self._cv:
            j = self._jobs.get(job_id)
            if j is None or j["state"] in ("done", "failed", "cancelled"):
                return False
            if job_id in self._queue:
                self._queue.remove(job_id)
                j["state"] = "cancelled"
            else:
                j["cancel_requested"] = True
                return True
        self._emit(j, "state")
        return True

    def get(self, job_id):
        with self._cv:
            j = self._jobs.get(job_id)
            return self._snap(j) if j else None

    def jobs(self):
        with self._cv:
            return [self._snap(j) for j in self._jobs.values()]

    # ---- internals --------------------------------------------------------------------------
    def _run_enhance(self, j):
        self.models.ensure((f"pe_{j['mode']}", j["quant"]))
        self._emit(j, "state")
        self.running.set()
        self.release.wait()
        outcome = self.outcome.popleft() if self.outcome else "done"
        words = ["a", "very", "detailed", "rewrite"]
        for i, phase in enumerate(["thinking", "writing", "writing"], 1):
            time.sleep(0.01)
            if j.get("cancel_requested"):
                j["state"] = "cancelled"
                self.running.clear()
                self._emit(j, "state")
                return
            j["phase"], j["tokens"], j["text"] = phase, i * 10, " ".join(words[:i + 1]) if phase == "writing" else ""
            self._emit(j, "progress")
        if outcome == "fail":
            j["state"], j["error"] = "failed", "boom"
        else:
            fb = self.enhance_fell_back
            j["state"] = "done"
            j["result"] = {
                "rewritten_prompt": j["prompt"] if fb else (self.enhance_text(j["prompt"]) if self.enhance_text else "ENHANCED: " + j["prompt"]),
                "fell_back": fb, "rewriter": "official-8bit",
                "finish_reason": "stop", "thinking_tokens": 10, "answer_tokens": 20, "tok_s": 50.0,
                "gen_s": 0.03, "load_s": 0.0, "peak_mem_bytes": 1, "mode": j["mode"], "quant": j["quant"],
                "thinking": j["thinking"], "language": j["language"],
            }
            j["text"] = j["result"]["rewritten_prompt"]
        self.running.clear()
        self._emit(j, "state")

    @staticmethod
    def _snap(j):
        return {k: v for k, v in j.items() if k != "cancel_requested"}

    def _emit(self, j, kind):
        with self._cv:
            snap = {**self._snap(j), "type": kind}
        for fn in list(self._listeners):
            fn(snap)

    def _worker(self):
        while True:
            with self._cv:
                while not self._stopping and not (self._queue and self.gate.is_set()):
                    self._cv.wait(0.05)
                if self._stopping:
                    return
                j = self._jobs[self._queue.popleft()]
                j["state"] = "running"
                if j["kind"] == "generate":
                    j["total_steps"] = self.steps_per_job
            if j["kind"] == "enhance":
                self._run_enhance(j)
                continue
            self.models.ensure(("image", j["image_model"], j["text_encoder"], j["turbo"]))
            self._emit(j, "state")
            self.running.set()
            self.release.wait()
            outcome = self.outcome.popleft() if self.outcome else "done"
            for s in range(1, self.steps_per_job + 1):
                time.sleep(0.01)
                if j.get("cancel_requested"):
                    j["state"] = "cancelled"
                    self.running.clear()
                    self._emit(j, "state")
                    break
                j["step"] = s
                self._emit(j, "progress")
            else:
                if outcome == "fail":
                    # partial output left behind on purpose; the app must clean it up
                    with open(j["output_path"], "wb") as f:
                        f.write(b"partial")
                    j["state"] = "failed"
                    j["error"] = "boom"
                else:
                    # Opaque by default: normal outputs are fully opaque (API.md, alpha rules).
                    img = self.image_factory(j) if self.image_factory else Image.new("RGBA", (64, 48), (255, 0, 0, 255))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    with open(j["output_path"], "wb") as f:
                        f.write(buf.getvalue())
                    j["state"] = "done"
                    j["result"] = {
                        "seed": j["seed"] if j["seed"] is not None else 1234, "steps": j["steps"],
                        "turbo": j["turbo"], "image_model": j["image_model"],
                        "text_encoder": j["text_encoder"], "transparent": j["transparent"], "width": 64, "height": 48,
                        "gen_s": 0.03, "load_s": 0.0, "output_path": j["output_path"],
                    }
                self.running.clear()
                self._emit(j, "state")


@pytest.fixture
def ready(monkeypatch):
    """Controllable part readiness. ready["image"]["official-8bit"] = False / ready["te"]["heretic-8bit"] = False
    make that part not_downloaded (the local official encoder reports not_installed). ready["started"] /
    ready["started_te"] record download requests."""
    state = {"image": {n: True for n in models.IMAGE_MODELS}, "te": {n: True for n in models.TEXT_ENCODERS},
             "started": [], "started_te": []}

    def image_status(name):
        m = models.IMAGE_MODELS[name]
        return {"kind": "image", "name": name, "label": m.label, "approx_gb": m.approx_gb,
                "state": "ready" if state["image"][name] else "not_downloaded", "turbo_lora_ready": True, "error": None}

    def te_status(name):
        t = models.TEXT_ENCODERS[name]
        base = {"kind": "text_encoder", "name": name, "label": t.label, "approx_gb": t.approx_gb, "source": t.source}
        if state["te"][name]:
            return {**base, "state": "ready"}
        if t.source == "local":
            return {**base, "state": "not_installed", "error": None, "hint": f"Copy the folder {t.folder}/ into models"}
        return {**base, "state": "not_downloaded", "error": None}

    def start_image_download(name):
        state["started"].append(name)
        return True

    def start_te_download(name):
        state["started_te"].append(name)
        return True

    monkeypatch.setattr(models, "image_status", image_status)
    monkeypatch.setattr(models, "te_status", te_status)
    monkeypatch.setattr(models, "start_image_download", start_image_download)
    monkeypatch.setattr(models, "start_te_download", start_te_download)
    monkeypatch.setattr(models, "is_image_downloading", lambda n: False)
    monkeypatch.setattr(models, "is_te_downloading", lambda n: False)
    return state


@pytest.fixture
def pe_ready(monkeypatch):
    """Controllable enhancer readiness: pe_ready['t2i-8bit'] = False makes it not_downloaded."""
    state = {n: True for n in models.PE_MODELS}
    started = []

    def pe_status(name):
        m = models.PE_MODELS[name]
        return {"name": name, "state": "ready" if state[name] else "not_downloaded", "mode": m.mode,
                "quant": m.quant, "error": None}

    def start_pe_download(name):
        started.append(name)
        return True

    monkeypatch.setattr(models, "pe_status", pe_status)
    monkeypatch.setattr(models, "start_pe_download", start_pe_download)
    monkeypatch.setattr(models, "is_pe_downloading", lambda n: False)
    state["started"] = started
    return state


@pytest.fixture
def engine():
    return StubEngine()


@pytest.fixture
def app(engine, ready, pe_ready, tmp_path):
    return create_app(engine=engine, data_dir=tmp_path / "data")


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---- helpers -------------------------------------------------------------------------------
def wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")


def new_project(client, **body):
    r = client.post("/api/projects", json=body)
    assert r.status_code in (200, 201), r.text
    return r.json()


def submit(client, pid, prompt="a cat", **extra):
    r = client.post(f"/api/projects/{pid}/runs", json={"prompt": prompt, **extra})
    assert r.status_code == 202, r.text
    return r.json()


def wait_run(client, pid, n):
    """Block until run n is saved (its json is visible via the project)."""
    def find():
        runs = client.get(f"/api/projects/{pid}").json()["runs"]
        return next((r for r in runs if r["run"] == n), None)
    return wait_until(find)


def wait_job(engine, job_id, states=("done", "failed", "cancelled")):
    wait_until(lambda: engine.get(job_id)["state"] in states)
    # the app saves/cleans up on its event loop; give it a moment to settle
    time.sleep(0.2)


def do_run(client, pid, prompt="a cat", **extra):
    r = submit(client, pid, prompt, **extra)
    wait_run(client, pid, r["run_reserved"])
    return r


def patch_parts(monkeypatch):
    """Stub part resolution and composition for real-Engine tests (no weights, no symlinks)."""
    def resolve_parts(image_model, text_encoder, turbo):
        return Path("/nowhere"), Path("/nowhere/te"), (Path("/nowhere/lora") if turbo else None)

    monkeypatch.setattr(models, "resolve_parts", resolve_parts)
    monkeypatch.setattr(models, "compose", lambda *a, **kw: Path("/nowhere/composed"))
