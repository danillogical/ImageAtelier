"""Strict job serialization and the `cancelling` state, using the real Engine with stubbed loaders/generation."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imageatelier.app import create_app

from conftest import patch_parts

START_STATES = ("loading", "running")
TERMINAL = ("done", "failed", "cancelled")


@pytest.fixture
def h(monkeypatch, tmp_path, ready, pe_ready):
    """Harness around a real (not yet started) Engine. Call h.start() for engine-level tests."""
    mlx_vlm = pytest.importorskip("mlx_vlm")
    from imageatelier import engine as eng_mod
    from imageatelier import models

    log: list[tuple] = []  # ordered: ("job", id, state, t) | ("sync_start"|"sync_end", t)
    behaviour: deque = deque()  # per job that reaches its GPU work: "block" | "fail" | (absent -> quick)
    release = threading.Event()  # unblocks "block" jobs
    ns = SimpleNamespace(log=log, behaviour=behaviour, release=release, sync_delay=0.0, sync_calls=0, step_sleep=0.01)

    def fake_sync():
        ns.sync_calls += 1
        log.append(("sync_start", time.monotonic()))
        time.sleep(ns.sync_delay)
        log.append(("sync_end", time.monotonic()))

    def work(engine):
        """The fake GPU work for the current job."""
        mode = behaviour.popleft() if behaviour else "quick"
        if mode == "fail":
            raise RuntimeError("boom")
        if mode == "block":
            i = 0
            while not release.is_set():
                i += 1
                engine._on_step(i, 1000)  # raises JobCancelled once a cancel was requested
                time.sleep(ns.step_sleep)

    def fake_generate_image(**kw):
        work(ns.engine)
        return SimpleNamespace(image=Image.new("RGBA", (16, 16), (1, 2, 3, 255)))

    def fake_load_image(self, key, snapshot, lora):
        self.model = SimpleNamespace(
            text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None), tiling_config=None,
            generate_image=fake_generate_image)
        self.snapshot, self.key = snapshot, key

    def fake_load_pe(self, key, path, system_prompt):
        self.model = object()
        self.processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
        self.system_prompt, self.key = system_prompt, key

    def fake_stream(model, processor, prompt, image=None, max_tokens=0, **kw):
        work(ns.engine)
        yield SimpleNamespace(text='{"rewritten_prompt": "better"}', finish_reason="stop", generation_tps=10.0)

    patch_parts(monkeypatch)
    monkeypatch.setattr(models, "resolve_pe", lambda m, q: (Path("/nowhere"), "system prompt"))
    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", fake_load_image)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_pe", fake_load_pe)
    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream)
    monkeypatch.setattr(eng_mod, "_synchronize_all", fake_sync)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)

    e = eng_mod.Engine(settings_provider=lambda: {"pe_thinking": False, "low_memory": False})
    ns.engine = e

    def on_event(ev):
        if ev.get("type") in ("state", "progress"):
            log.append(("job", ev["id"], ev["state"], time.monotonic()))

    e.on_event(on_event)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (16, 16)).save(ref)
    ns.ref = str(ref)
    ns.tmp = tmp_path
    ns.start = e.start
    ns.n = 0

    def enh():
        return e.submit({"kind": "enhance", "prompt": "a cat", "mode": "t2i", "quant": "8bit"})

    def gen():
        ns.n += 1
        return e.submit({"kind": "generate", "prompt": "a cat", "mode": "t2i", "steps": 6, "turbo": True,
                         "output_path": str(tmp_path / f"out{ns.n}.png")})

    ns.enh, ns.gen = enh, gen
    yield ns
    release.set()
    try:
        e.stop(timeout=10)
    except Exception:  # noqa: BLE001 - already stopped by the app lifespan
        pass


def wait_state(engine, jid, states, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if engine.get(jid)["state"] in states:
            return engine.get(jid)
        time.sleep(0.01)
    raise AssertionError(f"job {jid} never reached {states}: {engine.get(jid)}")


def states_of(log, jid):
    """The job's state sequence from the event log, consecutive duplicates (progress) collapsed."""
    out = []
    for e in log:
        if e[0] == "job" and e[1] == jid and (not out or out[-1] != e[2]):
            out.append(e[2])
    return out


def first_index(log, jid, states):
    return next(i for i, e in enumerate(log) if e[0] == "job" and e[1] == jid and e[2] in states)


def terminal_index(log, jid):
    return next(i for i, e in enumerate(log) if e[0] == "job" and e[1] == jid and e[2] in TERMINAL)


# 1 ---------------------------------------------------------------------------------------------
def test_no_overlap_in_mixed_sequence(h):
    ids = [h.enh(), h.gen(), h.enh()]
    h.start()
    for i in ids:
        assert wait_state(h.engine, i, TERMINAL)["state"] == "done"
    for prev, nxt in zip(ids, ids[1:]):
        assert terminal_index(h.log, prev) < first_index(h.log, nxt, START_STATES), h.log
    for i in ids:  # exactly one terminal event per job, and it is last
        seq = states_of(h.log, i)
        assert seq[-1] == "done" and sum(s in TERMINAL for s in seq) == 1, seq


def test_result_has_teardown_and_gpu_util(h):
    i = h.gen()
    h.start()
    res = wait_state(h.engine, i, TERMINAL)["result"]
    assert "teardown_s" in res and res["teardown_s"] >= 0
    assert "gpu_util_at_start" in res
    j = h.enh()
    res = wait_state(h.engine, j, TERMINAL)["result"]
    assert "teardown_s" in res and "gpu_util_at_start" in res


# 2 ---------------------------------------------------------------------------------------------
def test_cancel_running_goes_through_cancelling(h):
    h.sync_delay = 0.5  # widens the cancelling window (teardown) so it can be observed
    h.behaviour.append("block")
    a, b = h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    assert h.engine.cancel(a) is True
    assert h.engine.get(a)["state"] == "cancelling"
    time.sleep(0.25)  # the job has stopped at its next step and is inside teardown
    assert h.engine.get(a)["state"] == "cancelling"
    assert h.engine.get(b)["state"] == "queued"  # the next job has not been dequeued
    wait_state(h.engine, a, TERMINAL)
    assert h.engine.get(a)["state"] == "cancelled"
    assert h.engine.get(a)["result"] is None
    wait_state(h.engine, b, TERMINAL)
    seq = states_of(h.log, a)
    assert seq[seq.index("running"):] == ["running", "cancelling", "cancelled"], seq
    assert states_of(h.log, b)[0] == "queued" and h.engine.get(b)["state"] == "done"
    assert terminal_index(h.log, a) < first_index(h.log, b, START_STATES)


# 3 ---------------------------------------------------------------------------------------------
def test_cancel_queued_is_instant_and_leaves_running_job_alone(h):
    h.behaviour.append("block")
    a, b = h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    assert h.engine.cancel(b) is True
    assert h.engine.get(b)["state"] == "cancelled"  # immediately, no cancelling step
    assert states_of(h.log, b) == ["queued", "cancelled"]
    assert h.engine.get(a)["state"] == "running"
    h.release.set()
    assert wait_state(h.engine, a, TERMINAL)["state"] == "done"
    assert states_of(h.log, b) == ["queued", "cancelled"]  # never loaded or ran


# 4 ---------------------------------------------------------------------------------------------
def test_failure_runs_teardown_before_next_job(h):
    h.behaviour.append("fail")
    a, b = h.gen(), h.gen()
    h.start()
    assert wait_state(h.engine, a, TERMINAL)["state"] == "failed"
    assert "boom" in h.engine.get(a)["error"]
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"
    sync_ends = [i for i, e in enumerate(h.log) if e[0] == "sync_end"]
    a_running, a_failed = first_index(h.log, a, ("running",)), terminal_index(h.log, a)
    b_start = first_index(h.log, b, START_STATES)
    assert any(a_running < i < a_failed for i in sync_ends), "no synchronize between A's work and its failed event"
    assert a_failed < b_start


# 5 ---------------------------------------------------------------------------------------------
def test_slow_teardown_delays_next_job(h):
    h.sync_delay = 0.5
    a, b = h.gen(), h.enh()
    h.start()
    for i in (a, b):
        assert wait_state(h.engine, i, TERMINAL, timeout=15)["state"] == "done"
    t = lambda idx: h.log[idx][-1]  # noqa: E731
    a_run, a_done = first_index(h.log, a, ("running",)), terminal_index(h.log, a)
    b_start = first_index(h.log, b, START_STATES)
    assert t(a_done) - t(a_run) >= 0.45  # terminal event waits for the synchronize
    assert t(b_start) >= t(a_done)
    sync_end_after_a_run = next(i for i, e in enumerate(h.log) if e[0] == "sync_end" and i > a_run)
    assert sync_end_after_a_run < a_done < b_start
    assert h.engine.get(a)["result"]["teardown_s"] >= 0.45


# 6 ---------------------------------------------------------------------------------------------
def test_api_shows_cancelling_and_cancel_is_safe_while_cancelling(h, ready, pe_ready, tmp_path):
    h.sync_delay = 0.6
    h.behaviour.append("block")
    app = create_app(engine=h.engine, data_dir=tmp_path / "data")
    with TestClient(app) as c:  # lifespan starts the engine
        pid = c.post("/api/projects", json={}).json()["id"]
        r = c.post(f"/api/projects/{pid}/runs", json={"prompt": "a cat"})
        assert r.status_code == 202
        jid, n = r.json()["job_id"], r.json()["run_reserved"]

        def job():
            return next(j for j in c.get("/api/jobs").json() if j["id"] == jid)

        end = time.time() + 10
        while job()["state"] != "running" and time.time() < end:
            time.sleep(0.01)
        assert job()["state"] == "running"
        assert c.post(f"/api/jobs/{jid}/cancel").json() == {"cancelled": True}
        assert job()["state"] == "cancelling"
        time.sleep(0.2)
        assert job()["state"] == "cancelling"  # still tearing down
        again = c.post(f"/api/jobs/{jid}/cancel")  # API.md: cancelling a `cancelling` job returns true
        assert again.status_code == 200 and again.json() == {"cancelled": True}
        end = time.time() + 10
        while job()["state"] != "cancelled" and time.time() < end:
            time.sleep(0.02)
        assert job()["state"] == "cancelled"
        time.sleep(0.3)
        p = c.get(f"/api/projects/{pid}").json()
        assert p["runs"] == [] and p["current_run"] is None
        runs = tmp_path / "data" / "projects" / pid / "runs"
        assert not list(runs.glob(f"{n:04d}*")) if runs.exists() else True
        assert c.post(f"/api/jobs/{jid}/cancel").json() == {"cancelled": False}  # finished now
        h.release.set()
