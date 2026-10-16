"""Job controls (pause/resume/hold/release), job ordering, logs and SSE log events, against API.md.

Uses the real Engine with stubbed loaders/generation (the `h` harness from test_serialization).
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from imageatelier.app import create_app

from conftest import new_project
from test_api import parse_sse
from test_serialization import (  # noqa: F401 - h is a fixture
    TERMINAL, first_index, h, states_of, terminal_index, wait_state,
)

START = ("loading", "running")


def wait_for(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.01)
    raise AssertionError("timed out")


def lines_of(resp):
    body = resp.json()
    return body["lines"] if isinstance(body, dict) else body


@pytest.fixture
def api(h, tmp_path):
    """TestClient over the real engine (started by the app lifespan)."""
    app = create_app(engine=h.engine, data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c
    h.release.set()


def api_run(c, pid, prompt="x"):
    r = c.post(f"/api/projects/{pid}/runs", json={"prompt": prompt})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def api_job(c, jid):
    return next(j for j in c.get("/api/jobs").json() if j["id"] == jid)


def act(c, jid, action):
    return c.post(f"/api/jobs/{jid}/{action}")


# 1 ---------------------------------------------------------------------------------------------
def test_pause_running_job(h):
    h.behaviour.append("block")
    h.step_sleep = 0.3  # a wide window between pause() and the next step boundary
    a, b = h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    assert h.engine.pause(a) is True
    snap = h.engine.get(a)
    assert snap["pause_requested"] is True and snap["state"] == "running"  # not paused until the boundary
    paused = wait_state(h.engine, a, ("paused",))
    assert paused["pause_requested"] is False
    step = paused["step"]
    time.sleep(0.5)
    assert h.engine.get(a)["step"] == step and h.engine.get(a)["state"] == "paused"
    assert h.engine.get(b)["state"] == "queued"  # a paused job blocks the queue
    h.step_sleep = 0.01
    assert h.engine.resume(a) is True
    wait_state(h.engine, a, ("running",))
    wait_for(lambda: h.engine.get(a)["step"] > step)  # progress continues after the resume
    h.release.set()
    done = wait_state(h.engine, a, TERMINAL)
    assert done["state"] == "done" and done["paused_s"] > 0 and done["result"]["paused_s"] > 0
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"
    assert terminal_index(h.log, a) < first_index(h.log, b, START)
    seq = states_of(h.log, a)
    assert seq[seq.index("running"):][:4] == ["running", "paused", "running", "done"], seq


def test_resume_withdraws_unapplied_pause_request(h):
    h.behaviour.append("block")
    h.step_sleep = 0.3
    a = h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    assert h.engine.pause(a) is True
    assert h.engine.resume(a) is True
    assert h.engine.get(a)["pause_requested"] is False
    time.sleep(0.6)
    assert h.engine.get(a)["state"] == "running"
    h.release.set()
    assert wait_state(h.engine, a, TERMINAL)["state"] == "done"


# 2 ---------------------------------------------------------------------------------------------
def test_cancel_while_paused(h):
    h.sync_delay = 0.4
    h.behaviour.append("block")
    a, b = h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    h.engine.pause(a)
    wait_state(h.engine, a, ("paused",))
    assert h.engine.cancel(a) is True
    assert h.engine.get(a)["state"] == "cancelling"
    assert h.engine.get(b)["state"] == "queued"
    wait_state(h.engine, a, TERMINAL)
    assert h.engine.get(a)["state"] == "cancelled"
    seq = states_of(h.log, a)
    assert seq[seq.index("paused"):] == ["paused", "cancelling", "cancelled"], seq
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"
    assert terminal_index(h.log, a) < first_index(h.log, b, START)


# 3 ---------------------------------------------------------------------------------------------
def test_hold_release_ordering(h):
    h.behaviour.append("block")
    a, b, c = h.gen(), h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    assert [h.engine.get(x)["queue_position"] for x in (a, b, c)] == [None, 1, 2]
    assert h.engine.hold(b) is True
    hb = h.engine.get(b)
    assert hb["state"] == "held" and hb["queue_position"] == 1  # keeps its place
    h.release.set()
    wait_state(h.engine, c, TERMINAL)
    assert h.engine.get(b)["state"] == "held" and h.engine.get(b)["queue_position"] == 1
    assert h.engine.get(c)["state"] == "done"
    assert h.engine.release(b) is True
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"
    assert terminal_index(h.log, c) < first_index(h.log, b, START)


def test_release_keeps_original_relative_position(h):
    h.behaviour.append("block")
    a, b, c, d = h.gen(), h.gen(), h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    h.engine.hold(b)
    assert h.engine.release(b) is True
    assert [h.engine.get(x)["queue_position"] for x in (b, c, d)] == [1, 2, 3]
    h.engine.hold(c)
    h.engine.release(c)
    h.release.set()
    for x in (b, c, d):
        wait_state(h.engine, x, TERMINAL)
    starts = [first_index(h.log, x, START) for x in (b, c, d)]
    assert starts == sorted(starts)


def test_queue_positions_shift_as_jobs_start(h):
    h.behaviour.append("block")
    a, b, c = h.gen(), h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    h.release.set()
    wait_state(h.engine, b, TERMINAL)
    wait_state(h.engine, c, TERMINAL)
    positions = [e for e in h.log if e[0] == "job" and e[1] == c and e[2] == "queued"]
    assert len(positions) >= 2  # a fresh event was emitted when C's queue_position changed


# 4 ---------------------------------------------------------------------------------------------
def test_holding_only_queued_job_leaves_worker_idle(h):
    b = h.gen()
    h.engine.hold(b)  # before the worker exists: nothing can be dequeued
    h.start()
    time.sleep(0.5)
    assert h.engine.get(b)["state"] == "held"
    assert states_of(h.log, b) == ["queued", "held"]
    assert h.sync_calls == 0  # no job ever began, so no teardown ran
    assert h.engine.release(b) is True
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"


def test_holding_last_queued_job_while_another_runs(h):
    h.behaviour.append("block")
    a, b = h.gen(), h.gen()
    h.start()
    wait_state(h.engine, a, ("running",))
    h.engine.hold(b)
    h.release.set()
    wait_state(h.engine, a, TERMINAL)
    time.sleep(0.4)
    assert h.engine.get(b)["state"] == "held"  # worker idle though a job is waiting
    h.engine.release(b)
    assert wait_state(h.engine, b, TERMINAL)["state"] == "done"


# 5 ---------------------------------------------------------------------------------------------
class TestControlApi:
    def test_invalid_transitions_and_noops(self, h, api):
        pid = new_project(api, name="P")["id"]
        h.behaviour.append("block")
        h.step_sleep = 0.02
        a, b = api_run(api, pid), api_run(api, pid)
        wait_for(lambda: api_job(api, a)["state"] == "running")

        # queued job: pause is invalid, resume is invalid, release is a no-op
        r = act(api, b, "pause")
        assert r.status_code == 409 and "queued" in r.json()["detail"]
        assert act(api, b, "resume").status_code == 409
        r = act(api, b, "release")
        assert r.status_code == 200 and r.json()["changed"] is False and r.json()["id"] == b

        # running job: hold is invalid, resume is invalid, release is invalid
        assert act(api, a, "hold").status_code == 409
        assert act(api, a, "resume").status_code == 409
        assert act(api, a, "release").status_code == 409

        # hold: 200 + snapshot + changed, repeat is a no-op
        r = act(api, b, "hold")
        assert r.status_code == 200 and r.json()["changed"] is True
        assert r.json()["state"] == "held" and r.json()["queue_position"] == 1 and r.json()["project_name"] == "P"
        r = act(api, b, "hold")
        assert r.status_code == 200 and r.json()["changed"] is False and r.json()["state"] == "held"

        # pause: 200 + changed, repeat is a no-op, then hold/release are invalid on a paused job
        r = act(api, a, "pause")
        assert r.status_code == 200 and r.json()["changed"] is True
        assert r.json()["pause_requested"] is True or r.json()["state"] == "paused"
        r = act(api, a, "pause")
        assert r.status_code == 200 and r.json()["changed"] is False
        wait_for(lambda: api_job(api, a)["state"] == "paused")
        assert act(api, a, "pause").json()["changed"] is False
        assert act(api, a, "hold").status_code == 409
        assert act(api, a, "release").status_code == 409

        # resume: changed true, then a second resume (job not paused) is invalid
        r = act(api, a, "resume")
        assert r.status_code == 200 and r.json()["changed"] is True
        wait_for(lambda: api_job(api, a)["state"] == "running")
        assert act(api, a, "resume").status_code == 409

        assert act(api, b, "release").json()["changed"] is True
        h.release.set()
        wait_for(lambda: api_job(api, b)["state"] == "done")
        # terminal jobs: nothing is valid
        for action in ("pause", "resume", "hold"):
            assert act(api, b, action).status_code == 409

    @pytest.mark.parametrize("action", ["pause", "resume", "hold", "release"])
    def test_unknown_job_404(self, api, action):
        assert act(api, "nope", action).status_code == 404

    def test_unknown_job_log_404(self, api):
        assert api.get("/api/jobs/nope/log").status_code == 404


# 6 ---------------------------------------------------------------------------------------------
def test_jobs_ordering_and_project_name(h, api):
    pid = new_project(api, name="Alpha")["id"]
    f1, f2 = api_run(api, pid), None
    wait_for(lambda: api_job(api, f1)["state"] == "done")
    time.sleep(0.3)  # let the app settle the run
    f2 = api_run(api, pid)
    wait_for(lambda: api_job(api, f2)["state"] == "done")
    time.sleep(0.3)
    h.behaviour.append("block")
    a = api_run(api, pid)
    wait_for(lambda: api_job(api, a)["state"] == "running")
    b, c, d = api_run(api, pid), api_run(api, pid), api_run(api, pid)
    act(api, c, "hold")
    jobs = api.get("/api/jobs").json()
    ids = [j["id"] for j in jobs]
    assert ids == [a, b, c, d, f1, f2], [(j["id"], j["state"]) for j in jobs]  # active, queued/held by position, finished
    by = {j["id"]: j for j in jobs}
    assert [by[x]["queue_position"] for x in (a, b, c, d)] == [None, 1, 2, 3]
    assert by[c]["state"] == "held"
    assert all(j["project_name"] == "Alpha" for j in jobs)
    # a paused job is still listed with the active ones
    act(api, a, "pause")
    wait_for(lambda: api_job(api, a)["state"] == "paused")
    assert api.get("/api/jobs").json()[0]["id"] == a
    # renaming the project is reflected
    assert api.patch(f"/api/projects/{pid}", json={"name": "Beta"}).status_code == 200
    assert all(j["project_name"] == "Beta" for j in api.get("/api/jobs").json())
    assert act(api, b, "hold").json()["project_name"] == "Beta"
    act(api, a, "cancel")
    h.release.set()


# 7 ---------------------------------------------------------------------------------------------
def test_job_log_has_steps_and_pause_resume(h, api):
    pid = new_project(api)["id"]
    h.behaviour.append("block")
    a = api_run(api, pid)
    wait_for(lambda: api_job(api, a)["state"] == "running")
    act(api, a, "pause")
    wait_for(lambda: api_job(api, a)["state"] == "paused")
    act(api, a, "resume")
    wait_for(lambda: api_job(api, a)["state"] == "running")
    h.release.set()
    wait_for(lambda: api_job(api, a)["state"] == "done")
    r = api.get(f"/api/jobs/{a}/log")
    assert r.status_code == 200
    lines = lines_of(r)
    assert lines and all(isinstance(x, str) for x in lines)
    low = [x.lower() for x in lines]
    assert any("step" in x for x in low)
    assert any("pause" in x for x in low)
    assert any("resum" in x for x in low)


class TestServerLog:
    def _fill(self, n):
        lg = logging.getLogger("imageatelier.logprobe")
        for i in range(n):
            lg.info("probe line %d", i)

    def test_tail_clamping(self, h, api):
        self._fill(2100)
        assert len(lines_of(api.get("/api/logs/server?tail=5"))) == 5
        assert len(lines_of(api.get("/api/logs/server?tail=0"))) == 1  # clamped up to 1
        assert len(lines_of(api.get("/api/logs/server?tail=-10"))) == 1
        assert len(lines_of(api.get("/api/logs/server?tail=99999"))) == 2000  # clamped down to 2000
        assert len(lines_of(api.get("/api/logs/server"))) == 500  # default
        assert "probe line 2099" in lines_of(api.get("/api/logs/server?tail=1"))[0]

    def test_tail_returns_most_recent_in_order(self, h, api):
        self._fill(30)
        got = lines_of(api.get("/api/logs/server?tail=3"))
        assert [x.rsplit(" ", 1)[-1] for x in got] == ["27", "28", "29"]


async def _listen(app, query, chunks, started):
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/api/events", "raw_path": b"/api/events", "query_string": query, "headers": [],
             "server": ("test", 80), "client": ("test", 1)}

    async def receive():
        await asyncio.Event().wait()

    async def send(msg):
        if msg["type"] == "http.response.start":
            started.set()
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    await app(scope, receive, send)


def test_log_events_to_all_and_serverlog_only_when_opted_in(h, tmp_path):
    app = create_app(engine=h.engine, data_dir=tmp_path / "data")
    h.behaviour.append("block")

    async def main():
        plain, opted = [], []
        s1, s2 = asyncio.Event(), asyncio.Event()
        async with app.router.lifespan_context(app):
            t1 = asyncio.create_task(_listen(app, b"", plain, s1))
            t2 = asyncio.create_task(_listen(app, b"serverlog=1", opted, s2))
            await asyncio.wait_for(s1.wait(), 5)
            await asyncio.wait_for(s2.wait(), 5)
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                pid = (await c.post("/api/projects", json={})).json()["id"]
                jid = (await c.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})).json()["job_id"]
                await asyncio.sleep(0.3)
                await c.post(f"/api/jobs/{jid}/pause")
                await asyncio.sleep(0.3)
                logging.getLogger("imageatelier.logprobe").info("server-only-marker-42")
                await asyncio.sleep(1.5)  # the server-log flusher batches every ~0.5 s
                await c.post(f"/api/jobs/{jid}/resume")
                h.release.set()
                await asyncio.sleep(1.0)
            for t in (t1, t2):
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        return parse_sse(b"".join(plain).decode()), parse_sse(b"".join(opted).decode()), jid

    plain, opted, jid = asyncio.run(main())
    for events in (plain, opted):
        logs = [e["data"] for e in events if e["event"] == "log"]
        assert logs and all(x["job_id"] == jid and isinstance(x["line"], str) for x in logs)
        assert any("pause" in x["line"].lower() for x in logs)
    assert not [e for e in plain if e["event"] == "serverlog"]
    served = [e["data"]["line"] for e in opted if e["event"] == "serverlog"]
    assert any("server-only-marker-42" in x for x in served), served[-5:]


# 8 ---------------------------------------------------------------------------------------------
def test_project_delete_cancels_held_and_paused(h, api, tmp_path):
    doomed = new_project(api, name="doomed")["id"]
    other = new_project(api, name="other")["id"]
    h.behaviour.append("block")
    a = api_run(api, doomed)
    wait_for(lambda: api_job(api, a)["state"] == "running")
    b = api_run(api, doomed)
    c = api_run(api, other)
    act(api, b, "hold")
    act(api, a, "pause")
    wait_for(lambda: api_job(api, a)["state"] == "paused")
    assert api.delete(f"/api/projects/{doomed}").status_code == 204
    wait_for(lambda: api_job(api, a)["state"] == "cancelled")
    assert api_job(api, b)["state"] == "cancelled"
    assert not (tmp_path / "data" / "projects" / doomed).exists()
    # the queue is not stuck: the other project's job runs to completion and is saved
    wait_for(lambda: api_job(api, c)["state"] == "done")
    wait_for(lambda: api.get(f"/api/projects/{other}").json()["current_run"] == 1)
