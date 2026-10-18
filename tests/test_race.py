"""Regression: a job must only be reported `done` after its run is saved and current_run advanced.

The race is made deterministic by slowing Store.save_run, so the engine finishes (and the stub's
state is `done`) well before the app has saved the run.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from conftest import do_run, new_project, submit, wait_run
from test_api import parse_sse

SAVE_DELAY = 0.6


def slow_save(monkeypatch):
    from imageatelier.store import Store

    orig = Store.save_run

    def slow(self, *a, **kw):
        time.sleep(SAVE_DELAY)
        return orig(self, *a, **kw)

    monkeypatch.setattr(Store, "save_run", slow)


def test_run_posted_right_after_done_edits_that_run(client, engine, monkeypatch):
    slow_save(monkeypatch)
    pid = new_project(client)["id"]
    do_run(client, pid, "one")
    do_run(client, pid, "two")
    j3 = submit(client, pid, "three")
    assert j3["mode"] == "edit" and j3["run_reserved"] == 3

    deadline = time.time() + 10
    while time.time() < deadline:
        jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
        if jobs[j3["job_id"]]["state"] == "done":
            break
        time.sleep(0.005)
    else:
        raise AssertionError("job 3 never reported done")

    r4 = submit(client, pid, "four")  # immediately after observing done
    assert r4["mode"] == "edit"
    job = engine.submitted[-1]
    assert job["mode"] == "edit"
    assert job["ref_image_path"].endswith("runs/0003.png")
    assert wait_run(client, pid, r4["run_reserved"])["parent_run"] == 3


def test_done_implies_saved_and_run_event_precedes_done(app, engine, monkeypatch, tmp_path):
    slow_save(monkeypatch)
    runs_dir = tmp_path / "data" / "projects"

    async def main():
        chunks: list[bytes] = []
        started = asyncio.Event()

        async def listen():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                     "scheme": "http", "path": "/api/events", "raw_path": b"/api/events", "query_string": b"",
                     "headers": [], "server": ("test", 80), "client": ("127.0.0.1", 1)}

            async def receive():
                await asyncio.Event().wait()

            async def send(msg):
                if msg["type"] == "http.response.start":
                    started.set()
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            await app(scope, receive, send)

        violations = []
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(listen())
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                pid = (await c.post("/api/projects", json={})).json()["id"]
                r = (await c.post(f"/api/projects/{pid}/runs", json={"prompt": "a cat"})).json()
                n = r["run_reserved"]
                deadline = time.time() + 10
                saw_done = False
                while time.time() < deadline:
                    jobs = (await c.get("/api/jobs")).json()
                    job = next(j for j in jobs if j["id"] == r["job_id"])
                    if job["state"] == "done":
                        saw_done = True
                        if not (runs_dir / pid / "runs" / f"{n:04d}.json").is_file():
                            violations.append("done in /api/jobs but run json not on disk")
                        break
                    await asyncio.sleep(0.005)
                assert saw_done
                await asyncio.sleep(SAVE_DELAY + 0.5)  # let the SSE stream catch up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return parse_sse(b"".join(chunks).decode()), violations

    events, violations = asyncio.run(main())
    assert not violations, violations
    kinds = [(e["event"], e["data"].get("state")) for e in events if e["event"] in ("job", "run")]
    run_idx = next(i for i, k in enumerate(kinds) if k[0] == "run")
    done_idx = next(i for i, k in enumerate(kinds) if k == ("job", "done"))
    assert run_idx < done_idx, kinds
