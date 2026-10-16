"""ImageAtelier API tests, written against API.md (the contract), with a stub engine."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from conftest import do_run, new_project, submit, wait_job, wait_run, wait_until

RUN_KEYS = {"run", "project_id", "prompt", "mode", "parent_run", "seed", "steps", "turbo", "image_model", "text_encoder",
            "width", "height", "created_at", "gen_s", "load_s", "job_id"}


def runs_dir(tmp_path, pid):
    return tmp_path / "data" / "projects" / pid / "runs"


# ---- 1. project CRUD ------------------------------------------------------------------------
class TestProjects:
    def test_create_default_names(self, client):
        a = new_project(client)
        b = new_project(client)
        assert a["name"].startswith("Untitled ") and b["name"].startswith("Untitled ")
        assert a["name"] != b["name"]
        assert a["name"].split()[-1].isdigit()
        assert {"id", "name", "created_at", "updated_at", "current_run"} <= set(a)
        assert a["current_run"] is None
        assert isinstance(a["created_at"], float)

    def test_create_named(self, client):
        assert new_project(client, name="Robots")["name"] == "Robots"

    def test_get_and_404(self, client):
        p = new_project(client)
        r = client.get(f"/api/projects/{p['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == p["id"] and r.json()["runs"] == []
        assert client.get("/api/projects/nope").status_code == 404

    def test_rename(self, client):
        p = new_project(client)
        r = client.patch(f"/api/projects/{p['id']}", json={"name": "Renamed"})
        assert r.status_code == 200 and r.json()["name"] == "Renamed"
        assert client.get(f"/api/projects/{p['id']}").json()["name"] == "Renamed"

    def test_patch_missing_project_404(self, client):
        assert client.patch("/api/projects/nope", json={"name": "x"}).status_code == 404

    def test_delete(self, client, tmp_path):
        p = new_project(client)
        r = client.delete(f"/api/projects/{p['id']}")
        assert r.status_code == 204
        assert client.get(f"/api/projects/{p['id']}").status_code == 404
        assert not (tmp_path / "data" / "projects" / p["id"]).exists()

    def test_list_sorted_by_updated_desc(self, client):
        a, b, c = (new_project(client, name=n) for n in "abc")
        time.sleep(0.02)
        client.patch(f"/api/projects/{a['id']}", json={"name": "a2"})  # not required to bump; use a run instead
        do_run(client, b["id"])
        lst = client.get("/api/projects").json()
        ts = [p["updated_at"] for p in lst]
        assert ts == sorted(ts, reverse=True)
        assert lst[0]["id"] == b["id"]  # most recently updated by a saved run


# ---- 2-5. runs ------------------------------------------------------------------------------
class TestRuns:
    def test_first_run_is_t2i(self, client, engine):
        p = new_project(client)
        r = submit(client, p["id"], "a cat")
        assert r["mode"] == "t2i" and r["run_reserved"] == 1 and r["job_id"]
        job = engine.submitted[0]
        assert job["mode"] == "t2i"
        assert not job.get("ref_image_path")
        run = wait_run(client, p["id"], 1)
        assert run["mode"] == "t2i" and run["parent_run"] is None

    def test_second_run_edits_current(self, client, engine):
        p = new_project(client)
        do_run(client, p["id"], "a cat")
        r = submit(client, p["id"], "make it blue")
        assert r["mode"] == "edit"
        job = engine.submitted[1]
        assert job["mode"] == "edit"
        assert job["ref_image_path"].endswith("runs/0001.png")
        run = wait_run(client, p["id"], r["run_reserved"])
        assert run["parent_run"] == 1 and run["mode"] == "edit"

    def test_patch_current_run_branches(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        assert client.get(f"/api/projects/{pid}").json()["current_run"] == 2
        r = client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        assert r.status_code == 200 and r.json()["current_run"] == 1
        r3 = do_run(client, pid, "three")
        job = engine.submitted[-1]
        assert job["mode"] == "edit" and job["ref_image_path"].endswith("runs/0001.png")
        run3 = wait_run(client, pid, r3["run_reserved"])
        assert run3["parent_run"] == 1
        assert client.get(f"/api/projects/{pid}").json()["current_run"] == r3["run_reserved"]

    def test_patch_current_run_nonexistent_400(self, client):
        pid = new_project(client)["id"]
        do_run(client, pid)
        assert client.patch(f"/api/projects/{pid}", json={"current_run": 99}).status_code == 400
        assert client.get(f"/api/projects/{pid}").json()["current_run"] == 1

    def test_patch_current_run_on_empty_project_400(self, client):
        pid = new_project(client)["id"]
        assert client.patch(f"/api/projects/{pid}", json={"current_run": 1}).status_code == 400

    def test_done_writes_files_and_advances(self, client, tmp_path):
        pid = new_project(client)["id"]
        before = client.get(f"/api/projects/{pid}").json()["updated_at"]
        time.sleep(0.02)
        r = submit(client, pid, "a cat", seed=42, steps=8, turbo=True, image_model="official-8bit",
                   text_encoder="official-8bit")
        wait_run(client, pid, r["run_reserved"])
        d = runs_dir(tmp_path, pid)
        assert (d / "0001.png").is_file()
        assert (d / "0001.thumb.webp").is_file()
        meta = json.loads((d / "0001.json").read_text())
        assert RUN_KEYS <= set(meta)
        assert meta["prompt"] == "a cat" and meta["mode"] == "t2i" and meta["parent_run"] is None
        assert meta["seed"] == 42 and meta["steps"] == 8 and meta["turbo"] is True
        assert meta["image_model"] == "official-8bit" and meta["text_encoder"] == "official-8bit"
        assert "variant" not in meta
        p = client.get(f"/api/projects/{pid}").json()
        assert p["current_run"] == 1 and p["updated_at"] > before
        assert p["runs"][0]["image_url"] == f"/api/projects/{pid}/runs/1/image"
        assert p["runs"][0]["thumb_url"] == f"/api/projects/{pid}/runs/1/thumb"

    def test_runs_ascending_and_numbering(self, client):
        pid = new_project(client)["id"]
        for i in range(3):
            do_run(client, pid, f"p{i}")
        runs = client.get(f"/api/projects/{pid}").json()["runs"]
        assert [r["run"] for r in runs] == [1, 2, 3]

    def test_image_and_thumb_routes(self, client):
        pid = new_project(client)["id"]
        do_run(client, pid)
        r = client.get(f"/api/projects/{pid}/runs/1/image")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
        t = client.get(f"/api/projects/{pid}/runs/1/thumb")
        assert t.status_code == 200 and t.headers["content-type"] == "image/webp"
        assert client.get(f"/api/projects/{pid}/runs/9/image").status_code == 404
        assert client.get("/api/projects/nope/runs/1/image").status_code == 404


# ---- 6. failed / cancelled ------------------------------------------------------------------
class TestFailures:
    def test_failed_leaves_no_files_and_later_runs_work(self, client, engine, tmp_path):
        pid = new_project(client)["id"]
        engine.outcome.append("fail")
        r = submit(client, pid, "doomed")
        wait_job(engine, r["job_id"])
        assert engine.get(r["job_id"])["state"] == "failed"
        d = runs_dir(tmp_path, pid)
        assert not list(d.glob(f"{r['run_reserved']:04d}*")) if d.exists() else True
        p = client.get(f"/api/projects/{pid}").json()
        assert p["runs"] == [] and p["current_run"] is None
        ok = do_run(client, pid, "fine")
        p = client.get(f"/api/projects/{pid}").json()
        assert p["current_run"] == ok["run_reserved"]
        assert ok["mode"] == "t2i"  # nothing succeeded before, so still text-to-image
        assert p["runs"][0]["parent_run"] is None

    def test_cancelled_leaves_no_files_and_later_runs_work(self, client, engine, tmp_path):
        pid = new_project(client)["id"]
        do_run(client, pid, "base")
        engine.gate.clear()
        r = submit(client, pid, "to cancel")
        assert client.post(f"/api/jobs/{r['job_id']}/cancel").json() == {"cancelled": True}
        wait_job(engine, r["job_id"])
        engine.gate.set()
        d = runs_dir(tmp_path, pid)
        assert not list(d.glob(f"{r['run_reserved']:04d}*"))
        assert client.get(f"/api/projects/{pid}").json()["current_run"] == 1
        ok = do_run(client, pid, "after")
        assert wait_run(client, pid, ok["run_reserved"])["parent_run"] == 1


# ---- 7. gallery -----------------------------------------------------------------------------
class TestGallery:
    def test_gallery(self, client):
        empty = new_project(client, name="empty")
        full = new_project(client, name="full")
        do_run(client, full["id"], "one")
        do_run(client, full["id"], "two")
        by_id = {p["id"]: p for p in client.get("/api/projects").json()}
        assert by_id[empty["id"]]["latest"] is None and by_id[empty["id"]]["run_count"] == 0
        g = by_id[full["id"]]
        assert g["run_count"] == 2 and g["current_run"] == 2
        assert g["latest"]["run"] == 2 or g["latest"]["run"]["run"] == 2
        latest = g["latest"]
        img = client.get(latest["image_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        th = client.get(latest["thumb_url"])
        assert th.status_code == 200 and th.headers["content-type"] == "image/webp"

    def test_latest_follows_current_run(self, client):
        pid = new_project(client)["id"]
        do_run(client, pid)
        do_run(client, pid)
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        g = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
        assert g["latest"]["image_url"].endswith("/runs/1/image")
        assert g["run_count"] == 2


# ---- 8. settings ----------------------------------------------------------------------------
class TestSettings:
    def test_defaults(self, client):
        assert client.get("/api/settings").json() == {
            "image_model": "official-8bit", "text_encoder": "heretic-8bit", "turbo": True, "steps": 6, "low_memory": True,
            "vae_tiling": False, "resolution": 1024, "pe_quant": "8bit", "pe_thinking": True,
            "pe_max_new_tokens": 8192, "pe_language": "english",
            "pe_variant": "official", "ref_detail": "auto", "max_refs": 3}

    @pytest.mark.parametrize("steps,code", [(6, 200), (8, 200), (7, 400), (5, 400), (40, 400)])
    def test_turbo_steps(self, client, steps, code):
        r = client.put("/api/settings", json={"turbo": True, "steps": steps})
        assert r.status_code == code
        if code == 200:
            assert r.json()["steps"] == steps
        else:
            assert client.get("/api/settings").json()["steps"] == 6  # failed PUT changes nothing

    @pytest.mark.parametrize("steps,code", [(2, 200), (40, 200), (100, 200), (1, 400), (101, 400)])
    def test_non_turbo_steps(self, client, steps, code):
        r = client.put("/api/settings", json={"turbo": False, "steps": steps})
        assert r.status_code == code
        if code == 400:
            assert client.get("/api/settings").json()["turbo"] is True

    def test_toggle_turbo_resets_steps(self, client):
        assert client.put("/api/settings", json={"turbo": False}).json()["steps"] == 40
        assert client.put("/api/settings", json={"turbo": True}).json()["steps"] == 6
        client.put("/api/settings", json={"steps": 8})
        assert client.put("/api/settings", json={"turbo": False}).json()["steps"] == 40

    def test_unknown_key_400(self, client):
        r = client.put("/api/settings", json={"bogus": 1})
        assert r.status_code == 400 and "detail" in r.json()

    @pytest.mark.parametrize("body", [{"variant": "8bit"}, {"variant": "4bit"}, {"image_model": "nope"},
                                      {"text_encoder": "nope"}, {"image_model": "heretic-8bit"}])
    def test_bad_parts_400(self, client, body):
        before = client.get("/api/settings").json()
        assert client.put("/api/settings", json=body).status_code == 400
        assert client.get("/api/settings").json() == before

    @pytest.mark.parametrize("te", ["heretic-8bit", "official-8bit"])
    def test_text_encoder_choices(self, client, te):
        r = client.put("/api/settings", json={"text_encoder": te})
        assert r.status_code == 200 and r.json()["text_encoder"] == te and "variant" not in r.json()

    def test_resolution_range(self, client):
        assert client.put("/api/settings", json={"resolution": 512}).status_code == 200
        assert client.put("/api/settings", json={"resolution": 128}).status_code == 400
        assert client.put("/api/settings", json={"resolution": 4096}).status_code == 400

    def test_malformed_json_is_400(self, client):
        r = client.put("/api/settings", content=b"{not json", headers={"content-type": "application/json"})
        assert r.status_code == 400

    def test_settings_feed_run_defaults(self, client, engine):
        client.put("/api/settings", json={"turbo": False})
        pid = new_project(client)["id"]
        r = submit(client, pid)
        job = engine.submitted[-1]
        assert job["turbo"] is False and job["steps"] == 40
        assert job["image_model"] == "official-8bit" and job["text_encoder"] == "heretic-8bit"
        assert "variant" not in job
        wait_run(client, pid, r["run_reserved"])

    def test_turbo_override_uses_mode_default_steps(self, client, engine):
        client.put("/api/settings", json={"turbo": False, "steps": 25})
        pid = new_project(client)["id"]
        r = submit(client, pid, turbo=True)  # differs from settings -> mode default 6
        assert engine.submitted[-1]["steps"] == 6 and engine.submitted[-1]["turbo"] is True
        wait_run(client, pid, r["run_reserved"])


# ---- 9. validation --------------------------------------------------------------------------
class TestSubmitValidation:
    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_empty_prompt_400(self, client, prompt):
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": prompt})
        assert r.status_code == 400 and "detail" in r.json()

    def test_missing_prompt_and_bad_body_400(self, client):
        pid = new_project(client)["id"]
        assert client.post(f"/api/projects/{pid}/runs", json={}).status_code == 400
        r = client.post(f"/api/projects/{pid}/runs", content=b"nope", headers={"content-type": "application/json"})
        assert r.status_code == 400

    def test_bad_steps_400(self, client):
        pid = new_project(client)["id"]
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "steps": 7}).status_code == 400

    def test_image_model_not_downloaded_409(self, client, ready, engine):
        ready["image"]["official-8bit"] = False
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "image_model": "official-8bit"})
        assert r.status_code == 409 and "detail" in r.json()
        assert engine.submitted == []

    def test_default_text_encoder_not_downloaded_409(self, client, ready, engine):
        ready["te"]["heretic-8bit"] = False  # the default encoder
        pid = new_project(client)["id"]
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 409
        assert engine.submitted == []

    def test_chosen_text_encoder_not_installed_409(self, client, ready, engine):
        ready["te"]["official-8bit"] = False  # local encoder, not copied in
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "text_encoder": "official-8bit"})
        assert r.status_code == 409 and "installed" in r.json()["detail"]
        assert engine.submitted == []
        # the default encoder is unaffected
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 202

    @pytest.mark.parametrize("body", [{"image_model": "nope"}, {"text_encoder": "nope"}, {"variant": "8bit"},
                                      {"variant": "4bit"}, {"variant": "nope"}])
    def test_unknown_parts_and_legacy_variant_are_400(self, client, engine, body):
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", **body})
        assert r.status_code == 400 and "detail" in r.json()
        assert engine.submitted == []

    def test_unknown_project_404(self, client):
        assert client.post("/api/projects/nope/runs", json={"prompt": "x"}).status_code == 404

    def test_rejected_submission_leaves_project_runnable(self, client):
        pid = new_project(client)["id"]
        client.post(f"/api/projects/{pid}/runs", json={"prompt": ""})
        r = do_run(client, pid, "ok")
        assert r["mode"] == "t2i"


# ---- 10. pinning ----------------------------------------------------------------------------
class TestPinning:
    def test_back_to_back_on_fresh_project_are_all_t2i(self, client, engine):
        pid = new_project(client)["id"]
        engine.gate.clear()
        rs = [submit(client, pid, f"p{i}") for i in range(3)]
        assert [r["mode"] for r in rs] == ["t2i"] * 3
        assert [j["mode"] for j in engine.submitted] == ["t2i"] * 3
        assert all(not j.get("ref_image_path") for j in engine.submitted)
        assert len({r["run_reserved"] for r in rs}) == 3
        engine.gate.set()
        wait_run(client, pid, rs[-1]["run_reserved"])

    def test_queued_edit_pins_current_run_at_submit(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "base")  # run 1
        engine.gate.clear()
        a = submit(client, pid, "edit A")  # pinned to run 1
        b = submit(client, pid, "edit B")  # run A not finished: must also edit run 1, not A
        assert a["mode"] == b["mode"] == "edit"
        assert engine.submitted[1]["ref_image_path"].endswith("runs/0001.png")
        assert engine.submitted[2]["ref_image_path"].endswith("runs/0001.png")
        # moving current_run afterwards must not change what is already queued
        engine.gate.set()
        wait_run(client, pid, b["run_reserved"])
        runs = {r["run"]: r for r in client.get(f"/api/projects/{pid}").json()["runs"]}
        assert runs[a["run_reserved"]]["parent_run"] == 1
        assert runs[b["run_reserved"]]["parent_run"] == 1

    def test_patch_after_submit_does_not_repin(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        engine.gate.clear()
        r = submit(client, pid, "edit")  # pinned to run 2
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        engine.gate.set()
        assert wait_run(client, pid, r["run_reserved"])["parent_run"] == 2
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0002.png")


# ---- 12. jobs / cancel ----------------------------------------------------------------------
class TestJobs:
    def test_cancel_queued(self, client, engine):
        pid = new_project(client)["id"]
        engine.gate.clear()
        r = submit(client, pid)
        assert engine.get(r["job_id"])["state"] == "queued"  # precondition: the gate really held it
        assert client.post(f"/api/jobs/{r['job_id']}/cancel").json() == {"cancelled": True}
        assert engine.get(r["job_id"])["state"] == "cancelled"  # instantly, at the engine
        jobs = client.get("/api/jobs").json()
        assert next(j for j in jobs if j["id"] == r["job_id"])["state"] == "cancelled"
        engine.gate.set()

    def test_cancel_queued_never_reported_running(self, client, engine, monkeypatch):
        from imageatelier.store import Store

        orig = Store.discard_run

        def slow(self, *a, **kw):  # widens the app's settle window deterministically
            time.sleep(0.5)
            return orig(self, *a, **kw)

        monkeypatch.setattr(Store, "discard_run", slow)
        pid = new_project(client)["id"]
        engine.gate.clear()
        r = submit(client, pid)
        assert client.post(f"/api/jobs/{r['job_id']}/cancel").json() == {"cancelled": True}
        snap = next(j for j in client.get("/api/jobs").json() if j["id"] == r["job_id"])
        assert snap["state"] == "cancelled"  # a job that never started must not be reported as running
        engine.gate.set()

    def test_cancel_unknown(self, client):
        r = client.post("/api/jobs/doesnotexist/cancel")
        # contract: {cancelled: bool}
        assert r.status_code == 200 and r.json() == {"cancelled": False}

    def test_cancel_finished(self, client):
        pid = new_project(client)["id"]
        r = do_run(client, pid)
        assert client.post(f"/api/jobs/{r['job_id']}/cancel").json() == {"cancelled": False}

    def test_jobs_lists_submitted(self, client, engine):
        pid = new_project(client)["id"]
        r = do_run(client, pid)
        assert r["job_id"] in [j["id"] for j in client.get("/api/jobs").json()]

    def test_delete_project_cancels_queued(self, client, engine, tmp_path):
        pid = new_project(client)["id"]
        engine.gate.clear()
        r = submit(client, pid)
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert engine.get(r["job_id"])["state"] == "cancelled"
        engine.gate.set()
        time.sleep(0.2)
        assert not (tmp_path / "data" / "projects" / pid).exists()


# ---- models ---------------------------------------------------------------------------------
class TestModels:
    def test_list_and_download(self, client, ready):
        lst = client.get("/api/models").json()
        images = [m for m in lst if m["kind"] == "image"]
        tes = [m for m in lst if m["kind"] == "text_encoder"]
        assert [m["name"] for m in images] == ["official-8bit"]
        assert {m["name"]: m["source"] for m in tes} == {"heretic-8bit": "hf", "official-8bit": "local"}
        assert all({"name", "label", "approx_gb", "state"} <= set(m) for m in images + tes)
        assert "turbo_lora_ready" in images[0]
        assert [m["kind"] for m in lst[:3]] == ["image", "text_encoder", "text_encoder"]
        r = client.post("/api/models/image/official-8bit/download")
        assert r.status_code == 202 and r.json() == {"started": True}
        assert ready["started"] == ["official-8bit"]
        r = client.post("/api/models/text_encoder/heretic-8bit/download")
        assert r.status_code == 202 and r.json() == {"started": True}
        assert ready["started_te"] == ["heretic-8bit"]
        assert client.post("/api/models/image/nope/download").status_code == 404
        assert client.post("/api/models/text_encoder/nope/download").status_code == 404
        # the old per-variant endpoint was removed (no alias)
        assert client.post("/api/models/8bit/download").status_code in (404, 405)


# ---- 11. SSE --------------------------------------------------------------------------------
def parse_sse(raw: str):
    out = []
    for block in raw.split("\n\n"):
        ev = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                ev["data"] = json.loads(line[6:])
        if "event" in ev:
            out.append(ev)
    return out


def test_sse_job_and_run_events(app):
    async def main():
        chunks: list[bytes] = []
        head: dict = {}
        started = asyncio.Event()

        async def listen():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                     "scheme": "http", "path": "/api/events", "raw_path": b"/api/events", "query_string": b"",
                     "headers": [], "server": ("test", 80), "client": ("test", 1)}

            async def receive():
                await asyncio.Event().wait()  # never disconnects on its own

            async def send(msg):
                if msg["type"] == "http.response.start":
                    head["status"] = msg["status"]
                    head["headers"] = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
                    started.set()
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            await app(scope, receive, send)

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(listen())
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.2)  # let the subscriber register
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                pid = (await c.post("/api/projects", json={})).json()["id"]
                r = await c.post(f"/api/projects/{pid}/runs", json={"prompt": "a cat"})
                assert r.status_code == 202
                n = r.json()["run_reserved"]
                deadline = time.time() + 5
                events = []
                while time.time() < deadline:
                    events = parse_sse(b"".join(chunks).decode())
                    if any(e["event"] == "run" for e in events) and any(
                            e["event"] == "job" and e["data"]["state"] == "done" for e in events):
                        break
                    await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return head, events, pid, n

    head, events, pid, n = asyncio.run(main())
    assert head["status"] == 200
    assert head["headers"]["content-type"].startswith("text/event-stream")
    jobs = [e["data"] for e in events if e["event"] == "job"]
    runs = [e["data"] for e in events if e["event"] == "run"]
    assert jobs and runs
    assert all(j["project_id"] == pid and j["run"] == n for j in jobs)
    assert {"state", "step", "total_steps"} <= set(jobs[0])
    assert "running" in {j["state"] for j in jobs} and jobs[-1]["state"] == "done"
    assert runs[0]["project_id"] == pid and runs[0]["run"]["run"] == n
    assert runs[0]["run"]["image_url"] == f"/api/projects/{pid}/runs/{n}/image"
