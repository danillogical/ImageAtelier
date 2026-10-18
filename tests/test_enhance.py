"""Prompt enhancer + one-model-at-a-time tests, written against API.md."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from conftest import do_run, new_project, patch_parts, submit, wait_run, wait_until
from test_api import parse_sse


def enhance(client, pid, prompt="a cat", **extra):
    return client.post(f"/api/projects/{pid}/enhance", json={"prompt": prompt, **extra})


def wait_job_state(client, job_id, states=("done", "failed", "cancelled")):
    def find():
        j = next((j for j in client.get("/api/jobs").json() if j["id"] == job_id), None)
        return j if j and j["state"] in states else None
    return wait_until(find)


# ---- modes and pinning ----------------------------------------------------------------------
class TestEnhanceModes:
    def test_empty_project_is_t2i_without_ref(self, client, engine):
        pid = new_project(client)["id"]
        r = enhance(client, pid)
        assert r.status_code == 202
        body = r.json()
        assert body["mode"] == "t2i" and body["job_id"]
        job = engine.submitted[-1]
        assert job["kind"] == "enhance" and job["mode"] == "t2i"
        assert not job.get("ref_image_path")

    def test_project_with_run_is_i2i_with_current_ref(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        r = enhance(client, pid)
        assert r.json()["mode"] == "i2i"
        job = engine.submitted[-1]
        assert job["mode"] == "i2i" and job["ref_image_path"].endswith("runs/0002.png")

    def test_patch_current_run_changes_ref(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        enhance(client, pid)
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0001.png")

    def test_ref_pinned_at_submit(self, client, engine):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        engine.gate.clear()
        jid = enhance(client, pid).json()["job_id"]
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        snap = next(j for j in client.get("/api/jobs").json() if j["id"] == jid)
        assert snap["ref_image_path"].endswith("runs/0002.png")
        engine.gate.set()
        done = wait_job_state(client, jid)
        assert done["ref_image_path"].endswith("runs/0002.png")

    def test_unknown_project_404(self, client):
        assert enhance(client, "nope").status_code == 404


# ---- enhance is not a run -------------------------------------------------------------------
class TestEnhanceSavesNothing:
    def test_no_run_saved_and_job_listed(self, client, engine, tmp_path):
        pid = new_project(client)["id"]
        do_run(client, pid, "one")
        before = client.get(f"/api/projects/{pid}").json()
        jid = enhance(client, pid, "make it blue").json()["job_id"]
        snap = wait_job_state(client, jid)
        assert snap["state"] == "done" and snap["kind"] == "enhance"
        assert snap["project_id"] == pid and snap["run"] is None
        res = snap["result"]
        assert res["rewritten_prompt"] == "ENHANCED: make it blue" and res["fell_back"] is False
        time.sleep(0.2)
        after = client.get(f"/api/projects/{pid}").json()
        assert after["current_run"] == before["current_run"] == 1
        assert [r["run"] for r in after["runs"]] == [1]
        assert after["updated_at"] == before["updated_at"]
        g = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
        assert g["run_count"] == 1
        assert len(list((tmp_path / "data" / "projects" / pid / "runs").glob("*.json"))) == 1

    def test_fell_back_result(self, client, engine):
        engine.enhance_fell_back = True
        pid = new_project(client)["id"]
        snap = wait_job_state(client, enhance(client, pid, "x").json()["job_id"])
        assert snap["result"]["fell_back"] is True and snap["result"]["rewritten_prompt"] == "x"

    def test_generate_jobs_have_kind_generate(self, client):
        pid = new_project(client)["id"]
        r = do_run(client, pid)
        snap = next(j for j in client.get("/api/jobs").json() if j["id"] == r["job_id"])
        assert snap["kind"] == "generate"

    def test_cancel_queued_enhance(self, client, engine):
        pid = new_project(client)["id"]
        engine.gate.clear()
        jid = enhance(client, pid).json()["job_id"]
        assert client.post(f"/api/jobs/{jid}/cancel").json() == {"cancelled": True}
        assert wait_job_state(client, jid)["state"] == "cancelled"
        engine.gate.set()

    def test_run_still_works_after_enhance(self, client):
        pid = new_project(client)["id"]
        wait_job_state(client, enhance(client, pid).json()["job_id"])
        r = do_run(client, pid, "go")
        assert r["mode"] == "t2i"


# ---- options from settings / body -----------------------------------------------------------
class TestEnhanceOptions:
    def test_defaults_come_from_settings(self, client, engine):
        pid = new_project(client)["id"]
        enhance(client, pid)
        job = engine.submitted[-1]
        assert job["thinking"] is True and job["quant"] == "8bit"
        assert job["language"] == "english" and job["max_new_tokens"] == 8192

    def test_settings_changes_are_used(self, client, engine):
        client.put("/api/settings", json={"pe_thinking": False,
                                          "pe_language": "auto", "pe_max_new_tokens": 512})
        pid = new_project(client)["id"]
        enhance(client, pid)
        job = engine.submitted[-1]
        assert job["thinking"] is False and job["quant"] == "8bit"
        assert job["language"] == "auto" and job["max_new_tokens"] == 512

    def test_body_overrides_settings(self, client, engine):
        pid = new_project(client)["id"]
        r = enhance(client, pid, thinking=False, language="auto")
        assert r.status_code == 202
        job = engine.submitted[-1]
        assert job["thinking"] is False and job["quant"] == "8bit" and job["language"] == "auto"
        assert client.get("/api/settings").json()["pe_thinking"] is True  # override is per-request

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_empty_prompt_400(self, client, prompt):
        pid = new_project(client)["id"]
        r = enhance(client, pid, prompt)
        assert r.status_code == 400 and "detail" in r.json()

    @pytest.mark.parametrize("extra", [{"quant": "16bit"}, {"quant": "4bit"}, {"language": "french"}, {"thinking": "maybe"}])
    def test_bad_options_400(self, client, engine, extra):
        pid = new_project(client)["id"]
        assert enhance(client, pid, **extra).status_code == 400
        assert engine.submitted == []

    def test_malformed_json_400(self, client):
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/enhance", content=b"nope", headers={"content-type": "application/json"})
        assert r.status_code == 400

    def test_pe_not_downloaded_409_names_it(self, client, engine, pe_ready):
        pe_ready["t2i-8bit"] = False
        pid = new_project(client)["id"]
        r = enhance(client, pid)
        assert r.status_code == 409 and "suite" in r.json()["detail"]
        assert engine.submitted == []

    def test_either_missing_enhancer_blocks_the_whole_suite(self, client, pe_ready):
        pid = new_project(client)["id"]
        pe_ready["i2i-8bit"] = False
        assert enhance(client, pid).status_code == 409
        pe_ready["i2i-8bit"] = True
        assert enhance(client, pid).status_code == 202


# ---- settings -------------------------------------------------------------------------------
class TestPeSettings:
    def test_defaults(self, client):
        s = client.get("/api/settings").json()
        assert (s["pe_thinking"], s["pe_quant"], s["pe_max_new_tokens"], s["pe_language"]) == (
            True, "8bit", 8192, "english")

    @pytest.mark.parametrize("key,val", [("pe_language", "auto"),
                                         ("pe_language", "english"), ("pe_thinking", False),
                                         ("pe_max_new_tokens", 256), ("pe_max_new_tokens", 16384)])
    def test_valid(self, client, key, val):
        r = client.put("/api/settings", json={key: val})
        assert r.status_code == 200 and r.json()[key] == val
        assert client.get("/api/settings").json()[key] == val

    @pytest.mark.parametrize("key,val", [("pe_quant", "4bit"), ("pe_quant", "16bit"), ("pe_quant", 8), ("pe_language", "french"),
                                         ("pe_thinking", "yes"), ("pe_max_new_tokens", 255),
                                         ("pe_max_new_tokens", 16385), ("pe_max_new_tokens", "lots")])
    def test_invalid_changes_nothing(self, client, key, val):
        before = client.get("/api/settings").json()
        r = client.put("/api/settings", json={key: val})
        assert r.status_code == 400 and "detail" in r.json()
        assert client.get("/api/settings").json() == before

    def test_pe_keys_dont_disturb_turbo_rules(self, client):
        client.put("/api/settings", json={"pe_language": "auto"})
        s = client.get("/api/settings").json()
        assert s["turbo"] is False and s["steps"] == 40


# ---- models list ----------------------------------------------------------------------------
class TestModelsList:
    def test_only_suites_are_listed(self, client):
        rows = client.get("/api/models").json()
        assert [m["name"] for m in rows] == ["bf16", "q8", "q6", "q4"]
        assert all(m["kind"] == "suite" for m in rows)

    def test_enhancer_readiness_applies_to_whole_suite(self, client, pe_ready):
        pe_ready["i2i-8bit"] = False
        rows = {m["name"]: m for m in client.get("/api/models").json()}
        assert rows["q8"]["state"] != "ready" and rows["q6"]["state"] == "ready"

    def test_independent_enhancer_download_retired(self, client):
        assert client.post("/api/models/pe/t2i-8bit/download").status_code == 410


# ---- enhanced_from --------------------------------------------------------------------------
class TestEnhancedFrom:
    def test_persisted(self, client):
        pid = new_project(client)["id"]
        r = submit(client, pid, "rewritten text", enhanced_from="my original")
        run = wait_run(client, pid, r["run_reserved"])
        assert run["enhanced_from"] == "my original" and run["prompt"] == "rewritten text"
        again = client.get(f"/api/projects/{pid}").json()["runs"][0]
        assert again["enhanced_from"] == "my original"

    def test_null_when_absent(self, client, tmp_path):
        import json
        pid = new_project(client)["id"]
        r = do_run(client, pid, "plain")
        run = wait_run(client, pid, r["run_reserved"])
        assert "enhanced_from" in run and run["enhanced_from"] is None
        meta = json.loads((tmp_path / "data" / "projects" / pid / "runs" / "0001.json").read_text())
        assert meta["enhanced_from"] is None


# ---- SSE ------------------------------------------------------------------------------------
def collect_events(app, action, done, timeout=8.0):
    """Connect to /api/events, run `action(client)`, collect events until done(events) or timeout."""
    async def main():
        chunks: list[bytes] = []
        started = asyncio.Event()

        async def listen():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                     "scheme": "http", "path": "/api/events", "raw_path": b"/api/events", "query_string": b"",
                     "headers": [], "server": ("test", 80), "client": ("test", 1)}

            async def receive():
                await asyncio.Event().wait()

            async def send(msg):
                if msg["type"] == "http.response.start":
                    started.set()
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            await app(scope, receive, send)

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(listen())
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.2)
            events = []
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                ctx = await action(c)
                end = time.time() + timeout
                while time.time() < end:
                    events = parse_sse(b"".join(chunks).decode())
                    if done(events, ctx):
                        break
                    await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return events, ctx

    return asyncio.run(main())


class TestEnhanceSse:
    def test_first_event_is_model_with_resident_state(self, app, engine):
        engine._resident = {"key": ["pe_t2i", "8bit"], "state": "loaded"}

        async def action(c):
            return None

        events, _ = collect_events(app, action, lambda ev, _: len(ev) >= 1, timeout=2)
        assert events and events[0]["event"] == "model"
        assert events[0]["data"]["state"] == "loaded" and events[0]["data"]["key"] == ["pe_t2i", "8bit"]

    def test_first_event_unloaded_when_nothing_resident(self, app):
        async def action(c):
            return None

        events, _ = collect_events(app, action, lambda ev, _: len(ev) >= 1, timeout=2)
        assert events[0]["event"] == "model"
        assert events[0]["data"]["state"] == "unloaded" and events[0]["data"]["key"] is None

    def test_get_api_model_matches(self, client, engine):
        assert client.get("/api/model").json() == {"state": "unloaded", "key": None}
        engine._resident = {"key": ["image", "official-8bit", "official-8bit", False], "state": "loaded"}
        assert client.get("/api/model").json() == {"state": "loaded", "key": ["image", "official-8bit", "official-8bit", False]}

    def test_model_events_pass_through(self, app, engine):
        async def action(c):
            engine.emit_model("loading", ["image", "official-8bit", "official-8bit", False])
            engine.emit_model("loaded", ["image", "official-8bit", "official-8bit", False], load_s=1.5)
            engine.emit_model("unloaded", None, active_bytes=123)

        events, _ = collect_events(app, action, lambda ev, _: len([e for e in ev if e["event"] == "model"]) >= 4)
        m = [e["data"] for e in events if e["event"] == "model"][1:]  # [0] is the on-connect event
        assert [x["state"] for x in m] == ["loading", "loaded", "unloaded"]
        assert m[1]["key"] == ["image", "official-8bit", "official-8bit", False] and m[1]["load_s"] == 1.5
        assert m[2]["active_bytes"] == 123

    def test_enhance_job_events_carry_phase_tokens_text(self, app, engine):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            r = await c.post(f"/api/projects/{pid}/enhance", json={"prompt": "a cat"})
            assert r.status_code == 202
            return pid, r.json()["job_id"]

        def done(ev, ctx):
            return any(e["event"] == "job" and e["data"]["state"] == "done" for e in ev)

        events, (pid, jid) = collect_events(app, action, done)
        jobs = [e["data"] for e in events if e["event"] == "job" and e["data"]["id"] == jid]
        assert jobs and all(j["kind"] == "enhance" and j["project_id"] == pid and j["run"] is None for j in jobs)
        phases = [j["phase"] for j in jobs if j["phase"]]
        assert "thinking" in phases and "writing" in phases
        assert any(j["tokens"] > 0 for j in jobs)
        assert any(j["text"] for j in jobs if j["phase"] == "writing")
        assert jobs[-1]["state"] == "done" and jobs[-1]["result"]["rewritten_prompt"] == "ENHANCED: a cat"
        assert not [e for e in events if e["event"] == "run"]  # enhance never produces a run event


# ---- one-model rule (real Engine + ModelManager, loaders stubbed) ---------------------------
@pytest.fixture
def real_engine(monkeypatch, tmp_path):
    mlx_vlm = pytest.importorskip("mlx_vlm")
    from imageatelier import engine as eng_mod
    from imageatelier import models

    not_ready: set = set()
    loads: list[tuple] = []
    events: list[dict] = []

    def fake_resolve_pe(mode, quant):
        if (mode, quant) in not_ready:
            raise models.ModelNotReady(f"prompt enhancer {mode}-{quant!r} is not downloaded")
        return Path("/nowhere"), "system prompt"

    def fake_load_image(self, key, snapshot, lora):
        loads.append(key)
        self.model = SimpleNamespace(
            text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None), tiling_config=None,
            generate_image=lambda **kw: SimpleNamespace(image=Image.new("RGBA", (16, 16), (1, 2, 3, 255))))
        self.snapshot, self.key = snapshot, key

    def fake_load_pe(self, key, path, system_prompt):
        loads.append(key)
        self.model = object()
        self.processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
        self.system_prompt, self.key = system_prompt, key

    def fake_stream(model, processor, prompt, image=None, max_tokens=0, **kw):
        yield SimpleNamespace(text='{"rewritten_prompt": "better"}', finish_reason="stop", generation_tps=10.0)

    patch_parts(monkeypatch)
    monkeypatch.setattr(models, "resolve_pe", fake_resolve_pe)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", fake_load_image)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_pe", fake_load_pe)
    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)  # skip the real ioreg call per job

    e = eng_mod.Engine(settings_provider=lambda: {"pe_thinking": False, "low_memory": False})
    e.on_event(events.append)
    e.start()
    ref = tmp_path / "ref.png"
    Image.new("RGB", (16, 16)).save(ref)
    yield SimpleNamespace(engine=e, loads=loads, events=events, not_ready=not_ready, ref=str(ref),
                          out=tmp_path / "out.png")
    e.stop(timeout=10)


def _wait(engine, jid):
    end = time.time() + 15
    while time.time() < end:
        s = engine.get(jid)
        if s["state"] in ("done", "failed", "cancelled"):
            return s
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish: {engine.get(jid)}")


class TestOneModelRule:
    def _enh(self, h, mode="t2i", **kw):
        job = {"kind": "enhance", "prompt": "a cat", "mode": mode, "quant": "8bit", **kw}
        if mode == "i2i":
            job["ref_image_path"] = h.ref
        return h.engine.submit(job)

    def _gen(self, h, quant="8bit"):
        return h.engine.submit({"kind": "generate", "prompt": "a cat", "mode": "t2i", "steps": 6, "turbo": False, "image_model": f"official-{quant}", "text_encoder": f"official-{quant}",
                                "output_path": str(h.out)})

    def test_enhance_enhance_generate_enhance(self, real_engine):
        h = real_engine
        ids = [self._enh(h), self._enh(h), self._gen(h), self._enh(h, "i2i")]
        snaps = [_wait(h.engine, i) for i in ids]
        assert [s["state"] for s in snaps] == ["done"] * 4, snaps
        assert h.loads == [("pe_t2i", "8bit"), ("image", "official-8bit", "official-8bit", False), ("pe_i2i", "8bit")]
        loaded = [tuple(e["key"]) for e in h.events if e.get("type") == "model" and e["state"] == "loaded"]
        assert loaded == h.loads
        assert snaps[0]["result"]["rewritten_prompt"] == "better"

    def test_only_one_model_resident_at_a_time(self, real_engine):
        h = real_engine
        for i in [self._enh(h), self._gen(h), self._enh(h, "i2i")]:
            _wait(h.engine, i)
        resident = None
        for e in h.events:
            if e.get("type") != "model":
                continue
            if e["state"] == "loading":
                assert resident is None, f"loading {e['key']} while {resident} still resident"
            elif e["state"] == "loaded":
                resident = tuple(e["key"])
            elif e["state"] == "unloaded":
                resident = None

    def test_different_model_key_reloads(self, real_engine):
        h = real_engine
        _wait(h.engine, self._enh(h))
        _wait(h.engine, self._enh(h, "i2i"))  # other rewriter mode -> other key
        _wait(h.engine, self._gen(h))
        _wait(h.engine, self._gen(h, quant="6bit"))  # a different suite -> another image key
        _wait(h.engine, self._gen(h, quant="6bit"))  # same key again -> no reload
        assert h.loads == [("pe_t2i", "8bit"), ("pe_i2i", "8bit"), ("image", "official-8bit", "official-8bit", False), ("image", "official-6bit", "official-6bit", False)]

    def test_not_downloaded_pe_leaves_current_model_resident(self, real_engine):
        h = real_engine
        _wait(h.engine, self._gen(h))
        assert h.loads == [("image", "official-8bit", "official-8bit", False)]
        n_events = len(h.events)
        h.not_ready.add(("i2i", "8bit"))
        snap = _wait(h.engine, self._enh(h, "i2i"))
        assert snap["state"] == "failed" and "not downloaded" in snap["error"]
        assert h.loads == [("image", "official-8bit", "official-8bit", False)]
        assert h.engine.resident() == {"key": ["image", "official-8bit", "official-8bit", False], "state": "loaded"}
        assert not [e for e in h.events[n_events:] if e.get("type") == "model"]
        # and the image model is still usable without a reload
        _wait(h.engine, self._gen(h))
        assert h.loads == [("image", "official-8bit", "official-8bit", False)]
