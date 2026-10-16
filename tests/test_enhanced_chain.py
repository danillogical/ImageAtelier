"""Persisted enhanced prompt, use_enhanced, Enhance & Generate chains, and the startup orphan sweep (API.md)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imageatelier.app import create_app
from imageatelier.store import Store

from conftest import StubEngine, do_run, new_project, submit, wait_run, wait_until
from test_enhance import collect_events
from test_refs import add_ok as add_ref, colour, refs as list_refs
from test_transparent import half_transparent, wrapped
from test_upload import BLUE, GREEN, RED


@pytest.fixture
def pid(client):
    return new_project(client)["id"]


def project(client, pid):
    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    return r.json()


def job_snap(client, jid):
    return next((j for j in client.get("/api/jobs").json() if j["id"] == jid), None)


def enhance(client, pid, prompt="a cat", **extra):
    r = client.post(f"/api/projects/{pid}/enhance", json={"prompt": prompt, **extra})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def enhance_done(client, pid, prompt="a cat", **extra):
    jid = enhance(client, pid, prompt, **extra)
    wait_until(lambda: (job_snap(client, jid) or {}).get("state") in ("done", "failed", "cancelled"))
    return jid


def chain(client, pid, prompt="a cat", **extra):
    r = client.post(f"/api/projects/{pid}/enhance-and-generate", json={"prompt": prompt, **extra})
    assert r.status_code == 202, r.text
    return r.json()


def runs_dir(tmp_path, pid):
    return tmp_path / "data" / "projects" / pid / "runs"


def run_of(client, pid, n):
    return wait_until(lambda: next((r for r in project(client, pid)["runs"] if r["run"] == n), None), timeout=10)


def settled(client, jid, states=("done", "failed", "cancelled")):
    wait_until(lambda: (job_snap(client, jid) or {}).get("state") in states)
    return job_snap(client, jid)


def kind(job):
    return job.get("kind", "generate")  # run jobs carry no explicit kind


# ---- 1. the persisted enhanced prompt --------------------------------------------------------------
class TestEnhancedStore:
    def test_null_initially(self, client, pid):
        assert project(client, pid)["enhanced"] is None

    def test_set_on_done(self, client, pid):
        jid = enhance_done(client, pid, "a cat")
        e = project(client, pid)["enhanced"]
        assert e["text"] == "ENHANCED: a cat" and e["original"] == "a cat"
        assert e["mode"] == "t2i" and e["fell_back"] is False and e["job_id"] == jid
        assert e["rewriter"] == "official-8bit" and isinstance(e["created_at"], float)
        assert set(e) >= {"text", "original", "mode", "rewriter", "fell_back", "job_id", "created_at"}

    def test_mode_is_i2i_in_a_project_with_a_run(self, client, pid):
        do_run(client, pid, "base")
        enhance_done(client, pid, "make it blue")
        assert project(client, pid)["enhanced"]["mode"] == "i2i"

    def test_newer_done_replaces(self, client, pid):
        enhance_done(client, pid, "first")
        j2 = enhance_done(client, pid, "second")
        e = project(client, pid)["enhanced"]
        assert e["original"] == "second" and e["text"] == "ENHANCED: second" and e["job_id"] == j2

    def test_fell_back_stores_the_original(self, client, pid, engine):
        engine.enhance_fell_back = True
        enhance_done(client, pid, "a cat")
        e = project(client, pid)["enhanced"]
        assert e["text"] == "a cat" and e["original"] == "a cat" and e["fell_back"] is True

    def test_cancelled_enhance_leaves_previous_untouched(self, client, pid, engine):
        enhance_done(client, pid, "first")
        before = project(client, pid)["enhanced"]
        engine.gate.clear()
        jid = enhance(client, pid, "second")
        client.post(f"/api/jobs/{jid}/cancel")
        settled(client, jid)
        engine.gate.set()
        assert project(client, pid)["enhanced"] == before

    def test_running_cancel_leaves_previous_untouched(self, client, pid, engine):
        enhance_done(client, pid, "first")
        before = project(client, pid)["enhanced"]
        engine.release.clear()
        jid = enhance(client, pid, "second")
        wait_until(lambda: engine.get(jid)["state"] == "running")
        client.post(f"/api/jobs/{jid}/cancel")
        engine.release.set()
        assert settled(client, jid)["state"] == "cancelled"
        assert project(client, pid)["enhanced"] == before

    def test_failed_enhance_leaves_previous_untouched(self, client, pid, engine):
        enhance_done(client, pid, "first")
        before = project(client, pid)["enhanced"]
        engine.outcome.append("fail")
        jid = enhance(client, pid, "second")
        assert settled(client, jid)["state"] == "failed"
        assert project(client, pid)["enhanced"] == before

    def test_failed_or_cancelled_with_nothing_stored_stays_null(self, client, pid, engine):
        engine.outcome.append("fail")
        assert settled(client, enhance(client, pid))["state"] == "failed"
        assert project(client, pid)["enhanced"] is None

    def test_does_not_bump_updated_at(self, client, pid):
        before = project(client, pid)["updated_at"]
        time.sleep(0.05)
        enhance_done(client, pid, "a cat")
        p = project(client, pid)
        assert p["enhanced"] is not None and p["updated_at"] == before
        order = [x["id"] for x in client.get("/api/projects").json()]
        other = new_project(client)["id"]
        assert [x["id"] for x in client.get("/api/projects").json()][0] == other and pid in order

    def test_is_per_project(self, client, pid):
        other = new_project(client)["id"]
        enhance_done(client, pid)
        assert project(client, other)["enhanced"] is None

    def test_persists_across_app_recreation(self, client, pid, ready, pe_ready, tmp_path):
        enhance_done(client, pid, "a cat")
        want = project(client, pid)["enhanced"]
        with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "data")) as c2:
            assert project(c2, pid)["enhanced"] == want

    def test_sse_enhanced_event_precedes_the_terminal_job_event(self, app):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            r = await c.post(f"/api/projects/{pid}/enhance", json={"prompt": "a cat"})
            return pid, r.json()["job_id"]

        def done(ev, ctx):
            return any(e["event"] == "job" and e["data"]["id"] == ctx[1] and e["data"]["state"] == "done" for e in ev)

        events, (pid, jid) = collect_events(app, action, done)
        kinds = [(e["event"], e["data"].get("state") if e["event"] == "job" else None) for e in events]
        enh = next(i for i, e in enumerate(events) if e["event"] == "enhanced")
        term = next(i for i, e in enumerate(events) if e["event"] == "job" and e["data"]["id"] == jid
                    and e["data"]["state"] == "done")
        assert enh < term, kinds
        d = events[enh]["data"]
        assert d["project_id"] == pid and d["enhanced"]["original"] == "a cat" and d["enhanced"]["job_id"] == jid

    def test_jobs_report_running_until_the_enhanced_prompt_is_stored(self, client, pid, engine, monkeypatch):
        orig = Store.update_project

        def slow(self, *a, **kw):
            time.sleep(0.6)
            return orig(self, *a, **kw)

        monkeypatch.setattr(Store, "update_project", slow)
        jid = enhance(client, pid, "a cat")
        wait_until(lambda: engine.get(jid)["state"] == "done")
        snap = job_snap(client, jid)
        assert snap["state"] == "running", snap  # the engine is done but the prompt is not stored yet
        seen = []
        while True:  # whenever the API says done, the enhanced prompt is already there
            s = job_snap(client, jid)["state"]
            if s == "done":
                assert project(client, pid)["enhanced"]["job_id"] == jid
                break
            seen.append(s)
            time.sleep(0.02)
            assert len(seen) < 500

    def test_a_run_submitted_right_after_an_enhance_sees_it(self, client, pid, engine, monkeypatch):
        orig = Store.update_project

        def slow(self, *a, **kw):
            time.sleep(0.6)
            return orig(self, *a, **kw)

        monkeypatch.setattr(Store, "update_project", slow)
        jid = enhance(client, pid, "a cat")
        wait_until(lambda: engine.get(jid)["state"] == "done")  # engine done; the store write is still pending
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "ENHANCED: a cat", "use_enhanced": True})
        assert r.status_code == 202, r.text
        run = run_of(client, pid, r.json()["run_reserved"])
        assert run["enhanced_from"] == "a cat"  # it saw the stored value, so the default applied
        assert project(client, pid)["enhanced"] is None  # ...and consumed it


# ---- 2. discard and use_enhanced ------------------------------------------------------------------------
class TestDiscardAndUse:
    def test_delete_clears_with_and_without_a_value(self, client, pid):
        enhance_done(client, pid)
        assert client.delete(f"/api/projects/{pid}/enhanced").status_code == 204
        assert project(client, pid)["enhanced"] is None
        assert client.delete(f"/api/projects/{pid}/enhanced").status_code == 204
        assert client.delete("/api/projects/nope/enhanced").status_code == 404

    def test_delete_sends_a_null_event(self, app):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            jid = (await c.post(f"/api/projects/{pid}/enhance", json={"prompt": "x"})).json()["job_id"]
            for _ in range(200):
                if (await c.get(f"/api/projects/{pid}")).json()["enhanced"]:
                    break
                import asyncio
                await asyncio.sleep(0.02)
            await c.delete(f"/api/projects/{pid}/enhanced")
            return pid

        events, pid = collect_events(app, action, lambda ev, _: len([e for e in ev if e["event"] == "enhanced"]) >= 2)
        got = [e["data"] for e in events if e["event"] == "enhanced"]
        assert got[0]["enhanced"] is not None and got[-1] == {"project_id": pid, "enhanced": None}

    def test_use_enhanced_clears_and_defaults_enhanced_from(self, client, pid, engine):
        enhance_done(client, pid, "a cat")
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "ENHANCED: a cat", "use_enhanced": True})
        assert r.status_code == 202
        assert project(client, pid)["enhanced"] is None  # cleared once accepted
        run = run_of(client, pid, r.json()["run_reserved"])
        assert run["enhanced_from"] == "a cat" and run["prompt"] == "ENHANCED: a cat"

    def test_explicit_enhanced_from_wins(self, client, pid):
        enhance_done(client, pid, "a cat")
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "use_enhanced": True, "enhanced_from": "mine"})
        assert run_of(client, pid, r.json()["run_reserved"])["enhanced_from"] == "mine"
        assert project(client, pid)["enhanced"] is None

    def test_without_use_enhanced_the_value_stays(self, client, pid):
        enhance_done(client, pid, "a cat")
        for extra in ({}, {"use_enhanced": False}):
            r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "typed", **extra})
            run = run_of(client, pid, r.json()["run_reserved"])
            assert run["enhanced_from"] is None
            assert project(client, pid)["enhanced"]["original"] == "a cat"

    def test_use_enhanced_with_nothing_stored_is_harmless(self, client, pid):
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "use_enhanced": True})
        assert r.status_code == 202
        assert run_of(client, pid, r.json()["run_reserved"])["enhanced_from"] is None

    @pytest.mark.parametrize("why", ["part_missing", "empty_prompt", "bad_steps", "bad_flag"])
    def test_rejected_run_clears_nothing(self, client, pid, ready, why):
        enhance_done(client, pid, "a cat")
        before = project(client, pid)["enhanced"]
        body = {"prompt": "x", "use_enhanced": True}
        if why == "part_missing":
            ready["te"]["heretic-8bit"] = False
            want = 409
        elif why == "empty_prompt":
            body["prompt"] = ""
            want = 400
        elif why == "bad_steps":
            body["steps"] = 7
            want = 400
        else:
            body["transparent"] = "yes"
            want = 400
        assert client.post(f"/api/projects/{pid}/runs", json=body).status_code == want
        assert project(client, pid)["enhanced"] == before
        ready["te"]["heretic-8bit"] = True
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "use_enhanced": True}).status_code == 202
        assert project(client, pid)["enhanced"] is None

    def test_submit_error_in_the_engine_clears_nothing(self, client, pid, engine):
        enhance_done(client, pid, "a cat")

        def refuse(job):
            raise ValueError("refused")
        engine.reject_submit = refuse
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "use_enhanced": True})
        assert r.status_code == 400
        assert project(client, pid)["enhanced"] is not None


# ---- 3. Enhance & Generate ---------------------------------------------------------------------------------
def events_for_chain(app, engine, prompt="a cat", setup=None, finish=None, **extra):
    """Run a chain under an SSE listener; returns (events, response json, pid)."""
    async def action(c):
        pid = (await c.post("/api/projects", json={})).json()["id"]
        if setup:
            await setup(c, pid)
        r = await c.post(f"/api/projects/{pid}/enhance-and-generate", json={"prompt": prompt, **extra})
        assert r.status_code == 202, r.text
        if finish:
            await finish(c, pid, r.json())
        return pid, r.json()

    def done(ev, ctx):
        return any(e["event"] == "chain" for e in ev) and (
            any(e["event"] == "run" for e in ev) or any(e["event"] == "chain" and e["data"]["state"] != "generating" for e in ev))

    events, (pid, body) = collect_events(app, action, done)
    return events, body, pid


class TestChain:
    def test_response_shape(self, client, pid):
        body = chain(client, pid)
        assert body["mode"] == "t2i" and body["run_reserved"] == 1
        assert body["chain_id"] and body["enhance_job_id"]

    def test_generate_is_submitted_only_after_the_enhance_is_done(self, client, pid, engine):
        seen = []  # engine-side view at the instant each job is submitted (no sleeps, no polling windows)

        def record(job):
            enh = next((j for j in engine.jobs() if j["kind"] == "enhance"), None)
            seen.append((kind(job), enh["state"] if enh else None))

        engine.reject_submit = record  # observes only; never raises
        engine.release.clear()  # holds the enhance in "running" until we let it go
        body = chain(client, pid, "a cat")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        assert [kind(j) for j in engine.submitted] == ["enhance"]  # nothing generates while the enhance is held
        assert project(client, pid)["runs"] == []
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        wait_until(lambda: project(client, pid)["current_run"] == body["run_reserved"])
        assert [kind(j) for j in engine.submitted] == ["enhance", "generate"]
        # when the generate was submitted the enhance was already finished
        assert seen == [("enhance", None), ("generate", "done")], seen
        assert engine.submitted[1]["prompt"] == "ENHANCED: a cat"
        assert run["prompt"] == "ENHANCED: a cat" and run["enhanced_from"] == "a cat"
        assert run["run"] == body["run_reserved"] and run["mode"] == "t2i"

    def test_the_rewrite_is_what_the_model_gets(self, client, pid, engine):
        engine.enhance_text = lambda p: "A long, vivid description of " + p
        body = chain(client, pid, "a cat")
        run = run_of(client, pid, body["run_reserved"])
        assert run["prompt"] == "A long, vivid description of a cat" and run["enhanced_from"] == "a cat"
        assert engine.submitted[1]["prompt"] == "A long, vivid description of a cat"

    def test_transparent_flag_is_pinned_and_wraps_the_rewrite(self, client, pid, engine):
        body = chain(client, pid, "a cat", transparent=True)
        run = run_of(client, pid, body["run_reserved"])
        assert run["transparent"] is True and run["prompt"] == "ENHANCED: a cat"
        assert engine.submitted[1]["prompt"] == wrapped("ENHANCED: a cat") and run["model_prompt"] == wrapped("ENHANCED: a cat")

    def test_seed_option_reaches_the_generate_job(self, client, pid, engine):
        body = chain(client, pid, "a cat", seed=1234)
        assert run_of(client, pid, body["run_reserved"])["seed"] == 1234
        assert engine.submitted[1]["seed"] == 1234

    def test_enhance_options_reach_the_enhance_job(self, client, pid, engine):
        chain(client, pid, "a cat", thinking=False, language="auto")
        j = engine.submitted[0]
        assert kind(j) == "enhance" and j["thinking"] is False and j["language"] == "auto"

    def test_fallback_uses_the_original(self, client, pid, engine):
        engine.enhance_fell_back = True
        body = chain(client, pid, "a cat")
        run = run_of(client, pid, body["run_reserved"])
        assert run["prompt"] == "a cat" and run["enhanced_from"] == "a cat"
        assert engine.submitted[1]["prompt"] == "a cat"

    def test_chained_enhance_does_not_set_enhanced(self, client, pid):
        body = chain(client, pid)
        run_of(client, pid, body["run_reserved"])
        assert project(client, pid)["enhanced"] is None

    def test_chained_enhance_leaves_a_stored_value_alone(self, client, pid):
        enhance_done(client, pid, "stored one")
        before = project(client, pid)["enhanced"]
        body = chain(client, pid, "chained")
        run_of(client, pid, body["run_reserved"])
        assert project(client, pid)["enhanced"] == before

    def test_edit_mode_pins_the_parent(self, client, pid, engine):
        do_run(client, pid, "base")
        do_run(client, pid, "second")
        body = chain(client, pid, "make it red")
        assert body["mode"] == "edit" and body["run_reserved"] == 3
        enh = engine.submitted[-1] if engine.submitted[-1]["kind"] == "enhance" else next(
            j for j in engine.submitted if kind(j) == "enhance")
        assert enh["mode"] == "i2i" and enh["ref_image_path"].endswith("runs/0002.png")  # the same parent image
        run = run_of(client, pid, 3)
        assert run["mode"] == "edit" and run["parent_run"] == 2
        gen = [j for j in engine.submitted if kind(j) == "generate"][-1]
        assert gen["ref_image_path"].endswith("runs/0002.png")

    def test_parent_is_pinned_even_if_current_run_changes_mid_enhance(self, client, pid, engine):
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        engine.release.clear()
        body = chain(client, pid, "edit two")
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert run["parent_run"] == 2
        assert [j for j in engine.submitted if kind(j) == "generate"][-1]["ref_image_path"].endswith("runs/0002.png")

    def test_refs_are_pinned_at_submit_even_if_they_change_mid_enhance(self, client, pid, engine, tmp_path):
        a = add_ref(client, pid, RED, name="a")
        b = add_ref(client, pid, GREEN, name="b")
        engine.release.clear()
        body = chain(client, pid, "a person with the jacket")
        d = runs_dir(tmp_path, pid) / f"{body['run_reserved']:04d}.refs"
        assert sorted(p.name for p in d.iterdir()) == ["1.png", "2.png"]  # pinned immediately
        client.delete(f"/api/projects/{pid}/refs/{a['id']}")
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"enabled": False})
        add_ref(client, pid, BLUE, name="c")
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert [x["name"] for x in run["refs"]] == ["a", "b"] and [x["image_index"] for x in run["refs"]] == [1, 2]
        gen = [j for j in engine.submitted if kind(j) == "generate"][-1]
        assert [Path(p) for p in gen["ref_image_paths"]] == [d / "1.png", d / "2.png"]
        assert not [j for j in engine.submitted if kind(j) == "enhance"][0].get("ref_image_paths")

    def test_alpha_parent_is_flattened_at_submit_for_a_plain_chain(self, client, pid, engine, tmp_path):
        engine.image_factory = half_transparent
        do_run(client, pid, "cutout", transparent=True)
        engine.image_factory = None
        engine.release.clear()
        body = chain(client, pid, "fill the background")
        f = runs_dir(tmp_path, pid) / f"{body['run_reserved']:04d}.parent.png"
        assert f.is_file()
        engine.release.set()
        run_of(client, pid, body["run_reserved"])
        gen = [j for j in engine.submitted if kind(j) == "generate"][-1]
        assert gen["ref_image_path"].endswith(f"{body['run_reserved']:04d}.parent.png") and f.is_file()

    def test_run_number_is_reserved_immediately_and_later_runs_get_later_numbers(self, client, pid, engine):
        engine.release.clear()
        body = chain(client, pid, "a cat")
        assert body["run_reserved"] == 1
        later = submit(client, pid, "plain run")
        assert later["run_reserved"] == 2
        engine.release.set()
        assert run_of(client, pid, 1)["enhanced_from"] == "a cat"
        assert run_of(client, pid, 2)["prompt"] == "plain run"

    def test_snapshots_carry_chain_fields(self, client, pid, engine):
        plain = do_run(client, pid, "plain")
        body = chain(client, pid, "a cat")
        run_of(client, pid, body["run_reserved"])
        jobs = client.get("/api/jobs").json()
        enh = next(j for j in jobs if j["id"] == body["enhance_job_id"])
        assert enh["chain_id"] == body["chain_id"] and enh["chain_step"] == "enhance" and enh["kind"] == "enhance"
        gens = [j for j in jobs if j.get("chain_id") == body["chain_id"] and j["chain_step"] == "generate"]
        assert len(gens) == 1 and gens[0]["kind"] == "generate" and gens[0]["state"] == "done"
        ordinary = next(j for j in jobs if j["id"] == plain["job_id"])
        assert ordinary.get("chain_id") is None and ordinary.get("chain_step") is None

    def test_generate_job_appears_in_jobs_once_submitted(self, client, pid, engine):
        engine.release.clear()
        body = chain(client, pid)
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        engine.gate.clear()  # hold the generate that follows
        engine.release.set()
        settled(client, body["enhance_job_id"])
        gens = wait_until(lambda: [j for j in client.get("/api/jobs").json() if j.get("chain_step") == "generate"])
        assert gens[0]["state"] in ("queued", "running") and gens[0]["chain_id"] == body["chain_id"]
        engine.gate.set()
        run_of(client, pid, body["run_reserved"])

    # -- cancel / fail / delete / submit error: no run, number and pins released --
    def _pinned_project(self, client, pid, engine):
        add_ref(client, pid, RED, name="ref")
        engine.image_factory = half_transparent
        do_run(client, pid, "cutout", transparent=True)
        engine.image_factory = None

    def _assert_released(self, client, pid, tmp_path, body, engine):
        rd = runs_dir(tmp_path, pid)
        n = f"{body['run_reserved']:04d}"
        wait_until(lambda: not list(rd.glob(f"{n}*")))
        assert [r["run"] for r in project(client, pid)["runs"]] == [1]
        assert [j for j in engine.submitted if kind(j) == "generate"][-1:] and len(
            [j for j in engine.submitted if kind(j) == "generate"]) == 1  # only the setup run
        nxt = submit(client, pid, "after")
        assert nxt["run_reserved"] == body["run_reserved"]  # the reserved number was released and is reused
        run_of(client, pid, nxt["run_reserved"])

    def test_cancel_during_the_enhance_drops_the_chain(self, client, pid, engine, tmp_path):
        self._pinned_project(client, pid, engine)
        engine.release.clear()
        body = chain(client, pid, "doomed")
        rd = runs_dir(tmp_path, pid)
        n = f"{body['run_reserved']:04d}"
        assert (rd / f"{n}.refs").is_dir() and (rd / f"{n}.parent.png").is_file()
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        assert client.post(f"/api/jobs/{body['enhance_job_id']}/cancel").json() == {"cancelled": True}
        engine.release.set()
        assert settled(client, body["enhance_job_id"])["state"] == "cancelled"
        time.sleep(0.3)
        self._assert_released(client, pid, tmp_path, body, engine)

    def test_cancel_while_the_enhance_is_still_queued(self, client, pid, engine, tmp_path):
        self._pinned_project(client, pid, engine)
        engine.gate.clear()
        body = chain(client, pid, "doomed")
        assert client.post(f"/api/jobs/{body['enhance_job_id']}/cancel").json() == {"cancelled": True}
        settled(client, body["enhance_job_id"])
        engine.gate.set()
        self._assert_released(client, pid, tmp_path, body, engine)

    def test_enhance_failure_drops_the_chain(self, client, pid, engine, tmp_path):
        self._pinned_project(client, pid, engine)
        engine.outcome.append("fail")
        body = chain(client, pid, "doomed")
        assert settled(client, body["enhance_job_id"])["state"] == "failed"
        time.sleep(0.3)
        self._assert_released(client, pid, tmp_path, body, engine)

    def test_generate_submit_error_drops_the_chain(self, client, pid, engine, tmp_path):
        self._pinned_project(client, pid, engine)
        gens_before = len([j for j in engine.submitted if kind(j) == "generate"])

        def refuse(job):
            if kind(job) == "generate":
                raise ValueError("settings changed to something invalid")
        body = chain(client, pid, "doomed")  # the enhance is accepted...
        engine.reject_submit = refuse  # ...then the generate cannot be submitted
        settled(client, body["enhance_job_id"])
        time.sleep(0.5)
        engine.reject_submit = None
        rd = runs_dir(tmp_path, pid)
        wait_until(lambda: not list(rd.glob(f"{body['run_reserved']:04d}*")))
        assert len([j for j in engine.submitted if kind(j) == "generate"]) == gens_before
        assert [r["run"] for r in project(client, pid)["runs"]] == [1]
        assert submit(client, pid, "after")["run_reserved"] == body["run_reserved"]

    def test_project_deleted_during_the_enhance_drops_the_chain(self, client, pid, engine, tmp_path):
        engine.release.clear()
        body = chain(client, pid, "doomed")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        engine.release.set()
        settled(client, body["enhance_job_id"])
        time.sleep(0.3)
        assert [j for j in engine.submitted if kind(j) == "generate"] == []
        assert client.get(f"/api/projects/{pid}").status_code == 404
        assert not (tmp_path / "data" / "projects" / pid).exists()
        # the queue is not stuck
        other = new_project(client)["id"]
        do_run(client, other, "fine")

    # -- validation --
    def test_validation_errors_create_nothing(self, client, pid, engine, ready, pe_ready, tmp_path):
        for body, want in (({"prompt": ""}, 400), ({"prompt": "   "}, 400), ({"prompt": "x", "transparent": "yes"}, 400),
                           ({"prompt": "x", "thinking": "maybe"}, 400), ({"prompt": "x", "language": "klingon"}, 400)):
            assert client.post(f"/api/projects/{pid}/enhance-and-generate", json=body).status_code == want, body
        assert client.post("/api/projects/nope/enhance-and-generate", json={"prompt": "x"}).status_code == 404
        ready["te"]["heretic-8bit"] = False
        assert client.post(f"/api/projects/{pid}/enhance-and-generate", json={"prompt": "x"}).status_code == 409
        ready["te"]["heretic-8bit"] = True
        pe_ready["t2i-8bit"] = False
        assert client.post(f"/api/projects/{pid}/enhance-and-generate", json={"prompt": "x"}).status_code == 409
        pe_ready["t2i-8bit"] = True
        assert engine.submitted == []
        rd = runs_dir(tmp_path, pid)
        assert not rd.exists() or not any(rd.iterdir())
        assert chain(client, pid)["run_reserved"] == 1  # nothing was reserved by the failures


class TestChainEvents:
    def test_event_order_enhance_terminal_then_chain_then_generate(self, app, engine):
        events, body, pid = events_for_chain(app, engine, "a cat")
        idx = {}
        for i, e in enumerate(events):
            d = e["data"]
            if e["event"] == "job" and d["id"] == body["enhance_job_id"] and d["state"] == "done":
                idx.setdefault("enh_done", i)
            if e["event"] == "chain":
                idx.setdefault("chain", i)
            if e["event"] == "job" and d.get("chain_step") == "generate":
                idx.setdefault("gen_first", i)
            if e["event"] == "run":
                idx.setdefault("run", i)
        assert idx["enh_done"] < idx["chain"] < idx["gen_first"] < idx["run"], idx
        ch = events[idx["chain"]]["data"]
        assert ch["state"] == "generating" and ch["chain_id"] == body["chain_id"] and ch["project_id"] == pid
        assert ch["job_id"] and ch["run"] == body["run_reserved"] and ch["fell_back"] is False
        gen_job = events[idx["gen_first"]]["data"]
        assert gen_job["id"] == ch["job_id"] and gen_job["chain_id"] == body["chain_id"]
        assert not [e for e in events if e["event"] == "enhanced"]  # a chained enhance never announces one
        enh_events = [e["data"] for e in events if e["event"] == "job" and e["data"]["id"] == body["enhance_job_id"]]
        assert all(d["chain_step"] == "enhance" and d["chain_id"] == body["chain_id"] for d in enh_events)

    def test_fallback_is_reported_in_the_chain_event(self, app, engine):
        engine.enhance_fell_back = True
        events, body, _ = events_for_chain(app, engine, "a cat")
        ch = next(e["data"] for e in events if e["event"] == "chain")
        assert ch["state"] == "generating" and ch["fell_back"] is True

    def test_cancelled_chain_event_follows_the_terminal_job_event(self, app, engine):
        engine.release.clear()

        async def finish(c, pid, body):
            import asyncio
            for _ in range(200):
                if engine.get(body["enhance_job_id"])["state"] == "running":
                    break
                await asyncio.sleep(0.02)
            await c.post(f"/api/jobs/{body['enhance_job_id']}/cancel")
            engine.release.set()

        events, body, _ = events_for_chain(app, engine, "doomed", finish=finish)
        term = next(i for i, e in enumerate(events) if e["event"] == "job" and e["data"]["id"] == body["enhance_job_id"]
                    and e["data"]["state"] == "cancelled")
        ch = next(i for i, e in enumerate(events) if e["event"] == "chain")
        assert term < ch and events[ch]["data"]["state"] == "cancelled"
        assert not [e for e in events if e["event"] == "run"]
        assert not [e for e in events if e["event"] == "job" and e["data"].get("chain_step") == "generate"]

    def test_failed_chain_event_carries_the_error(self, app, engine):
        engine.outcome.append("fail")
        events, body, _ = events_for_chain(app, engine, "doomed")
        ch = next(e["data"] for e in events if e["event"] == "chain")
        assert ch["state"] == "failed" and ch["chain_id"] == body["chain_id"] and ch.get("error")
        assert not [e for e in events if e["event"] == "run"]

    def test_project_deleted_chain_event(self, app, engine):
        engine.release.clear()

        async def finish(c, pid, body):
            import asyncio
            for _ in range(200):
                if engine.get(body["enhance_job_id"])["state"] == "running":
                    break
                await asyncio.sleep(0.02)
            await c.delete(f"/api/projects/{pid}")
            engine.release.set()

        events, body, _ = events_for_chain(app, engine, "doomed", finish=finish)
        ch = [e["data"] for e in events if e["event"] == "chain"]
        # deleting the project cancels the waiting enhance; the chain is reported dropped either way
        assert ch and ch[-1]["state"] in ("project deleted", "cancelled") and ch[-1]["chain_id"] == body["chain_id"]
        assert not [e for e in events if e["event"] == "run"]


# ---- chain end to end with the real Engine (loaders stubbed) ---------------------------------------------------
from test_heretic import live, models_dir  # noqa: E402,F401 - fixtures


class TestChainWithRealEngine:
    def test_chain_through_the_real_engine(self, live):
        live.put("/api/settings", json={"low_memory": False})
        pid = new_project(live)["id"]
        body = chain(live, pid, "a cat")
        run = wait_until(lambda: next((r for r in project(live, pid)["runs"] if r["run"] == body["run_reserved"]), None),
                         timeout=20)
        assert run["prompt"] == "better" and run["enhanced_from"] == "a cat" and run["mode"] == "t2i"
        jobs = live.get("/api/jobs").json()
        enh = next(j for j in jobs if j["id"] == body["enhance_job_id"])
        assert enh["state"] == "done" and enh["chain_step"] == "enhance" and enh["result"]["rewritten_prompt"] == "better"
        gen = [j for j in jobs if j.get("chain_id") == body["chain_id"] and j["chain_step"] == "generate"]
        assert len(gen) == 1 and gen[0]["state"] == "done"
        assert project(live, pid)["enhanced"] is None
        # the enhance ran before the generate on the one engine
        assert enh["created_at"] <= gen[0]["created_at"]

    def test_stand_alone_enhance_through_the_real_engine_is_stored(self, live):
        pid = new_project(live)["id"]
        jid = enhance(live, pid, "a cat")
        settled(live, jid)
        e = project(live, pid)["enhanced"]
        assert e["text"] == "better" and e["original"] == "a cat" and e["fell_back"] is False and e["job_id"] == jid
        assert e["rewriter"] == "official-8bit"


# ---- 4. the startup orphan sweep ----------------------------------------------------------------------------------
def make_runs(root: Path):
    store = Store(root)
    pid = store.create_project("p")["id"]
    rd = store.runs_dir(pid)
    rd.mkdir(parents=True, exist_ok=True)
    # a saved run 1 with its pinned context
    (rd / "0001.png").write_bytes(b"png")
    (rd / "0001.json").write_text("{}")
    (rd / "0001.thumb.webp").write_bytes(b"w")
    (rd / "0001.refs").mkdir()
    (rd / "0001.refs" / "1.png").write_bytes(b"r")
    (rd / "0001.parent.png").write_bytes(b"p")
    # run 2 never saved: dangling pins
    (rd / "0002.refs").mkdir()
    (rd / "0002.refs" / "2.png").write_bytes(b"r")
    (rd / "0002.parent.png").write_bytes(b"p")
    # run 3 never saved, refs only
    (rd / "0003.refs").mkdir()
    return store, pid, rd


class TestSweepOrphans:
    def test_removes_only_unsaved_pins(self, tmp_path):
        store, pid, rd = make_runs(tmp_path / "data")
        n = store.sweep_orphans()
        assert n == 3
        assert not (rd / "0002.refs").exists() and not (rd / "0002.parent.png").exists() and not (rd / "0003.refs").exists()
        for keep in ("0001.png", "0001.json", "0001.thumb.webp", "0001.refs/1.png", "0001.parent.png"):
            assert (rd / keep).exists(), keep
        assert (store.project_dir(pid) / "project.json").is_file()

    def test_is_idempotent_and_handles_empty_stores(self, tmp_path):
        store, pid, rd = make_runs(tmp_path / "data")
        store.sweep_orphans()
        assert store.sweep_orphans() == 0
        assert Store(tmp_path / "empty").sweep_orphans() == 0

    def test_projects_without_runs_dirs_are_fine(self, tmp_path):
        store = Store(tmp_path / "data")
        store.create_project("no runs")
        assert store.sweep_orphans() == 0

    def test_runs_at_app_startup(self, tmp_path, ready, pe_ready):
        store, pid, rd = make_runs(tmp_path / "data")
        with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "data")) as c:
            assert not (rd / "0002.refs").exists() and not (rd / "0002.parent.png").exists()
            assert not (rd / "0003.refs").exists()
            assert (rd / "0001.refs" / "1.png").exists() and (rd / "0001.parent.png").exists()
            assert c.get(f"/api/projects/{pid}").status_code == 200
