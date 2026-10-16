"""Per-project art styles, written against the "Art styles" section of API.md (the contract)."""

from __future__ import annotations

import json

import pytest

from conftest import do_run, new_project, submit, wait_run, wait_until
from test_enhance import collect_events
from test_enhanced_chain import chain, enhance_done, job_snap, project, run_of
from test_transparent import PREFIX, SUFFIX
from test_upload import RED, png, upload
from PIL import Image

IDS = ["none", "photoreal", "cinematic", "anime", "stylized-anime", "hand-drawn", "watercolor", "oil-painting", "pixel-art",
       "3d-render", "low-poly", "ink-line-art", "comic-book", "custom"]


@pytest.fixture
def pid(client):
    return new_project(client)["id"]


def styles(client):
    r = client.get("/api/styles")
    assert r.status_code == 200
    return {s["id"]: s for s in r.json()}


def set_style(client, pid, style, **extra):
    return client.patch(f"/api/projects/{pid}", json={"style": style, **extra})


def set_ok(client, pid, style):
    r = set_style(client, pid, style)
    assert r.status_code == 200, r.text
    return r.json()


def base(prompt):
    """The prompt as API.md's composition rule reads it: trimmed, one trailing period dropped."""
    p = prompt.strip()
    return p[:-1] if p.endswith(".") else p


def t2i(prompt, snippet):
    return f"{base(prompt)}. {snippet}"


def edit(prompt, keep):
    return f"{base(prompt)}. {keep}"


def wrap(styled):
    """The transparency wrap drops one trailing period of the text it wraps (keep notes end in one)."""
    return PREFIX + base_no_ws(styled) + SUFFIX


def base_no_ws(text):
    return text[:-1] if text.endswith(".") else text


def gen_jobs(engine):
    return [j for j in engine.submitted if j.get("kind", "generate") == "generate"]


def enh_jobs(engine):
    return [j for j in engine.submitted if j.get("kind") == "enhance"]


def run_json(client, pid, n):
    return wait_run(client, pid, n)


def strip_hint(p):
    return p.split("\n\n")[0]


# ---- GET /api/styles --------------------------------------------------------------------------
class TestStyleList:
    def test_ids_in_display_order(self, client):
        assert [s["id"] for s in client.get("/api/styles").json()] == IDS

    def test_documented_fields(self, client):
        for s in client.get("/api/styles").json():
            assert {"id", "label", "description", "snippet", "keep"} <= set(s)
            assert all(isinstance(s[k], str) for k in ("id", "label", "description", "snippet", "keep"))
            assert s["label"]

    def test_none_adds_nothing(self, client):
        s = styles(client)["none"]
        assert s["snippet"] == "" and s["keep"] == ""
        assert not s.get("custom")

    def test_custom_entry_is_flagged_and_empty(self, client):
        s = styles(client)["custom"]
        assert s["custom"] is True and s["snippet"] == "" and s["keep"] == ""
        assert [x["id"] for x in client.get("/api/styles").json() if x.get("custom")] == ["custom"]

    def test_real_presets_have_text(self, client):
        for i, s in styles(client).items():
            if i not in ("none", "custom"):
                assert s["snippet"].strip() and s["keep"].strip() and not s.get("custom"), i

    def test_anime_label(self, client):
        assert styles(client)["anime"]["label"] == "Anime / cel-shaded"


# ---- project.style ----------------------------------------------------------------------------
class TestProjectStyle:
    def test_default_is_none(self, client, pid):
        assert new_project(client)["style"] == {"id": "none"}
        assert project(client, pid)["style"] == {"id": "none"}

    def test_in_list(self, client, pid):
        set_ok(client, pid, {"id": "anime"})
        item = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
        assert item["style"] == {"id": "anime"}
        assert all("style" in p for p in client.get("/api/projects").json())

    def test_patch_preset(self, client, pid):
        assert set_ok(client, pid, {"id": "watercolor"})["style"] == {"id": "watercolor"}
        assert project(client, pid)["style"] == {"id": "watercolor"}

    @pytest.mark.parametrize("i", [i for i in IDS if i != "custom"])
    def test_every_preset_is_accepted(self, client, pid, i):
        assert set_ok(client, pid, {"id": i})["style"] == {"id": i}

    def test_patch_custom(self, client, pid):
        got = set_ok(client, pid, {"id": "custom", "custom": "chalk on slate"})["style"]
        assert got == {"id": "custom", "custom": "chalk on slate"}
        assert project(client, pid)["style"] == got

    def test_custom_text_is_trimmed(self, client, pid):
        assert set_ok(client, pid, {"id": "custom", "custom": "  chalk \n"})["style"] == {"id": "custom", "custom": "chalk"}

    def test_custom_at_300_chars_ok(self, client, pid):
        t = "x" * 300
        assert set_ok(client, pid, {"id": "custom", "custom": t})["style"]["custom"] == t

    def test_custom_length_counted_after_trimming(self, client, pid):
        t = "x" * 300
        assert set_ok(client, pid, {"id": "custom", "custom": f"   {t}   "})["style"]["custom"] == t

    def test_whole_value_is_replaced(self, client, pid):
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        assert set_ok(client, pid, {"id": "pixel-art"})["style"] == {"id": "pixel-art"}
        assert set_ok(client, pid, {"id": "none"})["style"] == {"id": "none"}

    def test_custom_null_with_preset_is_allowed(self, client, pid):
        r = set_style(client, pid, {"id": "anime", "custom": None})
        assert r.status_code == 200 and r.json()["style"] == {"id": "anime"}

    def test_bumps_updated_at(self, client, pid):
        before = project(client, pid)["updated_at"]
        set_ok(client, pid, {"id": "anime"})
        assert project(client, pid)["updated_at"] > before

    def test_style_with_name_in_one_patch(self, client, pid):
        r = client.patch(f"/api/projects/{pid}", json={"name": "Renamed", "style": {"id": "anime"}})
        assert r.status_code == 200 and r.json()["name"] == "Renamed" and r.json()["style"] == {"id": "anime"}

    def test_legacy_project_without_the_field_reads_none(self, client, pid, tmp_path):
        set_ok(client, pid, {"id": "anime"})  # proves the on-disk edit below is what the server reads
        f = tmp_path / "data" / "projects" / pid / "project.json"
        d = json.loads(f.read_text())
        d.pop("style", None)
        f.write_text(json.dumps(d))
        assert project(client, pid)["style"] == {"id": "none"}
        item = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
        assert item["style"] == {"id": "none"}


BAD = [
    ("null", None),
    ("string", "anime"),
    ("list", ["anime"]),
    ("number", 3),
    ("empty object", {}),
    ("unknown id", {"id": "nope"}),
    ("non-string id", {"id": 5}),
    ("extra key", {"id": "anime", "label": "x"}),
    ("extra key only", {"foo": 1}),
    ("custom missing", {"id": "custom"}),
    ("custom null", {"id": "custom", "custom": None}),
    ("custom not a string", {"id": "custom", "custom": 5}),
    ("custom empty", {"id": "custom", "custom": ""}),
    ("custom blank", {"id": "custom", "custom": "  \n\t "}),
    ("custom too long", {"id": "custom", "custom": "x" * 301}),
    ("custom too long after trim", {"id": "custom", "custom": " " + "x" * 301 + " "}),
    ("custom text on preset", {"id": "anime", "custom": "chalk"}),
    ("custom text on none", {"id": "none", "custom": "chalk"}),
    ("empty custom text on preset", {"id": "anime", "custom": ""}),
]


class TestStyleValidation:
    @pytest.mark.parametrize("why,bad", BAD, ids=[b[0] for b in BAD])
    def test_rejected_with_400_detail(self, client, pid, why, bad):
        r = set_style(client, pid, bad)
        assert r.status_code == 400, r.text
        assert r.json().get("detail")

    @pytest.mark.parametrize("why,bad", BAD, ids=[b[0] for b in BAD])
    def test_failed_patch_changes_nothing(self, client, pid, why, bad):
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        do_run(client, pid, "one")
        do_run(client, pid, "two")
        before = project(client, pid)
        r = client.patch(f"/api/projects/{pid}", json={"style": bad, "name": "Changed", "current_run": 1})
        assert r.status_code == 400
        assert project(client, pid) == before

    def test_unknown_project_404(self, client):
        assert set_style(client, "nope", {"id": "anime"}).status_code == 404


# ---- SSE --------------------------------------------------------------------------------------
class TestStyleEvent:
    def _run(self, app, steps, expect):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            await steps(c, pid)
            return pid

        def done(ev, _):
            return len([e for e in ev if e["event"] == "style"]) >= expect

        events, pid = collect_events(app, action, done, timeout=3 if expect == 0 else 8)
        return [e["data"] for e in events if e["event"] == "style"], pid

    def test_event_on_change(self, app):
        async def steps(c, pid):
            r = await c.patch(f"/api/projects/{pid}", json={"style": {"id": "custom", "custom": " chalk "}})
            assert r.status_code == 200

        evs, pid = self._run(app, steps, 1)
        assert evs == [{"project_id": pid, "style": {"id": "custom", "custom": "chalk"}}]

    def test_resending_the_current_style_sends_nothing(self, app):
        async def steps(c, pid):
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "anime"}})
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "anime"}})
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "pixel-art"}})

        evs, pid = self._run(app, steps, 3)
        assert evs == [{"project_id": pid, "style": {"id": "anime"}}, {"project_id": pid, "style": {"id": "pixel-art"}}]

    def test_default_none_resent_sends_nothing(self, app):
        async def steps(c, pid):
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "none"}})

        evs, _ = self._run(app, steps, 0)
        assert evs == []

    def test_custom_text_change_is_a_change(self, app):
        async def steps(c, pid):
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "custom", "custom": "a"}})
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "custom", "custom": "a"}})
            await c.patch(f"/api/projects/{pid}", json={"style": {"id": "custom", "custom": "b"}})

        evs, pid = self._run(app, steps, 2)
        assert [e["style"]["custom"] for e in evs] == ["a", "b"]

    def test_failed_patch_sends_nothing(self, app):
        async def steps(c, pid):
            r = await c.patch(f"/api/projects/{pid}", json={"style": {"id": "nope"}})
            assert r.status_code == 400

        evs, _ = self._run(app, steps, 0)
        assert evs == []


# ---- composition ------------------------------------------------------------------------------
class TestComposition:
    def test_new_image_appends_snippet(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        r = submit(client, pid, "a cat")
        run = run_json(client, pid, r["run_reserved"])
        assert run["model_prompt"] == f"a cat. {snip}" and run["prompt"] == "a cat"
        assert gen_jobs(engine)[-1]["prompt"] == f"a cat. {snip}"

    @pytest.mark.parametrize("typed", ["a cat.", "  a cat.  ", "a cat", "a cat..", "  a cat  "])
    def test_trim_and_one_trailing_period(self, client, pid, engine, typed):
        snip = styles(client)["watercolor"]["snippet"]
        set_ok(client, pid, {"id": "watercolor"})
        r = submit(client, pid, typed)
        run = run_json(client, pid, r["run_reserved"])
        assert run["model_prompt"] == t2i(typed, snip)
        assert gen_jobs(engine)[-1]["prompt"] == t2i(typed, snip)

    def test_only_one_period_dropped(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        run = run_json(client, pid, submit(client, pid, "wait...")["run_reserved"])
        assert run["model_prompt"] == "wait.." + ". " + snip  # "wait..." -> drop one period -> "wait.." + ". <snippet>"

    def test_edit_appends_keep_note(self, client, pid, engine):
        do_run(client, pid, "base")
        keep = styles(client)["oil-painting"]["keep"]
        set_ok(client, pid, {"id": "oil-painting"})
        r = submit(client, pid, "make it blue.")
        assert r["mode"] == "edit"
        run = run_json(client, pid, r["run_reserved"])
        assert run["model_prompt"] == f"make it blue. {keep}" and run["prompt"] == "make it blue."
        assert gen_jobs(engine)[-1]["prompt"] == f"make it blue. {keep}"

    def test_none_leaves_prompt_unchanged_and_untrimmed(self, client, pid, engine):
        p = "  a cat.  "
        run = run_json(client, pid, submit(client, pid, p)["run_reserved"])
        assert run["model_prompt"] == p and gen_jobs(engine)[-1]["prompt"] == p

    def test_none_after_a_style_leaves_prompt_unchanged(self, client, pid, engine):
        set_ok(client, pid, {"id": "anime"})
        set_ok(client, pid, {"id": "none"})
        run = run_json(client, pid, submit(client, pid, "a cat.")["run_reserved"])
        assert run["model_prompt"] == "a cat."

    def test_custom_new_image_uses_users_text(self, client, pid, engine):
        set_ok(client, pid, {"id": "custom", "custom": "  chalk on slate "})
        run = run_json(client, pid, submit(client, pid, "a cat")["run_reserved"])
        assert run["model_prompt"] == "a cat. chalk on slate"

    def test_custom_edit_uses_derived_keep_note(self, client, pid, engine):
        do_run(client, pid, "base")
        set_ok(client, pid, {"id": "custom", "custom": "chalk on slate"})
        run = run_json(client, pid, submit(client, pid, "add a hat")["run_reserved"])
        assert run["model_prompt"] == "add a hat. Keep the same art style: chalk on slate"

    @pytest.mark.parametrize("sid", [i for i in IDS if i not in ("none", "custom")])
    def test_every_preset_composes_from_its_listing(self, client, pid, engine, sid):
        s = styles(client)[sid]
        set_ok(client, pid, {"id": sid})
        n1 = submit(client, pid, "a cat")["run_reserved"]
        assert run_json(client, pid, n1)["model_prompt"] == t2i("a cat", s["snippet"])
        n2 = submit(client, pid, "bigger")["run_reserved"]
        assert run_json(client, pid, n2)["model_prompt"] == edit("bigger", s["keep"])

    def test_style_is_read_when_prepared_not_later(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        engine.gate.clear()
        r = submit(client, pid, "a cat")
        set_ok(client, pid, {"id": "pixel-art"})
        engine.gate.set()
        run = run_json(client, pid, r["run_reserved"])
        assert run["model_prompt"] == t2i("a cat", snip) and run["style"]["id"] == "anime"

    def test_style_change_affects_only_later_runs(self, client, pid, engine):
        s = styles(client)
        set_ok(client, pid, {"id": "anime"})
        a = run_json(client, pid, submit(client, pid, "one")["run_reserved"])
        set_ok(client, pid, {"id": "pixel-art"})
        b = run_json(client, pid, submit(client, pid, "two")["run_reserved"])
        assert a["model_prompt"] == t2i("one", s["anime"]["snippet"])
        assert b["model_prompt"] == edit("two", s["pixel-art"]["keep"])

    def test_use_enhanced_false_always_adds_style(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        run = run_json(client, pid, submit(client, pid, "a cat", use_enhanced=False)["run_reserved"])
        assert run["model_prompt"] == t2i("a cat", snip)


class TestTransparencyOrder:
    def test_wrap_goes_around_styled_text_new_image(self, client, pid, engine):
        snip = styles(client)["pixel-art"]["snippet"]
        set_ok(client, pid, {"id": "pixel-art"})
        run = run_json(client, pid, submit(client, pid, "a cat.", transparent=True)["run_reserved"])
        want = wrap(f"a cat. {snip}")
        assert want == f"{PREFIX}a cat. {snip}{SUFFIX}"
        assert run["model_prompt"] == want and gen_jobs(engine)[-1]["prompt"] == want
        assert run["prompt"] == "a cat."

    def test_wrap_goes_around_styled_text_edit(self, client, pid, engine):
        do_run(client, pid, "base")
        keep = styles(client)["low-poly"]["keep"]
        set_ok(client, pid, {"id": "low-poly"})
        run = run_json(client, pid, submit(client, pid, "add a hat", transparent=True)["run_reserved"])
        assert run["model_prompt"] == wrap(f"add a hat. {keep}")

    def test_wrap_with_none_is_unchanged_behaviour(self, client, pid):
        run = run_json(client, pid, submit(client, pid, "a cat.", transparent=True)["run_reserved"])
        assert run["model_prompt"] == wrap("a cat")

    def test_wrap_with_custom(self, client, pid):
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        run = run_json(client, pid, submit(client, pid, "a cat", transparent=True)["run_reserved"])
        assert run["model_prompt"] == wrap("a cat. chalk")


# ---- run json style field ---------------------------------------------------------------------
class TestRunStyleField:
    def test_preset_run(self, client, pid):
        s = styles(client)["cinematic"]
        set_ok(client, pid, {"id": "cinematic"})
        run = run_json(client, pid, submit(client, pid, "a cat")["run_reserved"])
        assert run["style"] == {"id": "cinematic", "label": s["label"]}

    def test_custom_run(self, client, pid):
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        run = run_json(client, pid, submit(client, pid, "a cat")["run_reserved"])
        assert run["style"] == {"id": "custom", "label": "Custom", "custom": "chalk"}

    def test_none_run(self, client, pid):
        run = run_json(client, pid, submit(client, pid, "a cat")["run_reserved"])
        assert run["style"] == {"id": "none", "label": "None"}

    def test_job_snapshot_carries_style(self, client, pid, engine):
        s = styles(client)["anime"]
        set_ok(client, pid, {"id": "anime"})
        r = submit(client, pid, "a cat")
        snap = wait_until(lambda: job_snap(client, r["job_id"]))
        assert snap["style"] == {"id": "anime", "label": s["label"]}

    def test_upload_has_null_style(self, client, pid):
        set_ok(client, pid, {"id": "anime"})
        r = upload(client, pid, png(Image.new("RGB", (64, 48), RED)))
        assert r.status_code in (200, 201, 202), r.text
        runs = project(client, pid)["runs"]
        assert len(runs) == 1 and runs[0]["style"] is None

    def test_legacy_run_without_style_reads_null(self, client, pid, tmp_path):
        n = do_run(client, pid, "old")["run_reserved"]
        f = tmp_path / "data" / "projects" / pid / "runs" / f"{n:04d}.json"
        d = json.loads(f.read_text())
        d.pop("style", None)
        f.write_text(json.dumps(d))
        run = next(r for r in project(client, pid)["runs"] if r["run"] == n)
        assert run["style"] is None

    def test_legacy_style_run_does_not_restyle_later_runs(self, client, pid):
        do_run(client, pid, "old")
        run = run_json(client, pid, submit(client, pid, "next")["run_reserved"])
        assert run["model_prompt"] == "next" and run["style"] == {"id": "none", "label": "None"}


# ---- enhancer ---------------------------------------------------------------------------------
class TestEnhancerHint:
    def _enh(self, client, pid, engine, prompt="a cat"):
        engine.enhance_text = strip_hint
        enhance_done(client, pid, prompt)
        return enh_jobs(engine)[-1]

    def test_t2i_hint(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        job = self._enh(client, pid, engine)
        assert job["mode"] == "t2i"
        assert job["prompt"] == f"a cat\n\nArt style: {snip}"

    def test_i2i_hint(self, client, pid, engine):
        do_run(client, pid, "base")
        keep = styles(client)["anime"]["keep"]
        set_ok(client, pid, {"id": "anime"})
        job = self._enh(client, pid, engine, "make it blue")
        assert job["mode"] == "i2i"
        assert job["prompt"] == f"make it blue\n\n{keep}"

    def test_none_adds_nothing(self, client, pid, engine):
        job = self._enh(client, pid, engine)
        assert job["prompt"] == "a cat"
        do_run(client, pid, "base")
        job = self._enh(client, pid, engine, "make it blue")
        assert job["prompt"] == "make it blue"

    def test_custom_hints(self, client, pid, engine):
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        assert self._enh(client, pid, engine)["prompt"] == "a cat\n\nArt style: chalk"
        do_run(client, pid, "base")
        assert self._enh(client, pid, engine, "x")["prompt"] == "x\n\nKeep the same art style: chalk"

    def test_hint_comes_before_the_english_suffix(self, client, pid, engine):
        # The engine (not the app) adds the English-output suffix, so the text handed to it ends with the hint.
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        job = self._enh(client, pid, engine)
        assert job["prompt"].endswith(f"Art style: {snip}")

    def test_hint_is_not_in_snapshot_or_original(self, client, pid, engine):
        set_ok(client, pid, {"id": "anime"})
        jid = enhance_done(client, pid, "a cat")
        snap = job_snap(client, jid)
        assert snap["prompt"] == "a cat"
        assert project(client, pid)["enhanced"]["original"] == "a cat"

    def test_enhanced_records_style(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        assert project(client, pid)["enhanced"]["style"] == {"id": "anime"}

    def test_enhanced_records_none_and_custom(self, client, pid, engine):
        engine.enhance_text = strip_hint
        enhance_done(client, pid, "a cat")
        assert project(client, pid)["enhanced"]["style"] == {"id": "none"}
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        enhance_done(client, pid, "a cat")
        assert project(client, pid)["enhanced"]["style"] == {"id": "custom", "custom": "chalk"}

    def test_legacy_enhanced_reads_none(self, client, pid, engine, tmp_path):
        enhance_done(client, pid, "a cat")
        f = tmp_path / "data" / "projects" / pid / "project.json"
        d = json.loads(f.read_text())
        assert d.get("enhanced"), "enhanced is expected in project.json"
        d["enhanced"].pop("style", None)
        f.write_text(json.dumps(d))
        assert project(client, pid)["enhanced"]["style"] == {"id": "none"}

    def test_chain_enhance_step_gets_the_hint(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        body = chain(client, pid, "a cat")
        run_of(client, pid, body["run_reserved"])
        assert enh_jobs(engine)[-1]["prompt"] == f"a cat\n\nArt style: {snip}"
        assert job_snap(client, body["enhance_job_id"])["prompt"] == "a cat"


# ---- double-add -------------------------------------------------------------------------------
class TestNoDoubleAdd:
    def test_use_enhanced_unchanged_style_not_added(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        r = submit(client, pid, "ENHANCED: a cat", use_enhanced=True)
        run = run_json(client, pid, r["run_reserved"])
        assert run["model_prompt"] == "ENHANCED: a cat"
        assert gen_jobs(engine)[-1]["prompt"] == "ENHANCED: a cat"
        assert run["style"]["id"] == "anime"

    def test_use_enhanced_unchanged_custom_not_added(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        enhance_done(client, pid, "a cat")
        set_ok(client, pid, {"id": "custom", "custom": " chalk "})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == "R"
        assert run["style"] == {"id": "custom", "label": "Custom", "custom": "chalk"}

    def test_use_enhanced_changed_preset_added(self, client, pid, engine):
        s = styles(client)
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        set_ok(client, pid, {"id": "pixel-art"})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == t2i("R", s["pixel-art"]["snippet"])
        assert run["style"]["id"] == "pixel-art"

    def test_use_enhanced_custom_text_changed_added(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        enhance_done(client, pid, "a cat")
        set_ok(client, pid, {"id": "custom", "custom": "crayon"})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == "R. crayon"

    def test_use_enhanced_style_removed_adds_nothing(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        set_ok(client, pid, {"id": "none"})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == "R"

    def test_use_enhanced_style_set_after_a_none_enhance_is_added(self, client, pid, engine):
        snip = styles(client)["anime"]["snippet"]
        enhance_done(client, pid, "a cat")
        set_ok(client, pid, {"id": "anime"})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == t2i("R", snip)

    def test_use_enhanced_edit_unchanged_not_added(self, client, pid, engine):
        engine.enhance_text = strip_hint
        do_run(client, pid, "base")
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "blue")
        r = submit(client, pid, "R", use_enhanced=True)
        assert r["mode"] == "edit"
        assert run_json(client, pid, r["run_reserved"])["model_prompt"] == "R"

    def test_use_enhanced_without_stored_value_styles_normally(self, client, pid):
        snip = styles(client)["anime"]["snippet"]
        set_ok(client, pid, {"id": "anime"})
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True)["run_reserved"])
        assert run["model_prompt"] == t2i("R", snip)

    def test_unchanged_with_transparency_wraps_unstyled_rewrite(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        enhance_done(client, pid, "a cat")
        run = run_json(client, pid, submit(client, pid, "R", use_enhanced=True, transparent=True)["run_reserved"])
        assert run["model_prompt"] == wrap("R")

    def test_chain_unchanged_style_not_added(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        body = chain(client, pid, "a cat")
        run = run_of(client, pid, body["run_reserved"])
        assert run["prompt"] == "a cat" and run["model_prompt"] == "a cat"
        assert gen_jobs(engine)[-1]["prompt"] == "a cat"
        assert run["style"]["id"] == "anime"
        assert run["enhanced_from"] == "a cat"

    def test_chain_style_none_sends_rewrite_unchanged(self, client, pid, engine):
        body = chain(client, pid, "a cat")
        run = run_of(client, pid, body["run_reserved"])
        assert run["model_prompt"] == "ENHANCED: a cat" and run["style"] == {"id": "none", "label": "None"}

    def test_chain_style_changed_while_enhance_pending_is_added(self, client, pid, engine):
        s = styles(client)
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        engine.release.clear()
        body = chain(client, pid, "a cat")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        set_ok(client, pid, {"id": "watercolor"})
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert run["model_prompt"] == t2i("a cat", s["watercolor"]["snippet"])
        assert gen_jobs(engine)[-1]["prompt"] == run["model_prompt"]
        assert run["style"] == {"id": "watercolor", "label": s["watercolor"]["label"]}
        assert enh_jobs(engine)[-1]["prompt"] == f"a cat\n\nArt style: {s['anime']['snippet']}"

    def test_chain_style_changed_edit_uses_keep(self, client, pid, engine):
        s = styles(client)
        engine.enhance_text = strip_hint
        do_run(client, pid, "base")
        set_ok(client, pid, {"id": "anime"})
        engine.release.clear()
        body = chain(client, pid, "blue")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        set_ok(client, pid, {"id": "custom", "custom": "chalk"})
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert run["model_prompt"] == "blue. Keep the same art style: chalk"
        assert run["style"] == {"id": "custom", "label": "Custom", "custom": "chalk"}

    def test_chain_style_cleared_while_pending_adds_nothing(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        engine.release.clear()
        body = chain(client, pid, "a cat")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        set_ok(client, pid, {"id": "none"})
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert run["model_prompt"] == "a cat" and run["style"] == {"id": "none", "label": "None"}

    def test_chain_change_and_change_back_counts_as_unchanged(self, client, pid, engine):
        engine.enhance_text = strip_hint
        set_ok(client, pid, {"id": "anime"})
        engine.release.clear()
        body = chain(client, pid, "a cat")
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        set_ok(client, pid, {"id": "watercolor"})
        set_ok(client, pid, {"id": "anime"})
        engine.release.set()
        assert run_of(client, pid, body["run_reserved"])["model_prompt"] == "a cat"

    def test_chain_transparent_wraps_styled_current_style_when_changed(self, client, pid, engine):
        s = styles(client)
        engine.enhance_text = strip_hint
        engine.release.clear()
        body = chain(client, pid, "a cat", transparent=True)
        wait_until(lambda: engine.get(body["enhance_job_id"])["state"] == "running")
        set_ok(client, pid, {"id": "anime"})
        engine.release.set()
        run = run_of(client, pid, body["run_reserved"])
        assert run["model_prompt"] == wrap(t2i("a cat", s["anime"]["snippet"]))
