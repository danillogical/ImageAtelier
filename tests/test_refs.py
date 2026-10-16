"""Per-project reference images: library CRUD, numbering, pinning at submit, and engine conditioning (API.md)."""

from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image, ImageDraw

from conftest import do_run, new_project, patch_parts, submit, wait_run, wait_until
from test_enhance import collect_events
from test_heretic import live, models_dir  # noqa: F401 - fixtures: real Engine via create_app, loaders stubbed
from test_upload import BLUE, GREEN, RED, bands, jpeg, near, png


def add(client, pid, data, name=None):
    url = f"/api/projects/{pid}/refs" + (f"?name={name}" if name is not None else "")
    return client.post(url, content=data, headers={"content-type": "application/octet-stream"})


def colour(c, size=(64, 48)):
    return png(Image.new("RGB", size, c))


def refs(client, pid):
    r = client.get(f"/api/projects/{pid}/refs")
    assert r.status_code == 200
    return r.json()


def add_ok(client, pid, c=RED, name=None, size=(64, 48)):
    r = add(client, pid, colour(c, size), name)
    assert r.status_code == 201, r.text
    return r.json()


def proj_dir(tmp_path, pid):
    return tmp_path / "data" / "projects" / pid


@pytest.fixture
def pid(client):
    return new_project(client)["id"]


# ---- settings -------------------------------------------------------------------------------
class TestSettings:
    def test_defaults(self, client):
        s = client.get("/api/settings").json()
        assert s["ref_detail"] == "auto" and s["max_refs"] == 3

    @pytest.mark.parametrize("key,val", [("ref_detail", "auto"), ("ref_detail", "high"), ("ref_detail", "standard"),
                                         ("ref_detail", "low"), ("max_refs", 0), ("max_refs", 9), ("max_refs", 5)])
    def test_valid(self, client, key, val):
        r = client.put("/api/settings", json={key: val})
        assert r.status_code == 200 and r.json()[key] == val

    @pytest.mark.parametrize("key,val", [("ref_detail", "medium"), ("ref_detail", ""), ("ref_detail", 768),
                                         ("ref_detail", None), ("max_refs", -1), ("max_refs", 10), ("max_refs", "3"),
                                         ("max_refs", True), ("max_refs", 2.5), ("max_refs", None)])
    def test_invalid_changes_nothing(self, client, key, val):
        before = client.get("/api/settings").json()
        r = client.put("/api/settings", json={key: val})
        assert r.status_code == 400 and "detail" in r.json()
        assert client.get("/api/settings").json() == before


# ---- library CRUD and validation -------------------------------------------------------------
class TestLibrary:
    def test_empty_list_and_unknown_project(self, client, pid):
        assert refs(client, pid) == []
        assert client.get("/api/projects/nope/refs").status_code == 404
        assert add(client, "nope", colour(RED)).status_code == 404

    def test_create_shape_and_default_names(self, client, pid):
        a = add_ok(client, pid)
        b = add_ok(client, pid, GREEN)
        assert a["name"] == "Reference 1" and b["name"] == "Reference 2"
        for r in (a, b):
            assert set(r) >= {"id", "name", "enabled", "order", "width", "height", "created_at", "image_url",
                              "thumb_url", "image_index_edit", "image_index_new"}
            assert r["enabled"] is True and (r["width"], r["height"]) == (64, 48)
        assert a["id"] != b["id"] and [a["order"], b["order"]] == [0, 1]
        assert [r["id"] for r in refs(client, pid)] == [a["id"], b["id"]]

    def test_explicit_name(self, client, pid):
        assert add_ok(client, pid, name="Red jacket")["name"] == "Red jacket"

    def test_image_and_thumb_routes(self, client, pid):
        r = add_ok(client, pid, BLUE, size=(800, 600))
        img = client.get(r["image_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(img.content)).size == (800, 600)
        th = client.get(r["thumb_url"])
        assert th.status_code == 200 and th.headers["content-type"] == "image/webp"
        assert client.get(f"/api/projects/{pid}/refs/nope/image").status_code == 404
        assert client.get(f"/api/projects/{pid}/refs/nope/thumb").status_code == 404

    def test_exif_orientation_is_applied(self, client, pid):
        r = add(client, pid, jpeg(bands(400, 300, [RED, BLUE]), exif_orientation=6))
        assert r.status_code == 201
        ref = r.json()
        assert (ref["width"], ref["height"]) == (300, 400)
        im = Image.open(io.BytesIO(client.get(ref["image_url"]).content))
        assert near(im.getpixel((150, 10)), RED) and near(im.getpixel((150, 390)), BLUE)

    def test_transparency_flattened_to_white_rgb(self, client, pid):
        src = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        ImageDraw.Draw(src).rectangle([50, 0, 100, 100], fill=(220, 20, 20, 255))
        ref = add(client, pid, png(src)).json()
        im = Image.open(io.BytesIO(client.get(ref["image_url"]).content))
        assert im.mode == "RGB" and im.getpixel((10, 50)) == (255, 255, 255) and im.getpixel((90, 50)) == (220, 20, 20)

    @pytest.mark.parametrize("size,want", [((3000, 1500), (2048, 1024)), ((1500, 3000), (1024, 2048)),
                                           ((2048, 2048), (2048, 2048)), ((1000, 500), (1000, 500))])
    def test_long_side_capped_at_2048_keeping_aspect(self, client, pid, size, want):
        ref = add(client, pid, png(Image.new("RGB", size, RED))).json()
        assert (ref["width"], ref["height"]) == want
        im = Image.open(io.BytesIO(client.get(ref["image_url"]).content))
        assert im.size == want

    def test_odd_aspect_is_not_cropped_or_padded(self, client, pid):
        ref = add(client, pid, png(bands(3000, 2001, [RED, GREEN, BLUE]))).json()
        assert max(ref["width"], ref["height"]) == 2048
        assert ref["width"] / ref["height"] == pytest.approx(3000 / 2001, abs=0.002)
        im = Image.open(io.BytesIO(client.get(ref["image_url"]).content))
        assert near(im.getpixel((2, 2)), RED) and near(im.getpixel((im.width - 3, im.height - 3)), BLUE)

    def test_rejections(self, client, pid):
        assert add(client, pid, b"").status_code == 400
        assert add(client, pid, b"not an image").status_code == 415
        assert add(client, pid, png(Image.new("1", (9000, 9000)))).status_code == 413  # > 80 MP
        try:
            status = add(client, pid, b"x" * (26 * 1024 * 1024)).status_code
        except (httpx.HTTPError, OSError):
            status = None
        assert status in (None, 413)
        assert refs(client, pid) == []
        assert add(client, pid, colour(RED)).status_code == 201  # not stuck

    def test_cap_of_30_per_project(self, client, pid):
        client.put("/api/settings", json={"max_refs": 0})  # everything is added disabled; the cap counts all refs
        for i in range(30):
            assert add(client, pid, colour(RED, (8, 8))).status_code == 201, i
        r = add(client, pid, colour(RED, (8, 8)))
        assert r.status_code == 409 and "detail" in r.json()
        assert len(refs(client, pid)) == 30
        # deleting one makes room again
        assert client.delete(f"/api/projects/{pid}/refs/{refs(client, pid)[0]['id']}").status_code == 204
        assert add(client, pid, colour(RED, (8, 8))).status_code == 201

    def test_rename(self, client, pid):
        r = add_ok(client, pid)
        p = client.patch(f"/api/projects/{pid}/refs/{r['id']}", json={"name": "Jacket"})
        assert p.status_code == 200 and p.json()["name"] == "Jacket" and p.json()["id"] == r["id"]
        assert refs(client, pid)[0]["name"] == "Jacket"

    def test_delete(self, client, pid, tmp_path):
        r = add_ok(client, pid)
        assert client.delete(f"/api/projects/{pid}/refs/{r['id']}").status_code == 204
        assert refs(client, pid) == []
        assert client.get(r["image_url"]).status_code == 404 and client.get(r["thumb_url"]).status_code == 404
        assert client.delete(f"/api/projects/{pid}/refs/{r['id']}").status_code == 404
        assert not list((proj_dir(tmp_path, pid) / "refs").glob(f"{r['id']}*")) if (proj_dir(tmp_path, pid) / "refs").exists() else True

    def test_patch_unknown_ref_or_project_404(self, client, pid):
        assert client.patch(f"/api/projects/{pid}/refs/nope", json={"name": "x"}).status_code == 404
        assert client.patch("/api/projects/nope/refs/x", json={"name": "x"}).status_code == 404

    def test_reorder(self, client, pid):
        a, b, c = (add_ok(client, pid, col, name=n) for col, n in ((RED, "a"), (GREEN, "b"), (BLUE, "c")))
        r = client.patch(f"/api/projects/{pid}/refs/{c['id']}", json={"order": 0})
        assert r.status_code == 200
        lst = refs(client, pid)
        assert [x["name"] for x in lst] == ["c", "a", "b"] and [x["order"] for x in lst] == [0, 1, 2]
        client.patch(f"/api/projects/{pid}/refs/{c['id']}", json={"order": 2})
        assert [x["name"] for x in refs(client, pid)] == ["a", "b", "c"]
        client.patch(f"/api/projects/{pid}/refs/{a['id']}", json={"order": 1})
        assert [x["name"] for x in refs(client, pid)] == ["b", "a", "c"]

    def test_bad_patch_bodies_400(self, client, pid):
        r = add_ok(client, pid)
        for body in ({"enabled": "yes"}, {"order": "first"}, {"name": 5}):
            assert client.patch(f"/api/projects/{pid}/refs/{r['id']}", json=body).status_code == 400, body

    def test_delete_project_removes_refs(self, client, pid, tmp_path):
        add_ok(client, pid)
        assert (proj_dir(tmp_path, pid) / "refs").exists()
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert not proj_dir(tmp_path, pid).exists()
        assert client.get(f"/api/projects/{pid}/refs").status_code == 404

    def test_libraries_are_per_project(self, client, pid):
        other = new_project(client)["id"]
        add_ok(client, pid)
        assert refs(client, other) == []


# ---- max_refs: enabled set --------------------------------------------------------------------
class TestEnabledLimit:
    def test_added_disabled_with_note_past_the_limit(self, client, pid):
        rs = [add_ok(client, pid, name=f"r{i}") for i in range(3)]
        assert all(r["enabled"] for r in rs)
        fourth = add_ok(client, pid, name="r3")
        assert fourth["enabled"] is False and fourth["note"]
        assert fourth["image_index_edit"] is None and fourth["image_index_new"] is None
        assert sum(r["enabled"] for r in refs(client, pid)) == 3

    def test_limit_follows_the_setting(self, client, pid):
        client.put("/api/settings", json={"max_refs": 1})
        assert add_ok(client, pid)["enabled"] is True
        assert add_ok(client, pid)["enabled"] is False
        client.put("/api/settings", json={"max_refs": 0})
        assert add_ok(client, pid)["enabled"] is False

    def test_enabling_past_the_limit_is_409(self, client, pid):
        for i in range(3):
            add_ok(client, pid, name=f"r{i}")
        extra = add_ok(client, pid, name="extra")
        r = client.patch(f"/api/projects/{pid}/refs/{extra['id']}", json={"enabled": True})
        assert r.status_code == 409 and "max 3 references enabled" in r.json()["detail"]
        assert refs(client, pid)[3]["enabled"] is False  # unchanged
        # disable one, then it fits
        first = refs(client, pid)[0]
        assert client.patch(f"/api/projects/{pid}/refs/{first['id']}", json={"enabled": False}).status_code == 200
        assert client.patch(f"/api/projects/{pid}/refs/{extra['id']}", json={"enabled": True}).status_code == 200

    def test_enabling_an_already_enabled_ref_at_the_limit_is_fine(self, client, pid):
        rs = [add_ok(client, pid) for _ in range(3)]
        assert client.patch(f"/api/projects/{pid}/refs/{rs[0]['id']}", json={"enabled": True}).status_code == 200

    def test_max_refs_zero_blocks_enabling(self, client, pid):
        client.put("/api/settings", json={"max_refs": 0})
        r = add_ok(client, pid)
        assert client.patch(f"/api/projects/{pid}/refs/{r['id']}", json={"enabled": True}).status_code == 409


# ---- numbering ---------------------------------------------------------------------------------
class TestNumbering:
    def test_indexes_follow_enabled_order(self, client, pid):
        a, b, c = (add_ok(client, pid, name=n) for n in "abc")
        lst = refs(client, pid)
        assert [x["image_index_edit"] for x in lst] == [2, 3, 4]
        assert [x["image_index_new"] for x in lst] == [1, 2, 3]

    def test_disabled_refs_have_null_and_do_not_consume_numbers(self, client, pid):
        a, b, c = (add_ok(client, pid, name=n) for n in "abc")
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"enabled": False})
        by = {x["name"]: x for x in refs(client, pid)}
        assert by["b"]["enabled"] is False and by["b"]["image_index_edit"] is None and by["b"]["image_index_new"] is None
        assert (by["a"]["image_index_edit"], by["c"]["image_index_edit"]) == (2, 3)
        assert (by["a"]["image_index_new"], by["c"]["image_index_new"]) == (1, 2)
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"enabled": True})
        assert [x["image_index_new"] for x in refs(client, pid)] == [1, 2, 3]

    def test_reorder_and_delete_renumber(self, client, pid):
        a, b, c = (add_ok(client, pid, name=n) for n in "abc")
        client.patch(f"/api/projects/{pid}/refs/{c['id']}", json={"order": 0})
        assert {x["name"]: x["image_index_new"] for x in refs(client, pid)} == {"c": 1, "a": 2, "b": 3}
        client.delete(f"/api/projects/{pid}/refs/{c['id']}")
        assert {x["name"]: x["image_index_edit"] for x in refs(client, pid)} == {"a": 2, "b": 3}

    def test_same_in_a_project_with_a_current_run(self, client, pid):
        do_run(client, pid, "base")
        add_ok(client, pid, name="a")
        add_ok(client, pid, name="b")
        lst = refs(client, pid)
        assert [x["image_index_edit"] for x in lst] == [2, 3] and [x["image_index_new"] for x in lst] == [1, 2]


# ---- pinning at submit --------------------------------------------------------------------------
class TestPinning:
    def test_t2i_run_pins_refs_as_images_1_and_up(self, client, pid, engine, tmp_path):
        a = add_ok(client, pid, RED, name="jacket")
        b = add_ok(client, pid, GREEN, name="mug")
        r = submit(client, pid, "a person wearing the jacket")
        assert r["mode"] == "t2i"
        job = engine.submitted[-1]
        d = proj_dir(tmp_path, pid) / "runs" / f"{r['run_reserved']:04d}.refs"
        assert [Path(p) for p in job["ref_image_paths"]] == [d / "1.png", d / "2.png"]
        assert sorted(x.name for x in d.iterdir()) == ["1.png", "2.png"]
        assert near(Image.open(d / "1.png").getpixel((5, 5)), RED) and near(Image.open(d / "2.png").getpixel((5, 5)), GREEN)
        run = wait_run(client, pid, r["run_reserved"])
        assert [(x["id"], x["name"], x["image_index"]) for x in run["refs"]] == [(a["id"], "jacket", 1), (b["id"], "mug", 2)]
        for x in run["refs"]:
            assert set(x) >= {"id", "name", "image_index", "width", "height", "image_url"}
            assert x["image_url"] == f"/api/projects/{pid}/runs/{run['run']}/refs/{x['image_index']}"
            resp = client.get(x["image_url"])
            assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
        assert client.get(f"/api/projects/{pid}/runs/{run['run']}/refs/3").status_code == 404

    def test_edit_run_numbers_refs_from_2(self, client, pid, engine, tmp_path):
        do_run(client, pid, "base")
        add_ok(client, pid, RED, name="a")
        add_ok(client, pid, BLUE, name="b")
        r = submit(client, pid, "edit with refs")
        assert r["mode"] == "edit"
        job = engine.submitted[-1]
        d = proj_dir(tmp_path, pid) / "runs" / f"{r['run_reserved']:04d}.refs"
        assert [Path(p) for p in job["ref_image_paths"]] == [d / "2.png", d / "3.png"]
        assert job["ref_image_path"].endswith("runs/0001.png")
        run = wait_run(client, pid, r["run_reserved"])
        assert [x["image_index"] for x in run["refs"]] == [2, 3] and run["parent_run"] == 1

    def test_only_enabled_refs_in_order(self, client, pid, engine, tmp_path):
        a, b, c = (add_ok(client, pid, col, name=n) for col, n in ((RED, "a"), (GREEN, "b"), (BLUE, "c")))
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"enabled": False})
        client.patch(f"/api/projects/{pid}/refs/{c['id']}", json={"order": 0})
        r = submit(client, pid, "x")
        run = wait_run(client, pid, r["run_reserved"])
        assert [x["name"] for x in run["refs"]] == ["c", "a"] and [x["image_index"] for x in run["refs"]] == [1, 2]
        assert len(engine.submitted[-1]["ref_image_paths"]) == 2

    def test_no_refs_means_empty_list_and_no_refs_dir(self, client, pid, engine, tmp_path):
        r = submit(client, pid, "plain")
        job = engine.submitted[-1]
        assert not job.get("ref_image_paths")
        run = wait_run(client, pid, r["run_reserved"])
        assert run["refs"] == []
        assert not (proj_dir(tmp_path, pid) / "runs" / f"{r['run_reserved']:04d}.refs").exists()

    def test_later_changes_do_not_affect_a_queued_job(self, client, pid, engine, tmp_path):
        a = add_ok(client, pid, RED, name="a")
        b = add_ok(client, pid, GREEN, name="b")
        engine.gate.clear()
        r = submit(client, pid, "queued")
        paths = [Path(p) for p in engine.submitted[-1]["ref_image_paths"]]
        before = [p.read_bytes() for p in paths]
        client.delete(f"/api/projects/{pid}/refs/{a['id']}")
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"enabled": False, "name": "renamed"})
        add_ok(client, pid, BLUE, name="new")
        assert [p.read_bytes() for p in paths] == before and all(p.is_file() for p in paths)
        engine.gate.set()
        run = wait_run(client, pid, r["run_reserved"])
        assert [x["name"] for x in run["refs"]] == ["a", "b"]  # names as pinned
        for x in run["refs"]:
            assert client.get(x["image_url"]).status_code == 200  # history images remain after the library changed
        # the next run sees the new library only
        r2 = submit(client, pid, "after")
        run2 = wait_run(client, pid, r2["run_reserved"])
        assert [x["name"] for x in run2["refs"]] == ["new"]

    def test_reorder_after_submit_does_not_change_the_pinned_order(self, client, pid, engine):
        a = add_ok(client, pid, RED, name="a")
        b = add_ok(client, pid, GREEN, name="b")
        engine.gate.clear()
        r = submit(client, pid, "queued")
        client.patch(f"/api/projects/{pid}/refs/{b['id']}", json={"order": 0})
        engine.gate.set()
        run = wait_run(client, pid, r["run_reserved"])
        assert [x["name"] for x in run["refs"]] == ["a", "b"]

    def test_cancelled_run_leaves_no_refs_dir(self, client, pid, engine, tmp_path):
        add_ok(client, pid)
        engine.gate.clear()
        r = submit(client, pid, "doomed")
        d = proj_dir(tmp_path, pid) / "runs" / f"{r['run_reserved']:04d}.refs"
        assert d.is_dir()
        client.post(f"/api/jobs/{r['job_id']}/cancel")
        wait_until(lambda: not d.exists())
        engine.gate.set()
        assert not list((proj_dir(tmp_path, pid) / "runs").glob(f"{r['run_reserved']:04d}*"))
        assert refs(client, pid)[0]["enabled"] is True  # the library itself is untouched

    def test_failed_run_leaves_no_refs_dir(self, client, pid, engine, tmp_path):
        add_ok(client, pid)
        engine.outcome.append("fail")
        r = submit(client, pid, "doomed")
        d = proj_dir(tmp_path, pid) / "runs" / f"{r['run_reserved']:04d}.refs"
        wait_until(lambda: engine.get(r["job_id"])["state"] == "failed")
        wait_until(lambda: not d.exists())
        assert not list((proj_dir(tmp_path, pid) / "runs").glob(f"{r['run_reserved']:04d}*"))
        ok = do_run(client, pid, "fine")  # later runs still pin and finish
        assert len(wait_run(client, pid, ok["run_reserved"])["refs"]) == 1

    def test_project_delete_removes_runs_and_refs(self, client, pid, tmp_path):
        add_ok(client, pid)
        do_run(client, pid, "with ref")
        assert any((proj_dir(tmp_path, pid) / "runs").glob("*.refs"))
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert not proj_dir(tmp_path, pid).exists()

    def test_upload_run_has_empty_refs_and_null_detail(self, client, pid):
        add_ok(client, pid)
        run = client.post(f"/api/projects/{pid}/upload", content=colour(RED, (896, 1184))).json()
        assert run["refs"] == [] and run["reference_resolution"] is None and run["ref_detail"] is None


# ---- SSE ---------------------------------------------------------------------------------------
class TestRefsEvent:
    def test_refs_event_on_create_patch_delete(self, app):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            r = (await c.post(f"/api/projects/{pid}/refs?name=a", content=colour(RED))).json()
            await c.patch(f"/api/projects/{pid}/refs/{r['id']}", json={"name": "b"})
            await c.delete(f"/api/projects/{pid}/refs/{r['id']}")
            return pid

        events, pid = collect_events(app, action, lambda ev, _: len([e for e in ev if e["event"] == "refs"]) >= 3)
        got = [e["data"] for e in events if e["event"] == "refs"]
        assert len(got) == 3 and all(g["project_id"] == pid for g in got)
        assert [len(g["refs"]) for g in got] == [1, 1, 0]
        assert got[0]["refs"][0]["name"] == "a" and got[1]["refs"][0]["name"] == "b"
        assert got[0]["refs"][0]["image_index_new"] == 1


# ---- engine conditioning (real Engine, stub model capturing generate_image) ---------------------
@pytest.fixture
def eng(monkeypatch, tmp_path):
    pytest.importorskip("mlx_vlm")
    from imageatelier import engine as eng_mod

    patch_parts(monkeypatch)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    calls: list[dict] = []
    cfg = {"resolution": 1024, "max_refs": 3}

    def load_image(self, key, snapshot, lora):
        def generate_image(**kw):
            calls.append(kw)
            return SimpleNamespace(image=Image.new("RGBA", (16, 16), (1, 2, 3, 255)))

        self.model = SimpleNamespace(text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None),
                                     tiling_config=None, generate_image=generate_image)
        self.snapshot, self.key = snapshot, key

    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", load_image)
    e = eng_mod.Engine(settings_provider=lambda: {"low_memory": False, **cfg})
    e.start()
    n = [0]

    def img(w, h, name=None):
        n[0] += 1
        p = tmp_path / (name or f"img{n[0]}.png")
        Image.new("RGB", (w, h), (9, 9, 9)).save(p)
        return str(p)

    def job(mode="edit", parent=(896, 1184), nrefs=0, **extra):
        j = {"kind": "generate", "prompt": "a cat", "mode": mode, "steps": 6, "turbo": True,
             "output_path": str(tmp_path / f"out{n[0]}.png"), **extra}
        n[0] += 1
        if mode == "edit":
            j["ref_image_path"] = img(*parent)
        j["ref_image_paths"] = [img(100 + 10 * i, 80) for i in range(nrefs)]
        return j

    def run(mode="edit", parent=(896, 1184), nrefs=0, resolution=None, **extra):
        if resolution is not None:
            cfg["resolution"] = resolution
        j = job(mode, parent, nrefs, **extra)
        jid = e.submit(j)
        end = time.time() + 15
        while time.time() < end:
            s = e.get(jid)
            if s["state"] in ("done", "failed", "cancelled"):
                assert s["state"] == "done", s
                return SimpleNamespace(kw=calls[-1], job=j, snap=s)
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    yield SimpleNamespace(run=run, job=job, engine=e, cfg=cfg, calls=calls, img=img)
    e.stop(timeout=10)


class TestEngineConditioning:
    def test_edit_with_two_refs(self, eng):
        out = eng.run("edit", parent=(416, 224), nrefs=2, resolution=320)
        assert out.kw["image_paths"] == [out.job["ref_image_path"], *out.job["ref_image_paths"]]
        assert (out.kw["width"], out.kw["height"]) == (416, 224)  # the parent's size, not a ref's

    def test_edit_without_refs_is_just_the_parent(self, eng):
        out = eng.run("edit", parent=(896, 1184), nrefs=0)
        assert out.kw["image_paths"] == [out.job["ref_image_path"]]
        assert (out.kw["width"], out.kw["height"]) == (896, 1184)

    def test_t2i_with_one_ref(self, eng):
        out = eng.run("t2i", nrefs=1, resolution=768)
        assert out.kw["image_paths"] == out.job["ref_image_paths"] and len(out.kw["image_paths"]) == 1
        assert (out.kw["width"], out.kw["height"]) == (768, 768)

    def test_t2i_with_several_refs_keeps_order(self, eng):
        out = eng.run("t2i", nrefs=3, resolution=1024)
        assert out.kw["image_paths"] == out.job["ref_image_paths"]
        assert (out.kw["width"], out.kw["height"]) == (1024, 1024)

    def test_t2i_without_refs_is_unchanged(self, eng):
        out = eng.run("t2i", nrefs=0, resolution=1024)
        assert not out.kw.get("image_paths") and out.kw.get("width") is None and out.kw.get("height") is None

    @pytest.mark.parametrize("nrefs,want", [(0, 512), (1, 1024), (2, 768), (3, 768)])
    def test_auto_reference_resolution(self, eng, nrefs, want):
        out = eng.run("edit", nrefs=nrefs, resolution=512 if nrefs == 0 else 1024)
        if nrefs == 0:
            want = 512  # 0 extra keeps the normal resolution
        assert out.kw["reference_resolution"] == want
        assert out.snap["result"]["reference_resolution"] == want and out.snap["result"]["ref_detail"] == "auto"

    def test_auto_with_no_refs_follows_the_resolution_setting(self, eng):
        assert eng.run("edit", nrefs=0, resolution=1024).kw["reference_resolution"] == 1024
        assert eng.run("edit", nrefs=0, resolution=2048).kw["reference_resolution"] == 2048

    @pytest.mark.parametrize("nrefs", [0, 1, 2, 3])
    @pytest.mark.parametrize("detail,want", [("high", 1024), ("standard", 768), ("low", 512)])
    def test_explicit_ref_detail_applies_to_every_count(self, eng, detail, want, nrefs):
        out = eng.run("edit", nrefs=nrefs, ref_detail=detail)
        assert out.kw["reference_resolution"] == want and out.snap["result"]["ref_detail"] == detail

    def test_ref_detail_from_settings_and_job_override(self, eng):
        eng.cfg["ref_detail"] = "low"
        assert eng.run("edit", nrefs=1).kw["reference_resolution"] == 512
        assert eng.run("edit", nrefs=1, ref_detail="high").kw["reference_resolution"] == 1024
        eng.cfg["ref_detail"] = "auto"

    def test_submit_rejects_too_many_refs(self, eng):
        eng.cfg["max_refs"] = 2
        with pytest.raises(ValueError, match="max 2 references enabled \\(got 3\\)"):
            eng.engine.submit(eng.job("edit", nrefs=3))
        eng.engine.submit(eng.job("edit", nrefs=2))  # exactly at the limit is fine

    def test_submit_rejects_more_than_ten_images(self, eng):
        eng.cfg["max_refs"] = 9
        with pytest.raises(ValueError):
            eng.engine.submit(eng.job("t2i", nrefs=10))  # 10 refs > max_refs cap of 9 (and > what the fork takes with a parent)
        eng.engine.submit(eng.job("edit", nrefs=9))  # current + 9 refs = 10 images: the maximum
        assert eng.engine.jobs()

    def test_submit_rejects_missing_files(self, eng):
        j = eng.job("edit", nrefs=1)
        j["ref_image_paths"].append("/nonexistent/ref.png")
        with pytest.raises(ValueError, match="not found"):
            eng.engine.submit(j)

    def test_submit_rejects_bad_ref_detail_and_shape(self, eng):
        with pytest.raises(ValueError):
            eng.engine.submit(eng.job("edit", ref_detail="ultra"))
        j = eng.job("edit")
        j["ref_image_paths"] = "not-a-list.png"
        with pytest.raises(ValueError):
            eng.engine.submit(j)

    def test_rejected_jobs_are_not_queued(self, eng):
        eng.cfg["max_refs"] = 1
        with pytest.raises(ValueError):
            eng.engine.submit(eng.job("edit", nrefs=2))
        assert eng.engine.jobs() == []


# ---- through the API with the real engine --------------------------------------------------------
class TestApiWithEngine:
    def test_run_json_carries_reference_resolution_and_detail(self, live):
        live.put("/api/settings", json={"low_memory": False})
        pid = new_project(live)["id"]
        for c in (RED, GREEN):
            assert add(live, pid, colour(c)).status_code == 201
        r = live.post(f"/api/projects/{pid}/runs", json={"prompt": "with refs"})
        assert r.status_code == 202
        run = wait_until(lambda: next((x for x in live.get(f"/api/projects/{pid}").json()["runs"]
                                       if x["run"] == r.json()["run_reserved"]), None), timeout=15)
        assert run["reference_resolution"] == 768 and run["ref_detail"] == "auto" and len(run["refs"]) == 2
        live.put("/api/settings", json={"ref_detail": "high"})
        r2 = live.post(f"/api/projects/{pid}/runs", json={"prompt": "again"})
        run2 = wait_until(lambda: next((x for x in live.get(f"/api/projects/{pid}").json()["runs"]
                                        if x["run"] == r2.json()["run_reserved"]), None), timeout=15)
        assert run2["reference_resolution"] == 1024 and run2["ref_detail"] == "high"
        assert [x["image_index"] for x in run2["refs"]] == [2, 3]  # the second run is an edit of the first

    def test_lowering_max_refs_below_the_enabled_count_makes_runs_409(self, live):
        pid = new_project(live)["id"]
        for c in (RED, GREEN, BLUE):
            add(live, pid, colour(c))
        assert sum(r["enabled"] for r in refs(live, pid)) == 3
        live.put("/api/settings", json={"max_refs": 1})
        r = live.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})
        assert r.status_code == 409 and "max 1 references enabled (got 3)" in r.json()["detail"]
        assert live.get(f"/api/projects/{pid}").json()["runs"] == []
        # disabling down to the limit makes it work again
        for ref in refs(live, pid)[1:]:
            live.patch(f"/api/projects/{pid}/refs/{ref['id']}", json={"enabled": False})
        assert live.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 202

    def test_rejected_run_leaves_no_partial_files(self, live, tmp_path):
        pid = new_project(live)["id"]
        for c in (RED, GREEN):
            add(live, pid, colour(c))
        live.put("/api/settings", json={"max_refs": 1})
        assert live.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 409
        runs = proj_dir(tmp_path, pid) / "runs"
        assert not runs.exists() or not any(runs.iterdir())
