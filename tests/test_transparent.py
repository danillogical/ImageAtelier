"""Transparent backgrounds: prompt wrapping, alpha rules, and parent flattening for plain edits (API.md)."""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import do_run, new_project, patch_parts, submit, wait_run, wait_until

PREFIX = "This is an RGBA image with transparency. "
SUFFIX = ". The image has alpha channel and the background is transparent."


def wrapped(prompt: str) -> str:
    return PREFIX + (prompt[:-1] if prompt.endswith(".") else prompt) + SUFFIX


def half_transparent(_job=None) -> Image.Image:
    """Left half fully transparent, right half opaque red, plus one half-transparent blue column."""
    im = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    for x in range(32, 64):
        for y in range(48):
            im.putpixel((x, y), (255, 0, 0, 255))
    for y in range(48):
        im.putpixel((32, y), (0, 0, 255, 128))
    return im


def runs_dir(tmp_path, pid):
    return tmp_path / "data" / "projects" / pid / "runs"


@pytest.fixture
def pid(client):
    return new_project(client)["id"]


def post_run(client, pid, prompt="a cat", **extra):
    r = client.post(f"/api/projects/{pid}/runs", json={"prompt": prompt, **extra})
    assert r.status_code == 202, r.text
    return r.json()


# ---- wrapping -------------------------------------------------------------------------------
class TestWrapping:
    @pytest.mark.parametrize("prompt", ["a red fox", "a red fox.", "a red fox..", "a red fox!", "a red fox?",
                                        "a fox. with a hat", "Fuchs, rot."])
    def test_exact_wording_strips_one_trailing_period(self, client, pid, engine, prompt):
        post_run(client, pid, prompt, transparent=True)
        assert engine.submitted[-1]["prompt"] == wrapped(prompt)
        assert engine.submitted[-1]["transparent"] is True

    def test_no_double_period(self, client, pid, engine):
        post_run(client, pid, "a cat.", transparent=True)
        sent = engine.submitted[-1]["prompt"]
        assert "cat. The image" in sent and ".." not in sent

    def test_only_one_period_is_stripped(self, client, pid, engine):
        post_run(client, pid, "a cat..", transparent=True)
        assert engine.submitted[-1]["prompt"] == PREFIX + "a cat." + SUFFIX  # the second period stays

    def test_non_transparent_prompt_is_unchanged(self, client, pid, engine):
        for flag in ({}, {"transparent": False}):
            post_run(client, pid, "a cat.", **flag)
            assert engine.submitted[-1]["prompt"] == "a cat." and not engine.submitted[-1]["transparent"]

    def test_run_json_stores_prompt_flag_and_model_prompt(self, client, pid, tmp_path):
        r = post_run(client, pid, "a red fox.", transparent=True)
        run = wait_run(client, pid, r["run_reserved"])
        assert run["prompt"] == "a red fox." and run["transparent"] is True
        assert run["model_prompt"] == wrapped("a red fox.")
        import json
        on_disk = json.loads((runs_dir(tmp_path, pid) / "0001.json").read_text())
        assert on_disk["prompt"] == "a red fox." and on_disk["transparent"] is True
        assert on_disk["model_prompt"] == wrapped("a red fox.")

    def test_non_transparent_model_prompt_equals_prompt(self, client, pid):
        run = wait_run(client, pid, post_run(client, pid, "a red fox.")["run_reserved"])
        assert run["transparent"] is False and run["model_prompt"] == run["prompt"] == "a red fox."

    def test_default_is_false(self, client, pid):
        run = wait_run(client, pid, post_run(client, pid, "x")["run_reserved"])
        assert run["transparent"] is False

    @pytest.mark.parametrize("value", ["yes", "true", 1, 0, None, []])
    def test_non_boolean_flag_400(self, client, pid, engine, value):
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "transparent": value})
        assert r.status_code == 400 and "detail" in r.json()
        assert engine.submitted == []

    def test_job_snapshot_shows_user_text_and_flag(self, client, pid, engine):
        engine.gate.clear()
        r = post_run(client, pid, "a red fox.", transparent=True)
        snap = next(j for j in client.get("/api/jobs").json() if j["id"] == r["job_id"])
        assert snap["prompt"] == "a red fox." and snap["model_prompt"] == wrapped("a red fox.")
        assert snap["transparent"] is True
        engine.gate.set()
        wait_run(client, pid, r["run_reserved"])
        snap = next(j for j in client.get("/api/jobs").json() if j["id"] == r["job_id"])
        assert snap["prompt"] == "a red fox." and snap["transparent"] is True

    def test_enhanced_text_is_wrapped_like_typed_text(self, client, pid, engine):
        r = post_run(client, pid, "A sleek red fox, studio lighting.", transparent=True, enhanced_from="fox")
        run = wait_run(client, pid, r["run_reserved"])
        assert engine.submitted[-1]["prompt"] == wrapped("A sleek red fox, studio lighting.")
        assert run["enhanced_from"] == "fox" and run["prompt"] == "A sleek red fox, studio lighting."

    def test_enhance_itself_is_not_wrapped(self, client, pid, engine):
        r = client.post(f"/api/projects/{pid}/enhance", json={"prompt": "a fox."})
        assert r.status_code == 202
        assert engine.submitted[-1]["prompt"] == "a fox."

    def test_gallery_latest_includes_transparent(self, client, pid):
        do_run(client, pid, "cutout", transparent=True)
        do_run(client, pid, "filled", transparent=False)

        def latest():
            return next(p for p in client.get("/api/projects").json() if p["id"] == pid)["latest"]

        assert latest()["transparent"] is False and latest()["run"] == 2
        client.patch(f"/api/projects/{pid}", json={"current_run": 1})
        assert latest()["transparent"] is True and latest()["run"] == 1

    def test_gallery_latest_is_null_without_runs(self, client, pid):
        assert next(p for p in client.get("/api/projects").json() if p["id"] == pid)["latest"] is None


# ---- finalize_alpha on synthetic data -----------------------------------------------------------
class TestFinalizeAlpha:
    @pytest.fixture
    def fa(self):
        from imageatelier.engine import finalize_alpha
        return finalize_alpha

    @staticmethod
    def ramp():
        im = Image.new("RGBA", (256, 1))
        for a in range(256):
            im.putpixel((a, 0), (10, 20 + a % 50, 30, a))
        return im

    @pytest.mark.parametrize("a,want", [(0, 0), (1, 0), (8, 0), (15, 0), (16, 16), (17, 17), (100, 100), (200, 200),
                                        (238, 238), (239, 239), (240, 255), (241, 255), (250, 255), (254, 255),
                                        (255, 255)])
    def test_transparent_thresholds(self, fa, a, want):
        out = fa(self.ramp(), True)
        assert out.mode == "RGBA" and out.getpixel((a, 0))[3] == want

    def test_transparent_keeps_colour_and_size(self, fa):
        src = self.ramp()
        out = fa(src, True)
        assert out.size == src.size
        assert [out.getpixel((x, 0))[:3] for x in range(256)] == [src.getpixel((x, 0))[:3] for x in range(256)]

    def test_normal_forces_all_opaque(self, fa):
        out = fa(self.ramp(), False)
        assert out.mode == "RGBA" and {out.getpixel((x, 0))[3] for x in range(256)} == {255}
        assert [out.getpixel((x, 0))[:3] for x in range(256)] == [self.ramp().getpixel((x, 0))[:3] for x in range(256)]

    def test_normal_decoder_leftovers_252_to_255(self, fa):
        im = Image.new("RGBA", (4, 1))
        for x, a in enumerate((252, 253, 254, 255)):
            im.putpixel((x, 0), (1, 2, 3, a))
        assert {fa(im, False).getpixel((x, 0))[3] for x in range(4)} == {255}

    @pytest.mark.parametrize("transparent", [True, False])
    def test_rgb_input_comes_out_rgba_opaque(self, fa, transparent):
        out = fa(Image.new("RGB", (5, 5), (9, 9, 9)), transparent)
        assert out.mode == "RGBA" and out.getpixel((2, 2)) == (9, 9, 9, 255)

    def test_does_not_mutate_the_input(self, fa):
        src = self.ramp()
        before = src.tobytes()
        fa(src, True)
        fa(src, False)
        assert src.tobytes() == before


# ---- engine saves RGBA with the alpha rules -------------------------------------------------------
@pytest.fixture
def eng(monkeypatch, tmp_path):
    pytest.importorskip("mlx_vlm")
    from imageatelier import engine as eng_mod

    patch_parts(monkeypatch)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    state = {"image": None, "calls": []}

    def load_image(self, key, snapshot, lora):
        def generate_image(**kw):
            state["calls"].append(kw)
            return SimpleNamespace(image=state["image"].copy())

        self.model = SimpleNamespace(text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None),
                                     tiling_config=None, generate_image=generate_image)
        self.snapshot, self.key = snapshot, key

    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", load_image)
    e = eng_mod.Engine(settings_provider=lambda: {"low_memory": False})
    e.start()
    n = [0]

    def run(image, **extra):
        state["image"] = image
        n[0] += 1
        out = tmp_path / f"out{n[0]}.png"
        jid = e.submit({"kind": "generate", "prompt": "a cat", "mode": "t2i", "steps": 6, "turbo": True,
                        "output_path": str(out), **extra})
        end = time.time() + 15
        while time.time() < end:
            s = e.get(jid)
            if s["state"] in ("done", "failed", "cancelled"):
                assert s["state"] == "done", s
                return SimpleNamespace(snap=s, saved=Image.open(out), calls=state["calls"])
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    yield run
    e.stop(timeout=10)


def alpha_image():
    im = Image.new("RGBA", (32, 32), (10, 20, 30, 3))  # faint background like the MLX decode
    for x in range(8, 24):
        for y in range(8, 24):
            im.putpixel((x, y), (200, 50, 50, 245))  # subject decoded slightly see-through
    im.putpixel((2, 2), (0, 255, 0, 120))  # soft edge
    return im


class TestEngineAlpha:
    def test_transparent_job_cleans_alpha_and_reports_it(self, eng):
        out = eng(alpha_image(), transparent=True)
        assert out.saved.mode == "RGBA"
        assert out.saved.getpixel((0, 0))[3] == 0 and out.saved.getpixel((16, 16))[3] == 255
        assert out.saved.getpixel((2, 2))[3] == 120  # soft edges are left as decoded
        assert out.snap["result"]["transparent"] is True and out.snap["transparent"] is True

    def test_normal_job_saves_fully_opaque_rgba(self, eng):
        out = eng(alpha_image())
        assert out.saved.mode == "RGBA"
        assert out.saved.getchannel("A").getextrema() == (255, 255)
        assert out.snap["result"]["transparent"] is False

    def test_rgb_output_is_still_saved_as_rgba(self, eng):
        out = eng(Image.new("RGB", (32, 32), (1, 2, 3)))
        assert out.saved.mode == "RGBA" and out.saved.getpixel((5, 5)) == (1, 2, 3, 255)

    def test_engine_passes_the_prompt_it_is_given(self, eng):
        out = eng(alpha_image(), prompt="exactly this prompt", transparent=True)
        assert out.calls[-1]["prompt"] == "exactly this prompt"  # wrapping is the API layer's job


# ---- flattening a transparent parent for plain edits ----------------------------------------------
class TestParentFlattening:
    def with_alpha_parent(self, client, engine, pid, **first):
        engine.image_factory = half_transparent
        do_run(client, pid, "cutout", transparent=True, **first)
        engine.image_factory = None  # later outputs are opaque again

    def test_plain_edit_of_a_cutout_gets_a_flattened_ref(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        r = submit(client, pid, "fill the background", transparent=False)
        assert r["mode"] == "edit"
        ref = engine.submitted[-1]["ref_image_path"]
        assert ref.endswith("runs/0002.parent.png")
        flat = Image.open(ref)
        orig = Image.open(runs_dir(tmp_path, pid) / "0001.png")
        assert orig.mode == "RGBA" and orig.getpixel((0, 0))[3] == 0  # the original stays a cut-out
        assert flat.size == orig.size == (64, 48)
        assert flat.mode == "RGB" or flat.getchannel("A").getextrema() == (255, 255)  # opaque
        flat = flat.convert("RGB")
        assert flat.getpixel((5, 5)) == (255, 255, 255)  # white where the parent was transparent
        assert flat.getpixel((50, 5)) == (255, 0, 0)  # opaque content unchanged
        # semi-transparent blue over white: ~ (127, 127, 255)
        px = flat.getpixel((32, 5))
        assert abs(px[0] - 127) <= 3 and abs(px[1] - 127) <= 3 and px[2] >= 252
        wait_run(client, pid, r["run_reserved"])
        assert (runs_dir(tmp_path, pid) / "0002.parent.png").is_file()  # kept as provenance
        assert wait_run(client, pid, 2)["parent_run"] == 1

    def test_default_transparent_false_also_flattens(self, client, pid, engine):
        self.with_alpha_parent(client, engine, pid)
        submit(client, pid, "no flag given")
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0002.parent.png")

    def test_transparent_edit_gets_the_rgba_original(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        r = submit(client, pid, "change the colour", transparent=True)
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0001.png")
        wait_run(client, pid, r["run_reserved"])
        assert not (runs_dir(tmp_path, pid) / "0002.parent.png").exists()

    def test_opaque_parent_needs_no_flattened_copy(self, client, pid, engine, tmp_path):
        do_run(client, pid, "opaque base")  # the stub's default output is opaque
        r = submit(client, pid, "edit")
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0001.png")
        wait_run(client, pid, r["run_reserved"])
        assert not list(runs_dir(tmp_path, pid).glob("*.parent.png"))

    def test_a_single_faint_pixel_of_transparency_triggers_flattening(self, client, pid, engine):
        def nearly_opaque(_job):
            im = Image.new("RGBA", (64, 48), (10, 200, 10, 255))
            im.putpixel((3, 3), (10, 200, 10, 254))
            return im

        engine.image_factory = nearly_opaque
        do_run(client, pid, "base")
        engine.image_factory = None
        submit(client, pid, "edit")
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0002.parent.png")

    def test_t2i_never_gets_a_flattened_copy(self, client, pid, engine, tmp_path):
        r = submit(client, pid, "first", transparent=False)
        assert r["mode"] == "t2i" and not engine.submitted[-1].get("ref_image_path")
        wait_run(client, pid, r["run_reserved"])
        assert not list(runs_dir(tmp_path, pid).glob("*.parent.png"))

    def test_edit_of_a_flattened_result_is_a_plain_edit(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        do_run(client, pid, "fill", transparent=False)  # run 2, opaque output
        r = submit(client, pid, "again")
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0002.png")
        wait_run(client, pid, r["run_reserved"])
        assert sorted(p.name for p in runs_dir(tmp_path, pid).glob("*.parent.png")) == ["0002.parent.png"]

    def test_flattening_depends_on_the_actual_parent_not_the_history(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        do_run(client, pid, "second", transparent=True)
        engine.gate.clear()
        r = submit(client, pid, "queued edit", transparent=False)  # parent = run 2 (opaque): no flatten
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0002.png")
        engine.gate.set()
        wait_run(client, pid, r["run_reserved"])

    def test_cancelled_run_removes_the_flattened_copy(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        engine.gate.clear()
        r = submit(client, pid, "doomed")
        f = runs_dir(tmp_path, pid) / f"{r['run_reserved']:04d}.parent.png"
        assert f.is_file()  # written at submit
        client.post(f"/api/jobs/{r['job_id']}/cancel")
        wait_until(lambda: not f.exists())
        engine.gate.set()
        assert not list(runs_dir(tmp_path, pid).glob(f"{r['run_reserved']:04d}*"))
        assert (runs_dir(tmp_path, pid) / "0001.png").is_file()  # the parent itself is untouched

    def test_failed_run_removes_the_flattened_copy(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        engine.outcome.append("fail")
        r = submit(client, pid, "doomed")
        f = runs_dir(tmp_path, pid) / f"{r['run_reserved']:04d}.parent.png"
        wait_until(lambda: engine.get(r["job_id"])["state"] == "failed")
        wait_until(lambda: not f.exists())
        assert not list(runs_dir(tmp_path, pid).glob(f"{r['run_reserved']:04d}*"))
        # the next attempt flattens again and completes
        ok = do_run(client, pid, "retry")
        assert (runs_dir(tmp_path, pid) / f"{ok['run_reserved']:04d}.parent.png").is_file()

    def test_project_delete_removes_the_flattened_copy(self, client, pid, engine, tmp_path):
        self.with_alpha_parent(client, engine, pid)
        do_run(client, pid, "fill")
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert not (tmp_path / "data" / "projects" / pid).exists()
