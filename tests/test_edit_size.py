"""Every edit keeps its parent's exact size: the engine passes the reference image's width/height (API.md)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import patch_parts


@pytest.fixture
def eng(monkeypatch, tmp_path):
    pytest.importorskip("mlx_vlm")
    from webstudio import engine as eng_mod

    patch_parts(monkeypatch)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    calls: list[dict] = []
    cfg = {"resolution": 1024}

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

    def ref(w, h):
        p = tmp_path / f"ref{w}x{h}.png"
        Image.new("RGB", (w, h), (9, 9, 9)).save(p)
        return str(p)

    def run(mode="edit", ref_size=None, resolution=1024):
        cfg["resolution"] = resolution
        n[0] += 1
        job = {"kind": "generate", "prompt": "a cat", "mode": mode, "steps": 6, "turbo": True,
               "output_path": str(tmp_path / f"out{n[0]}.png")}
        if mode == "edit":
            job["ref_image_path"] = ref(*ref_size)
        jid = e.submit(job)
        end = time.time() + 15
        while time.time() < end:
            s = e.get(jid)
            if s["state"] in ("done", "failed", "cancelled"):
                assert s["state"] == "done", s
                return calls[-1]
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    yield SimpleNamespace(run=run, calls=calls)
    e.stop(timeout=10)


class TestEditKeepsParentSize:
    @pytest.mark.parametrize("resolution", [320, 1024])
    def test_edit_passes_the_reference_size(self, eng, resolution):
        kw = eng.run("edit", (416, 224), resolution)  # 16:9 at 320 would re-derive 448x224 from the ratio
        assert kw["width"] == 416 and kw["height"] == 224

    @pytest.mark.parametrize("size", [(896, 1184), (1376, 768), (768, 1376), (1024, 1024), (32, 32), (2048, 64)])
    def test_other_aligned_sizes(self, eng, size):
        kw = eng.run("edit", size, 1024)
        assert (kw["width"], kw["height"]) == size

    def test_size_follows_the_ref_not_the_resolution_setting(self, eng):
        a = eng.run("edit", (896, 1184), 512)
        b = eng.run("edit", (896, 1184), 2048)
        assert (a["width"], a["height"]) == (b["width"], b["height"]) == (896, 1184)

    def test_t2i_passes_no_size(self, eng):
        kw = eng.run("t2i", None, 320)
        assert kw.get("width") is None and kw.get("height") is None

    @pytest.mark.parametrize("size", [(1000, 700), (1024, 700), (1000, 704), (33, 64), (417, 224)])
    def test_unaligned_ref_passes_no_size(self, eng, size):
        kw = eng.run("edit", size, 1024)
        assert kw.get("width") is None and kw.get("height") is None  # falls back to the model's own sizing

    def test_resolution_is_still_passed(self, eng):
        kw = eng.run("edit", (416, 224), 320)
        assert kw.get("output_resolution") == 320 or kw.get("reference_resolution") == 320
