"""Model parts (image model + text encoder): statuses, composition, engine keys and run 409s."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webstudio import models
from webstudio.app import create_app

from conftest import StubEngine, new_project, wait_until
from test_heretic import live, models_dir  # noqa: F401 - fixtures: real Engine via create_app, loaders stubbed

IMG = "official-8bit"
TE_HF, TE_LOCAL = "heretic-8bit", "official-8bit"
LOCAL_FOLDER = "Qwen-Image-2.1-TextEncoder-MLX-8bit"


# ---- fake on-disk parts ---------------------------------------------------------------------
def make_part(path: Path, shard="model-00001-of-00001.safetensors", with_shard=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard}}))
    if with_shard:
        (path / shard).write_bytes(b"x")
    return path


def make_image_root(root: Path, model_index=True) -> Path:
    make_part(root / "transformer")
    make_part(root / "vae")
    (root / "processor").mkdir(parents=True, exist_ok=True)
    (root / "processor" / "tokenizer.json").write_text("{}")
    if model_index:
        (root / "model_index.json").write_text("{}")
    return root


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    hf, md = tmp_path / "hf", tmp_path / "models"
    hf.mkdir()
    md.mkdir()
    monkeypatch.setenv("WEBSTUDIO_HF_CACHE", str(hf))
    monkeypatch.setenv("WEBSTUDIO_MODELS_DIR", str(md))
    return SimpleNamespace(hf=hf, models=md)


def hf_snapshot(dirs, repo, revision) -> Path:
    return dirs.hf / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def install_hf_image(dirs) -> Path:
    m = models.IMAGE_MODELS[IMG]
    return make_image_root(hf_snapshot(dirs, m.repo, m.revision))


def install_hf_te(dirs) -> Path:
    t = models.TEXT_ENCODERS[TE_HF]
    return make_part(hf_snapshot(dirs, t.repo, t.revision) / "text_encoder")


def install_lora(dirs) -> Path:
    f = hf_snapshot(dirs, models.TURBO_LORA_REPO, models.TURBO_LORA_REVISION) / models.TURBO_LORA_FILENAME
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x")
    return f


def install_local_te(dirs) -> Path:
    return make_part(dirs.models / LOCAL_FOLDER / "text_encoder")


# ---- part statuses --------------------------------------------------------------------------
class TestStatuses:
    def test_image_not_downloaded_in_empty_cache(self, dirs):
        s = models.image_status(IMG)
        assert s["kind"] == "image" and s["name"] == IMG and s["state"] == "not_downloaded"
        assert s["label"] and s["approx_gb"] > 0 and s["turbo_lora_ready"] is False

    def test_image_ready_when_complete(self, dirs):
        root = install_hf_image(dirs)
        s = models.image_status(IMG)
        assert s["state"] == "ready" and Path(s["path"]) == root and s["turbo_lora_ready"] is False
        install_lora(dirs)
        assert models.image_status(IMG)["turbo_lora_ready"] is True

    def test_image_partial_is_not_ready(self, dirs):
        root = install_hf_image(dirs)
        (root / "transformer" / "model-00001-of-00001.safetensors").unlink()
        assert models.image_status(IMG)["state"] == "not_downloaded"

    def test_image_without_processor_is_not_ready(self, dirs):
        root = install_hf_image(dirs)
        (root / "processor" / "tokenizer.json").unlink()
        assert models.image_status(IMG)["state"] == "not_downloaded"

    def test_hf_text_encoder_not_downloaded_then_ready(self, dirs):
        s = models.te_status(TE_HF)
        assert s["kind"] == "text_encoder" and s["source"] == "hf" and s["state"] == "not_downloaded"
        te = install_hf_te(dirs)
        s = models.te_status(TE_HF)
        assert s["state"] == "ready" and Path(s["path"]) == te

    def test_local_text_encoder_not_installed_in_empty_models_dir(self, dirs):
        s = models.te_status(TE_LOCAL)
        assert s["source"] == "local" and s["state"] == "not_installed" and s["error"] is None
        assert LOCAL_FOLDER in s["hint"]

    def test_local_text_encoder_ready_with_fake_folder(self, dirs):
        te = install_local_te(dirs)
        s = models.te_status(TE_LOCAL)
        assert s["state"] == "ready" and Path(s["path"]) == te

    def test_local_text_encoder_incomplete_is_not_installed(self, dirs):
        make_part(dirs.models / LOCAL_FOLDER / "text_encoder", with_shard=False)
        assert models.te_status(TE_LOCAL)["state"] == "not_installed"

    def test_local_text_encoder_needs_text_encoder_subfolder(self, dirs):
        make_part(dirs.models / LOCAL_FOLDER)  # files directly in the folder are not enough
        assert models.te_status(TE_LOCAL)["state"] == "not_installed"

    def test_unknown_names_raise(self, dirs):
        with pytest.raises(ValueError):
            models.image_status("nope")
        with pytest.raises(ValueError):
            models.te_status("nope")


class TestResolveParts:
    def test_returns_parts(self, dirs):
        root, te = install_hf_image(dirs), install_hf_te(dirs)
        lora = install_lora(dirs)
        got_root, got_te, got_lora = models.resolve_parts(IMG, TE_HF, True)
        assert (got_root, got_te) == (root, te) and got_lora.name == lora.name
        assert models.resolve_parts(IMG, TE_HF, False)[2] is None

    def test_local_encoder(self, dirs):
        install_hf_image(dirs)
        te = install_local_te(dirs)
        assert models.resolve_parts(IMG, TE_LOCAL, False)[1] == te

    def test_errors_are_model_not_ready(self, dirs):
        with pytest.raises(models.ModelNotReady, match="not downloaded"):
            models.resolve_parts(IMG, TE_HF, False)  # image first
        install_hf_image(dirs)
        with pytest.raises(models.ModelNotReady, match="not downloaded"):
            models.resolve_parts(IMG, TE_HF, False)
        with pytest.raises(models.ModelNotReady, match="not installed"):
            models.resolve_parts(IMG, TE_LOCAL, False)
        install_hf_te(dirs)
        with pytest.raises(models.ModelNotReady, match="LoRA"):
            models.resolve_parts(IMG, TE_HF, True)
        assert models.resolve_parts(IMG, TE_HF, False)  # non-turbo needs no LoRA


# ---- compose --------------------------------------------------------------------------------
class TestCompose:
    @pytest.fixture
    def parts(self, tmp_path):
        return SimpleNamespace(
            root=make_image_root(tmp_path / "img"), te=make_part(tmp_path / "te1"),
            te2=make_part(tmp_path / "te2"), out=tmp_path / "compose")

    def test_builds_symlinks(self, parts):
        dest = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        assert dest == parts.out / f"{IMG}__{TE_HF}"
        expect = {"transformer": parts.root / "transformer", "vae": parts.root / "vae",
                  "processor": parts.root / "processor", "model_index.json": parts.root / "model_index.json",
                  "text_encoder": parts.te}
        assert {e.name for e in dest.iterdir()} == set(expect)
        for name, target in expect.items():
            link = dest / name
            assert link.is_symlink() and os.readlink(link) == str(target), name
        assert (dest / "transformer" / "config.json").is_file()  # resolves through the link
        assert (dest / "text_encoder" / "model-00001-of-00001.safetensors").is_file()

    def test_optional_parts_only_when_present(self, tmp_path):
        root = make_image_root(tmp_path / "img", model_index=False)
        dest = models.compose(IMG, TE_HF, root, make_part(tmp_path / "te"), tmp_path / "c")
        assert "model_index.json" not in {e.name for e in dest.iterdir()}

    def test_idempotent(self, parts):
        dest = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        before = {e.name: os.lstat(e).st_ino for e in dest.iterdir()}
        dir_ino = os.stat(dest).st_ino
        again = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        assert again == dest and os.stat(dest).st_ino == dir_ino
        assert {e.name: os.lstat(e).st_ino for e in dest.iterdir()} == before  # nothing was rebuilt
        assert sorted(p.name for p in parts.out.iterdir()) == [dest.name]  # no temp leftovers

    def test_repairs_wrong_link(self, parts):
        dest = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        (dest / "text_encoder").unlink()
        os.symlink(parts.te2, dest / "text_encoder")  # points at the wrong encoder
        fixed = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        assert fixed == dest and os.readlink(dest / "text_encoder") == str(parts.te)
        assert sorted(p.name for p in parts.out.iterdir()) == [dest.name]

    def test_repairs_dangling_and_extra_entries(self, parts):
        dest = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        (dest / "vae").unlink()
        os.symlink(parts.root / "missing", dest / "vae")  # dangling
        (dest / "stray.txt").write_text("x")
        models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        assert os.readlink(dest / "vae") == str(parts.root / "vae") and (dest / "vae" / "config.json").is_file()
        assert not (dest / "stray.txt").exists()

    def test_repairs_a_plain_directory_in_the_way(self, parts):
        dest = parts.out / f"{IMG}__{TE_HF}"
        dest.mkdir(parents=True)
        (dest / "junk").write_text("x")
        models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        assert (dest / "text_encoder").is_symlink() and not (dest / "junk").exists()

    def test_text_encoder_switch_gets_its_own_folder_sharing_the_image(self, parts):
        a = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        b = models.compose(IMG, TE_LOCAL, parts.root, parts.te2, parts.out)
        assert a != b and b.name == f"{IMG}__{TE_LOCAL}"
        assert os.readlink(a / "transformer") == os.readlink(b / "transformer") == str(parts.root / "transformer")
        assert os.readlink(a / "text_encoder") != os.readlink(b / "text_encoder")
        assert not any(p.is_file() and not p.is_symlink() for d in (a, b) for p in d.rglob("*") if p.parent == d)

    def test_validates_missing_shards(self, parts, tmp_path):
        bad_te = make_part(tmp_path / "bad_te", with_shard=False)
        with pytest.raises(models.ModelNotReady):
            models.compose(IMG, TE_HF, parts.root, bad_te, parts.out)
        assert not (parts.out / f"{IMG}__{TE_HF}").exists()
        assert [p for p in parts.out.iterdir()] == []  # the temp dir was cleaned up
        (parts.root / "vae" / "model-00001-of-00001.safetensors").unlink()
        with pytest.raises(models.ModelNotReady):
            models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)

    def test_failed_recompose_does_not_destroy_a_good_folder(self, parts, tmp_path):
        dest = models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)
        bad_te = make_part(tmp_path / "bad_te", with_shard=False)
        with pytest.raises(models.ModelNotReady):
            models.compose(IMG, TE_HF, parts.root, bad_te, parts.out)
        assert os.readlink(dest / "text_encoder") == str(parts.te)

    def test_missing_processor_tokenizer(self, parts):
        (parts.root / "processor" / "tokenizer.json").unlink()
        with pytest.raises(models.ModelNotReady):
            models.compose(IMG, TE_HF, parts.root, parts.te, parts.out)


# ---- engine: compose + keys (real Engine, real resolve/compose, loader stubbed) -----------------
@pytest.fixture
def composing(monkeypatch, dirs, tmp_path):
    pytest.importorskip("mlx_vlm")
    from webstudio import engine as eng_mod

    install_hf_image(dirs)
    install_hf_te(dirs)
    install_lora(dirs)
    install_local_te(dirs)
    loads: list[tuple] = []

    def load_image(self, key, snapshot, lora):
        loads.append((key, Path(snapshot), lora))
        self.model = SimpleNamespace(
            text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None), tiling_config=None,
            generate_image=lambda **kw: SimpleNamespace(image=Image.new("RGBA", (16, 16), (1, 2, 3, 255))))
        self.snapshot, self.key = snapshot, key

    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", load_image)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    e = eng_mod.Engine(settings_provider=lambda: {"low_memory": False}, compose_dir=tmp_path / "compose")
    e.start()
    n = [0]

    def gen(**kw):
        n[0] += 1
        job = {"kind": "generate", "prompt": "a cat", "mode": "t2i", "steps": 6, "turbo": True,
               "output_path": str(tmp_path / f"out{n[0]}.png"), **kw}
        jid = e.submit(job)
        end = time.time() + 15
        while time.time() < end:
            snap = e.get(jid)
            if snap["state"] in ("done", "failed", "cancelled"):
                return snap
            time.sleep(0.02)
        raise AssertionError(f"job did not finish: {e.get(jid)}")

    yield SimpleNamespace(engine=e, loads=loads, gen=gen, compose=tmp_path / "compose", dirs=dirs)
    e.stop(timeout=10)


class TestEngineParts:
    def test_default_key_and_composed_folder(self, composing):
        snap = composing.gen()
        assert snap["state"] == "done", snap
        (key, folder, lora), = composing.loads
        assert key == ("image", "official-8bit", "heretic-8bit", True)
        assert folder == composing.compose / "official-8bit__heretic-8bit"
        assert lora is not None and lora.name == models.TURBO_LORA_FILENAME
        assert os.readlink(folder / "text_encoder") == str(models.te_local_dir(TE_HF))
        assert composing.engine.resident() == {"key": list(key), "state": "loaded"}

    def test_results_and_snapshots_carry_parts_not_variant(self, composing):
        snap = composing.gen()
        for d in (snap, snap["result"]):
            assert d["image_model"] == "official-8bit" and d["text_encoder"] == "heretic-8bit"
            assert d["turbo"] is True
            assert "variant" not in d

    def test_text_encoder_switch_reloads_but_same_key_does_not(self, composing):
        composing.gen()
        composing.gen()
        assert len(composing.loads) == 1  # same key: no reload
        snap = composing.gen(text_encoder="official-8bit")
        assert snap["state"] == "done" and snap["result"]["text_encoder"] == "official-8bit"
        assert [k for k, _, _ in composing.loads] == [
            ("image", "official-8bit", "heretic-8bit", True), ("image", "official-8bit", "official-8bit", True)]
        assert composing.loads[1][1] == composing.compose / "official-8bit__official-8bit"
        assert os.readlink(composing.loads[1][1] / "text_encoder") == str(models.te_local_dir(TE_LOCAL))
        composing.gen(text_encoder="official-8bit")
        assert len(composing.loads) == 2
        composing.gen()  # back again: a reload, into the first composed folder
        assert len(composing.loads) == 3 and composing.loads[2][1] == composing.loads[0][1]

    def test_turbo_switch_reloads(self, composing):
        composing.gen()
        snap = composing.gen(turbo=False)
        assert snap["result"]["turbo"] is False
        assert [k[3] for k, _, _ in composing.loads] == [True, False]
        assert composing.loads[1][2] is None  # no LoRA when turbo is off

    def test_not_installed_encoder_fails_job_and_keeps_resident_model(self, composing):
        composing.gen()
        import shutil
        shutil.rmtree(composing.dirs.models / LOCAL_FOLDER)
        snap = composing.gen(text_encoder="official-8bit")
        assert snap["state"] == "failed" and "not installed" in snap["error"]
        assert len(composing.loads) == 1
        assert composing.engine.resident()["key"] == ["image", "official-8bit", "heretic-8bit", True]
        assert composing.gen()["state"] == "done" and len(composing.loads) == 1

    @pytest.mark.parametrize("variant", ["8bit", "4bit", "nope"])
    def test_legacy_variant_key_rejected_at_submit(self, composing, variant):
        with pytest.raises(ValueError, match="variant was replaced by image_model/text_encoder"):
            composing.engine.submit({"kind": "generate", "prompt": "x", "mode": "t2i", "output_path": "/tmp/x.png",
                                     "variant": variant})
        assert composing.engine.jobs() == []  # nothing was queued

    def test_unknown_part_rejected_at_submit(self, composing):
        with pytest.raises(ValueError):
            composing.engine.submit({"kind": "generate", "prompt": "x", "mode": "t2i", "output_path": "/tmp/x.png",
                                     "text_encoder": "nope"})
        with pytest.raises(ValueError):
            composing.engine.submit({"kind": "generate", "prompt": "x", "mode": "t2i", "output_path": "/tmp/x.png",
                                     "image_model": "nope"})


# ---- through the API: settings-driven defaults, run records, 409s ---------------------------------
def run_done(c, pid, **body):
    r = c.post(f"/api/projects/{pid}/runs", json={"prompt": "a cat", **body})
    assert r.status_code == 202, r.text
    n = r.json()["run_reserved"]

    def find():
        runs = c.get(f"/api/projects/{pid}").json()["runs"]
        return next((x for x in runs if x["run"] == n), None)
    return wait_until(find, timeout=15)


class TestApiParts:
    def test_defaults_from_settings_and_run_records(self, live):
        # low_memory offloads the text encoder between jobs and reloads it from disk, which the fake model can't do
        assert live.put("/api/settings", json={"low_memory": False}).status_code == 200
        pid = new_project(live)["id"]
        run = run_done(live, pid)
        assert run["image_model"] == "official-8bit" and run["text_encoder"] == "heretic-8bit"
        assert "variant" not in run
        assert live.loads[-1] == ("image", "official-8bit", "heretic-8bit", True)
        assert live.put("/api/settings", json={"text_encoder": "official-8bit"}).status_code == 200
        run = run_done(live, pid)
        assert run["text_encoder"] == "official-8bit"
        assert live.loads[-1] == ("image", "official-8bit", "official-8bit", True)
        n = len(live.loads)
        run_done(live, pid)
        assert len(live.loads) == n  # same key again: no reload
        run = run_done(live, pid, text_encoder="heretic-8bit")  # per-run override of the setting
        assert run["text_encoder"] == "heretic-8bit" and len(live.loads) == n + 1

    def test_job_snapshot_in_api_jobs(self, live):
        pid = new_project(live)["id"]
        r = live.post(f"/api/projects/{pid}/runs", json={"prompt": "a cat", "turbo": False, "steps": 10})
        jid = r.json()["job_id"]
        job = wait_until(lambda: next((j for j in live.get("/api/jobs").json()
                                       if j["id"] == jid and j["state"] == "done"), None), timeout=15)
        for d in (job, job["result"]):
            assert d["image_model"] == "official-8bit" and d["text_encoder"] == "heretic-8bit"
            assert d["turbo"] is False and "variant" not in d


class TestRunConflicts:
    """Real part statuses (nothing patched), pointed at empty tmp dirs."""

    @pytest.fixture
    def client(self, dirs, tmp_path):
        with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "data")) as c:
            yield c

    def test_everything_missing_is_409(self, client, dirs):
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})
        assert r.status_code == 409 and "image model" in r.json()["detail"]

    def test_image_ready_but_default_encoder_missing(self, client, dirs):
        install_hf_image(dirs)
        install_lora(dirs)
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})
        assert r.status_code == 409 and "text encoder" in r.json()["detail"]

    def test_local_encoder_not_installed_names_it_and_hints(self, client, dirs):
        install_hf_image(dirs)
        install_lora(dirs)
        install_hf_te(dirs)
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "text_encoder": "official-8bit"})
        assert r.status_code == 409
        assert "installed" in r.json()["detail"] and LOCAL_FOLDER in r.json()["detail"]
        client.put("/api/settings", json={"text_encoder": "official-8bit"})  # same via the setting
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"}).status_code == 409

    def test_turbo_without_lora_is_409_but_standard_mode_is_fine(self, client, dirs):
        install_hf_image(dirs)
        install_hf_te(dirs)
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})  # turbo is the default
        assert r.status_code == 409 and "turbo" in r.json()["detail"].lower()
        assert client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "turbo": False}).status_code == 202

    def test_all_parts_present_is_accepted(self, client, dirs):
        install_hf_image(dirs)
        install_hf_te(dirs)
        install_lora(dirs)
        pid = new_project(client)["id"]
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x"})
        assert r.status_code == 202
        install_local_te(dirs)
        r = client.post(f"/api/projects/{pid}/runs", json={"prompt": "x", "text_encoder": "official-8bit"})
        assert r.status_code == 202

    def test_models_listing_reflects_real_part_states(self, client, dirs):
        lst = {(m["kind"], m["name"]): m for m in client.get("/api/models").json()}
        assert lst[("image", IMG)]["state"] == "not_downloaded"
        assert lst[("text_encoder", TE_HF)]["state"] == "not_downloaded"
        assert lst[("text_encoder", TE_LOCAL)]["state"] == "not_installed" and lst[("text_encoder", TE_LOCAL)]["hint"]
        install_local_te(dirs)
        install_hf_image(dirs)
        lst = {(m["kind"], m["name"]): m for m in client.get("/api/models").json()}
        assert lst[("text_encoder", TE_LOCAL)]["state"] == "ready" and lst[("image", IMG)]["state"] == "ready"

    def test_local_encoder_download_is_409(self, client, dirs):
        r = client.post("/api/models/text_encoder/official-8bit/download")
        assert r.status_code == 409 and LOCAL_FOLDER in r.json()["detail"]
