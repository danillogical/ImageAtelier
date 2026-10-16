"""Heretic rewriters (local models, T2I and I2I halves) and the pe_variant setting, against API.md."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webstudio import models
from webstudio.app import create_app

from conftest import StubEngine, new_project, patch_parts, wait_until

HERETIC = "t2i-heretic-8bit"
HERETIC_I2I = "i2i-heretic-8bit"
FOLDERS = {"t2i": "Qwen-Image-2.1-PE-T2I-Heretic-MLX-8bit", "i2i": "Qwen-Image-2.1-PE-I2I-Heretic-MLX-8bit"}


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("WEBSTUDIO_MODELS_DIR", str(d))
    return d


def install_heretic(models_dir, prompt=True, shard=True, mode="t2i"):
    f = models_dir / FOLDERS[mode]
    f.mkdir(exist_ok=True)
    (f / "config.json").write_text("{}")
    (f / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}))
    if shard:
        (f / "model-00001-of-00001.safetensors").write_bytes(b"x")
    if prompt:
        (f / "system_prompt.txt").write_text("You rewrite prompts.")
    return f


def install_both(models_dir):
    install_heretic(models_dir, mode="t2i")
    install_heretic(models_dir, mode="i2i")


@pytest.fixture
def stub_client(ready, models_dir, tmp_path):
    """App over the stub engine; enhancer status is the real one (official ones are simply not downloaded)."""
    with TestClient(create_app(engine=StubEngine(), data_dir=tmp_path / "data")) as c:
        yield c


def pe_entries(c):
    return {m["name"]: m for m in c.get("/api/models").json() if m["kind"] == "pe"}


# ---- setting --------------------------------------------------------------------------------
class TestSetting:
    def test_default(self, stub_client):
        assert stub_client.get("/api/settings").json()["pe_variant"] == "official"

    @pytest.mark.parametrize("val", ["official", "heretic"])
    def test_valid(self, stub_client, val):
        r = stub_client.put("/api/settings", json={"pe_variant": val})
        assert r.status_code == 200 and r.json()["pe_variant"] == val
        assert stub_client.get("/api/settings").json()["pe_variant"] == val

    @pytest.mark.parametrize("val", ["Heretic", "other", "", 1, None, True])
    def test_invalid_changes_nothing(self, stub_client, val):
        before = stub_client.get("/api/settings").json()
        r = stub_client.put("/api/settings", json={"pe_variant": val})
        assert r.status_code == 400 and "detail" in r.json()
        assert stub_client.get("/api/settings").json() == before


# ---- models list / status / downloads ------------------------------------------------------
class TestLocalModel:
    def test_not_installed_in_empty_dir(self, stub_client, models_dir):
        m = pe_entries(stub_client)[HERETIC]
        assert m["kind"] == "pe" and m["mode"] == "t2i" and m["quant"] == "8bit"
        assert m["variant"] == "heretic" and m["source"] == "local" and m["approx_gb"] == pytest.approx(10.4)
        assert m["state"] == "not_installed" and m.get("hint")

    def test_official_entries_carry_variant_and_source(self, stub_client):
        pe = pe_entries(stub_client)
        assert set(pe) == {"t2i-8bit", "i2i-8bit", HERETIC, HERETIC_I2I}
        for name in ("t2i-8bit", "i2i-8bit"):
            assert pe[name]["variant"] == "official" and pe[name]["source"] == "hf"

    def test_i2i_half_listed_with_its_own_folder_status(self, stub_client, models_dir):
        m = pe_entries(stub_client)[HERETIC_I2I]
        assert m["kind"] == "pe" and m["mode"] == "i2i" and m["quant"] == "8bit" and m["variant"] == "heretic"
        assert m["source"] == "local" and m["approx_gb"] == pytest.approx(10.4)
        assert m["state"] == "not_installed" and m.get("hint") and FOLDERS["i2i"] in m["hint"]
        install_heretic(models_dir, mode="t2i")  # the halves are independent
        pe = pe_entries(stub_client)
        assert pe[HERETIC]["state"] == "ready" and pe[HERETIC_I2I]["state"] == "not_installed"
        install_heretic(models_dir, mode="i2i")
        assert pe_entries(stub_client)[HERETIC_I2I]["state"] == "ready"

    def test_both_heretic_halves_listed_and_ordered_after_the_official_ones(self, stub_client):
        names = [m["name"] for m in stub_client.get("/api/models").json() if m["kind"] == "pe"]
        assert sorted(names) == sorted(["t2i-8bit", "i2i-8bit", HERETIC, HERETIC_I2I]) and len(names) == 4

    def test_ready_with_complete_folder(self, stub_client, models_dir):
        install_heretic(models_dir)
        assert pe_entries(stub_client)[HERETIC]["state"] == "ready"

    @pytest.mark.parametrize("kw", [{"prompt": False}, {"shard": False}])
    def test_incomplete_folder_is_not_installed(self, stub_client, models_dir, kw):
        install_heretic(models_dir, **kw)
        assert pe_entries(stub_client)[HERETIC]["state"] == "not_installed"

    def test_missing_config_is_not_installed(self, stub_client, models_dir):
        (install_heretic(models_dir) / "config.json").unlink()
        assert pe_entries(stub_client)[HERETIC]["state"] == "not_installed"

    @pytest.mark.parametrize("installed", [False, True])
    @pytest.mark.parametrize("name", [HERETIC, HERETIC_I2I])
    def test_download_is_409(self, stub_client, models_dir, installed, name):
        if installed:
            install_both(models_dir)
        r = stub_client.post(f"/api/models/pe/{name}/download")
        assert r.status_code == 409 and "detail" in r.json()

    def test_enhance_409_when_selected_but_not_installed(self, stub_client, models_dir):
        stub_client.put("/api/settings", json={"pe_variant": "heretic"})
        pid = new_project(stub_client)["id"]
        r = stub_client.post(f"/api/projects/{pid}/enhance", json={"prompt": "a cat"})
        assert r.status_code == 409 and "not installed" in r.json()["detail"]
        assert HERETIC in r.json()["detail"]

    def test_enhance_ok_when_installed(self, stub_client, models_dir):
        install_heretic(models_dir)
        stub_client.put("/api/settings", json={"pe_variant": "heretic"})
        pid = new_project(stub_client)["id"]
        r = stub_client.post(f"/api/projects/{pid}/enhance", json={"prompt": "a cat"})
        assert r.status_code == 202 and r.json()["mode"] == "t2i"

    def test_i2i_enhance_409_names_the_missing_i2i_half(self, stub_client, models_dir):
        install_heretic(models_dir, mode="t2i")  # only the T2I half is installed
        stub_client.put("/api/settings", json={"pe_variant": "heretic"})
        pid = new_project(stub_client)["id"]
        stub_client.post(f"/api/projects/{pid}/runs", json={"prompt": "base", "turbo": False})
        wait_until(lambda: stub_client.get(f"/api/projects/{pid}").json()["current_run"] == 1, timeout=15)
        r = stub_client.post(f"/api/projects/{pid}/enhance", json={"prompt": "edit it"})
        assert r.status_code == 409 and HERETIC_I2I in r.json()["detail"] and "not installed" in r.json()["detail"]
        assert HERETIC not in r.json()["detail"].replace(HERETIC_I2I, "")
        stub_client.put("/api/settings", json={"pe_variant": "official"})  # official i2i is unaffected
        assert stub_client.post(f"/api/projects/{pid}/enhance", json={"prompt": "edit it"}).status_code == 202
        stub_client.put("/api/settings", json={"pe_variant": "heretic"})
        install_heretic(models_dir, mode="i2i")
        r = stub_client.post(f"/api/projects/{pid}/enhance", json={"prompt": "edit it"})
        assert r.status_code == 202 and r.json()["mode"] == "i2i"

    def test_old_setting_name_is_rejected(self, stub_client):
        before = stub_client.get("/api/settings").json()
        r = stub_client.put("/api/settings", json={"pe_t2i_variant": "heretic"})
        assert r.status_code == 400 and r.json()["detail"] == "pe_t2i_variant was replaced by pe_variant"
        assert stub_client.get("/api/settings").json() == before and "pe_t2i_variant" not in before


# ---- end to end with the real engine (loaders stubbed) --------------------------------------
@pytest.fixture
def live(monkeypatch, models_dir, ready, tmp_path):
    """create_app() builds the real Engine (settings flow from the API); only model loading is stubbed."""
    mlx_vlm = pytest.importorskip("mlx_vlm")
    from webstudio import engine as eng_mod

    loads: list[tuple] = []
    real_resolve_pe, real_pe_status = models.resolve_pe, models.pe_status

    def pe_status(name):
        if models.PE_MODELS[name].source == "local":
            return real_pe_status(name)
        m = models.PE_MODELS[name]
        return {"name": name, "state": "ready", "mode": m.mode, "quant": m.quant}

    def resolve_pe(mode, quant):
        if quant == "heretic-8bit":
            return real_resolve_pe(mode, quant)  # exercises the real installed-folder check
        return Path("/nowhere"), "official system prompt"

    def load_image(self, key, snapshot, lora):
        loads.append(key)
        self.model = SimpleNamespace(
            text_encoder=object(), prompt_cache=SimpleNamespace(clear=lambda: None), tiling_config=None,
            generate_image=lambda **kw: SimpleNamespace(image=Image.new("RGBA", (16, 16), (1, 2, 3, 255))))
        self.snapshot, self.key = snapshot, key

    def load_pe(self, key, path, system_prompt):
        loads.append(key)
        self.model = object()
        self.processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
        self.system_prompt, self.key = system_prompt, key

    def stream(model, processor, prompt, image=None, max_tokens=0, **kw):
        yield SimpleNamespace(text='</think>{"rewritten_prompt": "better"}', finish_reason="stop", generation_tps=10.0)

    monkeypatch.setattr(models, "pe_status", pe_status)
    patch_parts(monkeypatch)
    monkeypatch.setattr(models, "resolve_pe", resolve_pe)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", load_image)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_pe", load_pe)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    with TestClient(create_app(data_dir=tmp_path / "data")) as c:
        c.loads = loads
        yield c


def enhance_done(c, pid, **body):
    r = c.post(f"/api/projects/{pid}/enhance", json={"prompt": "a cat", **body})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]

    def find():
        j = next((j for j in c.get("/api/jobs").json() if j["id"] == jid), None)
        return j if j and j["state"] in ("done", "failed", "cancelled") else None
    job = wait_until(find, timeout=15)
    assert job["state"] == "done", job
    return job


def set_variant(c, v, **extra):
    assert c.put("/api/settings", json={"pe_variant": v, **extra}).status_code == 200


class TestEngineKeys:
    def test_heretic_t2i_uses_heretic_key_and_reports_rewriter(self, live, models_dir):
        install_heretic(models_dir)
        set_variant(live, "heretic")
        pid = new_project(live)["id"]
        job = enhance_done(live, pid)
        assert job["rewriter"] == "heretic-8bit" and job["result"]["rewriter"] == "heretic-8bit"
        assert job["result"]["rewritten_prompt"] == "better"
        assert live.loads == [("pe_t2i", "heretic-8bit")]
        assert live.get("/api/model").json() == {"state": "loaded", "key": ["pe_t2i", "heretic-8bit"]}

    def test_official_reports_rewriter(self, live):
        pid = new_project(live)["id"]
        job = enhance_done(live, pid)
        assert job["rewriter"] == "official-8bit" and job["result"]["rewriter"] == "official-8bit"
        enhance_done(live, pid)
        assert live.loads == [("pe_t2i", "8bit")]  # same official key: no reload
        assert live.post(f"/api/projects/{pid}/enhance", json={"prompt": "x", "quant": "4bit"}).status_code == 400

    def test_consecutive_heretic_enhances_do_not_reload(self, live, models_dir):
        install_heretic(models_dir)
        set_variant(live, "heretic")
        pid = new_project(live)["id"]
        enhance_done(live, pid)
        enhance_done(live, pid)
        enhance_done(live, pid)
        assert live.loads == [("pe_t2i", "heretic-8bit")]

    def test_switching_to_official_reloads_and_back(self, live, models_dir):
        install_heretic(models_dir)
        set_variant(live, "heretic")
        pid = new_project(live)["id"]
        enhance_done(live, pid)
        set_variant(live, "official")
        assert enhance_done(live, pid)["rewriter"] == "official-8bit"
        set_variant(live, "heretic")
        assert enhance_done(live, pid)["rewriter"] == "heretic-8bit"
        assert live.loads == [("pe_t2i", "heretic-8bit"), ("pe_t2i", "8bit"), ("pe_t2i", "heretic-8bit")]

    def test_heretic_t2i_regardless_of_quant_option(self, live, models_dir):
        install_heretic(models_dir)
        set_variant(live, "heretic", pe_quant="8bit")
        pid = new_project(live)["id"]
        assert enhance_done(live, pid)["rewriter"] == "heretic-8bit"
        assert enhance_done(live, pid, quant="8bit")["rewriter"] == "heretic-8bit"
        assert live.loads == [("pe_t2i", "heretic-8bit")]
        # 4-bit is gone: still a 400 even though Heretic ignores the quant
        r = live.post(f"/api/projects/{pid}/enhance", json={"prompt": "x", "quant": "4bit"})
        assert r.status_code == 400

    def _project_with_run(self, live):
        pid = new_project(live)["id"]
        assert live.post(f"/api/projects/{pid}/runs", json={"prompt": "base"}).status_code == 202
        wait_until(lambda: live.get(f"/api/projects/{pid}").json()["current_run"] == 1, timeout=15)
        return pid

    def test_heretic_i2i_uses_heretic_i2i_key(self, live, models_dir):
        install_both(models_dir)
        set_variant(live, "heretic")
        pid = self._project_with_run(live)
        job = enhance_done(live, pid)
        assert job["mode"] == "i2i" and job["rewriter"] == "heretic-8bit" and job["result"]["rewriter"] == "heretic-8bit"
        assert live.loads[-1] == ("pe_i2i", "heretic-8bit")
        assert ("pe_i2i", "8bit") not in live.loads
        assert live.get("/api/model").json() == {"state": "loaded", "key": ["pe_i2i", "heretic-8bit"]}

    def test_consecutive_heretic_i2i_enhances_do_not_reload(self, live, models_dir):
        install_both(models_dir)
        set_variant(live, "heretic")
        pid = self._project_with_run(live)
        n = len(live.loads)
        for _ in range(3):
            enhance_done(live, pid)
        assert live.loads[n:] == [("pe_i2i", "heretic-8bit")]

    def test_heretic_i2i_regardless_of_quant_option(self, live, models_dir):
        install_both(models_dir)
        set_variant(live, "heretic", pe_quant="8bit")
        pid = self._project_with_run(live)
        assert enhance_done(live, pid, quant="8bit")["rewriter"] == "heretic-8bit"
        assert live.post(f"/api/projects/{pid}/enhance", json={"prompt": "x", "quant": "4bit"}).status_code == 400

    def test_official_i2i_still_uses_official_key(self, live, models_dir):
        install_both(models_dir)
        pid = self._project_with_run(live)
        job = enhance_done(live, pid)
        assert job["mode"] == "i2i" and job["rewriter"] == "official-8bit"
        assert live.loads[-1] == ("pe_i2i", "8bit") and ("pe_i2i", "heretic-8bit") not in live.loads

    def test_i2i_switching_official_and_heretic_reloads(self, live, models_dir):
        install_both(models_dir)
        pid = self._project_with_run(live)
        enhance_done(live, pid)
        set_variant(live, "heretic")
        enhance_done(live, pid)
        set_variant(live, "official")
        enhance_done(live, pid)
        pe_loads = [k for k in live.loads if k[0] == "pe_i2i"]
        assert pe_loads == [("pe_i2i", "8bit"), ("pe_i2i", "heretic-8bit"), ("pe_i2i", "8bit")]

    def test_mode_change_with_heretic_reloads_between_halves(self, live, models_dir):
        install_both(models_dir)
        set_variant(live, "heretic")
        fresh = new_project(live)["id"]
        enhance_done(live, fresh)  # t2i half
        pid = self._project_with_run(live)
        enhance_done(live, pid)  # i2i half
        keys = [k for k in live.loads if k[0].startswith("pe_")]
        assert keys == [("pe_t2i", "heretic-8bit"), ("pe_i2i", "heretic-8bit")]

    def test_heretic_i2i_not_installed_is_409_while_official_i2i_works(self, live, models_dir):
        install_heretic(models_dir, mode="t2i")
        set_variant(live, "heretic")  # selected, but only the T2I half exists
        pid = self._project_with_run(live)
        r = live.post(f"/api/projects/{pid}/enhance", json={"prompt": "x"})
        assert r.status_code == 409 and HERETIC_I2I in r.json()["detail"]
        set_variant(live, "official")
        assert enhance_done(live, pid)["rewriter"] == "official-8bit"
        # ...and a T2I enhance works with the half that is installed
        set_variant(live, "heretic")
        fresh = new_project(live)["id"]
        assert enhance_done(live, fresh)["rewriter"] == "heretic-8bit"
        # with no Heretic halves at all, the T2I half is refused too
        import shutil
        shutil.rmtree(models_dir / FOLDERS["t2i"])
        assert live.post(f"/api/projects/{new_project(live)['id']}/enhance", json={"prompt": "x"}).status_code == 409

    def test_generate_job_has_no_rewriter_effect(self, live):
        pid = new_project(live)["id"]
        live.post(f"/api/projects/{pid}/runs", json={"prompt": "base"})
        wait_until(lambda: live.get(f"/api/projects/{pid}").json()["current_run"] == 1, timeout=15)
        assert ("image", "official-8bit", "heretic-8bit", True) in live.loads
