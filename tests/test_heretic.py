"""Legacy uncensored choices are rejected; shared real-engine fixture for image/edit tests."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from imageatelier import models
from imageatelier.app import create_app
from conftest import patch_parts

@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("IMAGEATELIER_MODELS_DIR", str(root))
    return root

@pytest.fixture
def live(monkeypatch, models_dir, ready, tmp_path):
    """create_app() builds the real Engine (settings flow from the API); only model loading is stubbed."""
    mlx_vlm = pytest.importorskip("mlx_vlm")
    from imageatelier import engine as eng_mod

    loads: list[tuple] = []
    def resolve_pe(mode, quant):
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

    patch_parts(monkeypatch)
    monkeypatch.setattr(models, "resolve_pe", resolve_pe)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_image", load_image)
    monkeypatch.setattr(eng_mod.ModelManager, "_load_pe", load_pe)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream)
    monkeypatch.setattr(eng_mod, "gpu_utilization", lambda: 0)
    with TestClient(create_app(data_dir=tmp_path / "data")) as c:
        c.loads = loads
        yield c



def test_uncensored_settings_rejected(client):
    assert client.put("/api/settings", json={"pe_variant": "heretic"}).status_code == 400
    assert client.put("/api/settings", json={"text_encoder": "heretic-8bit"}).status_code == 400
    assert all("heretic" not in name for registry in (models.IMAGE_MODELS, models.TEXT_ENCODERS, models.PE_MODELS) for name in registry)
