"""Small real MLX conversions exercise every suite precision and preserve the vision tower."""
import json
from pathlib import Path
import pytest
from imageatelier.convert_suite import convert_image_component

@pytest.mark.parametrize("bits", [8, 6, 4])
@pytest.mark.parametrize("name", ["transformer", "text_encoder", "vae"])
def test_component_conversion_round_trip(monkeypatch, tmp_path, bits, name):
    import mlx.core as mx
    import mlx.nn as nn
    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.qwen21.reference.model.qwen_image21_text_encoder import text_encoder
    from mflux.models.qwen21.reference.model.qwen_image21_transformer import transformer
    from mflux.models.qwen21.reference.model.qwen_image21_vae import vae

    class Tiny(nn.Module):
        def __init__(self, config):
            super().__init__()
            if name == "vae":
                self.conv = nn.Conv2d(1, 1, 1)
            else:
                self.language_model = nn.Linear(64, 4, bias=False)
                if name == "text_encoder": self.visual = nn.Linear(64, 4, bias=False)

    monkeypatch.setattr(text_encoder, "QwenImage21TextEncoder", Tiny)
    monkeypatch.setattr(transformer, "QwenImage21Transformer", Tiny)
    monkeypatch.setattr(vae, "QwenImage21VAE", Tiny)
    source = tmp_path / "source"
    ModelSaver._save_weights(str(source), None, Tiny({}), name)
    (source / name / "config.json").write_text("{}")
    output = tmp_path / "output"
    convert_image_component(source, output, name, bits)
    index = json.loads((output / name / "model.safetensors.index.json").read_text())
    assert index["metadata"]["quantization_level"] == ("None" if name == "vae" else str(bits))
    if name == "text_encoder":
        assert "visual.weight" in index["weight_map"] and "visual.scales" in index["weight_map"]
    mx.clear_cache()
