"""A suite is usable only when every official component, including vision, is present."""
import json
from pathlib import Path
import pytest
from imageatelier import models, suites


def make_part(folder: Path, keys=("weight",), shard="0.safetensors"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}")
    (folder / shard).write_bytes(b"weights")
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: shard for k in keys}}))
    return folder


def make_suite(root):
    image = root / "image"
    for part in ("transformer", "vae"):
        make_part(image / part)
    make_part(image / "text_encoder", ("language_model.layers.0.weight", "visual.weight"))
    (image / "processor").mkdir(exist_ok=True)
    for f in ("tokenizer.json", "preprocessor_config.json"):
        (image / "processor" / f).write_text("{}")
    for mode in ("t2i", "i2i"):
        folder = make_part(root / mode, ("language_model.weight", "vision_tower.weight"))
        for f in ("tokenizer.json", "processor_config.json"):
            (folder / f).write_text("{}")
        (folder / "system_prompt.txt").write_text("Rewrite prompts.")
    return suites.converted_parts(root)


@pytest.fixture
def source(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGEATELIER_MODELS_DIR", str(tmp_path / "models"))
    parts = make_suite(tmp_path / "source")
    def snapshot(repo, **kwargs):
        assert kwargs.get("local_files_only") is True
        kind = next(k for k, (r, rev) in suites.SOURCES.items() if r == repo)
        return str(parts[kind])
    monkeypatch.setattr(suites, "snapshot_download", snapshot)
    return parts


def publish(root, name):
    make_suite(root)
    (root / "suite.json").write_text(json.dumps({"format": 1, "variant": name,
        "sources": {k: list(v) for k, v in suites.SOURCES.items()}}))


def test_bf16_all_five_components_required(source):
    assert suites.parts("bf16") == source
    (source["i2i"] / "0.safetensors").unlink()
    assert suites.parts("bf16") is None
    assert suites.status("q8")["can_convert"] is False


def test_partial_image_and_text_only_encoder_rejected(source):
    (source["image"] / "text_encoder" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layers.0.weight": "0.safetensors"}}))
    assert suites.parts("bf16") is None


def test_raw_diffusers_index_and_single_vae_supported(source):
    root = source["image"]
    (root / "transformer" / "model.safetensors.index.json").rename(root / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
    (root / "vae" / "model.safetensors.index.json").unlink()
    assert suites.parts("bf16") == source


def test_all_converted_components_and_manifest_required(source):
    root = suites.suite_dir() / "q6"
    publish(root, "q6")
    assert suites.status("q6")["state"] == "ready"
    (root / "suite.json").unlink()
    assert suites.status("q6")["state"] == "not_converted"
    publish(root, "q6")
    (root / "t2i" / "0.safetensors").unlink()
    assert suites.status("q6")["state"] == "not_converted"


def test_mixing_and_turbo_rejected(source):
    with pytest.raises(ValueError, match="cannot be mixed"):
        models.resolve_parts("official-8bit", "official-4bit", False)
    with pytest.raises(ValueError, match="Turbo"):
        models.resolve_parts("official-bf16", "official-bf16", True)
    assert models.resolve_parts("official-bf16", "official-bf16", False) == (
        source["image"], source["image"] / "text_encoder", None)


def test_enhancer_uses_matching_suite(source):
    root = suites.suite_dir() / "q4"
    publish(root, "q4")
    path, prompt = models.resolve_pe("i2i", "4bit")
    assert path == root / "i2i" and prompt == "Rewrite prompts."
    assert models.resolve_parts("official-4bit", "official-4bit", False)[0] == root / "image"
