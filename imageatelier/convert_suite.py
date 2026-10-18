"""Offline conversion worker. Exports every image component and both vision-capable prompt enhancers."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path


def convert_image_component(source: Path, destination: Path, name: str, bits: int) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
    from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
    from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
    from mflux.models.qwen21.reference.qwen_image21_initializer import QwenImage21Initializer
    from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

    config = json.loads((source / name / "config.json").read_text())
    classes = {"transformer": QwenImage21Transformer, "text_encoder": QwenImage21TextEncoder, "vae": QwenImage21VAE}
    module = classes[name](config)
    component = next(c for c in QwenImage21WeightDefinition.get_components() if c.name == name)
    weights = WeightLoader.load_single_local(component, source)
    if weights.meta_data.quantization_level is not None:
        raise ValueError("Conversions must start from the original BF16 source")
    supplied = dict(tree_flatten(weights.components[name]))
    QwenImage21Initializer._validate_weights(name, module, supplied)
    if name == "text_encoder" and not any(k.startswith("visual.") for k in supplied):
        raise ValueError("The text encoder must include its vision tower")
    WeightApplier.apply_and_quantize_single(
        weights, module, component, bits, quantization_predicate=QwenImage21WeightDefinition.quantization_predicate)
    del supplied, weights
    mx.eval(module.parameters())
    mx.clear_cache()
    # The reference VAE uses convolution layers and intentionally stays in its supported full precision.
    saved_bits = None if component.skip_quantization else bits
    ModelSaver._save_weights(str(destination), saved_bits, module, name)
    (destination / name / "config.json").write_text(json.dumps(config, indent=2))
    # Reload and compare one component at a time; never hold the whole suite in memory.
    reloaded = WeightLoader.load_single_local(component, destination)
    original = dict(tree_flatten(module.parameters()))
    restored = dict(tree_flatten(reloaded.components[name]))
    if original.keys() != restored.keys() or reloaded.meta_data.quantization_level != saved_bits:
        raise ValueError(f"{name}: saved weights do not match the converted component")
    if any(original[k].dtype != restored[k].dtype or not mx.array_equal(original[k], restored[k]).item()
           for k in original):
        raise ValueError(f"{name}: conversion round trip changed weights")


def convert_enhancer(source: Path, destination: Path, bits: int) -> None:
    from mlx_vlm.convert import convert
    # Converters run with HF_HUB_OFFLINE=1; only downloaded, pinned source snapshots are used.
    convert(str(source), mlx_path=str(destination), quantize=True, q_bits=bits,
            q_group_size=64, trust_remote_code=False)
    shutil.copy2(source / "system_prompt.txt", destination / "system_prompt.txt")
    from safetensors import safe_open
    def vision_keys(root):
        keys = set()
        for shard in root.glob("*.safetensors"):
            with safe_open(str(shard), framework="numpy") as file:
                keys.update(k for k in file.keys() if any(p in k for p in ("visual.", "vision_tower.", "vision_model.")))
        return keys
    if not vision_keys(source) or not vision_keys(destination):
        raise ValueError("Prompt-enhancer conversion must preserve vision weights")
    from mlx_vlm import load
    model, processor = load(str(destination), lazy=True, trust_remote_code=False)
    if model is None or processor is None:
        raise ValueError("Converted prompt enhancer could not be reloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ("image", "t2i", "i2i", "out"):
        parser.add_argument(f"--{kind}", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(8, 6, 4))
    args = parser.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(256 * 1024**2)
    image_out = args.out / "image"
    for name in ("transformer", "text_encoder", "vae"):
        print(f"SUITE_PHASE Converting {name.replace('_', ' ')}", flush=True)
        convert_image_component(args.image, image_out, name, args.bits)
        gc.collect()
        mx.clear_cache()
    shutil.copytree(args.image / "processor", image_out / "processor")
    for relative in ("model_index.json", "scheduler"):
        source = args.image / relative
        if source.is_dir():
            shutil.copytree(source, image_out / relative)
        elif source.is_file():
            shutil.copy2(source, image_out / relative)
    for mode in ("t2i", "i2i"):
        print(f"SUITE_PHASE Converting {mode.upper()} prompt enhancer", flush=True)
        convert_enhancer(getattr(args, mode), args.out / mode, args.bits)
        gc.collect()
        mx.clear_cache()
    print("SUITE_PHASE Verifying complete suite", flush=True)


if __name__ == "__main__":
    main()
