"""Export a Qwen-Image-2.1 text encoder (Qwen3-VL, incl. vision tower) to mflux's quantized MLX format.

Loads ONLY the text encoder from a source root (official HF layout or an mflux save) exactly the way
QwenImage21Initializer loads that part, quantizes it with the fork's own predicate, and writes
<out>/text_encoder/ with ModelSaver._save_weights (numbered shards + index, mflux metadata) plus config.json,
the same bytes QwenImage21Edit.save_model would write for this part. Then it reloads the output through the
normal quantized-load path and checks every tensor is bit-identical, and writes <out>/PROVENANCE.md.

    python tools/export_text_encoder.py --source <root containing text_encoder/> --out <out root> \
        [--bits 8] [--source-repo Qwen/Qwen-Image-2.1]

Do not use mflux-save for this: it maps to the text-only class (drops the vision tower) and keeps the TE in bf16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.reference.qwen_image21_initializer import QwenImage21Initializer
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

PART = "text_encoder"


def _component():
    (component,) = [c for c in QwenImage21WeightDefinition.get_components() if c.name == PART]
    return component


def load_text_encoder(root: Path, bits: int | None) -> tuple[QwenImage21TextEncoder, dict, int | None, int | None]:
    """Build the TE from root/text_encoder as QwenImage21Initializer.init does; returns (module, config, stored_q, bits)."""
    config = json.loads((root / PART / "config.json").read_text())
    component = _component()
    module = QwenImage21TextEncoder(config)
    weights = WeightLoader.load_single_local(component, root)
    stored_q = weights.meta_data.quantization_level
    supplied = dict(tree_flatten(weights.components[component.name]))
    if stored_q is None:
        QwenImage21Initializer._validate_weights(component.name, module, supplied)
    applied = WeightApplier.apply_and_quantize_single(
        weights, module, component, bits, quantization_predicate=QwenImage21WeightDefinition.quantization_predicate
    )
    if stored_q is not None:
        QwenImage21Initializer._validate_weights(component.name, module, supplied)
    mx.eval(module.parameters())
    del weights, supplied
    mx.clear_cache()
    return module, config, stored_q, applied


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def mflux_source() -> str:
    try:
        direct = json.loads(metadata.distribution("mflux").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        return "mflux not installed"
    url = direct.get("url", "?")
    commit = direct.get("vcs_info", {}).get("commit_id") or url.rsplit("/", 1)[-1].removesuffix(".tar.gz")
    return f"{url} (commit {commit}), mflux {metadata.version('mflux')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="root containing text_encoder/ (HF snapshot dir)")
    ap.add_argument("--out", required=True, type=Path, help="output root; text_encoder/ is written inside it")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--source-repo", default=None, help="repo id recorded in PROVENANCE.md, e.g. Qwen/Qwen-Image-2.1")
    args = ap.parse_args()
    source, out = args.source.expanduser().resolve(), args.out.expanduser()
    if (out / PART).exists() and any((out / PART).iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out / PART}")

    t0 = time.perf_counter()
    module, config, stored_q, bits = load_text_encoder(source, args.bits)
    if stored_q is not None and stored_q != args.bits:
        sys.exit(f"source is already quantized at {stored_q} bits; re-quantizing to {args.bits} is not supported")
    load_s = time.perf_counter() - t0
    print(f"loaded + quantized {PART} from {source} (stored_q={stored_q}) to {bits} bits in {load_s:.1f} s; "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB")  # fmt: skip

    t1 = time.perf_counter()
    ModelSaver._save_weights(str(out), bits, module, PART)
    # Same serialization as QwenImage21Edit.save_model uses for a component's config.
    (out / PART / "config.json").write_text(json.dumps(config, indent=2))
    save_s = time.perf_counter() - t1
    print(f"saved to {out / PART} in {save_s:.1f} s")

    # Round trip: reload through the quantized-load path and compare every tensor exactly.
    original = dict(tree_flatten(module.parameters()))
    reloaded_module, _, reloaded_q, _ = load_text_encoder(out, None)
    reloaded = dict(tree_flatten(reloaded_module.parameters()))
    if set(original) != set(reloaded):
        sys.exit(f"round trip FAILED: key sets differ ({len(set(original) ^ set(reloaded))} keys)")
    bad = [k for k in original if original[k].dtype != reloaded[k].dtype or not mx.array_equal(original[k], reloaded[k])]
    if bad or reloaded_q != bits:
        sys.exit(f"round trip FAILED: quantization_level={reloaded_q}, {len(bad)} tensors differ, e.g. {bad[:5]}")
    print(f"round trip OK: {len(original)} tensors bit-identical after reload, quantization_level={reloaded_q}")

    index = json.loads((out / PART / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()), key=lambda s: int(s.split(".")[0]))
    hashes = {s: sha256(out / PART / s) for s in shards}
    snapshot_sha = source.name if len(source.name) == 40 else "unknown (not an HF snapshot dir)"
    script = Path(__file__).resolve()
    total_bytes = sum((out / PART / s).stat().st_size for s in shards)
    lines = [
        f"# Qwen-Image-2.1 text encoder, MLX {bits}-bit",
        "",
        f"- Source repo: `{args.source_repo or 'unspecified'}` @ `{snapshot_sha}`",
        f"- Source path: `{source}` (part `{PART}/`, bf16 HF layout)" if stored_q is None else f"- Source path: `{source}`",
        f"- mflux: {mflux_source()}",
        f"- mlx: {metadata.version('mlx')}; Python {platform.python_version()}; {platform.platform()}",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Quantization: {bits}-bit affine via `mlx.nn.quantize` (default group_size 64), applied by "
        "`WeightApplier.apply_and_quantize_single` with `QwenImage21WeightDefinition.quantization_predicate` "
        "(layers with `to_quantized` whose input dim is a multiple of 64, excluding modulation/time_text_embed/norm_out); "
        "key/weight transforms `text_key`/`text_weight` as in a normal load",
        f"- Shard metadata: `quantization_level: \"{bits}\"`, `mflux_version: \"{index['metadata']['mflux_version']}\"`",
        f"- Round trip: reloaded through the quantized-load path; all {len(original)} tensors bit-identical",
        f"- Size: {total_bytes / 1e9:.2f} GB in {len(shards)} shards",
        "",
        "## Shards (sha256)",
        "",
        *[f"- `{PART}/{s}`  `{h}`" for s, h in hashes.items()],
        f"- `{PART}/config.json`  `{sha256(out / PART / 'config.json')}`",
        f"- `{PART}/model.safetensors.index.json`  `{sha256(out / PART / 'model.safetensors.index.json')}`",
        "",
        "## Command",
        "",
        f"```\n{Path(sys.executable).name} {script} --source {args.source} --out {args.out} --bits {bits}"
        + (f" --source-repo {args.source_repo}" if args.source_repo else "") + "\n```",
        "",
        f"## Script (`{script.name}`, sha256 `{sha256(script)}`)",
        "",
        f"```python\n{script.read_text()}```",
        "",
    ]
    (out / "PROVENANCE.md").write_text("\n".join(lines))
    print(f"wrote {out / 'PROVENANCE.md'}; total {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
