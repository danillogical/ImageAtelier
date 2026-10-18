"""Offline, bounded-memory import of a Qwen transformer or ComfyUI text encoder."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path


def text_key(key: str) -> str | None:
    key = key.removeprefix('model.')
    if key.startswith('lm_head.') or key in ('norm.weight', 'language_model.norm.weight'):
        return None
    if key.startswith(('layers.', 'embed_tokens.')):
        key = 'language_model.' + key
    return 'model.' + key


def unrotate(weight, group: int):
    """Undo Comfy's regular Hadamard ConvRot, not the Sylvester Hadamard.

    Format: https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py
    """
    import torch
    if group < 4 or group > 4096 or 4 ** round(math.log(group, 4)) != group or weight.shape[-1] % group:
        raise ValueError('Unsupported ConvRot group size')
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < group:
        h = torch.kron(h, h4)
    return (weight.reshape(-1, group) @ (h / math.sqrt(group))).reshape(weight.shape)


def dequantize(reader, key: str):
    import torch
    tensor = reader.get_tensor(key)
    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return tensor.to(torch.bfloat16).contiguous()
    if tensor.dtype != torch.int8 or not key.endswith('.weight'):
        raise ValueError(f'Unsupported custom tensor dtype for {key}: {tensor.dtype}')
    prefix = key.removesuffix('.weight')
    marker = json.loads(bytes(reader.get_tensor(prefix + '.comfy_quant').tolist()))
    if marker.get('format') != 'int8_tensorwise':
        raise ValueError(f'Unsupported ComfyUI quantization for {key}')
    scale = reader.get_tensor(prefix + '.weight_scale').float()
    if scale.numel() not in (1, tensor.shape[0]) or scale.shape not in (torch.Size([]), torch.Size([1]), torch.Size([1, 1]), torch.Size([tensor.shape[0], 1])):
        raise ValueError(f'Invalid weight scale shape for {key}')
    weight = tensor.float() * scale
    if marker.get('convrot'):
        weight = unrotate(weight, int(marker.get('convrot_groupsize', 256)))
    return weight.to(torch.bfloat16).contiguous()


def validate_shapes(source: Path, kind: str, base: Path) -> None:
    """Use the real loader's expected keys and shapes, without evaluating its random initial weights."""
    from mlx.utils import tree_flatten
    from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
    from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
    from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
    from imageatelier.custom_models import header
    component = 'transformer' if kind == 'image' else kind
    config = json.loads((base / component / 'config.json').read_text())
    model = (QwenImage21Transformer if kind == 'image' else QwenImage21TextEncoder)(config)
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    supplied = {}
    for file in source.glob('*.safetensors'):
        for key, info in header(file).items():
            if key == '__metadata__':
                continue
            mapped = QwenImage21WeightDefinition.text_key(key) if kind == 'text_encoder' else key
            if mapped is None:
                continue
            shape = tuple(info['shape'])
            if kind == 'text_encoder' and mapped.endswith('patch_embed.proj.weight'):
                shape = (shape[0], *shape[2:], shape[1])
            supplied[mapped] = shape
    missing = {k for k in expected.keys() - supplied.keys() if not k.endswith('.inv_freq')}
    unexpected = supplied.keys() - expected.keys()
    mismatched = [k for k in expected.keys() & supplied.keys() if expected[k] != supplied[k]]
    if missing or unexpected or mismatched:
        raise ValueError(f'Checkpoint does not match Qwen-Image-2.1: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={mismatched}')


def convert(source: Path, kind: str, base: Path, out: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from imageatelier.custom_models import header
    torch.set_num_threads(min(4, torch.get_num_threads()))
    destination = out / kind
    destination.mkdir()
    component = 'transformer' if kind == 'image' else kind
    shutil.copy2(base / component / 'config.json', destination / 'config.json')
    if kind == 'image':
        from imageatelier.custom_models import header
        tensors = {k: v for k, v in header(source).items() if k != '__metadata__'}
        if not tensors or any(v['dtype'] not in ('BF16', 'F16', 'F32') for v in tensors.values()):
            raise ValueError('The transformer must use BF16, F16 or F32 weights')
    if kind == 'image' and not any('.img_mlp.gate_up.' in k for k in tensors):
        # A source link avoids duplicating an already compatible transformer.
        try:
            os.link(source, destination / 'weights.safetensors')
        except OSError:
            shutil.copy2(source, destination / 'weights.safetensors')
        (destination / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'weights.safetensors' for k in tensors}}))
        validate_shapes(destination, kind, base)
        return
    weights, index, size, shard = {}, {}, 0, 0
    def flush():
        nonlocal weights, size, shard
        if not weights:
            return
        filename = f'weights-{shard:04d}.safetensors'
        save_file(weights, str(destination / filename), metadata={'format': 'pt'})
        index.update({key: filename for key in weights})
        weights, size, shard = {}, 0, shard + 1
    with safe_open(str(source), framework='pt', device='cpu') as reader:
        for i, key in enumerate(reader.keys()):
            if key.endswith(('.weight_scale', '.comfy_quant')):
                continue
            mapped = text_key(key) if kind == 'text_encoder' else key
            if mapped is None:
                continue
            print(f'CUSTOM_PHASE Importing {kind}: tensor {i + 1}/{len(reader.keys())}', flush=True)
            tensor = reader.get_tensor(key) if kind == 'image' else dequantize(reader, key)
            if kind == 'image' and key.endswith('.img_mlp.gate_up.weight'):
                # ComfyUI fused layout is [gate_layer; proj] along the output dimension.
                if tensor.shape[0] % 2:
                    raise ValueError('Fused gate/up projection must have an even output dimension')
                gate, up = tensor.chunk(2, dim=0)
                weights[key.replace('.gate_up.', '.gate_layer.')] = gate.clone()
                weights[key.replace('.gate_up.', '.proj.')] = up.clone()
            else:
                weights[mapped] = tensor
            size += tensor.numel() * tensor.element_size()
            if size >= 256 * 1024**2:
                flush()
    flush()
    (destination / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    print('CUSTOM_PHASE Validating Qwen architecture', flush=True)
    validate_shapes(destination, kind, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--kind', choices=('image', 'text_encoder'), required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.kind, args.base, args.out)


if __name__ == '__main__':
    main()
