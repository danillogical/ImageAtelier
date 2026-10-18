"""Audit checkpoint formats and sampled dequantization error without loading whole models."""
from __future__ import annotations
import collections
import json
import struct
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from imageatelier import suites
OUT = Path(__file__).resolve().parent


def headers(root, rename=lambda key: key):
    weights = {}
    size = 0
    for path in sorted(root.glob('*.safetensors')):
        size += path.stat().st_size
        with path.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            content = json.loads(handle.read(length))
        for key, tensor in content.items():
            if key != '__metadata__':
                weights[rename(key)] = (path, 8 + length, tensor)
    return weights, size


def row(entry, index):
    path, base, tensor = entry
    dtype = {'BF16': '<u2', 'F32': '<f4', 'U32': '<u4'}[tensor['dtype']]
    width = tensor['shape'][-1]
    itemsize = np.dtype(dtype).itemsize
    with path.open('rb') as handle:
        handle.seek(base + tensor['data_offsets'][0] + index * width * itemsize)
        result = np.frombuffer(handle.read(width * itemsize), dtype=dtype)
    return (result.astype(np.uint32) << 16).view(np.float32) if tensor['dtype'] == 'BF16' else result


def pe_key(key):
    for a, b in [('model.language_model.visual', 'vision_tower'), ('model.language_model', 'language_model.model'),
                 ('model.visual', 'vision_tower'), ('lm_head', 'language_model.lm_head')]:
        if key.startswith(a):
            return key.replace(a, b, 1)
    return key


def main():
    source, output = suites.parts('bf16'), suites.parts('q8')
    assert source and output, 'Both suites must be complete'
    report = {'method': 'Checkpoint headers for all tensors; two sampled rows (first and middle) per packed matrix. No complete-tensor numerical comparison.',
              'sources': {k: list(v) for k,v in suites.SOURCES.items()}, 'parts': {}}
    for name in ('transformer', 'text_encoder', 'vae', 't2i', 'i2i'):
        root = lambda parts: parts['image'] / name if name in ('transformer', 'text_encoder', 'vae') else parts[name]
        rename = (lambda key:key.removeprefix('model.')) if name == 'text_encoder' else pe_key if name in ('t2i','i2i') else lambda key:key
        original, bf16_bytes = headers(root(source), rename)
        packed, q8_bytes = headers(root(output))
        config = json.loads((root(output) / 'config.json').read_text())
        index_path = next(iter(root(output).glob('*.safetensors.index.json')), None)
        index_metadata = json.loads(index_path.read_text()).get('metadata', {}) if index_path else {}
        if name in ('transformer', 'text_encoder'):
            assert index_metadata['quantization_level'] == '8'
        elif name in ('t2i','i2i'):
            assert config['quantization']['bits'] == 8 and config['quantization']['group_size'] == 64
        original_dtypes = collections.Counter(t[2]['dtype'] for t in original.values())
        assert set(original_dtypes) == ({'F32'} if name == 'vae' else {'BF16'}), original_dtypes
        errors = []
        layers = 0
        for key, entry in packed.items():
            if entry[2]['dtype'] != 'U32':
                continue
            assert key in original, (name, key)
            prefix = key.removesuffix('weight')
            shape = original[key][2]['shape']
            assert len(shape) == 2 and entry[2]['shape'] == [shape[0], shape[1]//4], (name,key)
            scales, biases = packed[prefix+'scales'], packed[prefix+'biases']
            assert scales[2]['shape'] == biases[2]['shape'] == [shape[0], shape[1]//64]
            for index in (0, shape[0]//2):
                integers = row(entry,index)
                values = ((integers[:,None] >> np.array([0,8,16,24],dtype=np.uint32)) & 255).reshape(-1,64)
                dequantized = (values * row(scales,index)[:,None] + row(biases,index)[:,None]).reshape(-1)
                reference = row(original[key],index)
                error = np.sqrt(np.mean((dequantized-reference)**2)) / max(np.sqrt(np.mean(reference**2)),1e-12)
                assert np.isfinite(error) and error < .03, (name,key,error)
                errors.append(float(error))
            layers += 1
        if name == 'vae':
            assert layers == 0 and set(t[2]['dtype'] for t in packed.values()) == {'F32'}
        else:
            assert layers > 0
        is_vision = lambda key: any(p in key for p in ('visual.', 'vision_tower.', 'vision_model.'))
        vision = [key for key in packed if is_vision(key)]
        vision_source = [key for key in original if is_vision(key)]
        for key in vision_source:
            assert key in packed, (name, 'vision tensor dropped', key)
            source_shape = original[key][2]['shape']
            expected_shape = list(source_shape)
            if packed[key][2]['dtype'] == 'U32':
                expected_shape[-1] //= 4
            elif key.endswith('patch_embed.proj.weight') and len(source_shape) == 5:
                expected_shape = [source_shape[i] for i in (0,2,3,4,1)]
            assert packed[key][2]['shape'] == expected_shape, (name, key, source_shape, packed[key][2]['shape'])
        if name in ('text_encoder','t2i','i2i'):
            assert vision, f'{name}: vision missing'
        report['parts'][name] = {'bf16_weight_bytes':bf16_bytes,'q8_weight_bytes':q8_bytes,
            'q8_packed_weight_bytes':sum(entry[2]['data_offsets'][1]-entry[2]['data_offsets'][0] for entry in packed.values() if entry[2]['dtype']=='U32'),
            'bf16_dtypes':dict(original_dtypes), 'q8_dtypes':dict(collections.Counter(t[2]['dtype'] for t in packed.values())),
            'quantized_layers':layers, 'sampled_rows':len(errors),
            'mean_relative_rms_error_percent':float(np.mean(errors)*100) if errors else None,
            'max_relative_rms_error_percent':max(errors)*100 if errors else None,
            'vision_tensors':len(vision), 'vision_quantized_layers':sum(packed[k][2]['dtype']=='U32' for k in vision),
            'source_vision_tensors_preserved':len(vision_source),
            'vision_dtypes':dict(collections.Counter(packed[k][2]['dtype'] for k in vision)),
            'config_quantization':config.get('quantization'), 'index_metadata':index_metadata}
    report['passed'] = True
    (OUT/'quantization-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
