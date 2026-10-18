"""Custom checkpoint discovery, conversion math, selection and base-suite composition."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from imageatelier import custom_models as custom, convert_custom, models, suites
from test_parts import source  # noqa: F401


@pytest.fixture
def checkpoint(source):
    path = models.models_dir() / 'anime.safetensors'
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({'img_in.weight': torch.ones(2, 2), 'transformer_blocks.0.attn.to_q.weight': torch.ones(2, 2)}, str(path))
    return path


def publish(path):
    row = custom.inspect(path)
    out = custom.folder(row['id'])
    part = out / row['kind']
    part.mkdir(parents=True)
    (part / 'config.json').write_text('{}')
    (part / 'weights.safetensors').write_bytes(b'weights')
    (out / 'custom.json').write_text(json.dumps({'format': 1, 'kind': row['kind'], 'source': custom.fingerprint(path)}))
    return row['id']


def test_discovery_classifies_by_weights_not_filename(checkpoint):
    row = custom.list_models()[0]
    assert row['kind'] == 'image' and row['state'] == 'not_prepared'
    assert row['filename'] == 'anime.safetensors'
    assert publish(checkpoint) == row['id']
    assert custom.get(row['id'], 'image', ready=True)['state'] == 'ready'
    with pytest.raises(ValueError, match='wrong component'):
        custom.get(row['id'], 'text_encoder')
    checkpoint.touch()
    assert custom.get(row['id'])['state'] == 'not_prepared'


def test_malformed_and_text_only_checkpoints_are_not_selectable(checkpoint):
    path = checkpoint.with_name('encoder.safetensors')
    save_file({'model.layers.0.weight': torch.ones(2, 2)}, str(path))
    row = custom.inspect(path)
    assert row['kind'] == 'text_encoder' and row['state'] == 'unsupported'
    assert 'vision' in row['error']
    checkpoint.write_bytes(b'bad')
    assert custom.list_models()[0]['state'] == 'unsupported'
    with pytest.raises(ValueError):
        custom.folder('../../outside')


def test_custom_overlay_keeps_base_vae_processor_and_encoder(checkpoint, source):
    identifier = publish(checkpoint)
    image, encoder = custom.model_ids('bf16', identifier)
    composed, te, lora = models.resolve_parts(image, encoder, False)
    assert (composed / 'transformer').resolve() == custom.folder(identifier) / 'image'
    assert (composed / 'vae').resolve() == source['image'] / 'vae'
    assert (composed / 'processor').resolve() == source['image'] / 'processor'
    assert te.resolve() == source['image'] / 'text_encoder'
    assert lora is None
    with pytest.raises(ValueError, match='same base'):
        custom.resolve(image, 'official-8bit')


def test_selection_and_generation_pin_custom_ids(checkpoint, client):
    row = custom.inspect(checkpoint)
    assert client.put('/api/settings', json={'custom_image': row['id']}).status_code == 409
    identifier = publish(checkpoint)
    response = client.put('/api/settings', json={'custom_image': identifier})
    assert response.status_code == 200, response.text
    assert response.json()['image_model'] == identifier + '@q8'
    pid = client.post('/api/projects', json={'name': 'Custom'}).json()['id']
    response = client.post(f'/api/projects/{pid}/runs', json={'prompt': 'a tree'})
    assert response.status_code == 202, response.text
    job = client.app.state.engine.submitted[-1]
    assert job['image_model'] == identifier + '@q8'
    assert job['text_encoder'] == 'official-8bit'
    assert client.get('/api/custom-models').json()[0]['state'] == 'ready'
    assert client.put('/api/settings', json={'custom_image': ''}).status_code == 200


def test_regular_hadamard_convrot_restores_original_basis():
    # A Sylvester Hadamard is also self-inverse but is the wrong matrix for these checkpoints.
    q = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    expected = torch.tensor([[1., 2., 3., 4.]]) @ torch.tensor(
        [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
    assert torch.equal(convert_custom.unrotate(q, 4), expected)
    for group in (0, 2, 8, 15):
        with pytest.raises(ValueError):
            convert_custom.unrotate(q, group)
    weight = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    assert torch.allclose(convert_custom.unrotate(convert_custom.unrotate(weight, 256), 256), weight)


def test_comfy_encoder_dequantizes_scales_and_rotation(checkpoint):
    path = checkpoint.with_name('encoder.safetensors')
    prefix = 'model.layers.0.self_attn.q_proj'
    marker = torch.tensor(list(json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 4}).encode()), dtype=torch.uint8)
    quantized = torch.tensor([[1, 2, 3, 4], [-1, -2, 4, 8]], dtype=torch.int8)
    scales = torch.tensor([[0.5], [2.0]])
    save_file({prefix + '.weight': quantized, prefix + '.weight_scale': scales, prefix + '.comfy_quant': marker}, str(path))
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as reader:
        restored = convert_custom.dequantize(reader, prefix + '.weight')
    assert torch.equal(restored, convert_custom.unrotate(quantized.float() * scales, 4).bfloat16())
    assert convert_custom.text_key(prefix + '.weight') == 'model.language_model.layers.0.self_attn.q_proj.weight'
    assert convert_custom.text_key('model.norm.weight') is None
    assert convert_custom.text_key('lm_head.weight') is None


def test_import_preserves_source_and_publishes_index(checkpoint, source, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    original = checkpoint.read_bytes()
    seen = []
    monkeypatch.setattr(convert_custom, 'validate_shapes', lambda *a: seen.append(a))
    convert_custom.convert(checkpoint, 'image', source['image'], out)
    part = out / 'image'
    index = json.loads((part / 'model.safetensors.index.json').read_text())['weight_map']
    assert set(index) == {'img_in.weight', 'transformer_blocks.0.attn.to_q.weight'}
    assert all((part / shard).is_file() for shard in index.values())
    assert load_file(str(part / index['img_in.weight']))['img_in.weight'].dtype == torch.float32
    assert checkpoint.read_bytes() == original
    assert seen


def test_fused_transformer_gate_and_up_are_split_in_order(checkpoint, source, tmp_path, monkeypatch):
    tensor = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    save_file({'transformer_blocks.0.img_mlp.gate_up.weight': tensor}, str(checkpoint))
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(convert_custom, 'validate_shapes', lambda *a: None)
    convert_custom.convert(checkpoint, 'image', source['image'], out)
    restored = load_file(str(out / 'image/weights-0000.safetensors'))
    assert torch.equal(restored['transformer_blocks.0.img_mlp.gate_layer.weight'], tensor[:4])
    assert torch.equal(restored['transformer_blocks.0.img_mlp.proj.weight'], tensor[4:])


@pytest.mark.parametrize('value', [42, True, [], {}, 'custom-1111111111111111@bf16'])
def test_invalid_custom_selection_returns_client_error(client, value):
    assert client.put('/api/settings', json={'custom_image': value}).status_code == 400


def test_preparation_refuses_insufficient_disk_before_starting_worker(checkpoint, monkeypatch):
    identifier = custom.inspect(checkpoint)['id']
    monkeypatch.setattr(custom.shutil, 'disk_usage', lambda p: __import__('types').SimpleNamespace(free=10))
    with pytest.raises(suites.SuiteError, match='disk space'):
        custom.prepare(identifier, 'bf16')


def make_enhancer(name):
    from test_parts import make_part
    root = models.models_dir() / name
    make_part(root, ('language_model.weight', 'model.visual.weight'))
    for filename in ('tokenizer.json', 'processor_config.json'):
        (root / filename).write_text('{}')
    (root / 'system_prompt.txt').write_text('Rewrite the prompt.')
    return root


def test_enhancer_discovery_selection_and_job_pinning(checkpoint, client):
    from imageatelier.engine import Engine
    path = make_enhancer('Example-PE-T2I-Heretic')
    row = custom.inspect_enhancer(path)
    assert row['kind'] == 'pe_t2i' and row['state'] == 'ready'
    assert client.put('/api/settings', json={'custom_pe_i2i': row['id']}).status_code == 400
    response = client.put('/api/settings', json={'custom_pe_t2i': row['id']})
    assert response.status_code == 200, response.text
    assert models.resolve_pe('t2i', row['id']) == (path, 'Rewrite the prompt.')
    engine = Engine(settings_provider=lambda: response.json())
    job = engine._make_enhance({'prompt': 'a tree', 'mode': 't2i'})
    assert job.rewriter == row['id']
    assert engine._make_enhance({'prompt': 'a tree', 'mode': 't2i', 'rewriter': None}).rewriter is None
    pid = client.post('/api/projects', json={'name': 'Custom enhancer'}).json()['id']
    result = client.post(f'/api/projects/{pid}/enhance', json={'prompt': 'a tree'})
    assert result.status_code == 202, result.text
    assert client.app.state.engine.submitted[-1]['rewriter'] == row['id']
    (path / '0.safetensors').unlink()
    assert custom.inspect_enhancer(path)['state'] == 'incomplete'
    assert client.post(f'/api/projects/{pid}/enhance', json={'prompt': 'a tree'}).status_code == 409


def test_delete_clears_only_its_category_and_preserves_projects(checkpoint, client):
    identifier = publish(checkpoint)
    path = make_enhancer('Example-PE-I2I-Heretic')
    enhancer = custom.inspect_enhancer(path)['id']
    assert client.put('/api/settings', json={'custom_image': identifier, 'custom_pe_i2i': enhancer}).status_code == 200
    pid = client.post('/api/projects', json={'name': 'Keep this'}).json()['id']
    result = client.delete(f'/api/custom-models/{identifier}')
    assert result.status_code == 200, result.text
    assert result.json()['deleted']
    assert not checkpoint.exists() and not custom.folder(identifier).exists()
    settings = client.get('/api/settings').json()
    assert settings['custom_image'] == '' and settings['custom_pe_i2i'] == enhancer
    assert path.is_dir() and client.get(f'/api/projects/{pid}').status_code == 200
    assert client.delete(f'/api/custom-models/{enhancer}').status_code == 200
    assert not path.exists()
    assert client.get('/api/settings').json()['custom_pe_i2i'] == ''


def test_deletion_refuses_during_preparation(checkpoint, monkeypatch):
    monkeypatch.setattr(suites, 'is_active', lambda: True)
    with pytest.raises(suites.SuiteError, match='Wait'):
        custom.delete(custom.inspect(checkpoint)['id'])
    assert checkpoint.exists()
