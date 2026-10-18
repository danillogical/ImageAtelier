"""LAN controls, authenticated isolation, shared-worker memory output and cleanup."""
import asyncio
import base64
import io
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imageatelier import models, network
from imageatelier.app import create_app
from imageatelier.network_api import create_api, TransientPNG
from conftest import StubEngine, wait_until
from test_serialization import h, wait_state, TERMINAL  # noqa: F401


class Config:
    def __init__(self):
        self.values = {'network_api_enabled': True, 'network_ui_enabled': True, 'model_suite': 'q8',
                       'image_model': 'official-8bit', 'text_encoder': 'official-8bit', 'steps': 6}
    def get(self): return dict(self.values)


@pytest.fixture
def memory_api(h, monkeypatch):
    monkeypatch.setattr(models, 'require_suite', lambda name: {})
    cfg = Config()
    app = create_api(h.engine, cfg, lambda value: value == 'Bearer test-key',
                     enabled=lambda: cfg.values['network_api_enabled'])
    h.start()
    with TestClient(app, headers={'Authorization': 'Bearer test-key'}) as client:
        yield client, h, cfg


def png_url(color=(255, 0, 0, 120)):
    with io.BytesIO() as buffer:
        Image.new('RGBA', (32, 32), color).save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def test_memory_generation_never_writes_images_and_releases_data(memory_api, monkeypatch):
    client, h, cfg = memory_api
    saved = []
    original = Image.Image.save
    def save(self, destination, *args, **kwargs):
        saved.append(destination)
        assert isinstance(destination, io.BytesIO), 'Transient image save touched the filesystem'
        return original(self, destination, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'save', save)
    before = set(h.tmp.rglob('*'))
    response = client.post('/generate', json={'prompt': 'a cat', 'width': 512, 'height': 768, 'seed': 42})
    assert response.status_code == 200, response.text
    assert response.content.startswith(b'\x89PNG')
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-image-seed'] == '42'
    assert len(saved) == 1
    assert set(h.tmp.rglob('*')) == before
    assert h.engine._jobs == {} and h.engine.jobs() == []
    assert not [event for event in h.log if event[0] == 'job']


def test_memory_edit_consumes_reference_pixels_without_temp_files(memory_api):
    client, h, cfg = memory_api
    before = set(h.tmp.rglob('*'))
    response = client.post('/generate', json={'prompt': 'make it blue', 'mode': 'edit', 'references': [png_url()]})
    assert response.status_code == 200, response.text
    assert set(h.tmp.rglob('*')) == before
    assert h.engine._jobs == {}


@pytest.mark.parametrize('count', [0, 3, 10])
def test_standalone_enhance_receives_all_images_in_order_without_saving_or_generating(memory_api, monkeypatch, count):
    import mlx_vlm
    client, h, cfg = memory_api
    cfg.values.update(pe_thinking=False, pe_language='auto', pe_max_new_tokens=512)
    colors = [(20 + i, 70, 130, 255) for i in range(count)]
    urls = [png_url(color) for color in colors]
    seen = []
    references = []
    original = mlx_vlm.stream_generate
    def stream(*args, **kw):
        refs = kw.get('image') or []
        references.extend(refs)
        job = h.engine._current
        seen.append((job.prompt, job.mode, job.thinking, job.language, kw['max_tokens'],
                     [image.getpixel((0, 0)) for image in refs]))
        yield from original(*args, **kw)
    monkeypatch.setattr(mlx_vlm, 'stream_generate', stream)
    def forbidden(*args, **kw): raise AssertionError('Enhancement must not generate/save an image')
    monkeypatch.setattr(h.engine, '_run', forbidden)
    monkeypatch.setattr(Image.Image, 'save', forbidden)
    before = set(h.tmp.rglob('*'))
    response = client.post('/enhance', json={'prompt': 'make the library warm', 'references': urls})
    assert response.status_code == 200, response.text
    result = response.json()
    mode = 'i2i' if count else 't2i'
    assert result['rewritten_prompt'] == 'better'
    assert result['fell_back'] is False and result['mode'] == mode
    assert result['rewriter'] == 'official-8bit'
    assert seen == [('make the library warm', mode, False, 'auto', 512, [c[:3] for c in colors])]
    assert response.headers['cache-control'] == 'no-store'
    assert set(h.tmp.rglob('*')) == before
    assert h.engine._jobs == {} and h.engine.jobs() == []
    assert not [event for event in h.log if event[0] == 'job']
    for image in references:
        with pytest.raises(ValueError): image.getpixel((0, 0))


def test_enhanced_prompt_can_be_modified_before_a_separate_generation(memory_api, monkeypatch):
    client, h, cfg = memory_api
    first = client.post('/enhance', json={'prompt': 'a cat', 'thinking': False, 'language': 'english', 'max_new_tokens': 256})
    assert first.status_code == 200, first.text
    prompt = first.json()['rewritten_prompt'] + ', with a red hat'
    submitted = []
    original = h.engine.submit_memory
    def submit(job, images):
        submitted.append(job['prompt'])
        return original(job, images)
    monkeypatch.setattr(h.engine, 'submit_memory', submit)
    response = client.post('/generate', json={'prompt': prompt})
    assert response.status_code == 200 and response.content.startswith(b'\x89PNG')
    assert submitted == ['better, with a red hat']


@pytest.mark.parametrize('mode', ['txt2img', 'edit'])
def test_memory_generation_accepts_ten_conditioning_images_despite_ui_limit(memory_api, monkeypatch, mode):
    from imageatelier import engine as eng_mod
    client, h, cfg = memory_api
    h.engine._settings_provider = lambda: {'max_refs': 0, 'low_memory': False}
    seen = []
    original = eng_mod.ModelManager._load_image
    def load(self, *args):
        original(self, *args)
        generate = self.model.generate_image
        def run(**kw):
            seen.extend(image.getpixel((0, 0)) for image in kw['image_paths'])
            return generate(**kw)
        self.model.generate_image = run
    monkeypatch.setattr(eng_mod.ModelManager, '_load_image', load)
    colors = [(i + 1, 60, 90, 255) for i in range(10)]
    response = client.post('/generate', json={'prompt': 'use these references', 'mode': mode,
                                             'references': [png_url(c) for c in colors]})
    assert response.status_code == 200, response.text
    assert seen == [c[:3] for c in colors]
    assert h.engine._jobs == {}


@pytest.mark.parametrize('body', [
    {'prompt': 'x', 'mode': 'i2i'},
    {'prompt': 'x', 'mode': 't2i', 'references': [png_url()]},
    {'prompt': 'x', 'references': ['file:///tmp/ref.png']},
    {'prompt': 'x', 'ref_image_path': '/tmp/ref.png'},
    {'prompt': 'x', 'output_path': '/tmp/out.png'},
    {'prompt': 'x', 'thinking': 'yes'},
    {'prompt': 'x', 'max_new_tokens': 255},
    {'prompt': 'x', 'language': 'french'},
    {'prompt': 'x', 'references': [png_url()] * 11},
])
def test_enhance_validates_modes_options_and_references(memory_api, body):
    client, h, cfg = memory_api
    response = client.post('/enhance', json=body)
    assert response.status_code == 400, response.text
    assert h.engine._jobs == {}


def test_enhance_missing_model_is_a_conflict_not_a_generation(memory_api, monkeypatch):
    client, h, cfg = memory_api
    def missing(*args): raise models.ModelNotReady('Select a ready prompt enhancer')
    monkeypatch.setattr(models, 'resolve_pe', missing)
    assert client.post('/enhance', json={'prompt': 'a cat'}).status_code == 409
    assert h.engine._jobs == {}


@pytest.mark.parametrize('action', ['enhance', 'generate'])
def test_memory_api_rejects_more_than_ten_images(memory_api, action):
    client, h, cfg = memory_api
    response = client.post('/'+action, json={'prompt': 'a cat', 'references': [png_url()] * 11})
    assert response.status_code == 400
    assert h.engine._jobs == {}


@pytest.mark.parametrize('action', ['generate', 'enhance'])
def test_polling_returns_an_opaque_token_and_a_single_use_result(memory_api, action):
    client, h, cfg = memory_api
    before = set(h.tmp.rglob('*'))
    response = client.post('/'+action, json={'prompt': 'a private cat prompt', 'wait': False})
    assert response.status_code == 202, response.text
    ticket = response.json()
    token = ticket['request_id']
    assert len(token) >= 32 and token not in h.engine._jobs
    assert response.headers['location'] == ticket['status_url']
    assert response.headers['cache-control'] == 'no-store'
    assert client.get(ticket['status_url'], headers={'Authorization': ''}).status_code == 401
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'done')
    status = client.get(ticket['status_url'])
    assert status.json()['kind'] == action and status.json()['expires_in'] > 0
    assert 'a private cat prompt' not in status.text
    assert 'job_id' not in status.json()
    assert client.get('/health').json()['active_requests'] == 1  # An unclaimed result occupies its slot.
    assert h.engine.jobs() == []
    result = client.get(ticket['result_url'])
    assert result.status_code == 200, result.text
    if action == 'enhance': assert result.json()['rewritten_prompt'] == 'better'
    else: assert result.content.startswith(b'\x89PNG')
    assert result.headers['cache-control'] == 'no-store'
    assert client.get(ticket['result_url']).status_code == 404
    assert client.get(ticket['status_url']).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0
    assert h.engine._jobs == {}
    assert set(h.tmp.rglob('*')) == before


def test_polling_reports_generation_steps_and_cancellation_releases_inputs(memory_api):
    client, h, cfg = memory_api
    h.behaviour.append('block')
    ticket = client.post('/generate', json={'prompt': 'a cat', 'references': [png_url()], 'wait': False}).json()
    wait_until(lambda: (client.get(ticket['status_url']).json()['progress']['step'] or 0) > 0)
    status = client.get(ticket['status_url']).json()
    assert status['state'] == 'running' and status['progress']['total_steps'] == 1000
    assert client.get(ticket['result_url']).status_code == 409
    assert client.delete(ticket['cancel_url']).status_code == 204
    assert client.get(ticket['status_url']).status_code == 404
    wait_until(lambda: not h.engine._jobs)
    assert client.get('/health').json()['active_requests'] == 0


def test_polling_reports_enhancement_tokens_without_exposing_thinking(memory_api, monkeypatch):
    import mlx_vlm
    import time
    client, h, cfg = memory_api
    def stream(*args, **kw):
        for part in ['{"rewritten_prompt": "', 'a ', 'better ', 'cat ', 'in ', 'a ', 'hat', '"}']:
            time.sleep(.08)
            yield SimpleNamespace(text=part, finish_reason='stop', generation_tps=10.0)
    monkeypatch.setattr(mlx_vlm, 'stream_generate', stream)
    ticket = client.post('/enhance', json={'prompt': 'a cat', 'wait': False, 'thinking': False}).json()
    wait_until(lambda: (client.get(ticket['status_url']).json()['progress']['tokens'] or 0) > 0)
    status = client.get(ticket['status_url']).json()
    assert status['progress']['phase'] == 'writing'
    assert 'text' not in status and 'thinking' not in status
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'done')
    assert client.get(ticket['result_url']).json()['rewritten_prompt'] == 'a better cat in a hat'


def test_polling_queue_position_and_shared_capacity(memory_api):
    client, h, cfg = memory_api
    h.behaviour.append('block')
    first = client.post('/generate', json={'prompt': 'first', 'wait': False}).json()
    wait_until(lambda: client.get(first['status_url']).json()['state'] == 'running')
    second = client.post('/enhance', json={'prompt': 'second', 'wait': False}).json()
    status = client.get(second['status_url']).json()
    assert status['state'] == 'queued' and status['queue_position'] == 1
    assert client.post('/generate', json={'prompt': 'third'}).status_code == 429
    assert client.delete(second['cancel_url']).status_code == 204
    assert client.delete(first['cancel_url']).status_code == 204
    wait_until(lambda: not h.engine._jobs)


def test_polling_unclaimed_result_expires_and_releases_memory(memory_api, monkeypatch):
    from imageatelier import network_api
    client, h, cfg = memory_api
    monkeypatch.setattr(network_api, 'RESULT_TTL', .1)
    ticket = client.post('/generate', json={'prompt': 'a cat', 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).status_code == 404)
    assert client.get('/health').json()['active_requests'] == 0
    assert h.engine._jobs == {}


def test_polling_timeout_is_reported_and_can_be_collected(memory_api, monkeypatch):
    from imageatelier import network_api
    client, h, cfg = memory_api
    h.behaviour.append('block')
    monkeypatch.setattr(network_api, 'REQUEST_TIMEOUT', -1)
    ticket = client.post('/generate', json={'prompt': 'a cat', 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'failed')
    assert 'timed out' in client.get(ticket['status_url']).json()['error']
    assert client.get(ticket['result_url']).status_code == 504
    assert client.get(ticket['status_url']).status_code == 404
    wait_until(lambda: not h.engine._jobs)
    assert client.get('/health').json()['active_requests'] == 0


def test_disabling_the_api_cancels_background_requests(memory_api):
    client, h, cfg = memory_api
    h.behaviour.append('block')
    ticket = client.post('/generate', json={'prompt': 'a cat', 'references': [png_url()], 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    cfg.values['network_api_enabled'] = False
    wait_until(lambda: not h.engine._jobs)
    cfg.values['network_api_enabled'] = True
    assert client.get(ticket['status_url']).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0


def test_failed_polling_job_reports_error_and_releases_its_slot(memory_api):
    client, h, cfg = memory_api
    h.behaviour.append('fail')
    ticket = client.post('/generate', json={'prompt': 'a cat', 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'failed')
    assert 'boom' in client.get(ticket['status_url']).json()['error']
    assert client.get(ticket['result_url']).status_code == 503
    assert client.get('/health').json()['active_requests'] == 0
    assert h.engine._jobs == {}


def test_api_shutdown_cancels_unclaimed_background_jobs(h, monkeypatch):
    monkeypatch.setattr(models, 'require_suite', lambda name: {})
    h.behaviour.append('block')
    cfg = Config()
    app = create_api(h.engine, cfg, lambda value: value == 'Bearer test-key')
    h.start()
    with TestClient(app, headers={'Authorization': 'Bearer test-key'}) as client:
        ticket = client.post('/generate', json={'prompt': 'a cat', 'wait': False}).json()
        wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    wait_until(lambda: not h.engine._jobs)


@pytest.mark.parametrize('body', [
    {'prompt': 'x', 'output_path': '/tmp/leak.png'},
    {'prompt': 'x', 'ref_image_path': '/private/image.png'},
    {'prompt': 'x', 'references': ['file:///private/image.png']},
    {'prompt': 'x', 'references': ['https://example.com/image.png']},
    {'prompt': 'x', 'mode': 'edit'},
    {'prompt': 'x', 'width': 513, 'height': 512},
    {'prompt': 'x', 'width': True, 'height': 512},
])
def test_memory_api_rejects_paths_and_invalid_requests(memory_api, body):
    client, h, cfg = memory_api
    response = client.post('/generate', json=body)
    assert response.status_code == 400, response.text
    assert h.engine._jobs == {}


def test_api_only_surface_is_authenticated_and_has_no_project_routes(memory_api):
    client, _, _ = memory_api
    assert client.get('/health', headers={'Authorization': ''}).status_code == 401
    assert client.get('/health').json()['stores_images'] is False
    assert client.get('/api/projects').status_code == 404
    assert client.get('/').status_code == 404
    assert client.get('/docs').status_code == 404


def test_abandoned_job_is_cancelled_and_its_images_released(h):
    h.behaviour.append('block')
    ref = Image.new('RGB', (32, 32))
    jid = h.engine.submit_memory({'prompt': 'a cat', 'steps': 6, 'mode': 'edit'}, [ref])
    h.start()
    wait_state(h.engine, jid, ('running',))
    h.engine.abandon_memory(jid)
    wait_until(lambda: h.engine.get(jid) is None)
    with pytest.raises(ValueError):
        ref.getpixel((0, 0))
    assert h.engine.jobs() == []


def test_abandon_does_not_cancel_normal_ui_job(h):
    jid = h.gen()
    h.engine.abandon_memory(jid)
    assert h.engine.get(jid)['state'] == 'queued'


def test_transient_png_clears_body_after_failed_send():
    async def scenario():
        response = TransientPNG(b'image-bytes')
        async def send(message): raise RuntimeError('disconnected')
        with pytest.raises(RuntimeError):
            await response({'type': 'http'}, None, send)
        assert response.body == b''
    asyncio.run(scenario())


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def network_client(tmp_path, monkeypatch, ready):
    monkeypatch.setenv('IMAGEATELIER_API_PORT', str(free_port()))
    monkeypatch.setenv('IMAGEATELIER_UI_PORT', str(free_port()))
    app = create_app(engine=StubEngine(), data_dir=tmp_path/'data')
    with TestClient(app, base_url='http://localhost') as client:
        yield client, app


def test_toggles_start_and_stop_independent_listeners(network_client):
    client, app = network_client
    status = client.get('/api/network').json()
    assert not status['api_listening'] and not status['ui_listening']
    assert len(status['access_key']) >= 32
    assert app.state.network.key_path.stat().st_mode & 0o777 == 0o600
    result = client.put('/api/settings', json={'network_api_enabled': True})
    assert result.status_code == 200, result.text
    status = client.get('/api/network').json()
    assert status['api_listening'] and not status['ui_listening']
    assert client.put('/api/settings', json={'network_ui_enabled': True}).status_code == 200
    status = client.get('/api/network').json()
    assert status['api_listening'] and status['ui_listening']
    assert client.put('/api/settings', json={'network_api_enabled': False}).status_code == 200
    status = client.get('/api/network').json()
    assert not status['api_listening'] and status['ui_listening']
    assert client.put('/api/settings', json={'network_ui_enabled': False}).status_code == 200
    assert not client.get('/api/network').json()['ui_listening']


def test_port_conflict_rolls_back_settings_and_other_listener(network_client):
    client, app = network_client
    with socket.socket() as occupied:
        occupied.bind(('0.0.0.0', app.state.network.ui_port))
        occupied.listen()
        response = client.put('/api/settings', json={'network_api_enabled': True, 'network_ui_enabled': True})
        assert response.status_code == 409
    assert not client.get('/api/settings').json()['network_api_enabled']
    status = client.get('/api/network').json()
    assert not status['api_listening'] and not status['ui_listening']


def test_network_controls_reject_foreign_browser_origins(network_client):
    client, app = network_client
    headers = {'Origin': 'https://another-computer.example'}
    assert client.get('/api/network', headers=headers).status_code == 403
    assert client.put('/api/settings', json={'network_api_enabled': True}, headers=headers).status_code == 403
    assert not app.state.network.listeners


def test_authenticated_network_ui_saves_normally_but_cannot_change_network_controls(network_client):
    client, app = network_client
    token = client.get('/api/network').json()['access_key']
    assert client.put('/api/settings', json={'network_ui_enabled': True}).status_code == 200
    from starlette.applications import Starlette
    from starlette.routing import Mount
    remote_app = Starlette(routes=[Mount('/', app=app.state.network.ui)])
    with TestClient(remote_app, base_url='http://laptop.local') as remote:
        assert remote.get('/api/projects').status_code == 401
        remote.auth = ('imageatelier', token)
        assert remote.get('/api/projects').status_code == 200
        response = remote.post('/api/projects', json={'name': 'Network project'})
        assert response.status_code == 200
        assert app.state.store.get_project(response.json()['id']) is not None
        assert remote.get('/api/network').status_code == 403
        assert remote.put('/api/settings', json={'network_api_enabled': True}).status_code == 403
        assert 'access_key' not in remote.get('/api/settings').json()


def test_invalid_setting_cannot_leave_an_unconfigured_listener(network_client):
    client, app = network_client
    response = client.put('/api/settings', json={'network_api_enabled': True, 'unknown': True})
    assert response.status_code == 400
    assert not app.state.network.listeners


@pytest.mark.parametrize('action', ['generate', 'enhance'])
def test_client_disconnect_cancels_transient_job(memory_api, monkeypatch, action):
    client, h, cfg = memory_api
    from starlette.requests import Request
    async def disconnected(self): return True
    monkeypatch.setattr(Request, 'is_disconnected', disconnected)
    response = client.post('/'+action, json={'prompt': 'a cat', 'references': [png_url()]})
    assert response.status_code == 499
    wait_until(lambda: not h.engine._jobs)


@pytest.mark.parametrize('action', ['generate', 'enhance'])
def test_request_timeout_releases_transient_job(memory_api, monkeypatch, action):
    from imageatelier import network_api
    client, h, cfg = memory_api
    h.behaviour.append('block')
    monkeypatch.setattr(network_api, 'REQUEST_TIMEOUT', -1)
    response = client.post('/'+action, json={'prompt': 'a cat'})
    assert response.status_code == 504
    wait_until(lambda: not h.engine._jobs)


def test_api_queue_is_bounded_while_gpu_worker_is_busy(memory_api):
    from concurrent.futures import ThreadPoolExecutor
    client, h, cfg = memory_api
    h.behaviour.append('block')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, '/generate', json={'prompt': 'first'})
        wait_until(lambda: client.get('/health').json()['active_requests'] == 1)
        second = pool.submit(client.post, '/enhance', json={'prompt': 'second'})
        wait_until(lambda: client.get('/health').json()['active_requests'] == 2)
        response = client.post('/generate', json={'prompt': 'third'})
        assert response.status_code == 429
        assert h.engine.jobs() == []
        h.release.set()
        assert first.result(timeout=10).status_code == 200
        assert second.result(timeout=10).status_code == 200
    assert h.engine._jobs == {}



def test_network_ui_listener_serves_real_http_with_authentication(network_client):
    import httpx
    client, app = network_client
    token = client.get('/api/network').json()['access_key']
    response = client.put('/api/settings', json={'network_ui_enabled': True})
    assert response.status_code == 200
    url = f'http://127.0.0.1:{app.state.network.ui_port}'
    with httpx.Client(trust_env=False) as remote:
        assert remote.get(url+'/api/projects').status_code == 401
        remote.auth = ('imageatelier', token)
        response = remote.get(url+'/api/projects')
        assert response.status_code == 200, response.text
        assert isinstance(response.json(), list)
        assert remote.get(url+'/api/network').status_code == 403
        assert remote.get(url+'/').status_code == 200
