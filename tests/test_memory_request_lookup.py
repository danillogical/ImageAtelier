"""Independent exact-request discovery checks for Worlds restart reconciliation."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import wait_until
from test_network import memory_api  # noqa: F401
from test_serialization import h  # noqa: F401

CLIENT = 'worlds-client-0123456789abcdef'
REQUEST = 'worlds-work-0123456789abcdef'
HEADERS = {'X-ImageAtelier-Client-Id': CLIENT, 'X-ImageAtelier-Request-Id': REQUEST}
LOOKUP = f'/request-lookup/{CLIENT}/{REQUEST}'


def post(client, action='generate', **kwargs):
    return client.post('/' + action, headers=HEADERS, json={'prompt': 'private original prompt', 'wait': False, **kwargs})


def test_health_advertises_exact_request_lookup(memory_api):
    client, _, _ = memory_api
    assert client.get('/health').json().get('request_lookup') is True


@pytest.mark.parametrize('action', ['generate', 'enhance'])
def test_exact_lookup_recovers_admitted_token_without_reposting(memory_api, action):
    client, h, _ = memory_api
    h.behaviour.append('block')
    admitted = post(client, action)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    recovered = client.get(LOOKUP)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()['request_id'] == ticket['request_id']
    assert recovered.json()['kind'] == action
    for key in ('status_url', 'result_url', 'cancel_url'):
        assert recovered.json()[key] == ticket[key]
    assert recovered.headers['cache-control'] == 'no-store'
    assert client.get('/health').json()['active_requests'] == 1
    assert client.delete(ticket['cancel_url']).status_code == 204
    assert client.get(LOOKUP).status_code == 404


@pytest.mark.parametrize('second_action', ['generate', 'enhance'])
def test_duplicate_identity_cannot_admit_a_second_engine_job(memory_api, second_action):
    client, h, _ = memory_api
    h.behaviour.append('block')
    first = post(client)
    assert first.status_code == 202, first.text
    ticket = first.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    job_ids = set(h.engine._jobs)
    second = post(client, second_action)
    assert second.status_code == 409, second.text
    assert set(h.engine._jobs) == job_ids
    assert client.get('/health').json()['active_requests'] == 1
    recovered = client.get(LOOKUP)
    assert recovered.status_code == 200 and recovered.json()['request_id'] == ticket['request_id']
    assert client.delete(ticket['cancel_url']).status_code == 204


def test_exact_lookup_is_authenticated_scoped_and_has_no_listing_or_prompt(memory_api):
    client, h, _ = memory_api
    h.behaviour.append('block')
    admitted = post(client)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    assert client.get(LOOKUP, headers={'Authorization': ''}).status_code == 401
    assert client.get(LOOKUP.replace(CLIENT, 'another-client-0123456789abcdef')).status_code == 404
    assert client.get(LOOKUP.replace(REQUEST, 'another-work-0123456789abcdef')).status_code == 404
    assert client.get('/request-lookup').status_code == 404
    assert client.get('/request-lookup/' + CLIENT).status_code == 404
    recovered = client.get(LOOKUP)
    assert recovered.status_code == 200, recovered.text
    assert 'private original prompt' not in recovered.text and 'test-key' not in recovered.text
    assert 'job_id' not in recovered.json() and 'references' not in recovered.json()
    assert client.delete(ticket['cancel_url']).status_code == 204


@pytest.mark.parametrize('headers', [
    {'X-ImageAtelier-Client-Id': CLIENT},
    {'X-ImageAtelier-Request-Id': REQUEST},
    {**HEADERS, 'X-ImageAtelier-Client-Id': 'short'},
    {**HEADERS, 'X-ImageAtelier-Request-Id': 'x' * 129},
    {**HEADERS, 'X-ImageAtelier-Request-Id': 'private/path-0123456789abcdef'},
])
def test_invalid_or_partial_identity_is_rejected_before_engine_admission(memory_api, headers):
    client, h, _ = memory_api
    response = client.post('/generate', headers=headers, json={'prompt': 'private original prompt', 'wait': False})
    if response.status_code == 202:
        client.delete(response.json()['cancel_url'])
    assert response.status_code == 400, response.text
    assert not h.engine._jobs
    assert client.get('/health').json()['active_requests'] == 0


def test_correlation_requires_async_and_failed_intake_releases_identity(memory_api):
    client, h, _ = memory_api
    refused = post(client, wait=True)
    assert refused.status_code == 400, refused.text
    assert client.get(LOOKUP).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0
    invalid = post(client, width=513, height=512)
    assert invalid.status_code == 400, invalid.text
    assert client.get(LOOKUP).status_code == 404
    accepted = post(client)
    assert accepted.status_code == 202, accepted.text
    recovered = client.get(LOOKUP)
    assert recovered.status_code == 200 and recovered.json()['request_id'] == accepted.json()['request_id']
    assert client.delete(accepted.json()['cancel_url']).status_code == 204


@pytest.mark.parametrize('release', ['consume', 'discard'])
def test_releasing_result_removes_exact_lookup_without_leaking_capacity(memory_api, release):
    client, _, _ = memory_api
    admitted = post(client)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'done')
    assert client.get(LOOKUP).status_code == 200
    response = client.get(ticket['result_url']) if release == 'consume' else client.delete(ticket['cancel_url'])
    assert response.status_code == (200 if release == 'consume' else 204)
    assert client.get(LOOKUP).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0


def test_terminal_expiry_removes_exact_lookup(memory_api, monkeypatch):
    from imageatelier import network_api
    client, _, _ = memory_api
    admitted = post(client)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'done')
    assert client.get(LOOKUP).status_code == 200
    monkeypatch.setattr(network_api, 'RESULT_TTL', .02)
    wait_until(lambda: client.get(ticket['status_url']).status_code == 404)
    assert client.get(LOOKUP).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0


def test_retained_cancel_keeps_exact_lookup_until_the_token_is_released(memory_api):
    client, h, _ = memory_api
    h.behaviour.append('block')
    admitted = post(client)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    assert client.delete(ticket['cancel_url'], params={'retain_status': 'true'}).status_code == 204
    recovered = client.get(LOOKUP)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()['request_id'] == ticket['request_id']
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'cancelled')
    assert client.get(LOOKUP).status_code == 200
    assert client.get(ticket['result_url']).status_code == 503
    assert client.get(LOOKUP).status_code == 404


def test_receiving_identity_is_visible_and_blocks_duplicate_before_body_admission(memory_api, monkeypatch):
    from starlette.requests import Request
    client, h, _ = memory_api
    entered, release = threading.Event(), threading.Event()
    original = Request.stream

    async def held_stream(request):
        if request.headers.get('x-imageatelier-request-id') == REQUEST and not entered.is_set():
            entered.set()
            while not release.is_set():
                await asyncio.sleep(.01)
        async for chunk in original(request):
            yield chunk

    monkeypatch.setattr(Request, 'stream', held_stream)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(post, client)
        try:
            assert entered.wait(5), 'Request never reached its body intake'
            receiving = client.get(LOOKUP)
            assert receiving.status_code == 202, receiving.text
            assert receiving.json()['state'] == 'receiving'
            assert receiving.json().get('request_id') is None
            assert receiving.headers.get('retry-after')
            duplicate = post(client)
            assert duplicate.status_code == 409, duplicate.text
            assert not h.engine._jobs
            assert client.get('/health').json()['active_requests'] == 1
        finally:
            release.set()
            admitted = pending.result(timeout=10)
            if admitted.status_code == 202:
                client.delete(admitted.json()['cancel_url'])
    assert admitted.status_code == 202, admitted.text


def test_legacy_requests_stay_unindexed_and_keep_their_existing_shapes(memory_api):
    client, _, _ = memory_api
    sync = client.post('/generate', json={'prompt': 'legacy synchronous'})
    assert sync.status_code == 200 and sync.content.startswith(b'\x89PNG')
    async_response = client.post('/generate', json={'prompt': 'legacy async', 'wait': False})
    assert async_response.status_code == 202
    ticket = async_response.json()
    assert set(ticket) == {'request_id', 'status_url', 'result_url', 'cancel_url'}
    assert client.get(LOOKUP).status_code == 404
    assert client.delete(ticket['cancel_url']).status_code == 204


@pytest.mark.parametrize('action,method', [('generate', 'submit_memory'), ('enhance', 'submit_memory_enhance')])
def test_identity_never_enters_engine_parameters_or_changes_settings(memory_api, monkeypatch, action, method):
    client, h, cfg = memory_api
    original = getattr(h.engine, method)
    seen = []
    before = dict(cfg.values)

    def capture(params, images):
        seen.append(dict(params))
        return original(params, images)

    monkeypatch.setattr(h.engine, method, capture)
    admitted = post(client, action)
    assert admitted.status_code == 202, admitted.text
    assert len(seen) == 1
    assert CLIENT not in repr(seen) and REQUEST not in repr(seen)
    assert not any('client' in key or 'request' in key for key in seen[0])
    assert cfg.values == before
    try:
        recovered = client.get(LOOKUP)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()['request_id'] == admitted.json()['request_id']
    finally:
        client.delete(admitted.json()['cancel_url'])


def test_full_capacity_does_not_erase_duplicates_or_reserve_refused_identities(memory_api):
    client, h, _ = memory_api
    h.behaviour.append('block')
    first = post(client)
    assert first.status_code == 202, first.text
    ticket = first.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    other_client = 'other-worlds-client-0123456789abcdef'
    other_headers = {**HEADERS, 'X-ImageAtelier-Client-Id': other_client}
    other_lookup = LOOKUP.replace(CLIENT, other_client)
    second = client.post('/generate', headers=other_headers, json={'prompt': 'second client', 'wait': False})
    assert second.status_code == 202, second.text
    try:
        assert client.get(LOOKUP).json().get('request_id') == ticket['request_id']
        assert client.get(other_lookup).json().get('request_id') == second.json()['request_id']
        assert ticket['request_id'] != second.json()['request_id']
        assert post(client).status_code == 409
        refused_id = 'third-worlds-work-0123456789abcdef'
        refused_headers = {**HEADERS, 'X-ImageAtelier-Request-Id': refused_id}
        refused_lookup = LOOKUP.replace(REQUEST, refused_id)
        refused = client.post('/enhance', headers=refused_headers, json={'prompt': 'third', 'wait': False})
        assert refused.status_code == 429, refused.text
        assert client.get(refused_lookup).status_code == 404
        assert client.delete(second.json()['cancel_url']).status_code == 204
        admitted = client.post('/enhance', headers=refused_headers, json={'prompt': 'third', 'wait': False})
        assert admitted.status_code == 202, admitted.text
        try:
            assert client.get(refused_lookup).json().get('request_id') == admitted.json()['request_id']
        finally:
            client.delete(admitted.json()['cancel_url'])
    finally:
        client.delete(ticket['cancel_url'])
        client.delete(second.json()['cancel_url'])


def hold_request_intake(monkeypatch):
    from starlette.requests import Request
    entered, release = threading.Event(), threading.Event()
    original = Request.stream

    async def stream(request):
        if request.headers.get('x-imageatelier-request-id') == REQUEST:
            entered.set()
            while not release.is_set():
                await asyncio.sleep(.01)
        async for chunk in original(request):
            yield chunk

    monkeypatch.setattr(Request, 'stream', stream)
    return entered, release


@pytest.mark.parametrize('interruption', ['shutdown', 'disabled'])
def test_interrupted_receiving_claim_cannot_resurrect_engine_work(memory_api, monkeypatch, interruption):
    client, h, cfg = memory_api
    entered, release = hold_request_intake(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(post, client)
        try:
            assert entered.wait(5), 'Request never reached its body intake'
            assert client.get(LOOKUP).status_code == 202
            if interruption == 'shutdown':
                client.portal.call(client.app.state.cancel_memory_requests)
                assert client.get(LOOKUP).status_code == 404
            else:
                cfg.values['network_api_enabled'] = False
            release.set()
            admitted = pending.result(timeout=10)
            assert admitted.status_code == 503, admitted.text
            assert not h.engine._jobs
        finally:
            release.set()
            admitted = pending.result(timeout=10)
            cfg.values['network_api_enabled'] = True
            if admitted.status_code == 202:
                client.delete(admitted.json()['cancel_url'])
    assert client.get(LOOKUP).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0


def test_listener_disable_clears_the_admitted_lookup_and_reservation(memory_api):
    client, h, cfg = memory_api
    h.behaviour.append('block')
    admitted = post(client)
    assert admitted.status_code == 202, admitted.text
    ticket = admitted.json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    try:
        assert client.get(LOOKUP).status_code == 200
        cfg.values['network_api_enabled'] = False
        wait_until(lambda: not h.engine._jobs)
    finally:
        cfg.values['network_api_enabled'] = True
        client.delete(ticket['cancel_url'])
    assert client.get(LOOKUP).status_code == 404
    assert client.get('/health').json()['active_requests'] == 0
