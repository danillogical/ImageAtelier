"""Independent memory cancellation contract tests against the real worker with stubbed inference."""
import threading
import time
from contextlib import contextmanager

import pytest

from conftest import wait_until
from test_network import memory_api, png_url  # noqa: F401
from test_serialization import h  # noqa: F401


@contextmanager
def held_teardown(h, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = h.engine._teardown

    def teardown(job):
        entered.set()
        if not release.wait(10):
            raise AssertionError('Test did not release worker teardown')
        return original(job)

    monkeypatch.setattr(h.engine, '_teardown', teardown)
    try:
        yield entered, release
    finally:
        release.set()


def retained_cancel(client, ticket):
    return client.delete(ticket['cancel_url'], params={'retain_status': 'true'})


def test_health_advertises_retained_cancellation(memory_api):
    client, _, _ = memory_api
    assert client.get('/health').json().get('cancellation_status') is True


@pytest.mark.parametrize('action', ['generate', 'enhance'])
def test_retained_cancel_waits_for_teardown_and_keeps_capacity(memory_api, monkeypatch, action):
    client, h, _ = memory_api
    h.behaviour.append('block')
    before = set(h.tmp.rglob('*'))
    with held_teardown(h, monkeypatch) as (entered, release):
        first = client.post('/' + action, json={
            'prompt': 'private cancellation prompt', 'references': [png_url()], 'wait': False,
        }).json()
        wait_until(lambda: client.get(first['status_url']).json()['state'] == 'running')
        second = client.post('/generate', json={'prompt': 'queued', 'wait': False}).json()
        assert retained_cancel(client, first).status_code == 204
        assert entered.wait(5), 'Worker did not enter teardown'
        for _ in range(2):
            response = client.get(first['status_url'])
            assert response.status_code == 200, response.text
            status = response.json()
            assert status['state'] == 'cancelling'
            assert status['expires_in'] is None and status['result_url'] is None
            assert 'private cancellation prompt' not in response.text
            assert client.get(first['result_url']).status_code == 409
            assert retained_cancel(client, first).status_code == 204
        assert client.get(second['status_url']).json()['state'] == 'queued'
        assert client.get('/health').json()['active_requests'] == 2
        full = client.post('/enhance', json={'prompt': 'third', 'wait': False})
        assert full.status_code == 429 and full.headers.get('retry-after')
        release.set()
        wait_until(lambda: client.get(first['status_url']).json()['state'] == 'cancelled')
        assert client.get(first['status_url']).json()['expires_in'] > 0
        assert client.get('/health').json()['active_requests'] == 2
        assert retained_cancel(client, first).status_code == 204
        assert client.get(first['status_url']).json()['state'] == 'cancelled'
        assert client.get(first['result_url']).status_code == 503
        assert client.get(first['status_url']).status_code == 404
        assert client.get('/health').json()['active_requests'] == 1
        wait_until(lambda: client.get(second['status_url']).json()['state'] == 'done')
        assert client.delete(second['cancel_url']).status_code == 204
        wait_until(lambda: not h.engine._jobs)
    assert client.get('/health').json()['active_requests'] == 0
    assert set(h.tmp.rglob('*')) == before


def test_retained_queued_cancellation_is_pollable_and_never_runs(memory_api):
    client, h, _ = memory_api
    h.behaviour.append('block')
    first = client.post('/generate', json={'prompt': 'running', 'wait': False}).json()
    wait_until(lambda: client.get(first['status_url']).json()['state'] == 'running')
    queued = client.post('/enhance', json={'prompt': 'queued', 'wait': False}).json()
    assert client.get(queued['status_url']).json()['state'] == 'queued'
    assert retained_cancel(client, queued).status_code == 204
    response = client.get(queued['status_url'])
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'cancelled'
    assert response.json()['queue_position'] is None
    assert client.get(first['status_url']).json()['state'] == 'running'
    assert client.get('/health').json()['active_requests'] == 2
    assert client.delete(queued['cancel_url']).status_code == 204
    assert client.get(queued['status_url']).status_code == 404
    assert client.delete(first['cancel_url']).status_code == 204
    wait_until(lambda: not h.engine._jobs)


def test_retained_cancel_does_not_expire_or_timeout_during_teardown(memory_api, monkeypatch):
    from imageatelier import network_api
    client, h, _ = memory_api
    h.behaviour.append('block')
    monkeypatch.setattr(network_api, 'REQUEST_TIMEOUT', .5)
    with held_teardown(h, monkeypatch) as (entered, release):
        ticket = client.post('/generate', json={'prompt': 'cancel', 'wait': False}).json()
        wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
        assert retained_cancel(client, ticket).status_code == 204
        assert entered.wait(5)
        monkeypatch.setattr(network_api, 'RESULT_TTL', .05)
        time.sleep(.65)
        response = client.get(ticket['status_url'])
        assert response.status_code == 200, response.text
        assert response.json()['state'] == 'cancelling'
        assert response.json()['expires_in'] is None
        assert client.get('/health').json()['active_requests'] == 1
        release.set()
        wait_until(lambda: client.get(ticket['status_url']).status_code == 404)
        wait_until(lambda: not h.engine._jobs)
        assert client.get('/health').json()['active_requests'] == 0


@pytest.mark.parametrize('outcome', ['done', 'failed'])
def test_retained_cancel_preserves_existing_terminal_outcome(memory_api, outcome):
    client, h, _ = memory_api
    if outcome == 'failed':
        h.behaviour.append('fail')
    ticket = client.post('/generate', json={'prompt': 'already terminal', 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == outcome)
    assert retained_cancel(client, ticket).status_code == 204
    response = client.get(ticket['status_url'])
    assert response.status_code == 200, response.text
    assert response.json()['state'] == outcome
    assert client.get(ticket['result_url']).status_code == (200 if outcome == 'done' else 503)
    assert client.get(ticket['status_url']).status_code == 404


def test_retained_cancel_requires_auth_and_unknown_token_stays_unknown(memory_api):
    client, h, _ = memory_api
    h.behaviour.append('block')
    ticket = client.post('/generate', json={'prompt': 'auth', 'wait': False}).json()
    wait_until(lambda: client.get(ticket['status_url']).json()['state'] == 'running')
    assert client.delete(ticket['cancel_url'], params={'retain_status': 'true'},
                         headers={'Authorization': ''}).status_code == 401
    assert client.get(ticket['status_url']).json()['state'] == 'running'
    assert client.delete('/requests/unknown', params={'retain_status': 'true'}).status_code == 404
    assert client.delete(ticket['cancel_url']).status_code == 204
    wait_until(lambda: not h.engine._jobs)
