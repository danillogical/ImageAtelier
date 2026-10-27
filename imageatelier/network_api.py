"""Transient LAN image API. References and output PNGs are never written to files."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import secrets
import warnings
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError
from starlette.responses import JSONResponse, Response

from imageatelier import custom_models, models

MAX_BODY = 48 * 1024**2
MAX_REFERENCE_BYTES = 8 * 1024**2
MAX_REFERENCE_PIXELS = 16_000_000
REQUEST_TIMEOUT = 1800
RESULT_TTL = 600
IDENTITY = re.compile(r'[A-Za-z0-9_-]{16,128}')
CLIENT_HEADER = 'X-ImageAtelier-Client-Id'
REQUEST_HEADER = 'X-ImageAtelier-Request-Id'


class Generate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=16000)
    width: StrictInt | None = None
    height: StrictInt | None = None
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    steps: StrictInt | None = Field(default=None, ge=2, le=100)
    mode: Literal['txt2img', 't2i', 'edit'] = 'txt2img'
    references: list[str] = Field(default_factory=list, max_length=10)
    transparent: StrictBool = False
    wait: StrictBool = True


class Enhance(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=16000)
    mode: Literal['t2i', 'i2i'] | None = None
    references: list[str] = Field(default_factory=list, max_length=10)
    thinking: StrictBool | None = None
    language: Literal['english', 'auto'] | None = None
    max_new_tokens: StrictInt | None = Field(default=None, ge=256, le=16384)
    wait: StrictBool = True


def decode_reference(data_url: str):
    from PIL import Image, ImageOps
    if not isinstance(data_url, str) or len(data_url) > MAX_REFERENCE_BYTES * 4 // 3 + 128:
        raise ValueError('Reference image is too large')
    try:
        prefix, encoded = data_url.split(',', 1)
        if prefix not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('References must be PNG, JPEG or WebP data URLs; paths and URLs are not accepted')
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError('Reference image is too large')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with io.BytesIO(raw) as buffer, Image.open(buffer) as image:
                if image.width * image.height > MAX_REFERENCE_PIXELS:
                    raise ValueError('Reference image exceeds the pixel limit')
                oriented = ImageOps.exif_transpose(image).convert('RGBA')
                # Match Worlds' opaque reference conditioning without storing intermediate files.
                background = Image.new('RGBA', oriented.size, 'white')
                background.alpha_composite(oriented)
                oriented.close()
                result = background.convert('RGB')
                background.close()
                return result
    except (OSError, binascii.Error, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Invalid reference image') from exc


class TransientPNG(Response):
    """Drop the response's PNG reference after ASGI send finishes, including a failed send."""
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.body = b''


def create_api(engine, settings, authorize, enabled=lambda: True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await cancel_requests()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    active = 0
    pending = {}
    # Live correlation claims, one per capacity reservation; never a content cache.
    lookups = {}

    def drop(claim):
        if claim is not None and lookups.get(claim['key']) is claim:
            del lookups[claim['key']]

    def ticket(token):
        return {'request_id': token, 'status_url': f'/requests/{token}',
                'result_url': f'/requests/{token}/result', 'cancel_url': f'/requests/{token}'}

    def release(token):
        nonlocal active
        entry = pending.pop(token, None)
        if entry is None:
            return
        drop(entry.get('claim'))
        engine.abandon_memory(entry['job_id'])
        task = entry.get('task')
        if task and task is not asyncio.current_task():
            task.cancel()
        active -= 1

    async def cancel_requests():
        tasks = [entry['task'] for entry in pending.values()]
        for token in list(pending):
            release(token)
        # Receiving claims are dropped too; their requests still own their capacity slots.
        lookups.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    app.state.cancel_memory_requests = cancel_requests

    def observe(entry):
        snapshot = engine.get(entry['job_id'])
        now = asyncio.get_running_loop().time()
        if entry['finished_at'] is None:
            if snapshot is None:
                entry.update(state='failed', error='Image job was interrupted', error_status=503, finished_at=now)
            elif snapshot['state'] in ('done', 'failed', 'cancelled'):
                entry.update(state=snapshot['state'], error=snapshot.get('error'), error_status=503, finished_at=now)
            elif now > entry['deadline'] and not entry['cancel_requested']:
                entry.update(state='failed', error='Memory API request timed out', error_status=504, finished_at=now)
                engine.abandon_memory(entry['job_id'])
        return snapshot

    async def watch(token):
        try:
            while token in pending:
                if not enabled():
                    release(token)
                    return
                entry = pending[token]
                observe(entry)
                if entry['finished_at'] is not None and asyncio.get_running_loop().time() - entry['finished_at'] >= RESULT_TTL:
                    release(token)
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A broken monitor must not leave an unbounded image/result reservation behind.
            release(token)

    def status(token):
        entry = pending.get(token)
        if entry is None:
            raise HTTPException(404, 'Unknown or expired request')
        snapshot = observe(entry) or {}
        state = entry['state'] or snapshot.get('state', 'queued')
        return {
            'request_id': token, 'kind': 'enhance' if entry['enhance'] else 'generate',
            'state': state, 'queue_position': snapshot.get('queue_position'),
            'progress': {key: snapshot.get(key) for key in ('phase', 'step', 'total_steps', 'tokens')},
            'error': entry['error'],
            'result_url': f'/requests/{token}/result' if state == 'done' else None,
            'expires_in': max(0, round(RESULT_TTL - (asyncio.get_running_loop().time() - entry['finished_at'])))
                if entry['finished_at'] is not None else None,
        }

    def take_result(job_id, enhance):
        if enhance:
            return JSONResponse(engine.take_memory_enhancement(job_id), headers={'Cache-Control': 'no-store'})
        data, metadata = engine.take_memory_result(job_id)
        return TransientPNG(data, media_type='image/png', headers={
            'Cache-Control': 'no-store', 'X-Image-Provider': 'qwen-image-2.1',
            'X-Image-Width': str(metadata['width']), 'X-Image-Height': str(metadata['height']),
            'X-Image-Seed': str(metadata['seed']),
        })

    @app.middleware('http')
    async def access(request, call_next):
        if not enabled():
            return JSONResponse({'detail': 'Memory API is disabled'}, status_code=503)
        if not authorize(request.headers.get('authorization', '')):
            return JSONResponse({'detail': 'A Bearer access key is required'}, status_code=401,
                                headers={'WWW-Authenticate': 'Bearer'})
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/health')
    async def health():
        return {'service': 'imageatelier', 'mode': 'memory', 'stores_images': False,
                'provider': 'qwen-image-2.1', 'active_requests': active,
                'actions': ['enhance', 'generate'], 'max_reference_images': 10,
                'polling': True, 'cancellation_status': True, 'request_lookup': True,
                'result_ttl_seconds': RESULT_TTL}

    @app.get('/requests/{token}')
    async def request_status(token: str):
        return JSONResponse(status(token), headers={'Retry-After': '1'})

    @app.get('/requests/{token}/result')
    async def request_result(token: str):
        info = status(token)
        if info['state'] not in ('done', 'failed', 'cancelled'):
            raise HTTPException(409, 'Result is not ready', headers={'Retry-After': '1'})
        entry = pending[token]
        try:
            if info['state'] != 'done':
                raise HTTPException(entry['error_status'], entry['error'] or 'Image job cancelled')
            return take_result(entry['job_id'], entry['enhance'])
        finally:
            release(token)

    @app.delete('/requests/{token}', status_code=204)
    async def cancel_request(token: str, retain_status: bool = False):
        entry = pending.get(token)
        if entry is None:
            raise HTTPException(404, 'Unknown or expired request')
        if not retain_status:
            release(token)
            return Response(status_code=204)
        observe(entry)
        if entry['finished_at'] is None:
            # Retained cancellation stays pollable until the worker publishes the job's terminal state.
            entry['cancel_requested'] = True
            engine.cancel(entry['job_id'])
        return Response(status_code=204)

    @app.get('/request-lookup/{client_id}/{client_request_id}')
    async def request_lookup(client_id: str, client_request_id: str):
        if not (IDENTITY.fullmatch(client_id) and IDENTITY.fullmatch(client_request_id)):
            raise HTTPException(400, 'Invalid request identity')
        claim = lookups.get((client_id, client_request_id))
        if claim is None or (claim['token'] is not None and claim['token'] not in pending):
            raise HTTPException(404, 'Unknown or expired request')
        if claim['token'] is None:
            return JSONResponse({'state': 'receiving', 'kind': claim['kind']},
                                status_code=202, headers={'Retry-After': '1'})
        return {**ticket(claim['token']), 'kind': claim['kind']}

    @app.post('/generate')
    async def generate(request: Request):
        return await execute(request, enhance=False)

    @app.post('/enhance')
    async def enhance(request: Request):
        return await execute(request, enhance=True)

    async def execute(request: Request, *, enhance: bool):
        nonlocal active
        clients = request.headers.getlist(CLIENT_HEADER)
        requests = request.headers.getlist(REQUEST_HEADER)
        claim = None
        if clients or requests:
            if (len(clients) != 1 or len(requests) != 1
                    or IDENTITY.fullmatch(clients[0]) is None or IDENTITY.fullmatch(requests[0]) is None):
                raise HTTPException(
                    400, 'X-ImageAtelier-Client-Id and X-ImageAtelier-Request-Id must be sent together '
                         'as 16-128 characters of A-Z, a-z, 0-9, _ or -')
            key = (clients[0], requests[0])
            if key in lookups:
                raise HTTPException(409, 'This client request is already in progress; use GET /request-lookup to recover it')
            claim = {'key': key, 'kind': 'enhance' if enhance else 'generate', 'token': None}
        if active >= 2:
            raise HTTPException(429, 'Image request queue is full; retry later', headers={'Retry-After': '5'})
        # Increment before awaiting: bounds both the model queue and in-flight request bodies.
        active += 1
        if claim is not None:
            # Reserve the exact pair together with its capacity slot, before any await.
            lookups[claim['key']] = claim
        images, job_id, retained = [], None, False
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(30):
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_BODY:
                            raise HTTPException(413, 'Request body is too large')
                        body.extend(chunk)
                payload = (Enhance if enhance else Generate).model_validate_json(body)
            except (ValueError, ValidationError) as exc:
                raise HTTPException(400, 'Invalid memory API request: ' + str(exc).split('\n')[0])
            except TimeoutError:
                raise HTTPException(408, 'Request body timed out')
            finally:
                body.clear()
            # No await occurs from here until the ticket is recorded.
            if claim is not None and lookups.get(claim['key']) is not claim:
                raise HTTPException(503, 'Memory request was interrupted')
            if claim is not None and not enabled():
                raise HTTPException(503, 'Memory API is disabled')
            if claim is not None and payload.wait:
                raise HTTPException(400, 'Correlated requests require wait: false')
            cfg = settings.get()
            try:
                if not enhance:
                    models.require_suite(cfg['model_suite'])
                    custom_models.validate_pair(cfg['image_model'], cfg['text_encoder'])
                for reference in payload.references:
                    images.append(decode_reference(reference))
                payload.references.clear()
                if enhance:
                    mode = payload.mode or ('i2i' if images else 't2i')
                    if mode == 'i2i' and not images:
                        raise ValueError('I2I enhancement requires at least one reference image')
                    if mode == 't2i' and images:
                        raise ValueError('Use i2i enhancement when supplying reference images')
                    quant = cfg.get('pe_quant', '8bit')
                    rewriter = cfg.get(f'custom_pe_{mode}') or None
                    models.resolve_pe(mode, rewriter or quant)
                    job_id = engine.submit_memory_enhance({
                        'prompt': payload.prompt, 'mode': mode, 'quant': quant, 'rewriter': rewriter,
                        'thinking': payload.thinking if payload.thinking is not None else cfg.get('pe_thinking', False),
                        'language': payload.language or cfg.get('pe_language', 'english'),
                        'max_new_tokens': payload.max_new_tokens or cfg.get('pe_max_new_tokens', 8192),
                    }, images)
                else:
                    if payload.mode == 'edit' and not images:
                        raise ValueError('Edit requests require at least one reference image')
                    job_id = engine.submit_memory({
                        'prompt': payload.prompt, 'mode': 'edit' if payload.mode == 'edit' else 't2i',
                        'width': payload.width, 'height': payload.height,
                        'seed': payload.seed, 'steps': payload.steps if payload.steps is not None else cfg['steps'],
                        'transparent': payload.transparent, 'image_model': cfg['image_model'],
                        'text_encoder': cfg['text_encoder'],
                    }, images)
                images.clear()  # The engine now owns and closes these images.
            except models.ModelNotReady as exc:
                raise HTTPException(409, str(exc))
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            except RuntimeError:
                raise HTTPException(503, 'Image engine is unavailable')
            if not payload.wait:
                token = secrets.token_urlsafe(24)
                pending[token] = {
                    'job_id': job_id, 'enhance': enhance, 'deadline': asyncio.get_running_loop().time() + REQUEST_TIMEOUT,
                    'finished_at': None, 'state': None, 'error': None, 'error_status': 503,
                    'cancel_requested': False,
                }
                pending[token]['claim'] = claim
                if claim is not None:
                    claim['token'] = token
                pending[token]['task'] = asyncio.create_task(watch(token))
                retained = True
                return JSONResponse(ticket(token), status_code=202,
                                    headers={'Location': f'/requests/{token}', 'Retry-After': '1'})
            del payload
            deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT
            while True:
                if await request.is_disconnected():
                    raise HTTPException(499, 'Client disconnected')
                snapshot = engine.get(job_id)
                if snapshot is None:
                    raise HTTPException(503, 'Image job was interrupted')
                if snapshot['state'] == 'done':
                    return take_result(job_id, enhance)
                if snapshot['state'] in ('failed', 'cancelled'):
                    raise HTTPException(503, snapshot.get('error') or 'Image job cancelled')
                if asyncio.get_running_loop().time() > deadline:
                    raise HTTPException(504, 'Image generation timed out')
                await asyncio.sleep(0.1)
        finally:
            if job_id is not None and not retained:
                engine.abandon_memory(job_id)
            for image in images:
                image.close()
            images.clear()
            if not retained:
                drop(claim)
                active -= 1

    return app
