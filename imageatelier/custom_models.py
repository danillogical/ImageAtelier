"""Local Qwen checkpoints: inspect without loading weights; prepare offline and compose over a suite."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from imageatelier import models, suites


@lru_cache(maxsize=64)
def _header(path: str, size: int, modified: int) -> dict:
    with open(path, 'rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= length <= min(size - 8, 64 * 1024**2):
            raise ValueError('Invalid safetensors header size')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors header')
    return header


def header(path: Path) -> dict:
    stat = path.stat()
    return _header(str(path), stat.st_size, stat.st_mtime_ns)


def folder(identifier: str) -> Path:
    if not isinstance(identifier, str):
        raise ValueError('Custom model ID must be a string')
    identifier = identifier.split('@', 1)[0]
    if not identifier.startswith('custom-') or len(identifier) != 23 or any(c not in '0123456789abcdef' for c in identifier[7:]):
        raise ValueError('Invalid custom model ID')
    return models.models_dir() / '.custom' / identifier


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {'file': path.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def inspect(path: Path) -> dict:
    h = header(path)
    keys = set(h) - {'__metadata__'}
    kind = 'image' if 'img_in.weight' in keys and 'transformer_blocks.0.attn.to_q.weight' in keys else 'text_encoder' if any(k.startswith(('model.layers.', 'model.language_model.', 'language_model.')) for k in keys) else 'unknown'
    error = None
    if kind == 'text_encoder' and not any(k.startswith(('visual.', 'model.visual.')) for k in keys):
        error = 'This text encoder has no vision tower; reference-image editing requires vision weights.'
    elif kind == 'unknown':
        error = 'Not a recognized Qwen-Image-2.1 transformer or text encoder.'
    identifier = 'custom-' + hashlib.sha256(path.name.encode()).hexdigest()[:16]
    root = folder(identifier)
    ready = False
    try:
        manifest = json.loads((root / 'custom.json').read_text())
        ready = manifest == {'format': 1, 'kind': kind, 'source': fingerprint(path)} and suites.weights_complete(root / kind)
    except (OSError, ValueError):
        pass
    with suites._lock:
        job = dict(suites._jobs.get(identifier, {}))
    preparing = bool(job.get('thread') and job['thread'].is_alive())
    return {'id': identifier, 'kind': kind, 'label': path.stem, 'filename': path.name,
            'bytes': path.stat().st_size, 'state': 'preparing' if preparing else 'ready' if ready else 'unsupported' if error else 'not_prepared',
            'phase': job.get('phase'), 'error': error or job.get('error'), 'path': str(path)}


def inspect_enhancer(path: Path) -> dict:
    label = path.name
    mode = 't2i' if '-PE-T2I-' in label else 'i2i'
    identifier = 'custom-' + hashlib.sha256(label.encode()).hexdigest()[:16]
    # Only complete, vision-capable Hugging Face/MLX folders are selectable.
    ready = suites.pe_complete(path)
    required = ('tokenizer.json', 'processor_config.json', 'system_prompt.txt')
    ready = ready and all((path / name).is_file() for name in required)
    return {'id': identifier, 'kind': f'pe_{mode}', 'label': label, 'filename': label,
            'path': str(path), 'bytes': sum(f.stat().st_size for f in path.glob('*.safetensors')),
            'state': 'ready' if ready else 'incomplete', 'phase': None, 'error': None}


def list_models() -> list[dict]:
    rows = []
    for path in sorted(models.models_dir().iterdir()) if models.models_dir().exists() else []:
        if path.is_symlink() or path.name.startswith('.'):
            continue
        enhancer = path.is_dir() and any(tag in path.name for tag in ('-PE-T2I-', '-PE-I2I-'))
        if not enhancer and (not path.is_file() or path.suffix != '.safetensors'):
            continue
        try:
            rows.append(inspect_enhancer(path) if enhancer else inspect(path))
        except (OSError, ValueError, struct.error, KeyError, TypeError) as exc:
            rows.append({'id': '', 'kind': 'unknown', 'label': path.name, 'filename': path.name,
                         'state': 'unsupported', 'error': str(exc)})
    return rows


def get(identifier: str, kind: str | None = None, *, ready=False) -> dict:
    folder(identifier)  # Validate before any path operation.
    identifier = identifier.split('@', 1)[0]
    row = next((r for r in list_models() if r['id'] == identifier), None)
    if row is None or (kind is not None and row['kind'] != kind):
        raise ValueError('Unknown custom model or wrong component type')
    if ready and row['state'] != 'ready':
        raise models.ModelNotReady(f"Prepare {row['label']} in Settings first")
    return row


def validate_pair(image: str, encoder: str) -> None:
    if not isinstance(image, str) or not isinstance(encoder, str):
        raise ValueError('Model IDs must be strings')
    custom_image, custom_encoder = image.startswith('custom-'), encoder.startswith('custom-')
    if custom_image:
        get(image, 'image', ready=True)
    elif image not in models.IMAGE_MODELS:
        raise ValueError(f'unknown image_model {image!r}')
    if custom_encoder:
        get(encoder, 'text_encoder', ready=True)
    elif encoder not in models.TEXT_ENCODERS:
        raise ValueError(f'unknown text_encoder {encoder!r}')
    if not custom_image and not custom_encoder and image != encoder:
        raise ValueError('Choose a complete suite; model parts cannot be mixed')


def compose(base_suite: str, image: str, encoder: str) -> Path:
    validate_pair(image, encoder)
    base = models.require_suite(base_suite)['image']
    sources = {name: base / name for name in ('transformer', 'text_encoder', 'vae', 'processor', 'scheduler', 'model_index.json')}
    if image.startswith('custom-'):
        sources['transformer'] = folder(image) / 'image'
    if encoder.startswith('custom-'):
        sources['text_encoder'] = folder(encoder) / 'text_encoder'
    identity = '\n'.join(f'{k}:{v.resolve()}' for k, v in sources.items())
    root = models.models_dir() / '.custom' / 'composed' / hashlib.sha256(identity.encode()).hexdigest()[:16]
    root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not source.exists():
            continue
        link = root / name
        try:
            link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != source.resolve():
                raise ValueError('Custom model composition contains an unexpected file')
    return root


@suites._serialized
def prepare(identifier: str, base_suite: str, maintenance=contextlib.nullcontext) -> bool:
    row = get(identifier)
    if row['state'] == 'ready':
        return False
    if row['kind'].startswith('pe_'):
        raise suites.SuiteError('The prompt-enhancer folder is incomplete; finish downloading its weights and supporting files')
    if row['state'] == 'unsupported':
        raise ValueError(row['error'])
    base = models.require_suite(base_suite)['image']
    tensors = {k: v for k, v in header(Path(row['path'])).items() if k != '__metadata__'}
    if row['kind'] == 'image':
        required = row['bytes'] if any('.img_mlp.gate_up.' in k for k in tensors) else 0
    else:
        required = sum(math.prod(info['shape']) * 2 for key, info in tensors.items()
                       if not key.endswith(('.weight_scale', '.comfy_quant')) and not key.startswith('lm_head.')
                       and key not in ('model.norm.weight', 'model.language_model.norm.weight'))
    if shutil.disk_usage(models.models_dir()).free < required + 1024**3:
        raise suites.SuiteError(f'Preparation needs about {required / 1e9:.1f} GB of additional disk space, plus 1 GB free')
    with suites._lock:
        if suites._stopping.is_set():
            raise suites.SuiteError('The app is shutting down')
        if any(j['thread'].is_alive() for j in suites._jobs.values()):
            raise suites.SuiteError('Another model preparation or conversion is running')
        job = {'phase': 'Preparing custom checkpoint', 'error': None, 'process': None}
        thread = threading.Thread(target=_prepare, args=(row, base, job, maintenance), daemon=True,
                                  name=f'prepare-{identifier}')
        job['thread'] = thread
        suites._jobs[identifier] = job
        thread.start()
    return True


def _prepare(row: dict, base: Path, job: dict, maintenance) -> None:
    destination = folder(row['id'])
    stage = destination.with_name(f'.{row["id"]}-{uuid.uuid4().hex}.tmp')
    original = fingerprint(Path(row['path']))
    tail = []
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with suites._deletion_guard(suites.suite_dir() / '.conversion.lock'), maintenance():
            if suites._stopping.is_set():
                raise suites.SuiteError('Preparation stopped')
            stage.mkdir()
            command = [sys.executable, '-m', 'imageatelier.convert_custom', '--source', row['path'],
                       '--kind', row['kind'], '--base', str(base), '--out', str(stage)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  bufsize=1, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}) as process:
                with suites._lock:
                    job['process'] = process
                    if suites._stopping.is_set():
                        process.terminate()
                for line in process.stdout:
                    tail = (tail + [line.rstrip()])[-12:]
                    if line.startswith('CUSTOM_PHASE '):
                        with suites._lock:
                            job['phase'] = line.removeprefix('CUSTOM_PHASE ').strip()
                if process.wait() != 0:
                    raise suites.SuiteError('Custom model preparation failed: ' + '\n'.join(tail))
            if original != fingerprint(Path(row['path'])):
                raise ValueError('The source checkpoint changed during preparation; retry')
            if not suites.weights_complete(stage / row['kind']):
                raise ValueError('Prepared component is incomplete')
            (stage / 'custom.json').write_text(json.dumps({'format': 1, 'kind': row['kind'], 'source': original}))
            if destination.exists():
                # Only this app's generated copy is replaced; imported source files are never changed.
                if destination.is_symlink():
                    raise ValueError('Refusing symbolic link destination')
                shutil.rmtree(destination)
            os.rename(stage, destination)
    except Exception as exc:
        with suites._lock:
            job['error'] = str(exc)
    finally:
        with suites._lock:
            job['process'] = None
        if stage.exists():
            shutil.rmtree(stage)


def model_ids(base_suite: str, image: str = '', encoder: str = '') -> tuple[str, str]:
    official = f'official-{models.SUITE_QUANTS[base_suite]}'
    return (f'{image}@{base_suite}' if image else official, f'{encoder}@{base_suite}' if encoder else official)


def resolve(image: str, encoder: str) -> Path:
    validate_pair(image, encoder)
    bases = {name.split('@', 1)[1] if name.startswith('custom-') and '@' in name else
             models.suite_for_model(name) for name in (image, encoder)}
    if len(bases) != 1 or not bases <= suites.VARIANTS.keys():
        raise ValueError('Custom components must use the same base suite')
    return compose(bases.pop(), image, encoder)


@suites._serialized
def delete(identifier: str, maintenance=contextlib.nullcontext) -> int:
    row = get(identifier)
    if suites.is_active():
        raise suites.SuiteError('Wait for model preparation or conversion to finish before deleting')
    source = Path(row['path'])
    prepared = folder(identifier)
    freed = 0
    with contextlib.ExitStack() as stack:
        stack.enter_context(suites._deletion_guard(suites.suite_dir() / '.conversion.lock'))
        # Hugging Face local-dir downloads use these locks; never delete an active download.
        if source.is_dir():
            for lock in sorted((source / '.cache/huggingface/download').rglob('*.lock')):
                stack.enter_context(suites._deletion_guard(lock))
        stack.enter_context(maintenance())
        for path in (prepared, source):
            if path.is_symlink():
                raise suites.SuiteError('Refusing to delete a symbolic link')
            if path.is_dir():
                freed += sum(f.stat().st_size for f in path.rglob('*') if f.is_file() and not f.is_symlink())
                shutil.rmtree(path)
            elif path.is_file():
                freed += path.stat().st_size
                path.unlink()
    return freed
