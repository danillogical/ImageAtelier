"""Official Qwen suite registry and background download primitives. No MLX imports."""
from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

log = logging.getLogger(__name__)
CACHE_ENV = "IMAGEATELIER_HF_CACHE"
MODELS_DIR_ENV = "IMAGEATELIER_MODELS_DIR"


class ModelNotReady(RuntimeError):
    pass


def cache_dir() -> str | None:
    return os.environ.get(CACHE_ENV) or None


def models_dir() -> Path:
    return Path(os.environ.get(MODELS_DIR_ENV, str(Path(__file__).resolve().parent.parent / "models"))).expanduser()


def _storage_folder(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    return Path(cache_dir() or HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


STALL_SECONDS = 120.0
POLL_SECONDS = 1.0
RETRY_SECONDS = 5.0
_lock = threading.Lock()
_downloads: dict[str, "_Download"] = {}


class _Download:
    def __init__(self, key, plan, verify):
        self.key, self.plan, self.verify = key, plan, verify
        self.stop = threading.Event()
        self.thread = threading.Thread(target=_run_download, args=(self,), name=f"download-{key}", daemon=True)


def _download_root(key: str) -> Path:
    return models_dir() / ".downloads" / key.replace(":", "-")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


def _intent(key: str, value: str | None = None, *, only_running=False) -> str | None:
    root = _download_root(key)
    if value is None:
        return _read(root / "intent.json").get("state")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "control.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        old = _read(root / "intent.json").get("state")
        if not only_running or old == "running":
            _write(root / "intent.json", {"state": value})
        return old


def _bytes_on_disk(files) -> int:
    """Completed cached files plus our contiguous, resumable partial files."""
    done = 0
    for repo, revision, filename, size in files:
        cached = try_to_load_from_cache(repo, filename, revision=revision, cache_dir=cache_dir())
        if isinstance(cached, str):
            done += size
    for repo in {f[0] for f in files}:
        for partial in (_storage_folder(repo) / "blobs").glob("*.imageatelier.incomplete"):
            try:
                done += partial.stat().st_size
            except FileNotFoundError:
                pass
    return done


def download_status(key: str) -> dict:
    root = _download_root(key)
    intent = _intent(key)
    runtime = _read(root / "status.json")
    progress = _read(root / "progress.json")
    state = {"paused": "paused", "failed": "failed"}.get(intent, "not_downloaded")
    if intent == "running":
        state = "retrying" if runtime.get("state") == "retrying" else "downloading"
    return {"state": state, "bytes_done": _bytes_on_disk(progress.get("files", [])),
            "bytes_total": progress.get("total"), "phase": runtime.get("phase"),
            "error": runtime.get("error") if state == "failed" else None,
            "retry_in": max(0, math.ceil(runtime.get("retry_at", 0) - time.time())) if state == "retrying" else None}


def _active(key: str) -> bool:
    return _intent(key) == "running"


def pause_download(key: str) -> bool:
    with _lock:
        if _intent(key) not in ("running", "paused"):
            return False
        return _intent(key, "paused") != "paused"


def _start(key: str, plan, verify: Callable[[], bool], *, resume_only=False) -> bool:
    with _lock:
        if resume_only and _intent(key) != "running":
            return False
        old = _intent(key, "running")
        d = _downloads.get(key)
        if d and d.thread.is_alive() and d.thread is not threading.current_thread():
            return old != "running"
        d = _Download(key, plan, verify)
        _downloads[key] = d
        d.thread.start()
        return True


def _terminate(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def _run_download(d: _Download) -> None:
    root = _download_root(d.key)
    process = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "worker.lock").open("a") as guard:
            while True:
                try:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if d.stop.wait(POLL_SECONDS):
                        return
            _write(root / "plan.json", {"plan": d.plan, "cache": cache_dir()})
            failures = 0
            while not d.stop.is_set():
                if _intent(d.key) == "paused":
                    d.stop.wait(POLL_SECONDS)
                    continue
                if _intent(d.key) != "running":
                    return
                _write(root / "status.json", {"state": "downloading"})
                (root / "result.json").unlink(missing_ok=True)
                with (root / "worker.log").open("w") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "imageatelier.download_worker", str(root)],
                        stdout=output, stderr=subprocess.STDOUT,
                        # Retain ownership until even an orphaned worker has exited.
                        pass_fds=(guard.fileno(),),
                        env={**os.environ, "IMAGEATELIER_DOWNLOAD_PARENT": str(os.getpid())},
                    )
                    last_activity = time.monotonic()
                    signature = None
                    stalled = False
                    while process.poll() is None:
                        if d.stop.wait(POLL_SECONDS):
                            _terminate(process)
                            return
                        if _intent(d.key) != "running":
                            _terminate(process)
                            break
                        progress = _read(root / "progress.json")
                        current = (_bytes_on_disk(progress.get("files", [])), progress.get("activity"))
                        if current != signature:
                            signature, last_activity = current, time.monotonic()
                        elif time.monotonic() - last_activity >= STALL_SECONDS:
                            stalled = True
                            _terminate(process)
                            break
                if d.stop.is_set():
                    return
                if _intent(d.key) != "running":
                    continue
                result = _read(root / "result.json")
                if not stalled and process.returncode == 0 and d.verify():
                    _intent(d.key, "complete", only_running=True)
                    _write(root / "status.json", {"state": "ready"})
                    return
                reason = "No download progress for two minutes" if stalled else result.get("error", "Download interrupted")
                if result.get("retryable") is False and not stalled:
                    _write(root / "status.json", {"state": "failed", "error": reason})
                    _intent(d.key, "failed", only_running=True)
                    return
                failures += 1
                delay = min(60, RETRY_SECONDS * 2 ** min(failures - 1, 4))
                _write(root / "status.json", {"state": "retrying", "phase": reason,
                                              "retry_at": time.time() + delay})
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    if d.stop.wait(min(POLL_SECONDS, max(0, deadline - time.monotonic()))):
                        return
                    if _intent(d.key) != "running":
                        break
    except Exception as exc:
        log.exception("download supervisor failed")
        _write(root / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _intent(d.key, "failed", only_running=True)
    finally:
        if process is not None:
            _terminate(process)
        # A Resume request can race with the supervisor publishing a terminal state.
        if not d.stop.is_set() and _intent(d.key) == "running":
            _start(d.key, d.plan, d.verify, resume_only=True)


def stop_downloads() -> None:
    # Leave running intent intact for automatic continuation on the next launch.
    with _lock:
        jobs = list(_downloads.values())
        for d in jobs:
            d.stop.set()
    for d in jobs:
        d.thread.join(timeout=5)


# Internal part identifiers are derived from the suite; there are no independent part choices.
from imageatelier import suites

SUITE_QUANTS = {"bf16": "bf16", "q8": "8bit", "q6": "6bit", "q4": "4bit"}
QUANT_SUITES = {q: name for name, q in SUITE_QUANTS.items()}


@dataclass(frozen=True)
class ImageModel:
    name: str
    suite: str
    registry_key: str = "qwen-image-2.1"


IMAGE_MODELS = {f"official-{quant}": ImageModel(f"official-{quant}", name) for name, quant in SUITE_QUANTS.items()}
TEXT_ENCODERS = dict(IMAGE_MODELS)
DEFAULT_IMAGE_MODEL = "official-8bit"
DEFAULT_TEXT_ENCODER = DEFAULT_IMAGE_MODEL


@dataclass(frozen=True)
class PEModel:
    name: str
    mode: str
    quant: str
    suite: str


PE_MODELS = {f"{mode}-{quant}": PEModel(f"{mode}-{quant}", mode, quant, name)
             for name, quant in SUITE_QUANTS.items() for mode in ("t2i", "i2i")}


def suite_for_model(name: str) -> str:
    try:
        return IMAGE_MODELS[name].suite
    except KeyError:
        raise ValueError(f"unknown official model {name!r}") from None


def require_suite(name: str) -> dict[str, Path]:
    if suites.status(name)["state"] != "ready":
        raise ModelNotReady(f"The complete {name.upper()} Qwen-Image-2.1 suite is not ready; open Settings")
    result = suites.parts(name)
    if result is None:
        raise ModelNotReady(f"The {name.upper()} suite is incomplete")
    return result


def resolve_parts(image_model: str, text_encoder: str, turbo: bool):
    if turbo:
        raise ValueError("Turbo is no longer supported")
    if image_model != text_encoder:
        raise ValueError("Choose a complete suite; model parts cannot be mixed")
    parts = require_suite(suite_for_model(image_model))
    return parts["image"], parts["image"] / "text_encoder", None


def pe_name(mode: str, quant: str) -> str:
    name = f"{mode}-{quant}"
    if name not in PE_MODELS:
        raise ValueError(f"unknown prompt enhancer {name!r}")
    return name


def resolve_pe(mode: str, quant: str):
    pe = PE_MODELS[pe_name(mode, quant)]
    root = require_suite(pe.suite)[mode]
    return root, (root / "system_prompt.txt").read_text(encoding="utf-8")


def image_status(name: str) -> dict:
    return {**suites.status(suite_for_model(name)), "kind": "image", "name": name}


def te_status(name: str) -> dict:
    return {**suites.status(suite_for_model(name)), "kind": "text_encoder", "name": name}


def pe_status(name: str) -> dict:
    pe = PE_MODELS[name]
    return {**suites.status(pe.suite), "kind": "pe", "name": name}
