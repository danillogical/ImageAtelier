"""Whole Qwen suites: pinned official sources, offline readiness and atomic local conversions."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

VARIANTS = {"bf16": None, "q8": 8, "q6": 6, "q4": 4}
SOURCES = {
    "image": ("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3"),
    "t2i": ("Qwen/Qwen-Image-2.1-PE-T2I", "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"),
    "i2i": ("Qwen/Qwen-Image-2.1-PE-I2I", "72927bc08afc99b7888ceb7d7d51a12db3700bbd"),
}
PATTERNS = {
    "image": ["model_index.json", "scheduler/*.json", "processor/**", "transformer/*.json",
              "transformer/*.safetensors", "text_encoder/*.json", "text_encoder/*.safetensors",
              "vae/*.json", "vae/*.safetensors"],
    "t2i": ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
    "i2i": ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
}
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_stopping = threading.Event()


class SuiteError(RuntimeError):
    pass


def suite_dir() -> Path:
    from imageatelier.models import models_dir
    return models_dir() / "Qwen-Image-2.1"


def _check_variant(name: str) -> None:
    if name not in VARIANTS:
        raise ValueError(f"unknown suite {name!r}; expected one of {list(VARIANTS)}")


def weights_complete(folder: Path) -> bool:
    if not (folder / "config.json").is_file():
        return False
    indices = list(folder.glob("*.safetensors.index.json"))
    if indices:
        try:
            shards = set(json.loads(indices[0].read_text())["weight_map"].values())
            return bool(shards) and all(isinstance(s, str) and Path(s).name == s
                                       and (folder / s).is_file() and (folder / s).stat().st_size > 0 for s in shards)
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return any(p.is_file() and p.stat().st_size > 0 for p in folder.glob("*.safetensors"))


def image_complete(root: Path) -> bool:
    if not all(weights_complete(root / part) for part in ("transformer", "text_encoder", "vae")):
        return False
    if not all((root / "processor" / f).is_file() for f in ("tokenizer.json", "preprocessor_config.json")):
        return False
    # Reject text-only encoder exports: edits need the vision tower as well as the language model.
    try:
        index = json.loads((root / "text_encoder" / "model.safetensors.index.json").read_text())
        return any(k.startswith(("visual.", "model.visual.")) for k in index["weight_map"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def pe_complete(root: Path) -> bool:
    if not weights_complete(root) or not all((root / f).is_file() for f in
                                            ("tokenizer.json", "processor_config.json", "system_prompt.txt")):
        return False
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        return any(any(p in k for p in ("visual.", "vision_tower.", "vision_model.")) for k in index["weight_map"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def complete(parts: dict[str, Path]) -> bool:
    return image_complete(parts["image"]) and all(pe_complete(parts[m]) for m in ("t2i", "i2i"))


def source_parts() -> dict[str, Path] | None:
    from imageatelier.models import cache_dir
    try:
        parts = {kind: Path(snapshot_download(repo, revision=rev, allow_patterns=PATTERNS[kind],
                                              cache_dir=cache_dir(), local_files_only=True))
                 for kind, (repo, rev) in SOURCES.items()}
    except LocalEntryNotFoundError:
        return None
    return parts if complete(parts) else None


def converted_parts(root: Path) -> dict[str, Path]:
    return {p: root / p for p in SOURCES}


def parts(name: str) -> dict[str, Path] | None:
    _check_variant(name)
    if name == "bf16":
        return source_parts()
    root = suite_dir() / name
    try:
        manifest = json.loads((root / "suite.json").read_text())
        if manifest != {"format": 1, "variant": name, "sources": {k: list(v) for k, v in SOURCES.items()}}:
            return None
    except (OSError, ValueError):
        return None
    result = converted_parts(root)
    return result if complete(result) else None


@lru_cache(maxsize=1)
def physical_memory() -> int | None:
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2, text=True))
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _base(name: str) -> dict:
    ram = physical_memory()
    recommended = None
    if ram is not None:
        if ram >= 48 * 1024**3:
            recommended = "bf16"
        elif ram >= 32 * 1024**3:
            recommended = "q8"
    return {"kind": "suite", "name": name, "label": name.upper(), "recommended": name == recommended,
            "approx_gb": 70.8 if name == "bf16" else None}


def status(name: str, *, bf16_ready: bool | None = None) -> dict:
    from imageatelier import models
    _check_variant(name)
    base = _base(name)
    if name == "bf16":
        if parts(name):
            return {**base, "state": "ready", "error": None}
        return {**base, **models.download_status("suite:bf16")}
    with _lock:
        job = dict(_jobs.get(name, {}))
    if job.get("thread") and job["thread"].is_alive():
        return {**base, "state": "converting", "phase": job.get("phase", "Preparing conversion")}
    ready = parts(name) is not None
    if bf16_ready is None:
        bf16_ready = status("bf16")["state"] == "ready"
    return {**base, "state": "ready" if ready else "not_converted", "can_convert": bf16_ready,
            "error": job.get("error")}


def list_suites() -> list[dict]:
    bf16 = status("bf16")
    return [bf16] + [status(n, bf16_ready=bf16["state"] == "ready") for n in VARIANTS if n != "bf16"]


def is_active() -> bool:
    from imageatelier import models
    with _lock:
        converting = any(j["thread"].is_alive() for j in _jobs.values())
    return converting or models._active("suite:bf16")


def start_download() -> bool:
    from imageatelier import models
    if source_parts() is not None:
        return False
    plan = [(repo, rev, PATTERNS[kind], None) for kind, (repo, rev) in SOURCES.items()]
    return models._start("suite:bf16", plan, lambda: source_parts() is not None)


def restore_download() -> None:
    """Continue an interrupted download at launch, preserving an explicit pause."""
    from imageatelier import models
    intent = models._intent("suite:bf16")
    # Old versions did not save intent. Partial blobs are evidence of a started download.
    legacy_partial = intent is None and any(
        any((models._storage_folder(repo) / "blobs").glob("*.incomplete")) for repo, _ in SOURCES.values())
    if intent == "running" or legacy_partial:
        start_download()


def start_conversion(name: str, maintenance=contextlib.nullcontext) -> bool:
    _check_variant(name)
    if name == "bf16":
        raise SuiteError("BF16 is downloaded, not converted")
    if status("bf16")["state"] != "ready":
        raise SuiteError("Download the complete BF16 suite before converting")
    if parts(name) is not None:
        return False
    with _lock:
        if _stopping.is_set():
            raise SuiteError("The app is shutting down")
        if any(j["thread"].is_alive() for j in _jobs.values()):
            raise SuiteError("A suite conversion is already running")
        job = {"phase": "Preparing conversion", "error": None, "process": None}
        thread = threading.Thread(target=_convert, args=(name, job, maintenance), daemon=True,
                                  name=f"convert-{name}")
        job["thread"] = thread
        _jobs[name] = job
        thread.start()
    return True


def _convert(name: str, job: dict, maintenance) -> None:
    import fcntl
    stage = suite_dir() / f".{name}-{uuid.uuid4().hex}.tmp"
    tail: list[str] = []
    try:
        suite_dir().mkdir(parents=True, exist_ok=True)
        # Multiple local/LAN app processes may share this model directory.
        with (suite_dir() / ".conversion.lock").open("w") as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SuiteError("Another app is converting a suite") from None
            with maintenance():
                if _stopping.is_set():
                    raise SuiteError("Conversion stopped")
                source = source_parts()
                if source is None:
                    raise SuiteError("The BF16 suite is no longer complete")
                stage.mkdir()
                command = [sys.executable, "-m", "imageatelier.convert_suite", "--bits", str(VARIANTS[name]),
                           "--out", str(stage)]
                for kind, path in source.items():
                    command += [f"--{kind}", str(path)]
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      bufsize=1, env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}) as process:
                    with _lock:
                        job["process"] = process
                        if _stopping.is_set():
                            process.terminate()
                    for line in process.stdout:
                        tail = (tail + [line.rstrip()])[-20:]
                        if line.startswith("SUITE_PHASE "):
                            with _lock:
                                job["phase"] = line.removeprefix("SUITE_PHASE ").strip()
                    if process.wait() != 0:
                        raise SuiteError("Conversion failed: " + "\n".join(tail))
                if not complete(converted_parts(stage)):
                    raise SuiteError("Conversion finished but the suite is incomplete")
                (stage / "suite.json").write_text(json.dumps(
                    {"format": 1, "variant": name, "sources": {k: list(v) for k, v in SOURCES.items()}}))
                destination = suite_dir() / name
                if destination.exists():
                    # Only incomplete, app-owned outputs can be replaced; ready suites were handled above.
                    if parts(name) is not None:
                        raise SuiteError("This suite is already converted")
                    raise SuiteError(f"Incomplete output already exists at {destination}; move it aside before retrying")
                os.rename(stage, destination)
    except Exception as exc:
        with _lock:
            job["error"] = str(exc)
    finally:
        with _lock:
            job["process"] = None
        if stage.exists():
            shutil.rmtree(stage)


def stop() -> None:
    _stopping.set()
    with _lock:
        jobs = list(_jobs.values())
        for job in jobs:
            process = job.get("process")
            if process and process.poll() is None:
                process.terminate()
    for job in jobs:
        process = job.get("process")
        if process and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for job in jobs:
        job["thread"].join(timeout=15)


def start() -> None:
    _stopping.clear()
