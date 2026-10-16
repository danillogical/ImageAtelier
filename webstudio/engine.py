"""Single-worker MLX engine: Qwen-Image-2.1-Heretic generation/edit plus Qwen prompt-enhancer rewrites.

All MLX work (model construction, generation, cache control) happens on one dedicated worker thread,
because MLX state such as the RNG seed and buffer cache is process-global. Exactly one model (an image
model or a prompt enhancer) is resident at a time; ModelManager swaps them on demand.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import gc
import json
import logging
import os
import random
import re
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from webstudio import models, serverlog

log = logging.getLogger(__name__)

TURBO_STEPS = (6, 8)
MIN_STEPS = 2
DEFAULT_STEPS = 40
DEFAULT_TURBO_STEPS = 6
DEFAULT_RESOLUTION = 1024
# low_memory expands to these settings; explicit "te_offload" / "vae_tiling" keys override it.
# Measured earlier at turbo 1024 (t2i + edit): te_offload cuts peak footprint 24.2 -> 19.5 GB for ~0.2 s/job,
# pixel-identical. vae_tiling cuts a further ~2.2 GB but adds ~3-5 s/job and changes pixels slightly, so it is opt-in.
LOW_MEMORY_PRESET = {"te_offload": True, "vae_tiling": False}
DEFAULT_LOW_MEMORY = True

# Prompt-enhancer defaults (each overridable by settings "pe_*" keys, then by the job itself).
PE_MODES = ("t2i", "i2i")
PE_QUANTS = ("8bit",)
PE_LANGUAGES = ("english", "auto")
DEFAULT_PE_QUANT = "8bit"
DEFAULT_PE_THINKING = False
# I2I thinking ran past 2560 tokens in the spike before answering, so the cap is generous.
DEFAULT_PE_MAX_NEW_TOKENS = 8192
DEFAULT_PE_LANGUAGE = "english"
# The I2I rewriter answers in Chinese for some English instructions; this line steers it back.
PE_ENGLISH_SUFFIX = "\n\nWrite the rewritten prompt in English."
PE_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
PE_PROGRESS_INTERVAL_S = 0.2  # at most 5 progress events per second
THINK_END = "</think>"
# Anything above this after unloading means something still references the old model's arrays.
UNLOAD_RESIDUE_WARN_BYTES = 500 * 1000**2

# Teardown (sync + cleanup) longer than this is logged with the job id.
TEARDOWN_WARN_S = 10.0
# GPU utilization at or above this with no job running is logged before the next job starts.
GPU_BUSY_WARN_PCT = 50
GPU_BUSY_RESAMPLES = 4  # re-reads while high, GPU_BUSY_RESAMPLE_S apart (~2 s worst case)
GPU_BUSY_RESAMPLE_S = 0.5

# Extra reference images on generate jobs. "auto" picks reference_resolution by the number of extra refs (measured on
# the M5 at 1024 output: the decode spike stays the peak through 3 refs at 768; 3 at 1024 pushes the footprint +2 GB
# and doubles the time; 512 visibly degrades the current image in edits, so auto never uses it).
REF_DETAILS = ("auto", "high", "standard", "low")
REF_DETAIL_RESOLUTION = {"high": 1024, "standard": 768, "low": 512}
AUTO_REF_RESOLUTION = ((1, 1024), (3, 768))  # (max extra refs, reference_resolution); 0 extra refs: unchanged
DEFAULT_REF_DETAIL = "auto"
DEFAULT_MAX_REFS = 3  # the turbo LoRA was trained on at most three references
MAX_REFS_CAP = 9
MAX_IMAGES = 10  # the fork accepts at most ten conditioning images (current + refs)

GENERATE, ENHANCE = "generate", "enhance"
QUEUED, LOADING, RUNNING, DONE, FAILED, CANCELLED = "queued", "loading", "running", "done", "failed", "cancelled"
# A running job whose cancel was requested; it becomes CANCELLED only after its teardown completes.
CANCELLING = "cancelling"
# A running job blocked at a safe boundary (GPU synchronized, model resident) until resume() or cancel().
PAUSED = "paused"
# A queued job skipped by the dequeue until release(); it keeps its queue position.
HELD = "held"
FINAL_STATES = (DONE, FAILED, CANCELLED)
JOB_LOG_CAP = 500  # per-job log lines kept; older ones are dropped behind a marker line
ENHANCE_LOG_EVERY_TOKENS = 50
PAUSE_POLL_S = 0.5  # how often a paused job re-checks for cancel/stop while waiting


def gpu_utilization() -> int | None:
    """IOAccelerator "Device Utilization %" (whole GPU, all processes), or None if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
    return int(m.group(1)) if m else None


def _synchronize_all() -> None:
    """Block until every MLX stream this (worker) thread uses has finished its queued GPU/CPU work."""
    import sys

    import mlx.core as mx

    mx.synchronize()  # this thread's default GPU stream: mflux image generation
    mx.synchronize(mx.cpu)
    common = sys.modules.get("mlx_vlm.generate.common")
    if common is not None:
        # mlx-vlm decodes on its own thread-local stream (mlx_vlm/generate/common.py:35).
        mx.synchronize(common.generation_stream)


class JobCancelled(Exception):
    """Raised from the progress callback to abort a running generation."""


class InvalidTransition(Exception):
    """The requested control action is not valid in the job's current state (the API maps this to 409)."""


class UnknownJob(KeyError):
    """No job with that id (the API maps this to 404)."""


class EngineBusy(RuntimeError):
    """maintenance() refused: a job is not finished, or another maintenance block is active (the API maps this to 409)."""


@dataclass
class _Job:
    id: str
    prompt: str
    mode: str
    ref_image_path: str | None
    kind: str = GENERATE
    # generate
    seed: int | None = None
    steps: int | None = None
    turbo: bool = False
    image_model: str = models.DEFAULT_IMAGE_MODEL
    text_encoder: str = models.DEFAULT_TEXT_ENCODER
    output_path: str = ""
    transparent: bool = False  # keep a cut-out background (alpha < 16 -> 0); otherwise the saved image is fully opaque
    ref_image_paths: list = field(default_factory=list)  # extra references, in order (Image 2.. for edits)
    ref_detail: str = DEFAULT_REF_DETAIL
    reference_resolution: int | None = None  # resolved when the job runs
    step: int = 0
    total_steps: int = 0
    # enhance (None = resolve from settings when the job runs)
    quant: str | None = None
    thinking: bool | None = None
    max_new_tokens: int | None = None
    language: str | None = None
    phase: str | None = None
    rewriter: str | None = None  # which rewriter model runs this job, e.g. "official-8bit" | "heretic-8bit"
    tokens: int = 0
    text: str = ""
    # common
    state: str = QUEUED
    error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False
    # pause / log bookkeeping (not all of it is in the snapshot)
    pause_requested: bool = False
    paused_s: float = 0.0
    resume_event: threading.Event = field(default_factory=threading.Event, repr=False)
    log: deque = field(default_factory=lambda: deque(maxlen=JOB_LOG_CAP), repr=False)
    log_dropped: int = 0
    last_step_t: float = 0.0

    def __post_init__(self) -> None:
        self.resume_event.set()  # set = not paused

    def snapshot(self) -> dict:
        snap = {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "mode": self.mode,
            "ref_image_path": self.ref_image_path,
        }
        if self.kind == ENHANCE:
            snap.update(
                quant=self.quant,
                thinking=self.thinking,
                max_new_tokens=self.max_new_tokens,
                language=self.language,
                phase=self.phase,
                rewriter=self.rewriter,
                tokens=self.tokens,
                text=self.text,
            )
        else:
            snap.update(
                seed=self.seed,
                steps=self.steps,
                turbo=self.turbo,
                transparent=self.transparent,
                image_model=self.image_model,
                text_encoder=self.text_encoder,
                output_path=self.output_path,
                refs=len(self.ref_image_paths),
                ref_image_paths=list(self.ref_image_paths),
                ref_detail=self.ref_detail,
                reference_resolution=self.reference_resolution,
                step=self.step,
                total_steps=self.total_steps,
            )
        snap.update(
            state=self.state,
            pause_requested=self.pause_requested,
            paused_s=round(self.paused_s, 3),
            error=self.error,
            result=dict(self.result) if self.result else None,
            created_at=self.created_at,
        )
        return snap


ALPHA_CLEAR_BELOW = 16  # transparent images: background alpha measured at 1..15 on MLX, so treat < 16 as fully clear
ALPHA_SOLID_FROM = 240  # ... and the subject sits at 240..254, so treat >= 240 as fully solid (16..239 stays: soft edges)


def finalize_alpha(image, transparent: bool):
    """The saved image is always RGBA. Normal images come out of the decoder with alpha 252..255; force them fully opaque.
    Transparent images clamp the near-zero background (< ALPHA_CLEAR_BELOW) to 0 and the near-solid subject (>= ALPHA_SOLID_FROM) to 255."""
    from PIL import Image

    rgba = image.convert("RGBA")
    if transparent:
        alpha = rgba.getchannel("A").point(lambda v: 0 if v < ALPHA_CLEAR_BELOW else 255 if v >= ALPHA_SOLID_FROM else v)
    else:
        alpha = Image.new("L", rgba.size, 255)
    rgba.putalpha(alpha)
    return rgba


def validate_steps(steps: int, turbo: bool) -> None:
    # The fork's terminal sigma shift needs at least two steps (QwenImage21LatentCreator.validate).
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < MIN_STEPS:
        raise ValueError(f"steps must be an integer >= {MIN_STEPS}, got {steps!r}")
    if turbo and steps not in TURBO_STEPS:
        raise ValueError(f"turbo supports only {TURBO_STEPS[0]} or {TURBO_STEPS[1]} steps, got {steps}")


def resolve_reference_resolution(ref_detail: str, extra_refs: int, resolution: int) -> int:
    """The reference_resolution for a generate job.

    auto: no extra refs keeps the previous behaviour (references sized like the output); 1 extra -> 1024; 2-3 -> 768
    (more than 3 also 768). high/standard/low: 1024/768/512 regardless of count.
    """
    if ref_detail != "auto":
        return REF_DETAIL_RESOLUTION[ref_detail]
    if extra_refs == 0:
        return resolution
    for max_refs, res in AUTO_REF_RESOLUTION:
        if extra_refs <= max_refs:
            return res
    return AUTO_REF_RESOLUTION[-1][1]


def _partial_rewrite(answer: str) -> str:
    """Best-effort live value of "rewritten_prompt" from a possibly incomplete JSON answer."""
    m = re.search(r'"rewritten_prompt"\s*:\s*"', answer)
    if not m:
        return ""
    out, i, s = [], m.end(), answer
    while i < len(s):
        c = s[i]
        if c == "\\":
            if i + 1 >= len(s):
                break
            if s[i + 1] == "u":
                if i + 6 > len(s):
                    break
                out.append(s[i : i + 6])
                i += 6
                continue
            out.append(s[i : i + 2])
            i += 2
            continue
        if c == '"':
            break
        out.append(c)
        i += 1
    try:
        return json.loads('"' + "".join(out) + '"')
    except ValueError:
        return "".join(out)


def _parse_rewrite(answer: str) -> str | None:
    """The final "rewritten_prompt" from the answer JSON (optionally ```-fenced), or None if unparseable."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    try:
        value = json.loads(text)["rewritten_prompt"]
    except (ValueError, TypeError, KeyError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


class _Progress:
    """mflux callback: phase memory stats, text-encoder offload, step progress, and cancellation."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def call_before_loop(self, **kwargs) -> None:
        # Fires after the prompt (and any reference image) is encoded, right before denoising.
        self._engine._end_phase("encode")
        self._engine._maybe_offload_text_encoder()

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        self._engine._on_step(t + 1 - config.init_time_step, config.num_inference_steps - config.init_time_step)

    def call_after_loop(self, **kwargs) -> None:
        self._engine._end_phase("denoise")


class ModelManager:
    """Holds exactly one resident model. Worker-thread only.

    Keys: ("image", image_model, text_encoder, turbo) | ("pe_t2i", quant) | ("pe_i2i", quant). Any component change
    is a different key, so it reloads; the same key never does.
    Attributes for the resident model: model; snapshot (image: the composed folder, which the per-job text-encoder
    reload also reads); parts (image: where each composed part comes from); processor and system_prompt (enhancers).
    """

    def __init__(self, emit: Callable[[dict], None], image_callback: object, compose_dir: Path | None = None):
        self._emit = emit
        self._sync = _synchronize_all
        self._image_callback = image_callback
        self._compose_dir = compose_dir
        self.key: tuple | None = None
        self.model = None
        self.snapshot: Path | None = None
        self.parts: dict | None = None
        self.processor = None
        self.system_prompt: str | None = None

    def ensure(self, key: tuple) -> float:
        """Make `key` the resident model; returns load seconds (0.0 if it was already resident).

        Raises models.ModelNotReady (current model kept) if the weights aren't downloaded, or the load error
        (nothing left resident) if construction fails.
        """
        if key == self.key:
            return 0.0
        # Resolve (and compose) before unloading so a not-ready request leaves the current model in place.
        if key[0] == "image":
            _, image_model, text_encoder, turbo = key
            root, te_dir, lora = models.resolve_parts(image_model, text_encoder, turbo)
            composed = models.compose(image_model, text_encoder, root, te_dir, self._compose_dir)
            source = (composed, lora)
            parts = {"composed": str(composed), "image": str(root), "text_encoder": str(te_dir)}
        elif key[0] in ("pe_t2i", "pe_i2i"):
            source = models.resolve_pe(key[0][3:], key[1])
        else:
            raise ValueError(f"unknown model key {key!r}")
        self.drop()
        self._emit({"type": "model", "state": "loading", "key": list(key), "load_s": None})
        t0 = time.perf_counter()
        try:
            if key[0] == "image":
                self._load_image(key, *source)
                self.parts = parts
            else:
                self._load_pe(key, *source)
        except Exception as exc:
            try:
                self._sync()
            except Exception:  # noqa: BLE001 - report the load error, not a secondary sync error
                log.exception("synchronize after failed load")
            self._clear()
            self._emit({"type": "model", "state": "unloaded", "key": list(key), "load_s": None,
                        "error": f"load failed: {type(exc).__name__}: {exc}"})  # fmt: skip
            raise
        load_s = time.perf_counter() - t0
        import mlx.core as mx

        self._emit({"type": "model", "state": "loaded", "key": list(key), "load_s": round(load_s, 3),
                    "active_bytes": mx.get_active_memory()})  # fmt: skip
        return load_s

    def drop(self) -> None:
        """Unload the resident model (if any) and report the MLX memory left behind."""
        if self.key is None and self.model is None:
            return
        import mlx.core as mx

        old = self.key
        # No GPU work may still be reading the old weights when they are released.
        self._sync()
        self._clear()
        active = mx.get_active_memory()
        if active > UNLOAD_RESIDUE_WARN_BYTES:
            gc.collect()
            mx.clear_cache()
            active = mx.get_active_memory()
            if active > UNLOAD_RESIDUE_WARN_BYTES:
                log.warning("%.2f GB still active after unloading %s", active / 1e9, old)
        self._emit({"type": "model", "state": "unloaded", "key": list(old) if old else None, "load_s": None,
                    "active_bytes": active})  # fmt: skip

    def _clear(self) -> None:
        import mlx.core as mx

        self.key = self.model = self.snapshot = self.processor = self.system_prompt = self.parts = None
        gc.collect()
        mx.clear_cache()

    def _load_image(self, key: tuple, snapshot: Path, lora: Path | None) -> None:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.models.qwen21.reference import QwenImage21Edit

        _, image_model, _text_encoder, turbo = key
        config = copy.copy(AVAILABLE_MODELS[models.IMAGE_MODELS[image_model].registry_key])
        model = QwenImage21Edit(
            model_path=str(snapshot),
            model_config=config,
            # An explicit local LoRA path keeps the preset from calling hf_hub_download at load time.
            lora_paths=[str(lora)] if turbo else None,
            lora_preset="viggle-turbo" if turbo else None,
        )
        model.callbacks.register(self._image_callback)
        self.model, self.snapshot, self.key = model, snapshot, key

    def _load_pe(self, key: tuple, path: Path, system_prompt: str) -> None:
        from mlx_vlm import load

        model, processor = load(str(path))
        self.model, self.processor, self.system_prompt, self.key = model, processor, system_prompt, key


class Engine:
    """FIFO engine with one worker thread for image generation/edit and prompt-enhancer jobs.

    settings_provider: zero-arg callable returning a Mapping (or None), read at the start of each job.
      Recognised keys, all optional: "resolution" (output long-side basis, default 1024),
      "default_steps" (non-turbo, default 40), "default_turbo_steps" (default 6),
      "low_memory" (default True: free the text encoder after each encode, reload it lazily next job),
      "te_offload" / "vae_tiling" (explicit overrides of what low_memory implies; vae_tiling default False),
      "pe_quant" ("8bit"), "pe_thinking" (False), "pe_max_new_tokens" (8192), "pe_language" ("english"|"auto").
    cache_gb: MLX buffer-cache limit; keeps peak memory near the working set instead of growing unbounded.
    server_log_access: also capture uvicorn.access lines in the server-log ring (default off).
    compose_dir: where composed model folders are built (default models.default_compose_dir(): <data>/compose).
    Settings "image_model" / "text_encoder" are the defaults for generate jobs that don't name their parts.
    """

    def __init__(
        self,
        settings_provider: Callable[[], Mapping | None] | None = None,
        cache_gb: float = 2,
        server_log_access: bool = False,
        compose_dir: Path | None = None,
    ):
        self._settings_provider = settings_provider or (lambda: None)
        self._cache_gb = cache_gb
        self._server_log_access = server_log_access
        self._ring: serverlog.RingLogHandler | None = None
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[str] = deque()
        self._current: _Job | None = None
        self._listeners: list[Callable[[dict], None]] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._resident: dict = {"key": None, "state": "unloaded"}
        # maintenance(): while set the worker dequeues nothing; _drop_requested asks it to unload, acked on _idle.
        self._maintenance = False
        self._drop_requested = False
        self._drop_error: str | None = None
        self._idle = threading.Condition(self._lock)
        self._worker_running = False  # true from start() until the worker thread's final cleanup, under the lock
        # Worker-thread-only state.
        # Composed model folders (<compose_dir>/<image>__<text_encoder>/); default models.default_compose_dir().
        self._mm = ModelManager(self._emit_model, _Progress(self), compose_dir)
        self._offload_te = False
        self._phases: dict[str, dict[str, int]] = {}

    # ---- public API (any thread) -------------------------------------------------------------

    def start(self) -> None:
        # Install after the server configured logging (uvicorn's dictConfig would otherwise drop the handler).
        self._ring = serverlog.install(include_access=self._server_log_access)
        self._ring.add_sink(self._on_server_log)
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stopping = False
            self._worker_running = True
            self._thread = threading.Thread(target=self._worker, name="webstudio-engine", daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Cancel queued and running jobs, release the model, and join the worker thread."""
        with self._lock:
            self._stopping = True
            queued = [self._jobs[i] for i in self._queue]
            self._queue.clear()
            current = self._current
            if current and current.state not in FINAL_STATES:
                current.cancel_requested = True
                current.resume_event.set()  # a paused job must wake up to be cancelled
            self._wake.notify_all()
        for job in queued:
            self._jlog(job, "cancelled: engine stopping")
            self._set_state(job, CANCELLED)
        if current and current.cancel_requested and current.state not in FINAL_STATES:
            self._jlog(current, "cancel requested: engine stopping")
            self._set_state(current, CANCELLING)
        if self._thread:
            self._thread.join(timeout)
        if self._ring is not None:
            self._ring.remove_sink(self._on_server_log)

    @contextlib.contextmanager
    def maintenance(self) -> Iterator[None]:
        """Context manager (any non-worker thread): inside it the worker is idle and no model is resident.

        Raises EngineBusy if any job is not in a final state (queued, held, loading, running, paused, cancelling) or
        another maintenance block is active (concurrent callers are refused, not queued). On entry the resident model
        is unloaded on the worker thread (which emits the usual "unloaded" model event) and this call blocks until
        the worker acknowledges; with no worker running (never started, or stopped: its exit already unloaded) the
        drop is skipped. Jobs submitted inside the block are queued but not started until it exits.
        """
        with self._lock:
            if self._maintenance:
                raise EngineBusy("another model operation is in progress")
            if any(j.state not in FINAL_STATES for j in self._jobs.values()):
                raise EngineBusy("finish or cancel the queued and running jobs first")
            self._maintenance = True  # set under the same lock as the check, so no job can start in between
        try:
            with self._lock:
                if self._worker_running:
                    self._drop_requested, self._drop_error = True, None
                    self._wake.notify_all()
                    # The worker clears the request after its drop, and also on exit, so this cannot outlive it.
                    while self._drop_requested:
                        self._idle.wait()
                    if self._drop_error:
                        raise RuntimeError(f"could not unload the resident model: {self._drop_error}")
            yield
        finally:
            with self._lock:
                self._maintenance = False
                self._wake.notify_all()

    def on_event(self, fn: Callable[[dict], None]) -> None:
        """Register a listener called on every job state/progress change and every model swap.

        Job events are the job snapshot plus "type": "state" | "progress". Job states: queued, loading, running,
        cancelling (cancel requested on a running job), then exactly one terminal state: done, failed or cancelled,
        emitted only after the job's GPU work has finished and its cleanup has run. Model events are
        {"type": "model", "state": "loading"|"loaded"|"unloaded", "key": [...]|None, "load_s": float|None}
        (unloads add "active_bytes"; a failed load is reported as "unloaded" with "error").
        Job states also include paused (blocked at a safe boundary) and held (queued but skipped by the
        dequeue); snapshots carry pause_requested, paused_s and queue_position (1-based among queued+held jobs,
        None otherwise). Log events: {"type": "log", "job_id", "line"} for every per-job log line, and
        {"type": "serverlog", "line"} for server-log lines (batched; at most 50 per 0.5 s, bursts summarised).
        Listeners run synchronously on the worker thread (or on the caller's thread for control calls, or the
        server-log flusher thread for serverlog events). They must return quickly and never block, e.g. by putting
        the event on a queue or calling loop.call_soon_threadsafe; a slow listener stalls generation. Exceptions
        are logged and ignored.
        """
        with self._lock:
            self._listeners.append(fn)

    def resident(self) -> dict:
        """The last model event's state and key, e.g. {"key": ["pe_t2i", "8bit"], "state": "loaded"}."""
        with self._lock:
            return dict(self._resident)

    def submit(self, job: Mapping[str, Any]) -> str:
        """Queue a job and return its id. Raises ValueError for malformed jobs.

        kind "generate" (default): prompt (str), mode ("t2i"|"edit"), ref_image_path (str, required for edit),
          seed (int|None -> random), steps (int|None -> default), turbo (bool),
          image_model (models.IMAGE_MODELS key) and text_encoder (models.TEXT_ENCODERS key), each defaulting to the
          settings value at submit time, output_path (str, PNG). The old "variant" key is rejected.
        kind "enhance": prompt (str), mode ("t2i"|"i2i"), ref_image_path (str, required for i2i),
          quant ("8bit"), thinking (bool), max_new_tokens (int), language ("english"|"auto");
          omitted options come from settings at run time.
        """
        kind = job.get("kind", GENERATE)
        if kind == GENERATE:
            j = self._make_generate(job)
        elif kind == ENHANCE:
            j = self._make_enhance(job)
        else:
            raise ValueError(f"kind must be {GENERATE!r} or {ENHANCE!r}, got {kind!r}")
        with self._lock:
            if self._stopping:
                raise RuntimeError("engine is stopping")
            self._jobs[j.id] = j
            self._queue.append(j.id)
            position = len(self._queue)
            self._wake.notify()
        self._jlog(j, f"submitted: kind={j.kind} mode={j.mode}; queued at position {position}")
        self._emit(j, "state")
        return j.id

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Queued or held: removed and CANCELLED immediately.

        Running or paused: a request. The job shows CANCELLING (a paused job is released from its wait), stops at
        its next denoising step / token, and becomes CANCELLED only after its teardown (GPU sync and cleanup) has
        finished. Returns False if the job is unknown or already finished.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in FINAL_STATES:
                return False
            if job_id in self._queue:
                was = job.state
                self._queue.remove(job_id)
                queued = True
            else:
                if job.cancel_requested:
                    return True
                was = job.state
                job.cancel_requested = True
                job.resume_event.set()
                queued = False
        if queued:
            self._jlog(job, f"cancelled while {was} (removed from the queue)")
            self._set_state(job, CANCELLED)
            self._emit_positions()
        else:
            self._jlog(job, f"cancel requested while {was}")
            self._set_state(job, CANCELLING)
        return True

    def pause(self, job_id: str) -> bool:
        """Ask a running (or loading) job to pause at its next safe boundary.

        Generate jobs pause after a denoising step (or right after model load / text-encoder reload); enhance jobs
        between tokens (or after model load). Model load, the prompt/reference encoding before step 1, mlx-vlm's
        prompt prefill, VAE decode and saving cannot pause: a request made then takes effect at the next boundary,
        and is dropped if the job finishes first (after the last step there is none). The state becomes PAUSED only
        once the job is actually blocked with its GPU work synchronized; until then pause_requested is true.
        Returns False if a pause is already requested or in effect. Raises InvalidTransition otherwise.
        """
        with self._lock:
            job = self._job(job_id)
            if job.state == PAUSED or job.pause_requested:
                return False
            if job.state not in (LOADING, RUNNING) or job.cancel_requested:
                raise InvalidTransition(f"cannot pause a {job.state} job" + (" (use hold)" if job.state == QUEUED else ""))
            job.pause_requested = True
            job.resume_event.clear()
        self._jlog(job, f"pause requested while {job.state}")
        self._emit(job, "state")
        return True

    def resume(self, job_id: str) -> bool:
        """Resume a paused job (or withdraw a pause request that has not taken effect yet).

        Raises InvalidTransition if the job is neither paused nor pause-requested.
        """
        with self._lock:
            job = self._job(job_id)
            if job.state != PAUSED and not job.pause_requested:
                raise InvalidTransition(f"cannot resume a {job.state} job that is not paused")
            paused = job.state == PAUSED
            job.pause_requested = False
            job.resume_event.set()
        self._jlog(job, "resume requested" if paused else "pause request withdrawn before it took effect")
        if not paused:
            self._emit(job, "state")
        return True

    def hold(self, job_id: str) -> bool:
        """Hold a queued job: the dequeue skips it until release(); it keeps its queue position.

        Returns False if it is already held. Raises InvalidTransition for any non-queued job.
        """
        with self._lock:
            job = self._job(job_id)
            if job.state == HELD:
                return False
            if job.state != QUEUED or job_id not in self._queue:
                raise InvalidTransition(f"cannot hold a {job.state} job" + (" (use pause)" if job.state in (LOADING, RUNNING) else ""))
            job.state = HELD
        self._jlog(job, "held")
        self._emit(job, "state")
        return True

    def release(self, job_id: str) -> bool:
        """Release a held job back to queued at its original position.

        Returns False if it is already queued. Raises InvalidTransition for any job that is neither held nor queued.
        """
        with self._lock:
            job = self._job(job_id)
            if job.state == QUEUED:
                return False
            if job.state != HELD:
                raise InvalidTransition(f"cannot release a {job.state} job")
            job.state = QUEUED
            position = self._position(job_id)
            self._wake.notify()
        self._jlog(job, f"released; queue position {position}")
        self._emit(job, "state")
        return True

    def job_log(self, job_id: str) -> list[str]:
        """The job's log lines, oldest first (a marker line replaces any dropped beyond the cap)."""
        with self._lock:
            job = self._job(job_id)
            lines = list(job.log)
            dropped = job.log_dropped
        return ([f"... {dropped} earlier log lines dropped ..."] if dropped else []) + lines

    def server_log(self, tail: int = 200) -> list[str]:
        """The last `tail` server-log lines (all kept lines if tail <= 0); empty before start()."""
        ring = self._ring or serverlog.handler()
        return ring.tail(tail) if ring else []

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snap(job) if job else None

    def jobs(self) -> list[dict]:
        with self._lock:
            return [self._snap(j) for j in self._jobs.values()]

    # ---- helpers (callers hold self._lock where noted) ---------------------------------------------

    def _job(self, job_id: str) -> _Job:
        """Caller holds the lock."""
        job = self._jobs.get(job_id)
        if job is None:
            raise UnknownJob(job_id)
        return job

    def _position(self, job_id: str) -> int | None:
        """1-based position among queued+held jobs, or None. Caller holds the lock."""
        try:
            return self._queue.index(job_id) + 1
        except ValueError:
            return None

    def _snap(self, job: _Job) -> dict:
        """Caller holds the lock."""
        snap = job.snapshot()
        snap["queue_position"] = self._position(job.id)
        return snap

    def _emit_positions(self) -> None:
        """Re-emit queued/held jobs after the queue shifted, so clients see their new queue_position."""
        with self._lock:
            waiting = [self._jobs[i] for i in self._queue]
        for job in waiting:
            self._emit(job, "state")

    def _jlog(self, job: _Job, msg: str) -> None:
        now = time.time()
        stamp = datetime.datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        line = f"{stamp} +{now - job.created_at:.2f}s {msg}"
        with self._lock:
            if len(job.log) == job.log.maxlen:
                job.log_dropped += 1
            job.log.append(line)
        self._dispatch({"type": "log", "job_id": job.id, "line": line})

    def _on_server_log(self, lines: list[str]) -> None:
        for line in lines:
            self._dispatch({"type": "serverlog", "line": line}, log_errors=False)

    # ---- job construction ------------------------------------------------------------------------

    @staticmethod
    def _prompt(job: Mapping[str, Any]) -> str:
        prompt = job.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return prompt

    def _make_generate(self, job: Mapping[str, Any]) -> _Job:
        mode = job.get("mode", "t2i")
        if mode not in ("t2i", "edit"):
            raise ValueError(f"mode must be 't2i' or 'edit', got {mode!r}")
        prompt = self._prompt(job)
        ref = job.get("ref_image_path")
        if mode == "edit" and not ref:
            raise ValueError("edit mode requires ref_image_path")
        if "variant" in job:
            raise ValueError("variant was replaced by image_model/text_encoder")
        defaults = dict(self._settings_provider() or {})
        # Resolved now, so a queued job keeps its parts even if the settings change before it runs.
        image_model = job.get("image_model") or defaults.get("image_model") or models.DEFAULT_IMAGE_MODEL
        text_encoder = job.get("text_encoder") or defaults.get("text_encoder") or models.DEFAULT_TEXT_ENCODER
        if image_model not in models.IMAGE_MODELS:
            raise ValueError(f"unknown image_model {image_model!r}; expected one of {sorted(models.IMAGE_MODELS)}")
        if text_encoder not in models.TEXT_ENCODERS:
            raise ValueError(f"unknown text_encoder {text_encoder!r}; expected one of {sorted(models.TEXT_ENCODERS)}")
        if not job.get("output_path"):
            raise ValueError("output_path is required")
        seed = job.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")
        turbo = bool(job.get("turbo", False))
        steps = job.get("steps")
        if steps is not None:
            validate_steps(steps, turbo)
        refs = job.get("ref_image_paths") or []
        if not isinstance(refs, (list, tuple)) or not all(isinstance(r, str) and r for r in refs):
            raise ValueError("ref_image_paths must be a list of file paths")
        max_refs = defaults.get("max_refs", DEFAULT_MAX_REFS)
        max_refs = max(0, min(MAX_REFS_CAP, max_refs if isinstance(max_refs, int) and not isinstance(max_refs, bool) else DEFAULT_MAX_REFS))
        if len(refs) > max_refs:
            raise ValueError(f"max {max_refs} references enabled (got {len(refs)})")
        total = len(refs) + (1 if mode == "edit" else 0)
        if total > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per generation (current image + references); got {total}")
        missing = [r for r in refs if not Path(r).is_file()]
        if missing:
            raise ValueError(f"reference image not found: {missing[0]}")
        ref_detail = job.get("ref_detail") or defaults.get("ref_detail") or DEFAULT_REF_DETAIL
        if ref_detail not in REF_DETAILS:
            raise ValueError(f"ref_detail must be one of {list(REF_DETAILS)}, got {ref_detail!r}")
        return _Job(
            id=uuid.uuid4().hex[:12],
            prompt=prompt,
            mode=mode,
            ref_image_path=str(ref) if mode == "edit" else None,
            ref_image_paths=[str(r) for r in refs],
            ref_detail=ref_detail,
            seed=seed,
            steps=steps,
            turbo=turbo,
            image_model=image_model,
            text_encoder=text_encoder,
            output_path=str(job["output_path"]),
            transparent=bool(job.get("transparent", False)),
        )

    def _make_enhance(self, job: Mapping[str, Any]) -> _Job:
        mode = job.get("mode", "t2i")
        if mode not in PE_MODES:
            raise ValueError(f"enhance mode must be 't2i' or 'i2i', got {mode!r}")
        prompt = self._prompt(job)
        ref = job.get("ref_image_path")
        if mode == "i2i" and not ref:
            raise ValueError("i2i enhance requires ref_image_path")
        quant, thinking = job.get("quant"), job.get("thinking")
        max_new_tokens, language = job.get("max_new_tokens"), job.get("language")
        if quant is not None and quant not in PE_QUANTS:
            raise ValueError(f"quant must be one of {PE_QUANTS}, got {quant!r}")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError(f"thinking must be a bool, got {thinking!r}")
        if max_new_tokens is not None:
            _validate_max_new_tokens(max_new_tokens)
        if language is not None and language not in PE_LANGUAGES:
            raise ValueError(f"language must be one of {PE_LANGUAGES}, got {language!r}")
        return _Job(
            id=uuid.uuid4().hex[:12],
            kind=ENHANCE,
            prompt=prompt,
            mode=mode,
            ref_image_path=str(ref) if mode == "i2i" else None,
            quant=quant,
            thinking=thinking,
            max_new_tokens=max_new_tokens,
            language=language,
        )

    # ---- events --------------------------------------------------------------------------------

    def _set_state(self, job: _Job, state: str, **fields) -> None:
        with self._lock:
            if job.cancel_requested and state in (LOADING, RUNNING, PAUSED):
                state = CANCELLING  # a pending cancel is never hidden by a later transition
            job.state = state
            for k, v in fields.items():
                setattr(job, k, v)
        self._emit(job, "state")

    def _emit(self, job: _Job, kind: str) -> None:
        with self._lock:
            event = self._snap(job)
        event["type"] = kind
        self._dispatch(event)

    def _emit_model(self, event: dict) -> None:
        with self._lock:
            self._resident = {"key": event["key"] if event["state"] != "unloaded" else None, "state": event["state"]}
            job = self._current
        if job is not None:
            active = f", active {event['active_bytes'] / 1e9:.2f} GB" if "active_bytes" in event else ""
            took = f" in {event['load_s']:.2f} s" if event.get("load_s") is not None else ""
            err = f" ({event['error']})" if event.get("error") else ""
            self._jlog(job, f"model {event['state']} {event['key']}{took}{active}{err}")
        self._dispatch(event)

    def _dispatch(self, event: dict, log_errors: bool = True) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken listener must not take down the worker
                if log_errors:
                    log.exception("event listener failed")

    def _on_step(self, step: int, total: int) -> None:
        """mflux in-loop callback, after mx.eval of the step's latents: progress, log, pause point, cancel."""
        job = self._current
        if job is None:
            return
        now = time.perf_counter()
        with self._lock:
            job.step, job.total_steps = step, total
            dt = now - job.last_step_t if job.last_step_t else 0.0
            job.last_step_t = now
        self._jlog(job, f"step {step}/{total} ({dt:.1f}s/step)")
        self._emit(job, "progress")
        self._pause_point(job, f"step {step}/{total}")
        if job.cancel_requested:
            raise JobCancelled(f"cancelled at step {step}/{total}")

    def _pause_point(self, job: _Job, where: str) -> None:
        """A safe boundary: if a pause is requested, sync the GPU and block until resume, cancel or stop.

        Worker thread only. Nothing is in flight while blocked; the model and its GPU memory stay resident and the
        queue behind this job waits. Returns normally on resume (state back to RUNNING) and on cancel (the caller
        then sees cancel_requested and unwinds).
        """
        if not job.pause_requested or job.cancel_requested:
            return
        import mlx.core as mx

        _synchronize_all()
        self._set_state(job, PAUSED, pause_requested=False)  # pause_requested means "not in effect yet"
        self._jlog(job, f"paused at {where} (GPU synchronized, active {mx.get_active_memory() / 1e9:.2f} GB)")
        t0 = time.perf_counter()
        while not job.resume_event.wait(PAUSE_POLL_S):
            if job.cancel_requested or self._stopping:
                break
        paused = time.perf_counter() - t0
        with self._lock:
            job.paused_s += paused
            job.last_step_t = time.perf_counter()  # step timing excludes the pause
        if job.cancel_requested or self._stopping:
            self._jlog(job, f"pause ended by cancel after {paused:.1f} s")
            return
        self._jlog(job, f"resumed at {where} after {paused:.1f} s paused")
        self._set_state(job, RUNNING)

    # ---- worker thread -------------------------------------------------------------------------

    def _worker(self) -> None:
        try:
            self._work()
        finally:
            # Release any maintenance() waiter: after this nothing is resident and nothing will run.
            with self._lock:
                self._worker_running = False
                self._drop_requested = False
                self._idle.notify_all()

    def _drop_for_maintenance(self) -> None:
        """Unload the resident model for maintenance() and acknowledge it (worker thread, lock not held)."""
        error = None
        try:
            self._mm.drop()
        except Exception as exc:  # noqa: BLE001 - reported to the maintenance() caller
            log.exception("unload for maintenance failed")
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._drop_requested, self._drop_error = False, error
            self._idle.notify_all()

    def _work(self) -> None:
        import mlx.core as mx

        mx.set_cache_limit(int(self._cache_gb * 1000**3))
        try:
            while True:
                with self._lock:
                    while not self._stopping and not self._drop_requested and (
                        self._maintenance or not any(self._jobs[i].state == QUEUED for i in self._queue)
                    ):
                        self._wake.wait()  # held jobs alone leave the worker idle; maintenance() pauses the dequeue
                    if self._stopping:
                        break
                    job = None
                    if not self._drop_requested:
                        jid = next(i for i in self._queue if self._jobs[i].state == QUEUED)  # FIFO, skipping held
                        self._queue.remove(jid)
                        job = self._jobs[jid]
                        self._current = job
                if job is None:
                    self._drop_for_maintenance()  # outside the lock: drop() emits a model event
                    continue
                self._jlog(job, f"dequeued after {time.time() - job.created_at:.1f} s")
                self._emit_positions()
                gpu_at_start = self._check_gpu_idle(job)
                state, fields = FAILED, {"error": "internal error: job produced no outcome"}
                try:
                    state, fields = self._run_enhance(job) if job.kind == ENHANCE else self._run(job)
                except JobCancelled:
                    state, fields = CANCELLED, {}
                except Exception as exc:  # noqa: BLE001 - one bad job must not stop the worker
                    log.exception("job %s failed", job.id)
                    state, fields = FAILED, {"error": f"{type(exc).__name__}: {exc}"}
                    tb = traceback.format_exception(exc)
                    self._jlog(job, f"error: {type(exc).__name__}: {exc}")
                    for line in "".join(tb[-3:]).rstrip().splitlines()[-10:]:
                        self._jlog(job, f"  {line}")
                finally:
                    # Strict serialization: the job's GPU work and cleanup finish before anyone sees it end.
                    teardown_s = self._teardown(job)
                    with self._lock:
                        self._current = None
                if job.pause_requested:
                    self._jlog(job, "pause request dropped: the job finished before another safe boundary")
                    with self._lock:
                        job.pause_requested = False
                        job.resume_event.set()
                if job.cancel_requested and state == DONE:
                    state, fields = CANCELLED, {}  # the cancel arrived after the work finished; honour it
                if fields.get("result") is not None:
                    fields["result"].update(teardown_s=round(teardown_s, 3), gpu_util_at_start=gpu_at_start)
                self._jlog(job, self._summary(state, fields))
                # Emitted before the next job is dequeued, so job N's terminal event precedes job N+1's start.
                self._set_state(job, state, **fields)
        finally:
            self._mm.drop()

    def _check_gpu_idle(self, job: _Job) -> int | None:
        """Log if the GPU is busy while no job of ours is running (our previous job's work is already synced)."""
        util = gpu_utilization()
        # ioreg's figure trails real work by about a second, so a high first reading right after our own
        # synchronized teardown is usually lag; only a reading that stays high means foreign/leaked GPU load.
        for _ in range(GPU_BUSY_RESAMPLES):
            if util is None or util < GPU_BUSY_WARN_PCT:
                break
            time.sleep(GPU_BUSY_RESAMPLE_S)
            util = gpu_utilization()
        if util is not None and util >= GPU_BUSY_WARN_PCT:
            log.warning("GPU utilization is %d%% with no job running, before starting job %s", util, job.id)
            self._jlog(job, f"WARNING: GPU utilization {util}% with no job running before start")
        else:
            log.info("GPU utilization %s%% before starting job %s", util, job.id)
            self._jlog(job, f"GPU guard: utilization {util}% before start")
        return util

    @staticmethod
    def _summary(state: str, fields: dict) -> str:
        res = fields.get("result") or {}
        if state == DONE and "rewritten_prompt" in res:
            return (f"done: {res['thinking_tokens']}+{res['answer_tokens']} tokens, {res['tok_s']} tok/s, "
                    f"gen {res['gen_s']} s, fell_back={res['fell_back']}, finish={res['finish_reason']}")  # fmt: skip
        if state == DONE:
            return (f"done: {res.get('width')}x{res.get('height')} seed={res.get('seed')} steps={res.get('steps')} "
                    f"gen {res.get('gen_s')} s, load {res.get('load_s')} s, peak {res.get('peak_mem_bytes', 0) / 1e9:.2f} GB")  # fmt: skip
        if state == FAILED:
            return f"failed: {fields.get('error')}"
        return state

    def _teardown(self, job: _Job) -> float:
        """Wait for all GPU work the job issued, then run its cleanup; returns the seconds it took."""
        import mlx.core as mx

        t0 = time.perf_counter()
        try:
            _synchronize_all()
        except Exception:  # noqa: BLE001 - cleanup must still run
            log.exception("synchronize after job %s failed", job.id)
        try:
            if self._mm.key is not None and self._mm.key[0] == "image" and self._mm.model is not None:
                # t2i prompt embeddings are cached on the model forever; drop them so a long-lived server doesn't leak.
                self._mm.model.prompt_cache.clear()
                self._maybe_offload_text_encoder()
        except Exception:  # noqa: BLE001
            log.exception("cleanup after job %s failed", job.id)
        gc.collect()
        mx.clear_cache()
        dt = time.perf_counter() - t0
        if dt > TEARDOWN_WARN_S:
            log.warning("teardown of job %s took %.1f s (waiting for GPU work and cleanup)", job.id, dt)
            self._jlog(job, f"WARNING: slow teardown {dt:.1f} s")
        else:
            self._jlog(job, f"teardown {dt:.2f} s (GPU synchronized, cleanup done)")
        return dt

    def _ensure(self, job: _Job, key: tuple) -> float:
        """Make `key` resident for `job` and return load seconds; raises models.ModelNotReady if not downloaded."""
        if self._mm.key != key:
            self._set_state(job, LOADING)
        else:
            self._jlog(job, f"model {list(key)} already resident (load skipped)")
        return self._mm.ensure(key)

    def _run(self, job: _Job) -> tuple[str, dict]:
        """Run a generate job; returns its terminal (state, fields). Terminal states are emitted by the worker."""
        import mlx.core as mx

        self._offload_te = False
        settings = dict(self._settings_provider() or {})
        steps = job.steps
        if steps is None:
            steps = settings.get("default_turbo_steps", DEFAULT_TURBO_STEPS) if job.turbo else settings.get(
                "default_steps", DEFAULT_STEPS
            )
        try:
            validate_steps(steps, job.turbo)
            if job.mode == "edit" and not Path(job.ref_image_path).is_file():
                raise ValueError(f"reference image not found: {job.ref_image_path}")
            for r in job.ref_image_paths:
                if not Path(r).is_file():
                    raise ValueError(f"reference image not found: {r}")
            seed = job.seed if job.seed is not None else random.SystemRandom().randrange(2**32)
        except ValueError as exc:
            return FAILED, {"error": str(exc)}
        resolution = int(settings.get("resolution", DEFAULT_RESOLUTION))
        reference_resolution = resolve_reference_resolution(job.ref_detail, len(job.ref_image_paths), resolution)
        with self._lock:
            job.seed, job.steps, job.reference_resolution = seed, steps, reference_resolution

        low_memory = bool(settings.get("low_memory", DEFAULT_LOW_MEMORY))
        preset = LOW_MEMORY_PRESET if low_memory else {"te_offload": False, "vae_tiling": False}
        te_offload = bool(settings.get("te_offload", preset["te_offload"]))
        vae_tiling = bool(settings.get("vae_tiling", preset["vae_tiling"]))

        mx.reset_peak_memory()
        self._phases = {}
        te_reload_s = 0.0
        try:
            load_s = self._ensure(job, ("image", job.image_model, job.text_encoder, job.turbo))
        except models.ModelNotReady as exc:
            return FAILED, {"error": str(exc)}
        parts = self._mm.parts or {}
        self._jlog(job, f"parts: image {job.image_model} (transformer/vae/processor from {parts.get('image')}), "
                        f"text encoder {job.text_encoder} (from {parts.get('text_encoder')}); "
                        f"composed {parts.get('composed')}")  # fmt: skip
        model = self._mm.model
        if model.text_encoder is None:
            # Freed by te_offload after the previous job's encode; every job re-encodes since prompt_cache is cleared.
            te_reload_s = self._reload_text_encoder()
            self._jlog(job, f"text encoder reloaded in {te_reload_s:.2f} s")
        self._end_phase("load")
        self._pause_point(job, "before generation")
        if job.cancel_requested:
            return CANCELLED, {}

        from mflux.models.common.vae.tiling_config import TilingConfig

        # Same config the CLI's --vae-tiling installs (callback_manager.py _apply_vae_tiling).
        model.tiling_config = TilingConfig(vae_decode_tile_size=512) if vae_tiling else None
        self._offload_te = te_offload
        self._set_state(job, RUNNING, step=0, total_steps=steps)
        self._jlog(job, f"generating {job.mode} seed={seed} steps={steps} turbo={job.turbo} "
                        f"image_model={job.image_model} text_encoder={job.text_encoder}")
        image_paths, size = self._conditioning(job, resolution)
        self._jlog(job, f"reference_resolution {reference_resolution} (ref_detail {job.ref_detail}, "
                        f"{len(job.ref_image_paths)} extra ref{'' if len(job.ref_image_paths) == 1 else 's'}); output {size.get('width', resolution)}x"
                        f"{size.get('height', resolution)}")  # fmt: skip
        t0 = time.perf_counter()
        with self._lock:
            job.last_step_t = t0  # step 1's time includes prompt (and reference) encoding
        try:
            generated = model.generate_image(
                seed=seed,
                prompt=job.prompt,
                num_inference_steps=steps,
                image_paths=image_paths,
                output_resolution=resolution,
                reference_resolution=reference_resolution,
                **size,
            )
        except JobCancelled:
            return CANCELLED, {}
        finally:
            del model
        image = generated.image
        del generated
        self._end_phase("decode")
        gen_s = time.perf_counter() - t0
        if job.cancel_requested:
            return CANCELLED, {}

        out = Path(job.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        image = finalize_alpha(image, job.transparent)
        image.save(tmp, format="PNG")
        os.replace(tmp, out)
        result = {
            "transparent": job.transparent,
            "seed": seed,
            "steps": steps,
            "turbo": job.turbo,
            "image_model": job.image_model,
            "text_encoder": job.text_encoder,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "output_path": str(out),
            "load_s": round(load_s, 3),
            "te_reload_s": round(te_reload_s, 3),
            "gen_s": round(gen_s, 3),
            "te_offload": te_offload,
            "vae_tiling": vae_tiling,
            "refs": len(job.ref_image_paths),
            "reference_resolution": reference_resolution,
            "ref_detail": job.ref_detail,
            "paused_s": round(job.paused_s, 3),
            "peak_mem_bytes": max(p["peak"] for p in self._phases.values()),
            "peak_phase": max(self._phases, key=lambda k: self._phases[k]["peak"]),
            # Per phase: "peak" since the previous phase ended, "active" at the phase boundary.
            "phases": copy.deepcopy(self._phases),
        }
        return DONE, {"result": result}

    def _conditioning(self, job: _Job, resolution: int) -> tuple[list[str], dict]:
        """(image_paths, size kwargs) for generate_image, and log the numbered images.

        Edit: [current] + refs (Image 1 = current), output = the parent's size. An exact parent size is passed when it
        is a multiple of 32. Otherwise, without refs nothing is passed (the fork sizes it from the parent's ratio), and
        with refs the size the fork would derive from the parent is passed explicitly, or it would follow the last ref. T2I with refs: refs only, square output at
        `resolution`. T2I without refs: no images, no size (unchanged).
        """
        from PIL import Image

        from webstudio import sizes

        def dims(path: str) -> tuple[int, int]:
            with Image.open(path) as im:
                return im.size

        images, lines = [], []
        size: dict = {}
        if job.mode == "edit":
            w, h = dims(job.ref_image_path)
            if w % 32 == 0 and h % 32 == 0:
                size = {"width": w, "height": h}
            elif job.ref_image_paths:
                # Same numbers the fork derives from a lone parent; explicit so the output can't follow a ref's aspect.
                ow, oh = sizes.dimensions(resolution, w / h)
                size = {"width": ow, "height": oh}
            stem = Path(job.ref_image_path).stem
            label = f"#{int(stem)}" if stem.isdigit() else repr(Path(job.ref_image_path).name)
            images.append(job.ref_image_path)
            lines.append(f"Image 1: current ({label}, {w}×{h})")
        elif job.ref_image_paths:
            size = {"width": resolution, "height": resolution}
        for path in job.ref_image_paths:
            w, h = dims(path)
            images.append(path)
            lines.append(f"Image {len(images)}: ref {Path(path).stem!r} ({w}×{h})")
        if lines:
            self._jlog(job, "; ".join(lines))
        return images, size

    def _run_enhance(self, job: _Job) -> tuple[str, dict]:
        """Run an enhance job; returns its terminal (state, fields). Terminal states are emitted by the worker."""
        import mlx.core as mx

        settings = dict(self._settings_provider() or {})
        quant = job.quant or settings.get("pe_quant", DEFAULT_PE_QUANT)
        thinking = job.thinking if job.thinking is not None else bool(settings.get("pe_thinking", DEFAULT_PE_THINKING))
        max_new_tokens = job.max_new_tokens or settings.get("pe_max_new_tokens", DEFAULT_PE_MAX_NEW_TOKENS)
        language = job.language or settings.get("pe_language", DEFAULT_PE_LANGUAGE)
        try:
            if quant not in PE_QUANTS:
                raise ValueError(f"pe_quant must be one of {PE_QUANTS}, got {quant!r}")
            if language not in PE_LANGUAGES:
                raise ValueError(f"pe_language must be one of {PE_LANGUAGES}, got {language!r}")
            _validate_max_new_tokens(max_new_tokens)
            if job.mode == "i2i" and not Path(job.ref_image_path).is_file():
                raise ValueError(f"reference image not found: {job.ref_image_path}")
        except ValueError as exc:
            return FAILED, {"error": str(exc)}
        # The Heretic rewriters (settings pe_variant) replace the official model of either mode whatever pe_quant says.
        heretic = settings.get("pe_variant", "official") == "heretic"
        model_quant = models.PE_HERETIC if heretic else quant
        with self._lock:
            job.quant, job.thinking, job.max_new_tokens, job.language = quant, thinking, max_new_tokens, language
            job.rewriter = models.PE_HERETIC if heretic else f"official-{quant}"

        mx.reset_peak_memory()
        try:
            load_s = self._ensure(job, (f"pe_{job.mode}", model_quant))
        except models.ModelNotReady as exc:
            return FAILED, {"error": str(exc)}
        self._pause_point(job, "before prefill")
        if job.cancel_requested:
            return CANCELLED, {}

        from mlx_vlm import stream_generate
        from PIL import Image

        image = None
        if job.mode == "i2i":
            src = Image.open(job.ref_image_path)
            if src.mode in ("RGBA", "LA", "P"):
                src = src.convert("RGBA")
                image = Image.new("RGB", src.size, (255, 255, 255))
                image.paste(src, mask=src.split()[3])
            else:
                image = src.convert("RGB")
        user_text = job.prompt + (PE_ENGLISH_SUFFIX if language == "english" else "")
        content = ([{"type": "image", "image": image}] if image is not None else []) + [
            {"type": "text", "text": user_text}
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._mm.system_prompt}]},
            {"role": "user", "content": content},
        ]
        processor = self._mm.processor
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

        self._set_state(job, RUNNING, phase="thinking" if thinking else "writing", tokens=0, text="")
        self._jlog(job, f"enhancing {job.mode} quant={quant} thinking={thinking} max_new_tokens={max_new_tokens} "
                        f"language={language}; prefill, then phase {'thinking' if thinking else 'writing'}")  # fmt: skip
        text, tokens, thinking_tokens, in_think, last = "", 0, 0, thinking, None
        cancelled = False
        last_emit = 0.0
        t0 = time.perf_counter()
        t_first = None  # first token time; tok/s below excludes prefill and pauses
        mark_t, mark_n = 0.0, 0  # for the per-50-token rate
        stream = stream_generate(
            self._mm.model,
            processor,
            prompt,
            image=[image] if image is not None else None,
            max_tokens=max_new_tokens,
            **PE_SAMPLING,
        )
        try:
            for last in stream:
                tokens += 1
                text += last.text
                now = time.perf_counter()
                if t_first is None:
                    t_first = mark_t = now
                    mark_n = tokens
                    self._jlog(job, f"first token after {now - t0:.1f} s (prefill)")
                if in_think:
                    thinking_tokens += 1
                    if THINK_END in text:
                        in_think = False
                        self._jlog(job, f"phase writing after {thinking_tokens} thinking tokens")
                if tokens % ENHANCE_LOG_EVERY_TOKENS == 0 and now > mark_t:
                    self._jlog(job, f"tokens {tokens} ({(tokens - mark_n) / (now - mark_t):.1f} tok/s)")
                    mark_t, mark_n = now, tokens
                if job.cancel_requested:
                    cancelled = True
                    break
                if job.pause_requested:
                    paused_before = job.paused_s
                    # Between tokens: mlx-vlm has already queued the next token; _pause_point syncs it first.
                    self._pause_point(job, f"token {tokens}")
                    mark_t += job.paused_s - paused_before
                    if job.cancel_requested:
                        cancelled = True
                        break
                if now - last_emit >= PE_PROGRESS_INTERVAL_S:
                    last_emit = now
                    answer = text.rsplit(THINK_END, 1)[1] if THINK_END in text else ("" if in_think else text)
                    with self._lock:
                        job.phase = "thinking" if in_think else "writing"
                        job.tokens = tokens
                        job.text = _partial_rewrite(answer)
                    self._emit(job, "progress")
        finally:
            # Closing runs stream_generate's wired_limit exit, which synchronizes mlx-vlm's generation stream
            # (mlx_vlm/generate/common.py:238-244); _teardown then syncs every stream again before the next job.
            stream.close()
            del stream
        t_end = time.perf_counter()
        gen_s = t_end - t0
        if cancelled:
            return CANCELLED, {"tokens": tokens}
        decode_s = (t_end - t_first - job.paused_s) if t_first is not None else 0.0

        answer = text.rsplit(THINK_END, 1)[1] if THINK_END in text else ("" if in_think else text)
        rewritten = _parse_rewrite(answer)
        result = {
            "rewritten_prompt": rewritten if rewritten is not None else job.prompt,
            "fell_back": rewritten is None,
            "finish_reason": last.finish_reason if last is not None else None,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": tokens - thinking_tokens,
            # mlx-vlm's generation_tps counts paused time, so compute the rate over active decode time instead.
            "tok_s": round((tokens - 1) / decode_s, 1) if job.paused_s > 0 and tokens > 1 and decode_s > 0 else (
                round(last.generation_tps, 1) if last is not None else 0.0),
            "paused_s": round(job.paused_s, 3),
            "gen_s": round(gen_s, 3),
            "load_s": round(load_s, 3),
            "peak_mem_bytes": mx.get_peak_memory(),
            "mode": job.mode,
            "quant": quant,
            "rewriter": job.rewriter,
            "thinking": thinking,
            "language": language,
        }
        fields = {
            "result": result,
            "phase": "thinking" if in_think else "writing",
            "tokens": tokens,
            "text": result["rewritten_prompt"],
        }
        return DONE, fields

    def _reload_text_encoder(self) -> float:
        """Rebuild only the text encoder from the local snapshot, as QwenImage21Initializer.init does per component.

        The LoRA touches only the transformer, so the TE reloads identically for turbo and non-turbo models.
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
        from mflux.models.qwen21.reference.qwen_image21_initializer import QwenImage21Initializer
        from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

        t0 = time.perf_counter()
        model = self._mm.model
        (component,) = [c for c in QwenImage21WeightDefinition.get_components() if c.name == "text_encoder"]
        module = QwenImage21TextEncoder(model._component_configs["text_encoder"])
        weights = WeightLoader.load_single_local(component, Path(self._mm.snapshot))
        supplied = dict(tree_flatten(weights.components[component.name]))
        if weights.meta_data.quantization_level is None:
            QwenImage21Initializer._validate_weights(component.name, module, supplied)
        WeightApplier.apply_and_quantize_single(
            weights, module, component, None, quantization_predicate=QwenImage21WeightDefinition.quantization_predicate
        )
        if weights.meta_data.quantization_level is not None:
            QwenImage21Initializer._validate_weights(component.name, module, supplied)
        mx.eval(module.parameters())
        del weights, supplied
        # Assign only once fully built so a failed reload leaves text_encoder None and the next job retries.
        model.text_encoder = module
        mx.clear_cache()
        return time.perf_counter() - t0

    def _maybe_offload_text_encoder(self) -> None:
        import mlx.core as mx

        model = self._mm.model
        if self._offload_te and model is not None and model.text_encoder is not None:
            model.text_encoder = None
            gc.collect()
            mx.clear_cache()

    def _end_phase(self, name: str) -> None:
        import mlx.core as mx

        self._phases[name] = {"peak": mx.get_peak_memory(), "active": mx.get_active_memory()}
        mx.reset_peak_memory()


def _validate_max_new_tokens(n: Any) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 32768:
        raise ValueError(f"max_new_tokens must be an integer in 1..32768, got {n!r}")
