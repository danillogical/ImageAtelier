"""FastAPI app: project/run API over the generation engine. See API.md for the contract."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import time
import uuid
import warnings
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictBool, StrictInt

from webstudio import ca, models, settings as settings_mod, sizes, styles
from webstudio.engine import EngineBusy, InvalidTransition, UnknownJob
from webstudio.store import Store, default_data_dir

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
FINAL_STATES = ("done", "failed", "cancelled")
RECENT_FINISHED_JOBS = 20
PING_SECONDS = 15
DOWNLOAD_EVENT_SECONDS = 1.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PIXELS = 80_000_000


def flatten_parent(src, dst) -> bool:
    """If the PNG at `src` has any transparency, write it composited over white (opaque RGB, same size) to `dst`.
    Returns whether a flattened copy was written. The model reads an empty RGBA area as noise, so a non-transparent edit of a
    cut-out must not be shown the raw alpha."""
    from PIL import Image

    with Image.open(src) as im:
        if "A" not in im.getbands() or im.getchannel("A").getextrema()[0] >= 255:
            return False
        rgba = im.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    tmp = dst.with_name(dst.name + f".{uuid.uuid4().hex[:8]}.tmp")
    bg.convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, dst)
    return True


def wrap_transparent(prompt: str) -> str:
    """The model-card wording that triggers a transparent background (one trailing period of the user's text is dropped)."""
    p = prompt.strip()
    if p.endswith("."):
        p = p[:-1]
    return f"This is an RGBA image with transparency. {p}. The image has alpha channel and the background is transparent."


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _flatten_rgb(im):
    """Opaque RGB: alpha would act as a mask in the reference path, so composite transparency over white."""
    from PIL import Image

    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info) or "A" in im.getbands():
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return im.convert("RGB")


def decode_image(data: bytes):
    """Bytes -> oriented, opaque RGB PIL image, enforcing the pixel cap. Raises UploadError (413/415)."""
    from PIL import Image, ImageOps

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.width * im.height > MAX_UPLOAD_PIXELS:
                raise UploadError(413, f"image is too large ({im.width}x{im.height}); the limit is {MAX_UPLOAD_PIXELS // 1_000_000} megapixels")
            im.load()
    except UploadError:
        raise
    except Image.DecompressionBombError:
        raise UploadError(413, f"image is too large; the limit is {MAX_UPLOAD_PIXELS // 1_000_000} megapixels")
    except (Image.UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise UploadError(415, "the file is not a readable image")
    return _flatten_rgb(ImageOps.exif_transpose(im))


REF_MAX_SIDE = 2048
MAX_REFS_PER_PROJECT = 30


def process_ref(data: bytes):
    """A reference image: EXIF-oriented, opaque RGB, long side downscaled to <= 2048, aspect kept, no crop."""
    from PIL import Image

    im = decode_image(data)
    if max(im.size) > REF_MAX_SIDE:
        k = REF_MAX_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def process_upload(data: bytes, resolution: int, preset: str | None, crop: tuple | None, src: tuple | None = None):
    """Decode, orient, flatten and cover-crop an uploaded image to the output size. Returns (RGB image, meta)."""
    from PIL import Image

    im = decode_image(data)
    sw, sh = im.size
    exact = next((lab for lab, r in sizes.PRESETS.items() if sizes.dimensions(resolution, r) == (sw, sh)), None)
    if exact and preset in (None, exact):
        # Already the output size: keep the pixels. A client that cropped locally sends its crop box and the original size
        # (`crop`/`src`, in original oriented pixels) purely as provenance for the run record.
        prov = [round(v) for v in crop] if crop else [0, 0, sw, sh]
        return im, {"source_size": list(src) if src else [sw, sh], "crop": prov, "preset": exact}
    label = preset or sizes.closest(sw, sh)
    dest = sizes.dimensions(resolution, sizes.PRESETS[label])
    ratio = dest[0] / dest[1]  # crop to the exact output ratio (not the nominal one) so the resize never stretches
    box = sizes.fit_box_to_ratio(crop, sw, sh, ratio) if crop else sizes.cover_crop_box(sw, sh, ratio)
    x, y, w, h = box
    out = im.crop((x, y, x + w, y + h)).resize(dest, Image.LANCZOS)
    return out, {"source_size": [sw, sh], "crop": [x, y, w, h], "preset": label}


class ProjectCreate(BaseModel):
    name: str | None = None


class ProjectPatch(BaseModel):
    name: str | None = None
    current_run: StrictInt | None = None
    style: dict | None = None


class RefPatch(BaseModel):
    enabled: StrictBool | None = None
    name: str | None = None
    order: StrictInt | None = None


class RunCreate(BaseModel):
    prompt: str
    seed: StrictInt | None = None
    steps: StrictInt | None = None
    turbo: StrictBool | None = None
    transparent: StrictBool = False
    use_enhanced: StrictBool = False
    image_model: str | None = None
    text_encoder: str | None = None
    variant: str | None = None  # legacy: rejected with a 400 (replaced by image_model/text_encoder)
    enhanced_from: str | None = None


class ChainCreate(BaseModel):
    prompt: str
    transparent: StrictBool = False
    seed: StrictInt | None = None
    thinking: StrictBool | None = None
    language: str | None = None


class EnhanceCreate(BaseModel):
    prompt: str
    thinking: StrictBool | None = None
    quant: str | None = None
    language: str | None = None


def create_app(engine=None, data_dir=None) -> FastAPI:
    store = Store(data_dir or default_data_dir())
    settings = settings_mod.SettingsStore(store.root / "settings.json")
    if engine is None:
        from webstudio.engine import Engine

        def provider():
            s = settings.get()
            return {**s, "default_steps": settings_mod.NON_TURBO_DEFAULT_STEPS, "default_turbo_steps": s["steps"]}

        engine = Engine(settings_provider=provider, compose_dir=store.root / "compose")

    jobmap: dict[str, dict] = {}  # job_id -> {project_id, run, parent_run, prompt, mode}
    chains: dict[str, dict] = {}  # Enhance & Generate units waiting for their enhance job to finish
    enhance_map: dict[str, dict] = {}  # job_id -> {project_id, mode} for prompt-enhance jobs (not runs)
    settled: dict[str, asyncio.Event] = {}  # job_id -> set once its terminal handling (save/cleanup) is complete
    subscribers: set[asyncio.Queue] = set()
    server_subs: set[asyncio.Queue] = set()  # connections that asked for ?serverlog=1
    project_names: dict[str, str] = {}
    events: asyncio.Queue = asyncio.Queue()  # engine events handed over from the worker thread

    # ---- event fan-out -------------------------------------------------------------------------
    def broadcast(kind: str, data) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        for q in list(subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                subscribers.discard(q)  # a stalled client is dropped; it reconnects and refetches

    def pname(pid: str | None) -> str | None:
        if not pid:
            return None
        if pid not in project_names:
            p = store.get_project(pid)
            if p is None:
                return None
            project_names[pid] = p["name"]
        return project_names[pid]

    def broadcast_quiet(kind: str, data, only: set | None = None) -> None:
        """Fan-out for log lines. Never logs: the server-log ring would feed its own output back into this path."""
        try:
            msg = f"event: {kind}\ndata: {json.dumps(data)}\n\n"
            for q in list(subscribers if only is None else only):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    subscribers.discard(q)
                    server_subs.discard(q)
        except Exception:  # noqa: BLE001
            pass

    def enrich_job(snap: dict) -> dict:
        info = jobmap.get(snap["id"]) or enhance_map.get(snap["id"]) or {}
        snap = {**snap, "chain_id": info.get("chain_id"), "chain_step": info.get("chain_step")}
        if "model_prompt" in info:  # show the user's text, not the wrapped prompt the model received
            snap = {**snap, "prompt": info["prompt"], "model_prompt": info["model_prompt"], "transparent": info["transparent"],
                    "style": info.get("style")}
        elif "prompt" in info:  # an enhance job: show the user's text, not the text with the art-style hint
            snap = {**snap, "prompt": info["prompt"]}
        ev = settled.get(snap["id"])
        if ev is not None and not ev.is_set() and snap["state"] in FINAL_STATES and (
            snap["state"] == "done" or os.path.exists(snap.get("output_path") or "")
        ):
            # Hold a terminal state until the run is saved (done) or its partial PNG is removed (failed/cancelled);
            # a job that never produced a file (e.g. cancelled while queued) reports its real state immediately.
            snap = {**snap, "state": "running", "result": None, "error": None}
        return {**snap, "project_id": info.get("project_id"), "project_name": pname(info.get("project_id")), "run": info.get("run")}

    async def settle(pid: str) -> None:
        """Wait until every engine-finished job of this project has been saved (or cleaned up)."""
        for jid, info in [*jobmap.items(), *enhance_map.items()]:
            ev = settled.get(jid)
            if info["project_id"] != pid or ev is None or ev.is_set():
                continue
            snap = engine.get(jid)
            if snap and snap["state"] in FINAL_STATES:
                try:
                    await asyncio.wait_for(ev.wait(), 30)
                except asyncio.TimeoutError:
                    log.error("timed out waiting for job %s to be saved", jid)

    def run_json(rec: dict) -> dict:
        base = f"/api/projects/{rec['project_id']}/runs/{rec['run']}"
        return {"style": None, **rec, "image_url": f"{base}/image", "thumb_url": f"{base}/thumb"}

    def finalize(snap: dict, info: dict) -> dict | None:
        """Blocking: persist a finished job (runs in a thread). Returns the saved run record or None."""
        pid, n = info["project_id"], info["run"]
        if snap["state"] != "done" or not snap.get("result"):
            store.discard_run(pid, n)
            return None
        res = snap["result"]
        record = {
            "run": n,
            "project_id": pid,
            "prompt": info["prompt"],
            "mode": info["mode"],
            "parent_run": info["parent_run"],
            "seed": res.get("seed"),
            "steps": res.get("steps"),
            "turbo": res.get("turbo"),
            "image_model": res.get("image_model"),
            "text_encoder": res.get("text_encoder"),
            "width": res.get("width"),
            "height": res.get("height"),
            "created_at": time.time(),
            "gen_s": res.get("gen_s"),
            "load_s": res.get("load_s"),
            "job_id": snap["id"],
            "enhanced_from": info.get("enhanced_from"),
            "transparent": info.get("transparent", False),
            "model_prompt": info.get("model_prompt", info["prompt"]),
            "style": info.get("style"),
            "refs": [{**r, "image_url": f"/api/projects/{pid}/runs/{n}/refs/{r['image_index']}"} for r in info.get("refs", [])],
            "reference_resolution": res.get("reference_resolution"),
            "ref_detail": res.get("ref_detail"),
        }
        try:
            saved = store.save_run(pid, record)
        except Exception:  # noqa: BLE001 - e.g. unreadable png; don't leave a half-run behind
            log.exception("saving run %s/%s failed", pid, n)
            saved = None
        if saved is None:
            store.discard_run(pid, n)
        return saved

    def set_enhanced(snap: dict, info: dict) -> dict | None:
        """Blocking: store a finished (non-chained) enhance as the project's `enhanced` value. Cancelled/failed leave it alone."""
        pid = info["project_id"]
        res = snap.get("result")
        p = store.get_project(pid)
        if snap["state"] != "done" or not res or p is None:
            return None
        fell_back = bool(res.get("fell_back"))
        original = info.get("prompt", snap["prompt"])  # the user's text; the engine's prompt also carries the art-style hint
        enhanced = {
            "text": original if fell_back or not res.get("rewritten_prompt") else res["rewritten_prompt"],
            "original": original, "mode": info["mode"], "rewriter": res.get("rewriter"), "fell_back": fell_back,
            "job_id": snap["id"], "created_at": time.time(), "style": styles.normalize(info.get("style")),
        }
        store.update_project(pid, enhanced=enhanced, updated_at=p["updated_at"])  # not a new activity: keep the gallery order
        return enhanced

    def clear_enhanced(pid: str) -> None:
        p = store.get_project(pid)
        if p is not None and p.get("enhanced") is not None:
            store.update_project(pid, enhanced=None, updated_at=p["updated_at"])
            broadcast("enhanced", {"project_id": pid, "enhanced": None})

    def chain_cleanup(prep: dict) -> None:
        store.discard_run(prep["pid"], prep["n"])
        store.release_run(prep["pid"], prep["n"])

    async def finish_enhance(snap: dict, info: dict, ev) -> tuple[dict, dict | None]:
        """Terminal handling of an enhance job (runs in the consumer, before the terminal `job` event is sent)."""
        pid, chain = info["project_id"], chains.pop(info.get("chain_id"), None)
        extra = None
        if chain is None:
            enhanced = await asyncio.to_thread(set_enhanced, snap, info)
            if enhanced is not None:
                broadcast("enhanced", {"project_id": pid, "enhanced": enhanced})
            ev.set()
            return snap, None
        prep = chain["prep"]
        done = snap["state"] == "done" and snap.get("result")
        if not done or store.get_project(pid) is None:
            await asyncio.to_thread(chain_cleanup, prep)
            why = "cancelled" if snap["state"] == "cancelled" else "failed" if snap["state"] == "failed" else "project deleted"
            if why == "project deleted":
                log.warning("dropping chain %s: project %s was deleted", chain["id"], pid)
            ev.set()
            return snap, {"chain_id": chain["id"], "project_id": pid, "state": why, "error": snap.get("error")}
        res = snap["result"]
        prompt = chain["original"] if res.get("fell_back") or not res.get("rewritten_prompt") else res["rewritten_prompt"]
        ev.set()  # the enhance is finished for clients; the generate is enqueued right behind it
        now_style = styles.normalize((store.get_project(pid) or {}).get("style"))  # the style as of now, which may have changed
        try:
            job_id = submit_prepared(prep, prompt, enhanced_from=chain["original"], chain_id=chain["id"],
                                     style=now_style, skip_style=styles.same(chain["style"], now_style))
        except GenSubmitError as exc:
            await asyncio.to_thread(chain_cleanup, prep)
            return snap, {"chain_id": chain["id"], "project_id": pid, "state": "failed", "error": exc.message}
        return snap, {"chain_id": chain["id"], "project_id": pid, "state": "generating", "job_id": job_id, "run": prep["n"], "fell_back": bool(res.get("fell_back"))}

    async def process(snap: dict) -> None:
        info = jobmap.get(snap["id"])
        saved = None
        ev = settled.get(snap["id"])
        einfo = enhance_map.get(snap["id"])
        if einfo is not None and ev is not None and snap["state"] in FINAL_STATES:
            chain_event = None
            try:
                snap, chain_event = await finish_enhance(snap, einfo, ev)
            finally:
                ev.set()
            broadcast("job", enrich_job(snap))  # the terminal enhance event first, then what the chain did about it
            if chain_event:
                broadcast("chain", chain_event)
            return
        try:
            if info and snap["state"] in FINAL_STATES:
                saved = await asyncio.to_thread(finalize, snap, info)
                if snap["state"] == "done" and saved is None:
                    snap = {**snap, "state": "failed", "error": "could not save run (project deleted or image unreadable)"}
        finally:
            if ev is not None and snap["state"] in FINAL_STATES:
                ev.set()  # from here on /api/jobs and create_run see the terminal state and the advanced current_run
        # Order for clients: `run` first, then the terminal `job` event.
        if saved:
            broadcast("run", {"project_id": saved["project_id"], "run": run_json(saved)})
        broadcast("job", enrich_job(snap))

    async def consume() -> None:
        while True:
            snap = await events.get()
            if snap is None:
                return
            try:
                if snap.get("type") == "model":  # model-manager events carry no job id
                    broadcast("model", snap)
                    continue
                if snap.get("type") == "log":  # per-job log line -> every client
                    broadcast_quiet("log", {"job_id": snap.get("job_id"), "line": snap.get("line")})
                    continue
                if snap.get("type") == "serverlog":  # server log line -> only clients subscribed with ?serverlog=1
                    broadcast_quiet("serverlog", {"line": snap.get("line")}, only=server_subs)
                    continue
                await process(snap)
            except Exception:  # noqa: BLE001
                log.exception("event handling failed")

    def image_models() -> list[dict]:
        return [models.image_status(n) for n in models.IMAGE_MODELS]

    def te_models() -> list[dict]:
        return [models.te_status(n) for n in models.TEXT_ENCODERS]

    def pe_models() -> list[dict]:
        return [
            {"kind": "pe", "name": m.name, "mode": m.mode, "quant": m.quant, "variant": m.variant, "source": m.source,
             "repo": m.repo or None, "approx_gb": m.approx_gb, **models.pe_status(m.name)}
            for m in models.PE_MODELS.values()
        ]

    def resident_event() -> dict:
        r = engine.resident() if hasattr(engine, "resident") else {}
        return {"type": "model", "state": r.get("state", "unloaded"), "key": r.get("key"), "load_s": None}

    async def download_ticker() -> None:
        was_active = False
        while True:
            await asyncio.sleep(DOWNLOAD_EVENT_SECONDS)
            active = (
                any(models.is_image_downloading(n) for n in models.IMAGE_MODELS)
                or any(models.is_te_downloading(n) for n in models.TEXT_ENCODERS)
                or any(models.is_pe_downloading(n) for n in models.PE_MODELS)
            )
            if active or was_active:  # one trailing event carries the final state
                broadcast("download", image_models() + te_models() + pe_models())
            was_active = active

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        def on_engine_event(snap: dict) -> None:
            try:
                loop.call_soon_threadsafe(events.put_nowait, snap)
            except RuntimeError:  # loop already closed during shutdown
                pass

        swept = store.sweep_orphans()
        if swept:
            log.info("removed %d orphaned pinned run file(s) from an interrupted session", swept)
        engine.on_event(on_engine_event)
        engine.start()
        if settings.migrated:
            log.info("settings: migrated %s from 4bit to 8bit (4-bit was removed)", ", ".join(settings.migrated))
        if getattr(settings, "migrated_pe_variant", None) is not None:
            log.info("settings: migrated pe_t2i_variant %r to pe_variant (now applies to new images and edits)", settings.migrated_pe_variant)
        if getattr(settings, "migrated_variant", None) is not None:
            s = settings.get()
            log.info("settings: migrated variant %r to image_model %s + text_encoder %s",
                     settings.migrated_variant, s["image_model"], s["text_encoder"])  # fmt: skip
        consumer = asyncio.create_task(consume())
        plain = ca.listener_from_env(app)
        if plain:
            await plain.start()
        ticker = asyncio.create_task(download_ticker())
        try:
            yield
        finally:
            if plain:
                await plain.stop()
            ticker.cancel()
            await asyncio.to_thread(engine.stop)
            loop.call_soon(events.put_nowait, None)  # FIFO after any events the worker just queued
            await consumer

    app = FastAPI(title="WebStudio", lifespan=lifespan)
    app.state.store, app.state.settings, app.state.engine = store, settings, engine

    @app.exception_handler(RequestValidationError)
    async def _validation_400(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return JSONResponse({"detail": msg or "invalid request"}, status_code=400)

    api = APIRouter(prefix="/api")

    def need_project(pid: str) -> dict:
        p = store.get_project(pid)
        if p is None:
            raise HTTPException(404, "project not found")
        return p

    def latest_of(p: dict) -> dict | None:
        n = p["current_run"]
        if n is None:
            return None
        base = f"/api/projects/{p['id']}/runs/{n}"
        run = store.get_run(p["id"], n) or {}
        return {"run": n, "thumb_url": f"{base}/thumb", "image_url": f"{base}/image", "transparent": bool(run.get("transparent"))}

    @api.get("/styles")
    def list_styles():
        return styles.list_presets()

    # ---- projects ------------------------------------------------------------------------------
    @api.get("/projects")
    def list_projects():
        return [{**p, "run_count": len(store.list_runs(p["id"])), "latest": latest_of(p)} for p in store.list_projects()]

    @api.post("/projects")
    def create_project(body: ProjectCreate | None = None):
        return store.create_project(body.name if body else None)

    @api.get("/projects/{pid}")
    def get_project(pid: str):
        p = need_project(pid)
        return {**p, "runs": [run_json(r) for r in store.list_runs(pid)]}

    @api.patch("/projects/{pid}")
    def patch_project(pid: str, body: ProjectPatch):
        need_project(pid)
        fields: dict = {}
        if "name" in body.model_fields_set:
            if body.name is None or not body.name.strip() or len(body.name) > 200:
                raise HTTPException(400, "name must be a non-empty string of at most 200 characters")
            fields["name"] = body.name.strip()
        if "current_run" in body.model_fields_set:
            if body.current_run is None or store.get_run(pid, body.current_run) is None:
                raise HTTPException(400, f"current_run must be an existing run, got {body.current_run!r}")
            fields["current_run"] = body.current_run
        style_changed = False
        if "style" in body.model_fields_set:
            try:
                new_style = styles.validate(body.style)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            style_changed = not styles.same(new_style, need_project(pid).get("style"))
            fields["style"] = new_style
        p = store.update_project(pid, **fields)
        if p:
            project_names[pid] = p["name"]
        if p is None:
            raise HTTPException(404, "project not found")
        if style_changed:
            broadcast("style", {"project_id": pid, "style": p["style"]})
        return p

    @api.delete("/projects/{pid}", status_code=204)
    def delete_project(pid: str):
        need_project(pid)
        for jid, info in [*jobmap.items(), *enhance_map.items()]:  # stop queued/running work for this project
            if info["project_id"] == pid:
                engine.cancel(jid)
        store.delete_project(pid)
        project_names.pop(pid, None)
        return Response(status_code=204)

    # ---- runs ----------------------------------------------------------------------------------
    class GenSubmitError(Exception):
        def __init__(self, status: int, message: str):
            super().__init__(message)
            self.status, self.message = status, message

    async def prepare_generate(pid: str, p: dict, body: RunCreate) -> dict:
        """Validate and pin everything a generation needs *now*: the parent (or a flattened copy), the enabled references,
        the run number. Returns a prepared context; submit_prepared() sends it to the engine later. Raises HTTPException."""
        cfg = settings.get()
        turbo = cfg["turbo"] if body.turbo is None else body.turbo
        if body.variant is not None:
            raise HTTPException(400, "variant was replaced by image_model/text_encoder")
        image_model = cfg["image_model"] if body.image_model is None else body.image_model
        text_encoder = cfg["text_encoder"] if body.text_encoder is None else body.text_encoder
        if image_model not in models.IMAGE_MODELS:
            raise HTTPException(400, f"unknown image_model {image_model!r}; expected one of {sorted(models.IMAGE_MODELS)}")
        if text_encoder not in models.TEXT_ENCODERS:
            raise HTTPException(400, f"unknown text_encoder {text_encoder!r}; expected one of {sorted(models.TEXT_ENCODERS)}")
        if body.steps is not None:
            steps = body.steps
        else:
            steps = cfg["steps"] if turbo == cfg["turbo"] else settings_mod.default_steps(turbo)
        if not body.prompt.strip():
            raise HTTPException(400, "prompt must be a non-empty string")
        try:
            settings_mod.validate_steps(steps, turbo)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        for part in (models.image_status(image_model), models.te_status(text_encoder)):
            if part["state"] != "ready":
                what = "image model" if part["kind"] == "image" else "text encoder"
                state = {"downloading": "is still downloading", "not_installed": "isn't installed"}.get(part["state"], "isn't downloaded yet")
                hint = f". {part['hint']}" if part.get("hint") else ""
                raise HTTPException(409, f"The {what} ({part['label']}) {state}{hint}")
            if part["kind"] == "image" and turbo and not part.get("turbo_lora_ready"):
                raise HTTPException(409, "The turbo add-on isn't downloaded yet; download it or switch to Standard mode")

        # A queued edit is pinned to the current run as of now, not as of when it starts.
        parent = p["current_run"]
        if parent is not None and not store.image_path(pid, parent).is_file():
            raise HTTPException(409, f"current run {parent} has no image on disk")
        mode = "t2i" if parent is None else "edit"
        n = store.reserve_run(pid)
        # Pin the enabled references (Image 2.. in an edit, Image 1.. otherwise) as immutable copies for this run.
        refs = [r for r in store.list_refs(pid) if r["enabled"]]
        first_index = 2 if mode == "edit" else 1
        try:
            pinned = store.pin_refs(pid, n, first_index, refs) if refs else []
        except OSError as exc:
            store.discard_run(pid, n)
            store.release_run(pid, n)
            raise HTTPException(500, f"could not pin the reference images: {exc}")
        ref_meta = [{"id": r["id"], "name": r["name"], "image_index": first_index + i, "width": r["width"], "height": r["height"]} for i, r in enumerate(refs)]
        ref_path = store.image_path(pid, parent) if parent is not None else None
        if parent is not None and not body.transparent:
            # A plain edit of a transparent image gets a flattened (over white) copy of the parent, stored next to the run.
            flat = store.runs_dir(pid) / f"{n:04d}.parent.png"
            try:
                if await asyncio.to_thread(flatten_parent, ref_path, flat):
                    ref_path = flat
            except OSError as exc:
                store.discard_run(pid, n)
                store.release_run(pid, n)
                raise HTTPException(500, f"could not prepare the parent image: {exc}")
        job = {
            "transparent": body.transparent,
            "mode": mode,
            "ref_image_path": str(ref_path) if ref_path is not None else None,
            "seed": body.seed,
            "steps": steps,
            "turbo": turbo,
            "image_model": image_model,
            "text_encoder": text_encoder,
            "ref_image_paths": [str(x) for x in pinned],
            "output_path": str(store.image_path(pid, n)),
        }
        return {"pid": pid, "n": n, "mode": mode, "parent": parent, "ref_meta": ref_meta, "job": job, "transparent": body.transparent,
                "style": styles.normalize(p.get("style"))}

    def submit_prepared(prep: dict, prompt: str, *, enhanced_from: str | None, chain_id: str | None = None,
                        style: dict | None = None, skip_style: bool = False) -> str:
        """Send a prepared generation to the engine with its final prompt. On failure the pinned files and run number are released.
        The art style (default: the one pinned in `prep`) is added first and the transparency wrap goes around the styled text;
        `skip_style` leaves the text unstyled (an enhanced prompt that already carries the style) but the run still records it."""
        pid, n = prep["pid"], prep["n"]
        style = styles.normalize(prep.get("style") if style is None else style)
        model_prompt = styles.compose(prompt, style, prep["mode"], skip=skip_style)
        if prep["transparent"]:
            model_prompt = wrap_transparent(model_prompt)
        try:
            job_id = engine.submit({**prep["job"], "prompt": model_prompt})
        except ValueError as exc:
            store.discard_run(pid, n)
            store.release_run(pid, n)
            raise GenSubmitError(409 if str(exc).startswith("max ") else 400, str(exc))
        except RuntimeError as exc:
            store.discard_run(pid, n)
            store.release_run(pid, n)
            raise GenSubmitError(503, str(exc))
        settled[job_id] = asyncio.Event()
        jobmap[job_id] = {"project_id": pid, "run": n, "parent_run": prep["parent"], "prompt": prompt, "mode": prep["mode"],
                         "enhanced_from": enhanced_from, "refs": prep["ref_meta"], "transparent": prep["transparent"],
                         "model_prompt": model_prompt, "style": styles.record(style), "chain_id": chain_id, "chain_step": "generate" if chain_id else None}
        return job_id

    @api.post("/projects/{pid}/runs", status_code=202)
    async def create_run(pid: str, body: RunCreate):
        # async on purpose: engine events are delivered via the loop, so the jobmap entry made below is
        # always in place before the first event for the job is processed.
        await settle(pid)  # a just-finished run (or enhance) must have been saved before we read the project
        p = need_project(pid)
        prep = await prepare_generate(pid, p, body)
        enhanced = p.get("enhanced") if body.use_enhanced else None
        enhanced_from = body.enhanced_from or (enhanced["original"] if enhanced else None)
        try:
            skip_style = bool(enhanced) and styles.same((enhanced.get("style") or {"id": styles.NONE_ID}), prep["style"])
            job_id = submit_prepared(prep, body.prompt, enhanced_from=enhanced_from, skip_style=skip_style)
        except GenSubmitError as exc:
            raise HTTPException(exc.status, exc.message)
        if enhanced:
            clear_enhanced(pid)  # consumed: the run was accepted
        return {"job_id": job_id, "run_reserved": prep["n"], "mode": prep["mode"]}

    @api.post("/projects/{pid}/enhance-and-generate", status_code=202)
    async def enhance_and_generate(pid: str, body: ChainCreate):
        """Enhance the prompt, then generate with the rewrite, as one unit. Everything the run needs (parent, references, run
        number, transparent flag) is pinned now; the generate job is only enqueued when the enhance job is done."""
        await settle(pid)
        p = need_project(pid)
        cfg = settings.get()
        language = cfg["pe_language"] if body.language is None else body.language
        thinking = cfg["pe_thinking"] if body.thinking is None else body.thinking
        if language not in settings_mod.PE_LANGUAGES:
            raise HTTPException(400, f"language must be one of {list(settings_mod.PE_LANGUAGES)}, got {language!r}")
        prep = await prepare_generate(pid, p, RunCreate(prompt=body.prompt, seed=body.seed, transparent=body.transparent))
        mode = "t2i" if prep["parent"] is None else "i2i"
        try:
            if cfg["pe_variant"] == "heretic":
                name = f"{mode}-heretic-8bit"
                if models.pe_status(name)["state"] != "ready":
                    raise HTTPException(409, f"prompt enhancer {name!r} is not installed; copy {models.PE_MODELS[name].folder}/ into {models.models_dir()}")
            else:
                name = models.pe_name(mode, cfg["pe_quant"])
                if models.pe_status(name)["state"] != "ready":
                    raise HTTPException(409, f"prompt enhancer {name!r} is not downloaded")
            chain_id = uuid.uuid4().hex[:10]
            style = prep["style"]
            job = {"kind": "enhance", "prompt": body.prompt + styles.enhancer_hint(style, mode), "mode": mode, "quant": cfg["pe_quant"], "thinking": thinking,
                   "language": language, "max_new_tokens": cfg["pe_max_new_tokens"], "ref_image_path": prep["job"]["ref_image_path"]}
            try:
                job_id = engine.submit(job)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            except RuntimeError as exc:
                raise HTTPException(503, str(exc))
        except HTTPException:
            chain_cleanup(prep)
            raise
        chains[chain_id] = {"id": chain_id, "prep": prep, "original": body.prompt, "style": style}
        enhance_map[job_id] = {"project_id": pid, "mode": mode, "chain_id": chain_id, "chain_step": "enhance",
                               "prompt": body.prompt, "style": style}
        settled[job_id] = asyncio.Event()
        return {"chain_id": chain_id, "enhance_job_id": job_id, "mode": "t2i" if prep["parent"] is None else "edit", "run_reserved": prep["n"]}

    @api.delete("/projects/{pid}/enhanced", status_code=204)
    def discard_enhanced(pid: str):
        need_project(pid)
        clear_enhanced(pid)
        return Response(status_code=204)

    # ---- per-project reference images ----
    def ref_json(pid: str, r: dict, edit_idx: int | None, new_idx: int | None) -> dict:
        base = f"/api/projects/{pid}/refs/{r['id']}"
        return {**r, "image_url": f"{base}/image", "thumb_url": f"{base}/thumb", "image_index_edit": edit_idx, "image_index_new": new_idx}

    def refs_json(pid: str) -> list[dict]:
        out, k = [], 0
        for r in store.list_refs(pid):
            if r["enabled"]:
                k += 1
                out.append(ref_json(pid, r, k + 1, k))  # edit: current image is Image 1, refs follow; new image: refs start at 1
            else:
                out.append(ref_json(pid, r, None, None))
        return out

    def announce_refs(pid: str) -> None:
        broadcast("refs", {"project_id": pid, "refs": refs_json(pid)})

    def need_ref(pid: str, rid: str) -> dict:
        need_project(pid)
        r = next((x for x in store.list_refs(pid) if x["id"] == rid), None)
        if r is None:
            raise HTTPException(404, "reference not found")
        return r

    @api.get("/projects/{pid}/refs")
    def list_refs(pid: str):
        need_project(pid)
        return refs_json(pid)

    @api.post("/projects/{pid}/refs", status_code=201)
    async def add_ref(pid: str, request: Request, name: str | None = None):
        need_project(pid)
        if len(store.list_refs(pid)) >= MAX_REFS_PER_PROJECT:
            raise HTTPException(409, f"a project can hold at most {MAX_REFS_PER_PROJECT} reference images")
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"the upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            image = await asyncio.to_thread(process_ref, bytes(data))
        except UploadError as exc:
            raise HTTPException(exc.status, exc.message)
        max_refs = settings.get()["max_refs"]
        fits = sum(1 for r in store.list_refs(pid) if r["enabled"]) < max_refs
        ref = await asyncio.to_thread(store.add_ref, pid, image, name, fits)
        if ref is None:
            raise HTTPException(404, "project not found")
        announce_refs(pid)
        item = next(x for x in refs_json(pid) if x["id"] == ref["id"])
        if not fits:
            item["note"] = f"Added disabled: max {max_refs} references enabled; disable one to enable this"
        return item

    @api.patch("/projects/{pid}/refs/{rid}")
    def patch_ref(pid: str, rid: str, body: RefPatch):
        r = need_ref(pid, rid)
        if body.name is not None and (not body.name.strip() or len(body.name) > 80):
            raise HTTPException(400, "name must be a non-empty string of at most 80 characters")
        if body.enabled and not r["enabled"]:
            max_refs = settings.get()["max_refs"]
            if sum(1 for x in store.list_refs(pid) if x["enabled"]) >= max_refs:
                raise HTTPException(409, f"max {max_refs} references enabled; disable one first")
        store.update_ref(pid, rid, enabled=body.enabled, name=body.name.strip() if body.name is not None else None, order=body.order)
        announce_refs(pid)
        return next(x for x in refs_json(pid) if x["id"] == rid)

    @api.delete("/projects/{pid}/refs/{rid}", status_code=204)
    def delete_ref(pid: str, rid: str):
        need_ref(pid, rid)
        store.delete_ref(pid, rid)
        announce_refs(pid)
        return Response(status_code=204)

    @api.get("/projects/{pid}/refs/{rid}/image")
    def ref_image(pid: str, rid: str):
        need_ref(pid, rid)
        return FileResponse(store.ref_path(pid, rid), media_type="image/png")

    @api.get("/projects/{pid}/refs/{rid}/thumb")
    def ref_thumb(pid: str, rid: str):
        need_ref(pid, rid)
        return FileResponse(store.ref_thumb_path(pid, rid), media_type="image/webp")

    @api.get("/projects/{pid}/runs/{n}/refs/{i}")
    def run_ref(pid: str, n: int, i: int):
        f = store.pinned_ref_path(pid, n, i) if store.valid_pid(pid) else None
        if f is None or store.get_run(pid, n) is None or not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f, media_type="image/png")

    # ---- upload: start an empty project from an image ----
    uploading: set[str] = set()

    @api.get("/sizes")
    def get_sizes(w: int | None = Query(None, ge=1), h: int | None = Query(None, ge=1)):
        res = settings.get()["resolution"]
        return {"resolution": res, "presets": sizes.presets_for(res), "closest": sizes.closest(w, h) if w and h else None}

    @api.post("/projects/{pid}/upload", status_code=201)
    async def upload(pid: str, request: Request, preset: str | None = None, crop: str | None = None, src: str | None = None):
        # The body is the raw image bytes (Content-Type is ignored; PIL sniffs the format). Options ride in the query string.
        await settle(pid)
        p = need_project(pid)
        if preset is not None and preset not in sizes.PRESETS:
            raise HTTPException(400, f"unknown preset {preset!r}; expected one of {list(sizes.PRESETS)}")
        box = None
        if crop is not None:
            try:
                box = tuple(float(v) for v in crop.split(","))
                if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                    raise ValueError
            except ValueError:
                raise HTTPException(400, "crop must be 'x,y,w,h' in source pixels with positive w and h")
        src_size = None
        if src is not None:
            try:
                sw_, sh_ = (int(v) for v in src.lower().split("x"))
                if sw_ < 1 or sh_ < 1:
                    raise ValueError
                src_size = (sw_, sh_)
            except ValueError:
                raise HTTPException(400, "src must be 'WxH' (the original image size in pixels)")
        busy = any(
            info["project_id"] == pid and (snap := engine.get(jid)) and snap["state"] not in FINAL_STATES
            for jid, info in [*jobmap.items(), *enhance_map.items()]
        )
        if p["current_run"] is not None or store.list_runs(pid) or busy or pid in uploading:
            raise HTTPException(409, "an upload can only start an empty project (this one already has images or jobs)")
        uploading.add(pid)
        try:
            data = bytearray()
            async for chunk in request.stream():  # count bytes as they arrive; never trust Content-Length
                data += chunk
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"the upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            if not data:
                raise HTTPException(400, "empty upload")
            res = settings.get()["resolution"]
            try:
                image, meta = await asyncio.to_thread(process_upload, bytes(data), res, preset, box, src_size)
            except UploadError as exc:
                raise HTTPException(exc.status, exc.message)
            n = store.reserve_run(pid)
            path = store.image_path(pid, n)

            def save() -> dict | None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
                image.save(tmp, format="PNG")
                os.replace(tmp, path)
                record = {
                    "run": n, "project_id": pid, "prompt": "", "mode": "upload", "parent_run": None,
                    "seed": None, "steps": None, "turbo": None, "image_model": None, "text_encoder": None,
                    "width": image.width, "height": image.height, "created_at": time.time(), "gen_s": None, "load_s": None,
                    "job_id": None, "enhanced_from": None, "transparent": False, "model_prompt": "", "refs": [], "reference_resolution": None, "ref_detail": None, **meta,
                }
                try:
                    return store.save_run(pid, record)
                except Exception:  # noqa: BLE001
                    log.exception("saving upload %s/%s failed", pid, n)
                    return None

            saved = await asyncio.to_thread(save)
            if saved is None:
                store.discard_run(pid, n)
                raise HTTPException(500, "could not save the upload")
            broadcast("run", {"project_id": pid, "run": run_json(saved)})
            return run_json(saved)
        finally:
            uploading.discard(pid)

    @api.post("/projects/{pid}/enhance", status_code=202)
    async def enhance(pid: str, body: EnhanceCreate):
        await settle(pid)  # a just-finished run must be the reference
        p = need_project(pid)
        cfg = settings.get()
        if not body.prompt.strip():
            raise HTTPException(400, "prompt must be a non-empty string")
        quant = cfg["pe_quant"] if body.quant is None else body.quant
        language = cfg["pe_language"] if body.language is None else body.language
        thinking = cfg["pe_thinking"] if body.thinking is None else body.thinking
        if quant not in settings_mod.PE_QUANTS:
            raise HTTPException(400, f"quant must be one of {list(settings_mod.PE_QUANTS)}, got {quant!r}")
        if language not in settings_mod.PE_LANGUAGES:
            raise HTTPException(400, f"language must be one of {list(settings_mod.PE_LANGUAGES)}, got {language!r}")
        parent = p["current_run"]  # pinned at submit, like runs
        if parent is not None and not store.image_path(pid, parent).is_file():
            raise HTTPException(409, f"current run {parent} has no image on disk")
        mode = "t2i" if parent is None else "i2i"
        if cfg["pe_variant"] == "heretic":
            name = f"{mode}-heretic-8bit"  # replaces the official rewriter of this mode whatever the quant
            if models.pe_status(name)["state"] != "ready":
                raise HTTPException(409, f"prompt enhancer {name!r} is not installed; copy {models.PE_MODELS[name].folder}/ into {models.models_dir()}")
        else:
            name = models.pe_name(mode, quant)
            if models.pe_status(name)["state"] != "ready":
                raise HTTPException(409, f"prompt enhancer {name!r} is not downloaded")
        style = styles.normalize(p.get("style"))
        job = {
            "kind": "enhance", "prompt": body.prompt + styles.enhancer_hint(style, mode), "mode": mode, "quant": quant, "thinking": thinking,
            "language": language, "max_new_tokens": cfg["pe_max_new_tokens"],
            "ref_image_path": str(store.image_path(pid, parent)) if parent is not None else None,
        }
        try:
            job_id = engine.submit(job)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        enhance_map[job_id] = {"project_id": pid, "mode": mode, "chain_id": None, "chain_step": None,
                               "prompt": body.prompt, "style": style}
        settled[job_id] = asyncio.Event()
        return {"job_id": job_id, "mode": mode}

    def run_file(pid: str, n: int, path: Path, media: str):
        if not store.valid_pid(pid) or store.get_run(pid, n) is None or not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type=media)

    @api.get("/projects/{pid}/runs/{n}/image")
    def run_image(pid: str, n: int):
        return run_file(pid, n, store.image_path(pid, n), "image/png")

    @api.get("/projects/{pid}/runs/{n}/thumb")
    def run_thumb(pid: str, n: int):
        return run_file(pid, n, store.thumb_path(pid, n), "image/webp")

    # ---- jobs / events -------------------------------------------------------------------------
    _RUNNING_STATES = ("loading", "running", "paused", "cancelling")

    @api.get("/jobs")
    def list_jobs():
        snaps = sorted(engine.jobs(), key=lambda s: s["created_at"])
        running = [s for s in snaps if s["state"] in _RUNNING_STATES]
        waiting = [s for s in snaps if s["state"] not in FINAL_STATES and s["state"] not in _RUNNING_STATES]
        # a queued job with no position is being dequeued right now: it goes first
        waiting.sort(key=lambda s: (s.get("queue_position") or 0, s["created_at"]))
        recent = [s for s in snaps if s["state"] in FINAL_STATES][-RECENT_FINISHED_JOBS:]
        return [enrich_job(s) for s in running + waiting + recent]

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return {"cancelled": bool(engine.cancel(job_id))}

    def _control(action: str):
        def handler(job_id: str):
            try:
                changed = bool(getattr(engine, action)(job_id))
            except InvalidTransition as exc:
                raise HTTPException(409, str(exc))
            except UnknownJob:
                raise HTTPException(404, "job not found")
            snap = engine.get(job_id)
            if snap is None:
                raise HTTPException(404, "job not found")
            return {**enrich_job(snap), "changed": changed}

        handler.__name__ = f"{action}_job"
        api.post(f"/jobs/{{job_id}}/{action}")(handler)

    for _action in ("pause", "resume", "hold", "release"):
        _control(_action)

    @api.get("/jobs/{job_id}/log")
    def job_log(job_id: str):
        try:
            return {"lines": engine.job_log(job_id)}
        except UnknownJob:
            raise HTTPException(404, "job not found")

    @api.get("/logs/server")
    def server_log(tail: int = Query(500)):
        return {"lines": engine.server_log(min(max(tail, 1), 2000))}

    @api.get("/events")
    async def sse(request: Request, serverlog: int = 0):
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        subscribers.add(q)
        if serverlog:  # opt-in: server-log lines are chatty and only the Jobs log view wants them
            server_subs.add(q)

        async def stream():
            try:
                yield ": connected\n\n"
                r = resident_event()
                yield f"event: model\ndata: {json.dumps(r)}\n\n"
                while True:
                    try:
                        yield await asyncio.wait_for(q.get(), PING_SECONDS)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                    if q not in subscribers:
                        return
            finally:
                subscribers.discard(q)
                server_subs.discard(q)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # ---- settings / models ---------------------------------------------------------------------
    @api.get("/settings")
    def get_settings():
        return settings.get()

    @api.put("/settings")
    def put_settings(partial: dict):
        try:
            return settings.update(partial)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @api.get("/model")
    def resident_model():
        r = resident_event()
        return {"state": r["state"], "key": r["key"]}

    @api.get("/models")
    def list_models():
        return image_models() + te_models() + pe_models()

    @api.post("/models/pe/{name}/download", status_code=202)
    def download_pe(name: str):
        if name not in models.PE_MODELS:
            raise HTTPException(404, f"unknown prompt enhancer {name!r}")
        if models.PE_MODELS[name].source == "local":
            raise HTTPException(409, f"{name!r} is a local model and cannot be downloaded; copy {models.PE_MODELS[name].folder}/ into {models.models_dir()}")
        return {"started": models.start_pe_download(name)}

    @api.post("/models/image/{name}/download", status_code=202)
    def download_image(name: str):
        if name not in models.IMAGE_MODELS:
            raise HTTPException(404, f"unknown image model {name!r}")
        return {"started": models.start_image_download(name)}

    @api.post("/models/text_encoder/{name}/download", status_code=202)
    def download_text_encoder(name: str):
        te = models.TEXT_ENCODERS.get(name)
        if te is None:
            raise HTTPException(404, f"unknown text encoder {name!r}")
        if te.source == "local":
            raise HTTPException(409, f"{te.label} is a local model and cannot be downloaded. {models.te_status(name).get('hint', '')}".strip())
        return {"started": models.start_te_download(name)}

    def delete_idle(delete, name: str) -> int:
        with engine.maintenance():
            return delete(name)

    async def delete_part(delete, name: str) -> dict:
        """Delete a downloaded part with the engine idle and nothing resident; broadcasts on the loop thread."""
        try:
            freed = await asyncio.to_thread(delete_idle, delete, name)  # maintenance() blocks until the worker drops
        except (EngineBusy, models.DeleteRefused) as exc:
            raise HTTPException(409, str(exc))
        broadcast("download", image_models() + te_models() + pe_models())
        return {"deleted": True, "freed_bytes": freed}

    @api.delete("/models/image/{name}")
    async def delete_image(name: str):
        if name not in models.IMAGE_MODELS:
            raise HTTPException(404, f"unknown image model {name!r}")
        return await delete_part(models.delete_image, name)

    @api.delete("/models/text_encoder/{name}")
    async def delete_text_encoder(name: str):
        if name not in models.TEXT_ENCODERS:
            raise HTTPException(404, f"unknown text encoder {name!r}")
        return await delete_part(models.delete_te, name)

    @api.delete("/models/pe/{name}")
    async def delete_pe(name: str):
        if name not in models.PE_MODELS:
            raise HTTPException(404, f"unknown prompt enhancer {name!r}")
        return await delete_part(models.delete_pe, name)

    app.include_router(api)
    app.router.routes.extend(ca.ROUTES)  # /ca and /ca/rootCA.crt, ahead of the static catch-all mount
    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
