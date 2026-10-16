"""Model parts registry: image models and text encoders (composed per job), prompt enhancers; offline status
and background downloads.

Nothing here touches MLX, so every function is safe to call from any thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError

log = logging.getLogger(__name__)

# Overrides the HF hub cache directory; unset means huggingface_hub's default (honours HF_HOME / HF_HUB_CACHE).
CACHE_ENV = "WEBSTUDIO_HF_CACHE"


@dataclass(frozen=True)
class ImageModel:
    """The image model part: transformer + VAE (+ processor), taken from an HF snapshot."""

    name: str
    repo: str
    revision: str
    registry_key: str  # mflux AVAILABLE_MODELS key whose sampling config matches this transformer
    approx_gb: float
    label: str

    # Everything a composed folder takes from the image part (model_index.json/scheduler only if the repo has them).
    patterns: tuple[str, ...] = (
        "model_index.json",
        "scheduler/*.json",
        "processor/**",
        "vae/*.json",
        "vae/*.safetensors",
        "transformer/*.json",
        "transformer/*.safetensors",
    )


@dataclass(frozen=True)
class TextEncoder:
    """The text-encoder part (Qwen3-VL incl. vision tower): an HF snapshot subfolder or a local converted folder."""

    name: str
    label: str
    approx_gb: float
    source: str  # "hf" (downloadable) | "local" (a folder under models_dir(), never downloaded)
    repo: str = ""
    revision: str = ""
    folder: str = ""  # local: <models_dir()>/<folder>/text_encoder/
    patterns: tuple[str, ...] = ("text_encoder/*.json", "text_encoder/*.safetensors")


_VANCH007_8BIT = ("vanch007/Qwen-Image-2.1-Heretic-mlx-8bit", "6b81cbc376cb23069cc92df6084cf0f8cb9c9b88")

IMAGE_MODELS: dict[str, ImageModel] = {
    # vanch007's transformer/VAE are 8-bit quantizations of the official Qwen-Image-2.1 weights (Heretic only changes
    # the text encoder), so this part is the official image model.
    "official-8bit": ImageModel(
        "official-8bit", *_VANCH007_8BIT, "qwen-image-2.1-heretic-8bit", 9.6, "Qwen-Image-2.1 (official), 8-bit"
    ),
}
TEXT_ENCODERS: dict[str, TextEncoder] = {
    "heretic-8bit": TextEncoder(
        "heretic-8bit", "Heretic (abliterated), 8-bit", 8.8, "hf", repo=_VANCH007_8BIT[0], revision=_VANCH007_8BIT[1]
    ),
    # Converted by tools/export_text_encoder.py from Qwen/Qwen-Image-2.1@d26bb61; see its PROVENANCE.md.
    "official-8bit": TextEncoder(
        "official-8bit", "Official, 8-bit", 8.8, "local", folder="Qwen-Image-2.1-TextEncoder-MLX-8bit"
    ),
}
DEFAULT_IMAGE_MODEL = "official-8bit"
DEFAULT_TEXT_ENCODER = "heretic-8bit"
COMPOSE_DIR_ENV = "WEBSTUDIO_COMPOSE_DIR"
COMPOSE_IMAGE_PARTS = ("transformer", "vae", "processor")
COMPOSE_OPTIONAL_PARTS = ("model_index.json", "scheduler")

# Mirrors mflux.models.qwen21.qwen21_lora.Qwen21LoRA (REPO/FILENAME/REVISION) at the pinned fork commit.
TURBO_LORA_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_LORA_FILENAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
TURBO_LORA_REVISION = "bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13"


class ModelNotReady(RuntimeError):
    """Raised when a model part (or the turbo LoRA) is not fully present locally."""


class DeleteRefused(RuntimeError):
    """Base for a delete_* refusal on a known part (a conflict, unlike the ValueError for an unknown name)."""


class LocalModelError(DeleteRefused):
    """Raised when asked to delete a local (copied-in, never downloaded) part; only HF-cache parts are deletable."""


class ModelBusy(DeleteRefused):
    """Raised when asked to delete a part while a download touching its repo is running."""


def cache_dir() -> str | None:
    return os.environ.get(CACHE_ENV) or None


def _image(name: str) -> ImageModel:
    try:
        return IMAGE_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown image model {name!r}; expected one of {sorted(IMAGE_MODELS)}") from None


def _te(name: str) -> TextEncoder:
    try:
        return TEXT_ENCODERS[name]
    except KeyError:
        raise ValueError(f"unknown text encoder {name!r}; expected one of {sorted(TEXT_ENCODERS)}") from None


def _storage_folder(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(cache_dir() or HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _cached_snapshot(repo: str, revision: str, patterns: tuple[str, ...]) -> Path | None:
    """The local snapshot dir if every file matching `patterns` is cached (per the tree cache). Never hits the network."""
    try:
        return Path(
            snapshot_download(
                repo, revision=revision, allow_patterns=list(patterns), cache_dir=cache_dir(), local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return None


def image_local_dir(name: str) -> Path | None:
    """Snapshot root holding complete transformer/, vae/ and processor/ for the image model, or None (offline)."""
    m = _image(name)
    root = _cached_snapshot(m.repo, m.revision, m.patterns)
    # snapshot_download only spots partial snapshots when its tree cache exists, so check the shards too.
    ok = root is not None and all(_shards_present(root / p) for p in ("transformer", "vae"))
    return root if ok and (root / "processor" / "tokenizer.json").is_file() else None


def te_local_dir(name: str) -> Path | None:
    """The complete text_encoder/ folder for this encoder, or None (offline)."""
    te = _te(name)
    if te.source == "local":
        path = models_dir() / te.folder / "text_encoder"
    else:
        root = _cached_snapshot(te.repo, te.revision, te.patterns)
        if root is None:
            return None
        path = root / "text_encoder"
    return path if _shards_present(path) else None


def turbo_lora_path() -> Path | None:
    """Return the cached turbo LoRA file, or None. Never hits the network."""
    hit = try_to_load_from_cache(
        TURBO_LORA_REPO, TURBO_LORA_FILENAME, revision=TURBO_LORA_REVISION, cache_dir=cache_dir()
    )
    return Path(hit) if isinstance(hit, str) else None


def resolve_parts(image_model: str, text_encoder: str, turbo: bool) -> tuple[Path, Path, Path | None]:
    """(image snapshot root, text_encoder dir, turbo LoRA or None) for loading; raises ModelNotReady with a user message."""
    if _active(_image_key(image_model)):
        raise ModelNotReady(f"image model {image_model!r} is still downloading")
    root = image_local_dir(image_model)
    if root is None:
        raise ModelNotReady(f"image model {image_model!r} is not downloaded")
    te = _te(text_encoder)
    if te.source != "local" and _active(_te_key(text_encoder)):
        raise ModelNotReady(f"text encoder {text_encoder!r} is still downloading")
    te_dir = te_local_dir(text_encoder)
    if te_dir is None:
        if te.source == "local":
            raise ModelNotReady(f"text encoder {text_encoder!r} is not installed; copy {te.folder}/ into {models_dir()}")
        raise ModelNotReady(f"text encoder {text_encoder!r} is not downloaded")
    lora = turbo_lora_path() if turbo else None
    if turbo and lora is None:
        raise ModelNotReady(f"turbo LoRA is not downloaded (download the image model {image_model!r})")
    return root, te_dir, lora


def default_compose_dir() -> Path:
    env = os.environ.get(COMPOSE_DIR_ENV)
    if env:
        return Path(env).expanduser()
    from webstudio.store import default_data_dir

    return default_data_dir() / "compose"


def _compose_targets(image_root: Path, te_dir: Path) -> dict[str, Path]:
    targets = {p: image_root / p for p in COMPOSE_IMAGE_PARTS}
    targets.update({p: image_root / p for p in COMPOSE_OPTIONAL_PARTS if (image_root / p).exists()})
    targets["text_encoder"] = te_dir
    return targets


def _compose_valid(dest: Path, targets: dict[str, Path]) -> bool:
    try:
        entries = {e.name for e in dest.iterdir()}
    except OSError:
        return False
    if entries != set(targets):
        return False
    for name, target in targets.items():
        link = dest / name
        if not link.is_symlink() or os.readlink(link) != str(target) or not link.exists():
            return False
    return all(_shards_present(dest / p) for p in ("transformer", "vae", "text_encoder")) and (
        dest / "processor" / "tokenizer.json"
    ).is_file()


def compose(image_model: str, text_encoder: str, image_root: Path, te_dir: Path, parent: Path | None = None) -> Path:
    """Return <parent>/<image>__<te>/: symlinks transformer/, vae/, processor/ -> the image part, text_encoder/ -> the
    encoder part. Reused when already correct; otherwise rebuilt in a temp dir and renamed into place. Validated."""
    import shutil
    import uuid

    parent = Path(parent) if parent is not None else default_compose_dir()
    dest = parent / f"{image_model}__{text_encoder}"
    targets = _compose_targets(Path(image_root), Path(te_dir))
    if _compose_valid(dest, targets):
        return dest
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    for name, target in targets.items():
        os.symlink(target, tmp / name)
    if not _compose_valid(tmp, targets):
        shutil.rmtree(tmp)  # only symlinks inside; rmtree unlinks them without following
        raise ModelNotReady(f"cannot compose {dest.name}: a part folder is incomplete")
    if dest.exists() or dest.is_symlink():
        old = parent / f".{dest.name}.old-{uuid.uuid4().hex[:8]}"
        os.rename(dest, old)
        os.rename(tmp, dest)
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old)
        else:
            old.unlink()
    else:
        os.rename(tmp, dest)
    log.info("composed %s (image %s, text encoder %s)", dest, image_root, te_dir)
    return dest


class _Download:
    def __init__(self, key: str, plan: list[tuple[str, str, list[str] | None, str | None]]):
        self.key = key
        # Each step is (repo, revision, allow_patterns, None) for a snapshot or (repo, revision, None, filename).
        self.plan = plan
        self.thread: threading.Thread | None = None
        self.total: int | None = None
        self.files: list[tuple[str, str, str, int]] = []  # (repo, revision, filename, size)
        self.error: str | None = None


_lock = threading.Lock()
_downloads: dict[str, _Download] = {}


def _alive(d: _Download | None) -> bool:
    return bool(d and d.thread and d.thread.is_alive())


def _active(key: str) -> bool:
    with _lock:
        return _alive(_downloads.get(key))


def _progress(key: str) -> tuple[bool, str | None, list[tuple[str, str, str, int]], int | None]:
    with _lock:
        d = _downloads.get(key)
        active = bool(d and d.thread and d.thread.is_alive())
        return active, (d.error if d else None), (list(d.files) if d else []), (d.total if d else None)


def _image_key(name: str) -> str:
    return "image:" + name


def _te_key(name: str) -> str:
    return "te:" + name


def is_image_downloading(name: str) -> bool:
    _image(name)
    return _active(_image_key(name))


def is_te_downloading(name: str) -> bool:
    return _te(name).source != "local" and _active(_te_key(name))


def image_status(name: str) -> dict:
    """Offline status of an image model part.

    {"kind": "image", "name", "label", "approx_gb", "state": "downloading", "bytes_done", "bytes_total"}
    | {..., "state": "ready", "path", "turbo_lora_ready"[, "error"]}
    | {..., "state": "not_downloaded", "turbo_lora_ready", "error"}
    """
    m = _image(name)
    base = {"kind": "image", "name": name, "label": m.label, "approx_gb": m.approx_gb}
    active, error, files, total = _progress(_image_key(name))
    if active:
        return {**base, "state": "downloading", "bytes_done": _bytes_on_disk(files), "bytes_total": total}
    root, lora_ready = image_local_dir(name), turbo_lora_path() is not None
    if root is not None:
        out = {**base, "state": "ready", "path": str(root), "turbo_lora_ready": lora_ready}
        return {**out, "error": error} if error else out
    return {**base, "state": "not_downloaded", "turbo_lora_ready": lora_ready, "error": error}


def te_status(name: str) -> dict:
    """Offline status of a text-encoder part.

    {"kind": "text_encoder", "name", "label", "approx_gb", "source", "state": "ready", "path"}
    | hf: "downloading" (+bytes_done/bytes_total) | "not_downloaded" (+error)
    | local: "not_installed" (+hint, error None)
    """
    te = _te(name)
    base = {"kind": "text_encoder", "name": name, "label": te.label, "approx_gb": te.approx_gb, "source": te.source}
    if te.source == "local":
        path = te_local_dir(name)
        if path is not None:
            return {**base, "state": "ready", "path": str(path)}
        return {**base, "state": "not_installed", "error": None,
                "hint": f"Copy the folder {te.folder}/ (with text_encoder/ inside) into {models_dir()}"}  # fmt: skip
    active, error, files, total = _progress(_te_key(name))
    if active:
        return {**base, "state": "downloading", "bytes_done": _bytes_on_disk(files), "bytes_total": total}
    path = te_local_dir(name)
    if path is not None:
        out = {**base, "state": "ready", "path": str(path)}
        return {**out, "error": error} if error else out
    return {**base, "state": "not_downloaded", "error": error}


def start_image_download(name: str) -> bool:
    """Download the image part (transformer/vae/processor) plus the turbo LoRA; idempotent (False if running).

    Coexists with a text-encoder download from the same repo: each fetches only its own allow_patterns.
    """
    m = _image(name)
    plan = [
        (m.repo, m.revision, list(m.patterns), None),
        (TURBO_LORA_REPO, TURBO_LORA_REVISION, None, TURBO_LORA_FILENAME),
    ]
    return _start(_image_key(name), plan, lambda: image_local_dir(name) is not None)


def start_te_download(name: str) -> bool:
    """Download an HF text-encoder part (text_encoder/* only); idempotent. Local encoders can't be downloaded (False)."""
    te = _te(name)
    if te.source == "local":
        return False
    plan = [(te.repo, te.revision, list(te.patterns), None)]
    return _start(_te_key(name), plan, lambda: te_local_dir(name) is not None)


def _bytes_on_disk(files: list[tuple[str, str, str, int]]) -> int:
    """Bytes of finished files plus partial blobs; an estimate good enough for a progress bar."""
    done = 0
    for repo, revision, filename, size in files:
        if isinstance(try_to_load_from_cache(repo, filename, revision=revision, cache_dir=cache_dir()), str):
            done += size
    for repo in {f[0] for f in files}:
        blobs = _storage_folder(repo) / "blobs"
        if not blobs.is_dir():
            continue
        for partial in blobs.glob("*.incomplete"):
            try:
                st = partial.stat()
            except OSError:
                continue
            # Count allocated bytes, not st_size, in case the downloader preallocates a sparse file.
            done += min(st.st_size, st.st_blocks * 512)
    return done


def _start(key: str, plan, verify: Callable[[], bool]) -> bool:
    with _lock:
        d = _downloads.get(key)
        if d and d.thread and d.thread.is_alive():
            return False
        d = _Download(key, plan)
        d.thread = threading.Thread(target=_run_download, args=(d, verify), name=f"download-{key}", daemon=True)
        _downloads[key] = d
        d.thread.start()
    return True


def _run_download(d: _Download, verify: Callable[[], bool]) -> None:
    try:
        # Size everything first so status() can report a real total.
        files = []
        for repo, revision, patterns, filename in d.plan:
            if filename is None:
                plan = snapshot_download(
                    repo, revision=revision, allow_patterns=patterns, cache_dir=cache_dir(), dry_run=True
                )
                files += [(repo, revision, f.filename, f.file_size or 0) for f in plan]
            else:
                info = hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir(), dry_run=True)
                files.append((repo, revision, filename, info.file_size or 0))
        with _lock:
            d.files = files
            d.total = sum(f[3] for f in files)
        log.info("downloading %s: %d files, %.2f GB", d.key, len(files), d.total / 1e9)
        for repo, revision, patterns, filename in d.plan:
            if filename is None:
                snapshot_download(repo, revision=revision, allow_patterns=patterns, cache_dir=cache_dir())
            else:
                hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir())
        if not verify():
            raise RuntimeError("download finished but the snapshot is still incomplete")
    except Exception as exc:  # noqa: BLE001 - any failure must land in status, not kill silently
        log.exception("download of %s failed", d.key)
        with _lock:
            d.error = f"{type(exc).__name__}: {exc}"


def _snapshot_links(snapshots: Path) -> set[str]:
    """Real paths of every symlink target under <repo>/snapshots/ (all revisions)."""
    targets = set()
    for dirpath, dirnames, filenames in os.walk(snapshots):
        for n in filenames + dirnames:
            p = os.path.join(dirpath, n)
            if os.path.islink(p):
                targets.add(os.path.realpath(p))
    return targets


def _delete_cached(repo: str, revision: str, patterns=None, filename: str | None = None) -> int:
    """Remove the files matching `patterns` (or the single `filename`) from repo@revision's HF-cache snapshot.

    A blob is deleted only once no symlink in any snapshot of the repo still points at it, so a part sharing the
    repo (and identical-content files across parts) stays intact. Returns the bytes actually freed.
    """
    from huggingface_hub.utils import filter_repo_objects

    repo_dir = _storage_folder(repo)
    snap = repo_dir / "snapshots" / revision
    if snap.is_symlink() or not snap.is_dir():
        return 0
    rels = []
    for dirpath, dirnames, filenames in os.walk(snap):
        for n in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            rels.append(Path(dirpath, n).relative_to(snap).as_posix())
    if filename is not None:
        matched = [r for r in rels if r == filename]
    else:
        matched = list(filter_repo_objects(rels, allow_patterns=list(patterns)))
    blobs_dir = os.path.realpath(repo_dir / "blobs")
    freed, blobs = 0, set()
    for rel in matched:
        p = snap / rel
        if p.is_symlink():
            target = os.path.realpath(p)
            # Only blobs of this repo are candidates; a link pointing anywhere else is just unlinked.
            if os.path.dirname(target) == blobs_dir:
                blobs.add(target)
            p.unlink()
        elif p.is_file():
            freed += p.stat().st_size  # no-symlink cache: the snapshot file is the only copy
            p.unlink()
    still_linked = _snapshot_links(repo_dir / "snapshots")
    for blob in blobs - still_linked:
        try:
            st = os.lstat(blob)
        except FileNotFoundError:
            continue
        if not os.path.isfile(blob) or os.path.islink(blob):
            continue
        os.unlink(blob)
        freed += st.st_size
    # Prune directories left empty, bottom-up, up to and including the snapshot dir itself.
    for dirpath, _dirnames, _filenames in os.walk(snap, topdown=False):
        if not os.path.islink(dirpath) and not os.listdir(dirpath):
            os.rmdir(dirpath)
    return freed


def _delete_part(key: str, steps: list[tuple[str, str, tuple[str, ...] | list[str] | None, str | None]]) -> int:
    """Run _delete_cached over `steps` under _lock, refusing (ModelBusy) if any download touches one of the repos."""
    repos = {s[0] for s in steps}
    with _lock:
        # Holding _lock across check and delete keeps _start from beginning a download mid-delete.
        if _alive(_downloads.get(key)):
            raise ModelBusy(f"{key} is downloading")
        # A sibling download on a shared repo could link a blob we are about to delete.
        for other, d in _downloads.items():
            if _alive(d) and any(step[0] in repos for step in d.plan):
                raise ModelBusy(f"{other} is downloading from the same repo")
        freed = sum(_delete_cached(repo, rev, pats, fn) for repo, rev, pats, fn in steps)
    log.info("deleted %s: %.2f GB freed", key, freed / 1e9)
    return freed


def delete_image(name: str) -> int:
    """Delete the downloaded image part and the turbo LoRA from the HF cache; returns bytes freed (0 if absent).

    Raises ValueError for an unknown name and ModelBusy while a download touching its repos runs. A text encoder
    sharing the repo/revision is left intact.
    """
    m = _image(name)
    return _delete_part(
        _image_key(name),
        [(m.repo, m.revision, m.patterns, None), (TURBO_LORA_REPO, TURBO_LORA_REVISION, None, TURBO_LORA_FILENAME)],
    )


def delete_te(name: str) -> int:
    """Delete a downloaded HF text-encoder part (text_encoder/*); returns bytes freed (0 if absent).

    Raises ValueError for an unknown name, LocalModelError for a local encoder and ModelBusy while a download
    touching its repo runs. An image part sharing the repo/revision is left intact.
    """
    te = _te(name)
    if te.source == "local":
        raise LocalModelError(f"text encoder {name!r} is a local folder, not a download; remove {te.folder}/ by hand")
    return _delete_part(_te_key(name), [(te.repo, te.revision, te.patterns, None)])


# ---- prompt-enhancer (rewriter) models -------------------------------------------------------------


@dataclass(frozen=True)
class PEModel:
    name: str  # "<mode>-<quant>", e.g. "t2i-8bit"
    mode: str  # "t2i" | "i2i"
    quant: str  # "8bit" (the only official quantisation offered)
    repo: str
    revision: str
    prompt_repo: str  # official Qwen repo holding system_prompt.txt
    prompt_revision: str
    approx_gb: float
    variant: str = "official"  # "official" | "heretic"
    source: str = "hf"  # "hf" (downloadable) | "local" (a folder under models_dir(), never downloaded)
    folder: str | None = None  # local models: folder name under models_dir()

    @property
    def patterns(self) -> list[str]:
        return [f"{self.quant}/*"]


_PE_REPOS = {
    "t2i": ("prithivMLmods/Qwen-Image-2.1-PE-T2I-MLX", "2d26ac502b16396fdd4b0cc21cc667831a7f6454",
            "Qwen/Qwen-Image-2.1-PE-T2I", "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"),
    "i2i": ("prithivMLmods/Qwen-Image-2.1-PE-I2I-MLX", "75172ef3e7b681eae8f82b9ca20f81885557018d",
            "Qwen/Qwen-Image-2.1-PE-I2I", "72927bc08afc99b7888ceb7d7d51a12db3700bbd"),
}  # fmt: skip
_PE_GB = {"8bit": 10.45}  # per subfolder, from the repo tree listing

PE_MODELS: dict[str, PEModel] = {
    f"{mode}-{quant}": PEModel(f"{mode}-{quant}", mode, quant, repo, rev, prepo, prev, _PE_GB[quant])
    for mode, (repo, rev, prepo, prev) in _PE_REPOS.items()
    for quant in ("8bit",)
}
# Local-only rewriters: converted on disk, not on the Hub. Only T2I/8-bit exists; I2I keeps the official model.
PE_MODELS["t2i-heretic-8bit"] = PEModel(
    "t2i-heretic-8bit", "t2i", "8bit", "", "", "", "", 10.4,
    variant="heretic", source="local", folder="Qwen-Image-2.1-PE-T2I-Heretic-MLX-8bit",
)  # fmt: skip
PE_MODELS["i2i-heretic-8bit"] = PEModel(
    "i2i-heretic-8bit", "i2i", "8bit", "", "", "", "", 10.4,
    variant="heretic", source="local", folder="Qwen-Image-2.1-PE-I2I-Heretic-MLX-8bit",
)  # fmt: skip
PE_HERETIC = "heretic-8bit"  # the "quant" part of the model name / manager key for the Heretic rewriters (T2I and I2I)
PE_HERETIC_T2I = PE_HERETIC  # old name
PE_SYSTEM_PROMPT_FILE = "system_prompt.txt"
MODELS_DIR_ENV = "WEBSTUDIO_MODELS_DIR"


def models_dir() -> Path:
    env = os.environ.get(MODELS_DIR_ENV)
    return Path(env).expanduser() if env else Path(__file__).resolve().parent.parent / "models"


def _shards_present(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").is_file() or not index.is_file():
        return False
    try:
        shards = set(json.loads(index.read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return bool(shards) and all((path / s).is_file() for s in shards)


def pe_name(mode: str, quant: str) -> str:
    name = f"{mode}-{quant}"
    if name not in PE_MODELS:
        raise ValueError(f"unknown prompt enhancer {name!r}; expected one of {sorted(PE_MODELS)}")
    return name


def _pe(name: str) -> PEModel:
    try:
        return PE_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown prompt enhancer {name!r}; expected one of {sorted(PE_MODELS)}") from None


def _pe_key(name: str) -> str:
    return "pe:" + name


def pe_local_dir(name: str) -> Path | None:
    """Return the complete local <snapshot>/<quant> model dir, or None. Never hits the network."""
    pe = _pe(name)
    if pe.source == "local":
        path = models_dir() / pe.folder
        return path if _shards_present(path) and (path / PE_SYSTEM_PROMPT_FILE).is_file() else None
    try:
        root = Path(
            snapshot_download(
                pe.repo, revision=pe.revision, allow_patterns=pe.patterns, cache_dir=cache_dir(), local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return None
    path = root / pe.quant
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file() or not index.is_file():
        return None
    try:
        shards = set(json.loads(index.read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return None
    return path if shards and all((path / s).is_file() for s in shards) else None


def pe_system_prompt_path(name: str) -> Path | None:
    pe = _pe(name)
    if pe.source == "local":
        f = models_dir() / pe.folder / PE_SYSTEM_PROMPT_FILE
        return f if f.is_file() else None
    hit = try_to_load_from_cache(
        pe.prompt_repo, PE_SYSTEM_PROMPT_FILE, revision=pe.prompt_revision, cache_dir=cache_dir()
    )
    return Path(hit) if isinstance(hit, str) else None


def is_pe_downloading(name: str) -> bool:
    pe = _pe(name)
    return pe.source != "local" and _active(_pe_key(pe.name))


def pe_status(name: str) -> dict:
    """Offline status of a prompt enhancer, same shapes as status() but keyed by "name" instead of "variant"."""
    pe = _pe(name)
    if pe.source == "local":
        path = pe_local_dir(name)
        base = {"name": name, "mode": pe.mode, "quant": pe.quant, "variant": pe.variant, "source": "local"}
        if path is not None:
            return {**base, "state": "ready", "path": str(path)}
        return {**base, "state": "not_installed", "error": None, "hint": f"Copy the model folder {pe.folder}/ into {models_dir()}"}
    active, error, files, total = _progress(_pe_key(name))
    if active:
        return {"name": name, "state": "downloading", "bytes_done": _bytes_on_disk(files), "bytes_total": total}
    path, prompt = pe_local_dir(name), pe_system_prompt_path(name)
    if path is not None and prompt is not None:
        return {"name": name, "state": "ready", "path": str(path), "mode": pe.mode, "quant": pe.quant}
    return {"name": name, "state": "not_downloaded", "mode": pe.mode, "quant": pe.quant, "error": error}


def start_pe_download(name: str) -> bool:
    """Background-download the enhancer subfolder plus its official system prompt; idempotent like start_download."""
    pe = _pe(name)
    if pe.source == "local":
        return False  # not on the Hub: it can only be installed by copying the folder into models/
    plan = [
        (pe.repo, pe.revision, pe.patterns, None),
        (pe.prompt_repo, pe.prompt_revision, None, PE_SYSTEM_PROMPT_FILE),
    ]
    return _start(_pe_key(name), plan, lambda: pe_local_dir(name) is not None and pe_system_prompt_path(name) is not None)


def delete_pe(name: str) -> int:
    """Delete a downloaded official enhancer (<quant>/* plus its system_prompt.txt); returns bytes freed (0 if absent).

    Raises ValueError for an unknown name, LocalModelError for a local (Heretic) enhancer and ModelBusy while a
    download touching its repos runs.
    """
    pe = _pe(name)
    if pe.source == "local":
        raise LocalModelError(f"prompt enhancer {name!r} is a local folder, not a download; remove {pe.folder}/ by hand")
    return _delete_part(
        _pe_key(name),
        [(pe.repo, pe.revision, pe.patterns, None), (pe.prompt_repo, pe.prompt_revision, None, PE_SYSTEM_PROMPT_FILE)],
    )


def resolve_pe(mode: str, quant: str) -> tuple[Path, str]:
    """Return (model dir, system prompt text) for loading, or raise ModelNotReady."""
    name = pe_name(mode, quant)
    if _pe(name).source == "local":
        path, prompt = pe_local_dir(name), pe_system_prompt_path(name)
        if path is None or prompt is None:
            raise ModelNotReady(f"prompt enhancer {name!r} is not installed; copy {_pe(name).folder}/ into {models_dir()}")
        return path, prompt.read_text(encoding="utf-8")
    if is_pe_downloading(name):
        raise ModelNotReady(f"prompt enhancer {name!r} is still downloading")
    path, prompt = pe_local_dir(name), pe_system_prompt_path(name)
    if path is None or prompt is None:
        raise ModelNotReady(f"prompt enhancer {name!r} is not downloaded")
    return path, prompt.read_text(encoding="utf-8")
