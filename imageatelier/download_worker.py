"""Resumable, hash-checked HTTP downloads into the standard Hugging Face cache.

Runs in a disposable child so pause/watchdog/shutdown can interrupt blocked I/O.
Only this worker's contiguous partial files are resumed; Xet's sparse files are not.
"""
from __future__ import annotations

import errno
import hashlib
import os
import re
import sys
import threading
import time
from pathlib import Path

import httpx
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

from imageatelier.models import _read, _write


class DownloadError(RuntimeError):
    pass


def transfer(client, url: str, partial: Path, size: int) -> None:
    """Resume a contiguous partial; restart safely if the server ignores Range."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if offset == size:
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[3])) != (offset, size) or not offset <= int(match[2]) < size:
                raise DownloadError("Server returned an invalid resume range")
        elif response.status_code == 200:
            offset = 0
        else:
            raise DownloadError(f"Unexpected HTTP status {response.status_code}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise DownloadError("Server returned compressed model bytes")
        with partial.open("ab" if offset else "wb", buffering=0) as output:
            for chunk in response.iter_raw(chunk_size=1024 * 1024):
                if output.tell() + len(chunk) > size:
                    raise DownloadError("Server returned more bytes than expected")
                output.write(chunk)
    if partial.stat().st_size != size:
        raise DownloadError("Connection ended before the file was complete")


def verified(path: Path, etag: str, heartbeat) -> bool:
    if len(etag) == 64:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    last = time.monotonic()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            if time.monotonic() - last > 1:
                heartbeat()
                last = time.monotonic()
    return digest.hexdigest() == etag


def download_file(client, repo, revision, filename, size, cache, heartbeat):
    cached = try_to_load_from_cache(repo, filename, revision=revision, cache_dir=cache)
    if isinstance(cached, str) and Path(cached).stat().st_size == size:
        return
    metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision), token=False, timeout=15)
    if metadata.commit_hash != revision or metadata.size != size or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", metadata.etag or ""):
        raise DownloadError("Model metadata does not match the pinned download")
    storage = Path(cache or HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    blob = storage / "blobs" / metadata.etag
    blob.parent.mkdir(parents=True, exist_ok=True)
    partial = blob.with_name(blob.name + ".imageatelier.incomplete")
    if not blob.exists():
        transfer(client, metadata.location, partial, size)
        heartbeat()
        if not verified(partial, metadata.etag, heartbeat):
            partial.unlink()
            raise DownloadError("Model checksum mismatch; retrying the file")
        os.replace(partial, blob)
    elif blob.stat().st_size != size:
        raise DownloadError("An existing cached model file has the wrong size")
    destination = storage / "snapshots" / revision / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".imageatelier-link")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(blob, destination.parent))
    os.replace(temporary, destination)


def retryable(exc) -> bool:
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS):
        return False
    response = getattr(exc, "response", None)
    if response is not None and response.status_code in (400, 401, 403, 404):
        return False
    return True


def run(root: Path) -> None:
    job = _read(root / "plan.json")
    cache = job.get("cache")
    files = []
    previous = _read(root / "progress.json").get("files", [])
    planned = False

    def heartbeat():
        visible = files if planned else previous
        _write(root / "progress.json", {"files": visible, "total": sum(f[3] for f in visible), "activity": time.time_ns()})

    heartbeat()
    for repo, revision, patterns, filename in job["plan"]:
        if filename is None:
            plan = snapshot_download(repo, revision=revision, allow_patterns=patterns, cache_dir=cache, dry_run=True)
        else:
            plan = [hf_hub_download(repo, filename, revision=revision, cache_dir=cache, dry_run=True)]
        for item in plan:
            if item.file_size is None:
                raise DownloadError("Model file size is unavailable")
            files.append((repo, revision, item.filename, item.file_size))
        heartbeat()
    planned = True
    heartbeat()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, connect=10)) as client:
        for repo, revision, filename, size in files:
            download_file(client, repo, revision, filename, size, cache, heartbeat)
            heartbeat()


def main() -> int:
    root = Path(sys.argv[1])
    parent = int(os.environ.get("IMAGEATELIER_DOWNLOAD_PARENT", os.getppid()))
    def watch_parent():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)  # Preserve partial files even if the app crashes during blocked I/O.
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        run(root)
    except Exception as exc:
        # Do not expose signed CDN URLs or tokens through status/API errors.
        response = getattr(exc, "response", None)
        message = f"HTTP {response.status_code}" if response is not None else str(exc) if isinstance(exc, (DownloadError, OSError)) else type(exc).__name__
        _write(root / "result.json", {"error": message, "retryable": retryable(exc)})
        return 1
    _write(root / "result.json", {"complete": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
