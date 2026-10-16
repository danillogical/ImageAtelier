"""File-based project/run storage.

Layout under the data dir:
  projects/<pid>/project.json
  projects/<pid>/runs/<NNNN>.png | .json | .thumb.webp
A run exists once its .json exists; the json is written last, after the png and thumbnail.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

THUMB_MAX = 384
_PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RUN_FILE_RE = re.compile(r"^(\d+)\.(?:png|json)$")


def default_data_dir() -> Path:
    env = os.environ.get("IMAGEATELIER_DATA")
    return Path(env).expanduser() if env else Path(__file__).resolve().parent.parent / "data"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Store:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_data_dir()
        self.projects_dir = self.root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._reserved: dict[str, int] = {}  # pid -> highest run number handed out this process

    # ---- paths -------------------------------------------------------------------------------
    def valid_pid(self, pid: str) -> bool:
        return bool(_PID_RE.match(pid))

    def project_dir(self, pid: str) -> Path:
        return self.projects_dir / pid

    def runs_dir(self, pid: str) -> Path:
        return self.project_dir(pid) / "runs"

    def image_path(self, pid: str, n: int) -> Path:
        return self.runs_dir(pid) / f"{n:04d}.png"

    def thumb_path(self, pid: str, n: int) -> Path:
        return self.runs_dir(pid) / f"{n:04d}.thumb.webp"

    def _run_json_path(self, pid: str, n: int) -> Path:
        return self.runs_dir(pid) / f"{n:04d}.json"

    # ---- projects ----------------------------------------------------------------------------
    def get_project(self, pid: str) -> dict | None:
        if not self.valid_pid(pid):
            return None
        p = _read_json(self.project_dir(pid) / "project.json")
        if p is not None:
            p.setdefault("enhanced", None)  # projects written before the field existed
            p.setdefault("style", {"id": "none"})
            if isinstance(p["enhanced"], dict):
                p["enhanced"].setdefault("style", {"id": "none"})  # enhanced values stored before art styles
        return p

    def list_projects(self) -> list[dict]:
        out = []
        for d in self.projects_dir.iterdir():
            p = self.get_project(d.name) if d.is_dir() else None
            if p:
                out.append(p)
        out.sort(key=lambda p: p["updated_at"], reverse=True)
        return out

    def create_project(self, name: str | None = None) -> dict:
        with self._lock:
            if not name or not name.strip():
                used = [
                    int(m.group(1))
                    for p in self.list_projects()
                    if (m := re.fullmatch(r"Untitled (\d+)", p["name"]))
                ]
                name = f"Untitled {max(used, default=0) + 1}"
            now = time.time()
            pid = secrets.token_urlsafe(6)
            project = {"id": pid, "name": name.strip(), "created_at": now, "updated_at": now, "current_run": None, "enhanced": None, "style": {"id": "none"}}
            self.runs_dir(pid).mkdir(parents=True, exist_ok=True)
            _write_json(self.project_dir(pid) / "project.json", project)
            return project

    def update_project(self, pid: str, **fields) -> dict | None:
        """Set fields on project.json and bump updated_at (unless updated_at is given)."""
        with self._lock:
            p = self.get_project(pid)
            if p is None:
                return None
            p.update(fields)
            if "updated_at" not in fields:
                p["updated_at"] = time.time()
            _write_json(self.project_dir(pid) / "project.json", p)
            return p

    def delete_project(self, pid: str) -> bool:
        with self._lock:
            if self.get_project(pid) is None:
                return False
            shutil.rmtree(self.project_dir(pid), ignore_errors=True)
            self._reserved.pop(pid, None)
            return True

    # ---- runs --------------------------------------------------------------------------------
    def reserve_run(self, pid: str) -> int:
        """Hand out the next run number (above anything on disk or already reserved)."""
        with self._lock:
            on_disk = 0
            if self.runs_dir(pid).is_dir():
                for f in self.runs_dir(pid).iterdir():
                    m = _RUN_FILE_RE.match(f.name)
                    if m:
                        on_disk = max(on_disk, int(m.group(1)))
            n = max(on_disk, self._reserved.get(pid, 0)) + 1
            self._reserved[pid] = n
            return n

    def release_run(self, pid: str, n: int) -> None:
        """Undo a reservation that never reached the engine, if it is still the latest one."""
        with self._lock:
            if self._reserved.get(pid) == n:
                self._reserved[pid] = n - 1

    def get_run(self, pid: str, n: int) -> dict | None:
        if not self.valid_pid(pid):
            return None
        return _read_json(self._run_json_path(pid, n))

    def list_runs(self, pid: str) -> list[dict]:
        runs = []
        if self.runs_dir(pid).is_dir():
            for f in self.runs_dir(pid).glob("*.json"):
                if re.fullmatch(r"\d+\.json", f.name):
                    r = _read_json(f)
                    if r:
                        runs.append(r)
        runs.sort(key=lambda r: r["run"])
        return runs

    def make_thumb(self, pid: str, n: int) -> None:
        with Image.open(self.image_path(pid, n)) as im:
            im.thumbnail((THUMB_MAX, THUMB_MAX))
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            tmp = self.thumb_path(pid, n).with_name(f"{n:04d}.thumb.{uuid.uuid4().hex[:8]}.tmp")
            im.save(tmp, format="WEBP", quality=80)
        os.replace(tmp, self.thumb_path(pid, n))

    def save_run(self, pid: str, record: dict) -> dict | None:
        """Thumbnail + run json (last) + project.current_run/updated_at. None if the project is gone."""
        n = record["run"]
        with self._lock:
            if self.get_project(pid) is None:
                return None
        self.make_thumb(pid, n)
        with self._lock:
            if self.get_project(pid) is None:
                return None
            _write_json(self._run_json_path(pid, n), record)
            self.update_project(pid, current_run=n)
        return record

    def discard_run(self, pid: str, n: int) -> None:
        """Remove every file of a run (partial png, tmp files, thumb, json, pinned reference copies)."""
        d = self.runs_dir(pid)
        if not d.is_dir():
            return
        for f in d.glob(f"{n:04d}.*"):
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)

    def sweep_orphans(self) -> int:
        """Remove pinned `.refs` folders and `.parent.png` copies whose run never got saved (e.g. the server stopped while an
        Enhance & Generate chain was waiting). Only safe at startup, when no job is in flight."""
        n = 0
        for proj in self.projects_dir.iterdir():
            runs = proj / "runs"
            if not runs.is_dir():
                continue
            for f in runs.iterdir():
                m = re.fullmatch(r"(\d+)\.(refs|parent\.png)", f.name)
                if m and not (runs / f"{m.group(1)}.json").exists():
                    shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
                    n += 1
        return n

    # ---- per-project reference images ---------------------------------------------------------------
    # projects/<pid>/refs/<rid>.png + <rid>.thumb.webp, with refs.json = [{id, name, enabled, order, width, height, created_at}].
    # Reference files are immutable once written, so a run can hard-link them to pin exactly what it was submitted with.
    def refs_dir(self, pid: str) -> Path:
        return self.project_dir(pid) / "refs"

    def ref_path(self, pid: str, rid: str) -> Path:
        return self.refs_dir(pid) / f"{rid}.png"

    def ref_thumb_path(self, pid: str, rid: str) -> Path:
        return self.refs_dir(pid) / f"{rid}.thumb.webp"

    def _refs_file(self, pid: str) -> Path:
        return self.refs_dir(pid) / "refs.json"

    def list_refs(self, pid: str) -> list[dict]:
        if not self.valid_pid(pid):
            return []
        refs = _read_json(self._refs_file(pid)) or []
        return sorted(refs, key=lambda r: r["order"])

    def _save_refs(self, pid: str, refs: list[dict]) -> None:
        for i, r in enumerate(refs):
            r["order"] = i
        _write_json(self._refs_file(pid), refs)

    def add_ref(self, pid: str, image, name: str | None, enabled: bool) -> dict | None:
        """Store an RGB PIL image as a new reference (appended last). None if the project is gone."""
        with self._lock:
            if self.get_project(pid) is None:
                return None
            refs = self.list_refs(pid)
            if not name or not name.strip():
                used = [int(m.group(1)) for r in refs if (m := re.fullmatch(r"Reference (\d+)", r["name"]))]
                name = f"Reference {max(used, default=0) + 1}"
            rid = secrets.token_urlsafe(5)
            self.refs_dir(pid).mkdir(parents=True, exist_ok=True)
            tmp = self.refs_dir(pid) / f"{rid}.{uuid.uuid4().hex[:8]}.tmp"
            image.save(tmp, format="PNG")
            os.replace(tmp, self.ref_path(pid, rid))
            thumb = image.copy()
            thumb.thumbnail((THUMB_MAX, THUMB_MAX))
            ttmp = self.refs_dir(pid) / f"{rid}.thumb.{uuid.uuid4().hex[:8]}.tmp"
            thumb.save(ttmp, format="WEBP", quality=80)
            os.replace(ttmp, self.ref_thumb_path(pid, rid))
            ref = {"id": rid, "name": name.strip()[:80], "enabled": bool(enabled), "order": len(refs),
                   "width": image.width, "height": image.height, "created_at": time.time()}  # fmt: skip
            self._save_refs(pid, [*refs, ref])
            return ref

    def update_ref(self, pid: str, rid: str, *, enabled: bool | None = None, name: str | None = None, order: int | None = None) -> dict | None:
        with self._lock:
            refs = self.list_refs(pid)
            ref = next((r for r in refs if r["id"] == rid), None)
            if ref is None:
                return None
            if enabled is not None:
                ref["enabled"] = enabled
            if name is not None:
                ref["name"] = name
            if order is not None:
                refs.remove(ref)
                refs.insert(max(0, min(len(refs), order)), ref)
            self._save_refs(pid, refs)
            return ref

    def delete_ref(self, pid: str, rid: str) -> bool:
        with self._lock:
            refs = self.list_refs(pid)
            if not any(r["id"] == rid for r in refs):
                return False
            self._save_refs(pid, [r for r in refs if r["id"] != rid])
            self.ref_path(pid, rid).unlink(missing_ok=True)
            self.ref_thumb_path(pid, rid).unlink(missing_ok=True)
            return True

    def pin_refs(self, pid: str, n: int, start_index: int, refs: list[dict]) -> list[Path]:
        """Hard-link (or copy) the given refs into runs/NNNN.refs/<image number>.png; returns the pinned paths in order."""
        d = self.runs_dir(pid) / f"{n:04d}.refs"
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for i, r in enumerate(refs):
            dst = d / f"{start_index + i}.png"
            src = self.ref_path(pid, r["id"])
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            out.append(dst)
        return out

    def pinned_ref_path(self, pid: str, n: int, index: int) -> Path:
        return self.runs_dir(pid) / f"{n:04d}.refs" / f"{index}.png"
