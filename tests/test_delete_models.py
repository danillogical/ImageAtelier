"""Deleting downloaded parts: models.delete_image/delete_te/delete_pe against a fake HF cache, and the DELETE API."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from imageatelier import models
from imageatelier.app import create_app
from imageatelier.engine import EngineBusy

from conftest import StubEngine, new_project, submit, wait_until
from test_serialization import h, wait_state  # noqa: F401 - fixtures: real Engine with stubbed loaders

IMG, TE_HF, TE_LOCAL = "official-8bit", "heretic-8bit", "official-8bit"
PE_T2I, PE_I2I, PE_LOCAL = "t2i-8bit", "i2i-8bit", "t2i-heretic-8bit"


# ---- fake HF cache --------------------------------------------------------------------------
class FakeCache:
    """Builds `models--org--name/{blobs/<hash>, snapshots/<rev>/<path> -> ../../blobs/<hash>}` under a tmp root."""

    def __init__(self, root: Path):
        self.root = root

    def repo_dir(self, repo: str) -> Path:
        return self.root / ("models--" + repo.replace("/", "--"))

    def add(self, repo: str, rev: str, rel: str, content: bytes) -> Path:
        d = self.repo_dir(repo)
        blob = d / "blobs" / hashlib.sha256(content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        link = d / "snapshots" / rev / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link

    def blobs(self) -> dict[str, int]:
        """Every existing blob (path -> size) in every repo of the cache."""
        out = {}
        for blobs in self.root.glob("models--*/blobs"):
            for b in blobs.iterdir():
                out[str(b)] = b.stat().st_size
        return out

    def dangling(self) -> list[str]:
        bad = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            for n in filenames + dirnames:
                p = os.path.join(dirpath, n)
                if os.path.islink(p) and not os.path.exists(p):
                    bad.append(p)
        return bad

    def part(self, repo: str, rev: str, folder: str, tag: str, shard_content: bytes | None = None, files=None):
        """A complete model subfolder: config + index + one shard (+ extra files)."""
        shard = "model-00001-of-00001.safetensors"
        self.add(repo, rev, f"{folder}/config.json", json.dumps({"tag": tag}).encode())
        self.add(repo, rev, f"{folder}/model.safetensors.index.json", json.dumps({"weight_map": {"w": shard}, "tag": tag}).encode())
        self.add(repo, rev, f"{folder}/{shard}", shard_content or (b"shard-" + tag.encode() + b"-" * 100))
        for rel, content in (files or {}).items():
            self.add(repo, rev, f"{folder}/{rel}", content)


SHARED = b"identical-shard-bytes-" + b"z" * 500


@pytest.fixture
def cache(monkeypatch, tmp_path):
    root = tmp_path / "hf"
    root.mkdir()
    monkeypatch.setenv("IMAGEATELIER_HF_CACHE", str(root))
    monkeypatch.setenv("IMAGEATELIER_MODELS_DIR", str(tmp_path / "models"))
    return FakeCache(root)


@pytest.fixture
def shared_cache(cache):
    """official-8bit image + heretic-8bit text encoder in ONE repo@revision, plus the turbo LoRA; one shared blob."""
    m, t = models.IMAGE_MODELS[IMG], models.TEXT_ENCODERS[TE_HF]
    assert (m.repo, m.revision) == (t.repo, t.revision)
    cache.part(m.repo, m.revision, "transformer", "transformer", shard_content=SHARED)
    cache.part(m.repo, m.revision, "vae", "vae")
    cache.add(m.repo, m.revision, "processor/tokenizer.json", b'{"tok": 1}')
    cache.part(m.repo, m.revision, "text_encoder", "text_encoder", shard_content=SHARED)  # same bytes -> same blob
    cache.add(models.TURBO_LORA_REPO, models.TURBO_LORA_REVISION, models.TURBO_LORA_FILENAME, b"lora-weights" * 50)
    cache.shared_blob = cache.repo_dir(m.repo) / "blobs" / hashlib.sha256(SHARED).hexdigest()
    cache.snap = cache.repo_dir(m.repo) / "snapshots" / m.revision
    assert models.image_status(IMG)["state"] == "ready"
    assert models.te_status(TE_HF)["state"] == "ready"
    assert models.image_status(IMG)["turbo_lora_ready"] is True
    return cache


def make_pe(cache: FakeCache, name: str):
    pe = models.PE_MODELS[name]
    cache.part(pe.repo, pe.revision, pe.quant, name, files={"tokenizer.json": b"tok-" + name.encode()})
    cache.add(pe.prompt_repo, pe.prompt_revision, models.PE_SYSTEM_PROMPT_FILE, b"system prompt " + name.encode())


@pytest.fixture
def pe_cache(cache):
    make_pe(cache, PE_T2I)
    make_pe(cache, PE_I2I)
    assert models.pe_status(PE_T2I)["state"] == "ready"
    assert models.pe_status(PE_I2I)["state"] == "ready"
    return cache


@contextlib.contextmanager
def fake_download(key: str):
    """Register a live-thread _Download under `key` (as _start would) for the duration of the block."""
    release = threading.Event()
    d = models._Download(key, [])
    d.thread = threading.Thread(target=release.wait, daemon=True)
    d.thread.start()
    with models._lock:
        models._downloads[key] = d
    try:
        yield d
    finally:
        release.set()
        d.thread.join(5)
        with models._lock:
            models._downloads.pop(key, None)


# ---- A: models.delete_* ---------------------------------------------------------------------
def test_delete_image_keeps_shared_blob_and_te(shared_cache):
    c = shared_cache
    before = c.blobs()
    freed = models.delete_image(IMG)
    after = c.blobs()
    gone = set(before) - set(after)
    assert models.image_status(IMG)["state"] == "not_downloaded"
    assert models.image_status(IMG)["turbo_lora_ready"] is False
    assert models.te_status(TE_HF)["state"] == "ready"
    assert str(c.shared_blob) in after
    assert c.dangling() == []
    assert freed > 0 and isinstance(freed, int)
    assert freed == sum(before[b] for b in gone)
    assert str(c.shared_blob) not in gone


def test_delete_te_after_image_removes_shared_blob(shared_cache):
    c = shared_cache
    models.delete_image(IMG)
    before = c.blobs()
    freed = models.delete_te(TE_HF)
    after = c.blobs()
    assert models.te_status(TE_HF)["state"] == "not_downloaded"
    assert not c.shared_blob.exists()
    assert freed == sum(before[b] for b in set(before) - set(after))
    assert freed >= len(SHARED)
    assert c.dangling() == []


def test_delete_te_first_keeps_image_ready(shared_cache):
    c = shared_cache
    before = c.blobs()
    freed = models.delete_te(TE_HF)
    after = c.blobs()
    assert models.te_status(TE_HF)["state"] == "not_downloaded"
    assert models.image_status(IMG)["state"] == "ready"
    assert models.image_status(IMG)["turbo_lora_ready"] is True
    assert c.shared_blob.exists()
    assert c.dangling() == []
    assert freed > 0 and freed == sum(before[b] for b in set(before) - set(after))
    # now the image goes too and the shared blob finally disappears
    models.delete_image(IMG)
    assert not c.shared_blob.exists()


@pytest.mark.parametrize("which", ["image", "te", "pe"])
def test_second_delete_returns_zero(shared_cache, which):
    make_pe(shared_cache, PE_T2I)
    fn, name = {"image": (models.delete_image, IMG), "te": (models.delete_te, TE_HF), "pe": (models.delete_pe, PE_T2I)}[which]
    assert fn(name) > 0
    assert fn(name) == 0


def test_second_delete_with_nothing_ever_cached(cache):
    assert models.delete_image(IMG) == 0
    assert models.delete_te(TE_HF) == 0
    assert models.delete_pe(PE_T2I) == 0


def test_delete_pe_removes_enhancer_and_prompt_keeps_i2i(pe_cache):
    c = pe_cache
    before = c.blobs()
    freed = models.delete_pe(PE_T2I)
    after = c.blobs()
    assert models.pe_status(PE_T2I)["state"] == "not_downloaded"
    assert models.pe_system_prompt_path(PE_T2I) is None
    assert models.pe_status(PE_I2I)["state"] == "ready"
    assert models.pe_system_prompt_path(PE_I2I) is not None
    assert freed > 0 and freed == sum(before[b] for b in set(before) - set(after))
    assert c.dangling() == []


def test_refusals_local_and_unknown(cache):
    with pytest.raises(models.LocalModelError):
        models.delete_te(TE_LOCAL)
    with pytest.raises(models.LocalModelError):
        models.delete_pe(PE_LOCAL)
    with pytest.raises(ValueError):
        models.delete_image("nope")
    with pytest.raises(ValueError):
        models.delete_te("nope")
    with pytest.raises(ValueError):
        models.delete_pe("nope")
    assert issubclass(models.LocalModelError, models.DeleteRefused)
    assert issubclass(models.ModelBusy, models.DeleteRefused)


def test_local_refusal_leaves_local_folder_alone(cache, tmp_path):
    folder = tmp_path / "models" / models.TEXT_ENCODERS[TE_LOCAL].folder / "text_encoder"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    with pytest.raises(models.LocalModelError):
        models.delete_te(TE_LOCAL)
    assert (folder / "config.json").exists()


@pytest.mark.parametrize("key,fn,name", [
    ("image:" + IMG, models.delete_image, IMG),
    ("te:" + TE_HF, models.delete_te, TE_HF),
    ("pe:" + PE_T2I, models.delete_pe, PE_T2I),
])
def test_busy_while_downloading(shared_cache, key, fn, name):
    make_pe(shared_cache, PE_T2I)
    before = shared_cache.blobs()
    with fake_download(key):
        with pytest.raises(models.ModelBusy):
            fn(name)
    assert shared_cache.blobs() == before  # nothing was deleted
    assert fn(name) > 0  # the download finished: delete works again


def test_delete_touches_nothing_outside_the_repo(shared_cache):
    c = shared_cache
    make_pe(c, PE_I2I)
    stray = c.root / "stray.txt"
    stray.write_text("keep me")
    other = c.add("other/repo", "r" * 40, "transformer/model.safetensors", b"other repo weights")
    other_blob = os.path.realpath(other)
    models.delete_image(IMG)
    models.delete_te(TE_HF)
    assert stray.read_text() == "keep me"
    assert os.path.exists(other) and os.path.exists(other_blob)
    assert models.pe_status(PE_I2I)["state"] == "ready"


# ---- B: HTTP --------------------------------------------------------------------------------
def _maintenance(self):
    """Stand-in for Engine.maintenance on the StubEngine: refuse while any job is not final."""
    @contextlib.contextmanager
    def cm():
        if any(j["state"] not in ("done", "failed", "cancelled") for j in self.jobs()):
            raise EngineBusy("finish or cancel the queued and running jobs first")
        yield
    return cm()


@pytest.fixture
def stub_with_maintenance(monkeypatch):
    monkeypatch.setattr(StubEngine, "maintenance", _maintenance, raising=False)


@pytest.fixture
def dclient(stub_with_maintenance, shared_cache, client):
    make_pe(shared_cache, PE_T2I)
    make_pe(shared_cache, PE_I2I)
    client.cache = shared_cache
    return client


DELETE_URLS = [
    ("/api/models/image/" + IMG, "image"),
    ("/api/models/text_encoder/" + TE_HF, "te"),
    ("/api/models/pe/" + PE_T2I, "pe"),
]


@pytest.mark.parametrize("url,_", DELETE_URLS)
def test_http_delete_ok(dclient, url, _):
    r = dclient.delete(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] is True
    assert isinstance(body["freed_bytes"], int) and body["freed_bytes"] > 0


def test_http_delete_really_deletes(dclient):
    assert dclient.delete("/api/models/pe/" + PE_T2I).status_code == 200
    # the conftest fixtures stub the *_status functions, so check the cache through the real resolvers
    assert models.pe_local_dir(PE_T2I) is None and models.pe_system_prompt_path(PE_T2I) is None
    assert models.pe_local_dir(PE_I2I) is not None and models.pe_system_prompt_path(PE_I2I) is not None
    assert dclient.delete("/api/models/pe/" + PE_T2I).json() == {"deleted": True, "freed_bytes": 0}


@pytest.mark.parametrize("kind", ["image", "text_encoder", "pe"])
def test_http_delete_unknown_404(dclient, kind):
    assert dclient.delete(f"/api/models/{kind}/nope").status_code == 404


@pytest.mark.parametrize("url", ["/api/models/text_encoder/" + TE_LOCAL, "/api/models/pe/" + PE_LOCAL])
def test_http_delete_local_409(dclient, url):
    r = dclient.delete(url)
    assert r.status_code == 409
    assert r.json()["detail"]


@pytest.mark.parametrize("url,key", [
    ("/api/models/image/" + IMG, "image:" + IMG),
    ("/api/models/text_encoder/" + TE_HF, "te:" + TE_HF),
    ("/api/models/pe/" + PE_T2I, "pe:" + PE_T2I),
])
def test_http_delete_active_download_409(dclient, url, key):
    with fake_download(key):
        r = dclient.delete(url)
    assert r.status_code == 409
    assert r.json()["detail"]
    assert dclient.delete(url).status_code == 200


@pytest.mark.parametrize("url,_", DELETE_URLS)
def test_http_delete_409_while_job_queued(dclient, engine, url, _):
    engine.gate.clear()
    pid = new_project(dclient)["id"]
    submit(dclient, pid)
    assert [j["state"] for j in engine.jobs()] == ["queued"]
    before = dclient.cache.blobs()
    r = dclient.delete(url)
    assert r.status_code == 409 and r.json()["detail"]
    assert dclient.cache.blobs() == before


def test_http_delete_409_while_job_running(dclient, engine):
    engine.release.clear()
    pid = new_project(dclient)["id"]
    submit(dclient, pid)
    wait_until(engine.running.is_set)
    r = dclient.delete("/api/models/image/" + IMG)
    assert r.status_code == 409 and r.json()["detail"]
    engine.release.set()


# ---- B: real Engine.maintenance -------------------------------------------------------------
def test_maintenance_refuses_queued_job(h):
    h.gen()
    with pytest.raises(EngineBusy):
        with h.engine.maintenance():
            pytest.fail("should not enter")


def test_maintenance_refuses_held_job(h):
    jid = h.gen()
    assert h.engine.hold(jid) is True
    assert h.engine.get(jid)["state"] == "held"
    with pytest.raises(EngineBusy):
        with h.engine.maintenance():
            pytest.fail("should not enter")


def test_maintenance_refuses_running_and_paused_jobs(h):
    h.behaviour.append("block")
    jid = h.gen()
    h.start()
    wait_state(h.engine, jid, ("running",))
    with pytest.raises(EngineBusy):
        with h.engine.maintenance():
            pytest.fail("should not enter")
    h.release.set()
    wait_state(h.engine, jid, ("done",))


def test_maintenance_concurrent_second_refused(h):
    h.start()
    with h.engine.maintenance():
        with pytest.raises(EngineBusy):
            with h.engine.maintenance():
                pytest.fail("should not enter")
    with h.engine.maintenance():  # and the flag was cleared on exit
        pass


def test_maintenance_defers_submitted_job_and_unloads_resident(h):
    h.start()
    first = h.gen()
    wait_state(h.engine, first, ("done",))
    assert h.engine.resident()["key"] is not None  # fake loader left a model resident
    with h.engine.maintenance():
        assert h.engine.resident()["key"] is None
        jid = h.gen()
        time.sleep(0.4)
        assert h.engine.get(jid)["state"] == "queued"
    assert wait_state(h.engine, jid, ("done",))["state"] == "done"


def test_http_delete_with_real_engine(h, shared_cache):
    """End to end: delete_part runs under the real maintenance(); a queued job turns it into a 409."""
    h.gen()
    with TestClient(create_app(engine=h.engine, data_dir=h.tmp / "data")) as c:
        wait_state(h.engine, h.engine.jobs()[0]["id"], ("done",))
        assert c.delete("/api/models/text_encoder/" + TE_HF).status_code == 200
        assert models.te_local_dir(TE_HF) is None  # (*_status is stubbed by conftest; use the real resolver)
        h.behaviour.append("block")
        wait_state(h.engine, h.gen(), ("running",))
        r = c.delete("/api/models/image/" + IMG)
        assert r.status_code == 409 and r.json()["detail"]
        assert models.image_local_dir(IMG) is not None
