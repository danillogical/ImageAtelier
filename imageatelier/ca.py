"""Serving the local CA certificate so phones can trust the HTTPS server.

The main app exposes /ca (instructions) and /ca/rootCA.crt (the public root certificate only, never the key).
When IMAGEATELIER_CA_PORT is set, a second plain-HTTP listener runs too. By default it serves the full app (no PWA
install over plain HTTP, but everything else works with no CA setup); IMAGEATELIER_HTTP_MODE=ca restricts it to /ca
plus a redirect to the HTTPS URL. Either way a device can fetch the CA before it trusts the HTTPS certificate.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "static" / "ca.html"


def certs_dir() -> Path:
    env = os.environ.get("IMAGEATELIER_CERTS")
    return Path(env).expanduser() if env else ROOT / "certs"


def root_ca() -> Path:
    return certs_dir() / "ca" / "rootCA.pem"


async def page(request: Request):
    return HTMLResponse(PAGE.read_text())


async def crt(request: Request):
    f = root_ca()
    if not f.is_file():
        return PlainTextResponse("No CA certificate yet: run ./setup.sh on the server.", status_code=404)
    return FileResponse(f, media_type="application/x-x509-ca-cert", filename="ImageAtelier-CA.crt")


ROUTES = [Route("/ca", page), Route("/ca/", page), Route("/ca/rootCA.crt", crt)]


def plain_app(https_port: int) -> Starlette:
    async def home(request: Request):
        hostport = request.headers.get("host", "localhost")
        host = hostport[: hostport.index("]") + 1] if hostport.startswith("[") else hostport.split(":")[0]
        suffix = "" if https_port == 443 else f":{https_port}"
        return RedirectResponse(f"https://{host}{suffix}/", status_code=307)

    return Starlette(routes=[Route("/", home), *ROUTES])


class _EmbeddedServer(uvicorn.Server):
    """A uvicorn server that leaves signal handling to the main server running in the same process."""

    def install_signal_handlers(self) -> None:  # uvicorn < 0.29
        pass

    @contextlib.contextmanager
    def capture_signals(self):  # uvicorn >= 0.29
        yield


class PlainListener:
    def __init__(self, port: int, https_port: int, app=None):
        # app given: serve the full app over plain HTTP too (same instance, so engine/queue/events are shared; its
        # lifespan already ran on the HTTPS server). Otherwise: only /ca plus a redirect to HTTPS.
        target = app if app is not None else plain_app(https_port)
        config = uvicorn.Config(target, host=os.environ.get("IMAGEATELIER_BIND", "0.0.0.0"), port=port, log_config=None, log_level="warning", lifespan="off")
        self.port = port
        self.server = _EmbeddedServer(config)
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        async def run():
            try:
                await self.server.serve()
            except SystemExit:  # uvicorn exits the process on bind failure; don't take the main server down
                log.error("CA listener could not bind port %s (in use?); /ca is still served over HTTPS", self.port)

        self.task = asyncio.create_task(run())
        for _ in range(50):
            if self.server.started or self.task.done():
                break
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self.server.should_exit = True
        if self.task:
            try:
                await asyncio.wait_for(self.task, 5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.task.cancel()


def listener_from_env(app=None) -> PlainListener | None:
    # IMAGEATELIER_HTTP_MODE: "app" (default) serves the whole app on the HTTP port; "ca" serves only /ca + a redirect.
    port = os.environ.get("IMAGEATELIER_CA_PORT")
    if not port:
        return None
    full = os.environ.get("IMAGEATELIER_HTTP_MODE", "app") == "app"
    return PlainListener(int(port), int(os.environ.get("IMAGEATELIER_HTTPS_PORT", "443")), app if full else None)
