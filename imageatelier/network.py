"""Independent, authenticated LAN listeners sharing the local app and GPU worker."""
from __future__ import annotations

import asyncio
import base64
import hmac
import ipaddress
import os
import secrets
import socket
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from starlette.responses import JSONResponse

from imageatelier.ca import _EmbeddedServer
from imageatelier.network_api import create_api

NETWORK_KEYS = {'network_api_enabled', 'network_ui_enabled'}


def loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == 'testclient'


def local_control(request) -> bool:
    if request.scope.get('imageatelier.network_ui'):
        return False
    host = request.client.host if request.client else ''
    if not loopback(host):
        return False
    origin = request.headers.get('origin')
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            return False
    allowed = {'localhost', '127.0.0.1', '::1'}
    if host == 'testclient':
        allowed.add('testserver')
    return request.url.hostname in allowed


class Listener:
    def __init__(self, app, port: int):
        self.port, self.task, self.socket = port, None, None
        self.server = _EmbeddedServer(uvicorn.Config(
            app, host='0.0.0.0', port=port, interface='asgi3', lifespan='off', log_config=None,
            log_level='warning', access_log=False, proxy_headers=False, timeout_graceful_shutdown=2))

    async def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.port))
            sock.listen(32)
            sock.setblocking(False)
            self.socket = sock
        except BaseException:
            sock.close()
            raise

        async def run():
            try:
                await self.server.serve(sockets=[sock])
            except SystemExit as exc:
                raise RuntimeError(f'Listener could not start on port {self.port}') from exc
        self.task = asyncio.create_task(run())
        for _ in range(100):
            if self.server.started:
                return
            if self.task.done():
                await self.task
                raise RuntimeError(f'Listener stopped on port {self.port}')
            await asyncio.sleep(0.02)
        raise RuntimeError(f'Listener startup timed out on port {self.port}')

    async def stop(self):
        self.server.should_exit = True
        try:
            if self.task:
                await asyncio.wait_for(asyncio.shield(self.task), 5)
        except (TimeoutError, Exception):
            if self.task:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        finally:
            if self.socket:
                self.socket.close()

    @property
    def listening(self):
        return bool(self.server.started and self.task and not self.task.done() and not self.server.should_exit)


class Network:
    def __init__(self, app, engine, settings, root: Path):
        self.app, self.engine, self.settings = app, engine, settings
        self.key_path = root / 'network-access-key'
        self._key = None
        self.api_port = int(os.environ.get('IMAGEATELIER_API_PORT', '8001'))
        self.ui_port = int(os.environ.get('IMAGEATELIER_UI_PORT', '8002'))
        self.hostname = socket.gethostname().removesuffix('.local') + '.local'
        self.listeners = {}
        self.lock = asyncio.Lock()
        self.error = None
        self.api = create_api(engine, settings, self.authorize,
                              enabled=lambda: settings.get()['network_api_enabled'])

    def key(self):
        if self._key is None:
            try:
                self._key = self.key_path.read_text().strip()
            except FileNotFoundError:
                value = secrets.token_urlsafe(32)
                try:
                    fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    value = self.key_path.read_text().strip()
                else:
                    with os.fdopen(fd, 'w') as stream:
                        stream.write(value + '\n')
                self._key = value
            if not self._key:
                raise RuntimeError('Network access key is empty')
        return self._key

    def authorize(self, authorization: str) -> bool:
        scheme, _, value = authorization.partition(' ')
        return scheme.lower() == 'bearer' and hmac.compare_digest(value.encode(), self.key().encode())

    def ui_authorized(self, authorization: str) -> bool:
        try:
            scheme, _, value = authorization.partition(' ')
            user, password = base64.b64decode(value, validate=True).decode().split(':', 1)
            return scheme.lower() == 'basic' and user == 'imageatelier' and hmac.compare_digest(password.encode(), self.key().encode())
        except (ValueError, UnicodeError):
            return False

    async def ui(self, scope, receive, send):
        if scope['type'] != 'http':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        scope = {**scope, 'imageatelier.network_ui': True}
        if not self.settings.get()['network_ui_enabled']:
            await JSONResponse({'detail': 'Network UI is disabled'}, status_code=503)(scope, receive, send)
            return
        headers = dict(scope.get('headers', []))
        if not self.ui_authorized(headers.get(b'authorization', b'').decode()):
            await JSONResponse({'detail': 'Sign in with your ImageAtelier network access key'}, status_code=401,
                               headers={'WWW-Authenticate': 'Basic realm="ImageAtelier", charset="UTF-8"'})(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def status(self, *, include_key=False):
        values = self.settings.get()
        status = {key: values[key] for key in NETWORK_KEYS}
        status.update(api_listening=bool(self.listeners.get('api') and self.listeners['api'].listening),
                      ui_listening=bool(self.listeners.get('ui') and self.listeners['ui'].listening),
                      api_url=f'http://{self.hostname}:{self.api_port}', ui_url=f'http://{self.hostname}:{self.ui_port}',
                      ui_username='imageatelier', error=self.error)
        if include_key:
            status['access_key'] = self.key()
        return status

    async def apply(self, values):
        """Prepare all requested sockets before replacing existing listeners; rollback new binds on failure."""
        async with self.lock:
            added = {}
            try:
                for name, key, target, port in (
                    ('api', 'network_api_enabled', self.api, self.api_port),
                    ('ui', 'network_ui_enabled', self.ui, self.ui_port)):
                    if values[key] and (name not in self.listeners or not self.listeners[name].listening):
                        listener = Listener(target, port)
                        added[name] = listener
                        await listener.start()
                for name, key in (('api', 'network_api_enabled'), ('ui', 'network_ui_enabled')):
                    if not values[key] and name in self.listeners:
                        if name == 'api':
                            await self.api.state.cancel_memory_requests()
                        await self.listeners.pop(name).stop()
                self.listeners.update(added)
                self.error = None
            except BaseException as exc:
                for listener in added.values():
                    await listener.stop()
                self.error = f'Could not update network listeners: {exc}'
                raise

    async def stop(self):
        async with self.lock:
            await self.api.state.cancel_memory_requests()
            for listener in self.listeners.values():
                await listener.stop()
            self.listeners.clear()
