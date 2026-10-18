"""Native macOS window for the local ImageAtelier server: starts uvicorn, shows the UI in pywebview, stops the server on quit."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HOST = "127.0.0.1"
START_TIMEOUT = 120.0
STOP_TIMEOUT = 15.0
_SIGNALS = {signal.SIGINT, signal.SIGTERM}
ICON = Path(__file__).resolve().parent / "static" / "icons" / "icon-512.png"
SHUTDOWN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ImageAtelier</title>
<style>
:root { color-scheme: light dark; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center;
  background: #fbf6e8; color: #221a46; font: 17px/1.5 system-ui, sans-serif; }
main { text-align: center; padding: 32px; }
h1 { font: 600 28px/1.3 ui-serif, Georgia, serif; }
p { color: #575077; }
.spinner { width: 32px; height: 32px; margin: auto; border: 3px solid #e3d7b8;
  border-top-color: #1d6a71; border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
@media (prefers-color-scheme: dark) { body { background: #161230; color: #f3ead6; }
  p { color: #b9b1d6; } .spinner { border-color: #372e6b; border-top-color: #4db4ba; } }
</style></head><body><main role="status" aria-live="polite"><div class="spinner" aria-hidden="true"></div>
<h1>Shutting down background server…</h1><p>ImageAtelier will close when shutdown is complete.</p>
</main></body></html>"""


def server_command(port: int) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "imageatelier.app:app", "--host", HOST, "--port", str(port),
            "--timeout-graceful-shutdown", "3"]


def port_open(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(port: int, proc: subprocess.Popen, timeout: float = START_TIMEOUT) -> bool:
    """True once the port accepts connections; False if the child exits or the timeout passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if port_open(port):
            return True
        time.sleep(0.2)
    return False


def stop_server(proc: subprocess.Popen, timeout: float = STOP_TIMEOUT) -> None:
    """Ask uvicorn to shut down gracefully (SIGINT), then kill it if it does not exit in time."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _set_dock_icon() -> None:
    # The window runs in a bare Python process, so without this the Dock shows Python's generic icon.
    try:
        from AppKit import NSApplication, NSImage
    except ImportError:  # pyobjc comes with pywebview on macOS; anywhere else keep the default icon
        return
    image = NSImage.alloc().initWithContentsOfFile_(str(ICON))
    if image is not None:
        NSApplication.sharedApplication().setApplicationIconImage_(image)


class Shutdown:
    """Keep the native window open until the server has actually exited."""
    def __init__(self, window, proc):
        self.window, self.proc = window, proc
        self.complete = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def request(self) -> bool:
        with self.lock:
            if self.complete.is_set():
                return True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="shutdown-server")
                self.thread.start()
        return False  # Cancel this close; _run closes the window once the server is gone.

    def _run(self) -> None:
        try:
            self.window.load_html(SHUTDOWN_HTML)
        except Exception as exc:
            print(f"could not display shutdown screen: {exc}", file=sys.stderr)
        try:
            stop_server(self.proc)
        except Exception as exc:
            print(f"could not stop background server: {exc}", file=sys.stderr)
            return  # Keep the window open; another close request can retry.
        self.complete.set()
        self.window.destroy()

    def join(self) -> None:
        if self.thread:
            self.thread.join()


def _close_on_signal(shutdown: Shutdown, stop: threading.Event) -> None:
    # Python signal handlers never run inside the Cocoa runloop, so wait for signals on a helper thread instead.
    signal.sigwait(_SIGNALS)
    if not stop.is_set():
        shutdown.request()


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    if port_open(port):
        print(f"error: something is already listening on {HOST}:{port}; stop it or set PORT", file=sys.stderr)
        return 1
    env = {**os.environ, "TQDM_DISABLE": "1"}
    proc = subprocess.Popen(server_command(port), env=env)
    try:
        if not wait_for_port(port, proc):
            code = proc.poll()
            if code is None:
                print(f"error: server did not start listening on port {port} in time", file=sys.stderr)
                return 1
            print(f"error: server exited during startup (code {code})", file=sys.stderr)
            return code or 1
        try:
            import webview
        except ImportError:
            print("pywebview is missing; run: uv sync --inexact --extra gui", file=sys.stderr)
            return 1
        _set_dock_icon()
        window = webview.create_window("ImageAtelier", f"http://{HOST}:{port}", width=1280, height=900)
        shutdown = Shutdown(window, proc)
        window.events.closing += shutdown.request
        # Block the signals before pywebview spawns threads so only the helper thread receives them.
        signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        stop = threading.Event()
        watcher = threading.Thread(target=_close_on_signal, args=(shutdown, stop), daemon=True)
        watcher.start()
        try:
            webview.start()
        finally:
            stop.set()
            if watcher.is_alive():
                signal.pthread_kill(watcher.ident, signal.SIGTERM)  # wakes sigwait if no signal arrived
            watcher.join()
            shutdown.join()
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
        return 0
    finally:
        stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
