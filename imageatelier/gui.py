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


def _close_on_signal(window, stop: threading.Event) -> None:
    # Python signal handlers never run inside the Cocoa runloop, so wait for signals on a helper thread instead.
    signal.sigwait(_SIGNALS)
    if not stop.is_set():
        window.destroy()


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
        # Block the signals before pywebview spawns threads so only the helper thread receives them.
        signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        stop = threading.Event()
        watcher = threading.Thread(target=_close_on_signal, args=(window, stop), daemon=True)
        watcher.start()
        try:
            webview.start()
        finally:
            stop.set()
            if watcher.is_alive():
                signal.pthread_kill(watcher.ident, signal.SIGTERM)  # wakes sigwait if no signal arrived
            watcher.join()
            signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
        return 0
    finally:
        stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
