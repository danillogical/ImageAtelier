"""webstudio.gui launcher: starts the server child, opens the window, stops the child (pywebview faked)."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from webstudio import gui

LISTENER = (
    "import socket, time\n"
    "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', {port})); s.listen()\n"
    "try:\n"
    "    time.sleep(600)\n"
    "except KeyboardInterrupt:\n"
    "    pass\n"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def blocked_signals() -> set:
    return set(signal.pthread_sigmask(signal.SIG_BLOCK, []))


def pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def port(monkeypatch):
    p = free_port()
    monkeypatch.setenv("PORT", str(p))
    return p


@pytest.fixture
def spawned(monkeypatch):
    """Record every Popen the launcher makes."""
    procs: list[subprocess.Popen] = []
    real = subprocess.Popen

    def spy(*a, **kw):
        p = real(*a, **kw)
        procs.append(p)
        return p

    monkeypatch.setattr(gui.subprocess, "Popen", spy)
    yield procs
    for p in procs:  # never leak a child if a test fails midway
        if p.poll() is None:
            p.kill()
            p.wait()


def fake_webview(monkeypatch, port, on_start=None):
    state = SimpleNamespace(created=[], started=0, port_open_at_start=None, destroyed=0)
    mod = ModuleType("webview")

    def create_window(*a, **kw):
        state.created.append((a, kw))
        return SimpleNamespace(destroy=lambda: setattr(state, "destroyed", state.destroyed + 1))

    def start(*a, **kw):
        state.started += 1
        state.port_open_at_start = listening(port)
        if on_start:
            on_start()

    mod.create_window, mod.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", mod)
    return state


def test_main_runs_window_and_stops_child(monkeypatch, port, spawned):
    monkeypatch.setattr(gui, "server_command", lambda p: [sys.executable, "-c", LISTENER.format(port=p)])
    state = fake_webview(monkeypatch, port)
    assert gui.main() == 0
    assert state.started == 1
    assert state.port_open_at_start is True
    assert len(state.created) == 1 and f"127.0.0.1:{port}" in state.created[0][0][1]
    assert len(spawned) == 1
    assert spawned[0].poll() is not None and pid_gone(spawned[0].pid)
    assert not listening(port)
    assert not (blocked_signals() & {signal.SIGINT, signal.SIGTERM})


def test_main_signal_closes_window(monkeypatch, port, spawned):
    """A SIGTERM arriving while the window is open is taken by the sigwait helper thread, which destroys the window."""
    monkeypatch.setattr(gui, "server_command", lambda p: [sys.executable, "-c", LISTENER.format(port=p)])

    def on_start():
        # Aim at the helper thread: a process-directed signal could be taken by any unrelated thread of the test run.
        watcher = next(t for t in threading.enumerate() if getattr(t, "_target", None) is gui._close_on_signal)
        signal.pthread_kill(watcher.ident, signal.SIGTERM)
        deadline = time.time() + 5
        while state.destroyed == 0 and time.time() < deadline:
            time.sleep(0.02)

    state = fake_webview(monkeypatch, port, on_start)
    assert gui.main() == 0
    assert state.destroyed == 1
    assert not listening(port)
    assert not (blocked_signals() & {signal.SIGINT, signal.SIGTERM})


def test_port_in_use_returns_1_without_spawning(monkeypatch, port):
    def boom(*a, **kw):
        pytest.fail("Popen must not be called when the port is taken")

    monkeypatch.setattr(gui.subprocess, "Popen", boom)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
        s.listen()
        assert gui.main() == 1
    assert not (blocked_signals() & {signal.SIGINT, signal.SIGTERM})


def test_child_exits_during_startup(monkeypatch, port, spawned):
    monkeypatch.setattr(gui, "server_command", lambda p: [sys.executable, "-c", "import sys; sys.exit(3)"])
    state = fake_webview(monkeypatch, port)
    assert gui.main() != 0
    assert state.started == 0 and state.created == []
    assert spawned[0].poll() is not None
    assert not (blocked_signals() & {signal.SIGINT, signal.SIGTERM})
