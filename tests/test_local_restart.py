"""Restart only the local ImageAtelier on the requested port, including its window/workers."""
import os
import signal
import subprocess
import sys

from imageatelier import local_restart as restart


def test_selects_window_server_and_workers_only():
    commands = {
        1: (0, "python -m imageatelier.gui"),
        2: (1, "python -m uvicorn imageatelier.app:app --host 127.0.0.1 --port 8400 --timeout-graceful-shutdown 3"),
        3: (2, "python -m imageatelier.convert_suite"),
        4: (0, "python -m uvicorn imageatelier.app:app --host 127.0.0.1 --port 8010"),
        5: (0, "python -m uvicorn imageatelier.app:app --host 0.0.0.0 --port 8400"),
        6: (0, "python -m uvicorn another.app:app --host 127.0.0.1 --port 8400"),
        7: (0, "python -m imageatelier.gui"),
    }
    current = {pid: restart.Process(pid, parent, cmd) for pid, (parent, cmd) in commands.items()}
    assert set(restart.targets(current, 8400)) == {1, 2, 3}
    assert set(restart.targets(current, 8010)) == {4}
    assert restart.targets(current, 9999) == {}


def test_stops_real_process_gracefully():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
    )
    try:
        selected = {proc.pid: restart.processes()[proc.pid]}
        restart.stop(selected, timeout=1)
        assert proc.wait(timeout=2) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_force_stops_unresponsive_worker_even_after_parent_exits():
    worker = (
        "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(os.getpid(), flush=True); time.sleep(600)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c",
         f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-u', '-c', {worker!r}]); time.sleep(600)"],
        stdout=subprocess.PIPE, text=True,
    )
    child_pid = None
    try:
        child_pid = int(proc.stdout.readline())
        current = restart.processes()
        restart.stop({pid: current[pid] for pid in (proc.pid, child_pid)}, timeout=0.2)
        assert proc.wait(timeout=2) == -signal.SIGTERM
        assert child_pid not in restart.processes()
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


def test_does_not_signal_reused_pid(monkeypatch):
    original = restart.Process(42, 1, "python -m imageatelier.gui")
    monkeypatch.setattr(restart, "processes", lambda: {42: restart.Process(42, 1, "unrelated")})
    monkeypatch.setattr(restart.os, "kill", lambda *args: (_ for _ in ()).throw(AssertionError("must not signal")))
    restart.stop({42: original}, timeout=0)
