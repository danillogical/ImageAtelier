"""Stop this user's existing local server (and its window) before local.sh restarts it."""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    command: str

    @property
    def args(self) -> list[str]:
        try:
            return shlex.split(self.command)
        except ValueError:
            return []


def processes() -> dict[int, Process]:
    # -x lists only our user's processes, including those with no terminal.
    output = subprocess.check_output(
        ["ps", "-x", "-ww", "-o", "pid=,ppid=,stat=,command="], text=True,
    )
    result = {}
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and not fields[2].startswith("Z"):
            pid, parent, _, command = fields
            result[int(pid)] = Process(int(pid), int(parent), command)
    return result


def targets(current: dict[int, Process], port: int) -> dict[int, Process]:
    roots = set()
    for proc in current.values():
        args = proc.args
        if args[1:4] != ["-m", "uvicorn", "imageatelier.app:app"]:
            continue
        options = dict(zip(args[4::2], args[5::2]))
        if options.get("--host") != "127.0.0.1" or options.get("--port") != str(port):
            continue
        parent = current.get(proc.parent)
        if parent and parent.args[1:] == ["-m", "imageatelier.gui"]:
            roots.add(parent.pid)
        else:
            roots.add(proc.pid)
    selected = {pid: current[pid] for pid in roots}
    # Include conversion workers so a forced restart cannot leave GPU work running.
    while True:
        children = {pid: proc for pid, proc in current.items()
                    if proc.parent in selected and pid not in selected}
        if not children:
            return selected
        selected.update(children)


def stop(selected: dict[int, Process], timeout: float = 15.0) -> None:
    def remaining() -> dict[int, Process]:
        current = processes()
        # Check identity again before signaling, in case a PID has been reused.
        return {pid: proc for pid, proc in selected.items()
                if pid in current and current[pid].command == proc.command}

    def send(procs: dict[int, Process], sig: int) -> None:
        for pid in procs:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def wait(seconds: float) -> dict[int, Process]:
        deadline = time.monotonic() + seconds
        while pending := remaining():
            if time.monotonic() >= deadline:
                return pending
            time.sleep(0.1)
        return {}

    pending = remaining()
    # Let the GUI shut down its server, or uvicorn shut down its workers, first.
    send({pid: proc for pid, proc in pending.items() if proc.parent not in pending}, signal.SIGTERM)
    pending = wait(timeout)
    if pending:
        send(pending, signal.SIGTERM)
        pending = wait(3.0)
    if pending:
        send(pending, signal.SIGKILL)
        if wait(3.0):
            raise RuntimeError("the previous ImageAtelier did not stop")


def main() -> int:
    try:
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")
        selected = targets(processes(), port)
        if selected:
            print(f"Stopping existing ImageAtelier on port {port}…", flush=True)
            stop(selected)
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"error: cannot restart ImageAtelier: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
