"""Server-log ring buffer: a logging.Handler keeping the last N formatted records and streaming them in batches.

Installed on the root logger and the uvicorn loggers; `uvicorn.access` is excluded unless asked for. Records are
deduplicated across loggers (uvicorn.error propagates to uvicorn), so each appears once. tqdm writes progress
bars straight to stderr, not through logging, so they never reach the ring; carriage returns are stripped anyway.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

DEFAULT_CAPACITY = 2000
FLUSH_INTERVAL_S = 0.5
MAX_LINES_PER_FLUSH = 50  # streamed lines per flush; a burst beyond this is summarised (still kept in the ring)
DEFAULT_LOGGERS = ("", "uvicorn", "uvicorn.error")
ACCESS_LOGGER = "uvicorn.access"
# Records from this logger never enter the ring: it reports failures of the ring's own sinks (avoids feedback loops).
SELF_LOGGER = "imageatelier.serverlog"

_self_log = logging.getLogger(SELF_LOGGER)


class RingLogHandler(logging.Handler):
    def __init__(self, capacity: int = DEFAULT_CAPACITY, include_access: bool = False):
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self.include_access = include_access
        self._lines: deque[str] = deque(maxlen=capacity)
        self._pending: list[str] = []
        self._sinks: list[Callable[[list[str]], None]] = []
        self._ring_lock = threading.Lock()
        self._flusher: threading.Thread | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_ws_ring_seen", False):
            return
        record._ws_ring_seen = True
        if record.name.startswith(SELF_LOGGER) or (record.name.startswith(ACCESS_LOGGER) and not self.include_access):
            return
        try:
            line = self.format(record).replace("\r", "")
        except Exception:  # noqa: BLE001
            self.handleError(record)
            return
        with self._ring_lock:
            self._lines.append(line)
            if self._sinks:
                self._pending.append(line)

    def tail(self, n: int | None = None) -> list[str]:
        with self._ring_lock:
            lines = list(self._lines)
        return lines if not n or n <= 0 else lines[-n:]

    def add_sink(self, fn: Callable[[list[str]], None]) -> None:
        """fn(lines) is called from a background thread at most every FLUSH_INTERVAL_S with new lines."""
        with self._ring_lock:
            if fn not in self._sinks:
                self._sinks.append(fn)
            if self._flusher is None:
                self._flusher = threading.Thread(target=self._flush_loop, name="serverlog-flush", daemon=True)
                self._flusher.start()

    def remove_sink(self, fn: Callable[[list[str]], None]) -> None:
        with self._ring_lock:
            if fn in self._sinks:
                self._sinks.remove(fn)

    def _flush_loop(self) -> None:
        stop = threading.Event()
        while not stop.wait(FLUSH_INTERVAL_S):
            with self._ring_lock:
                pending, self._pending = self._pending, []
                sinks = list(self._sinks)
            if not pending or not sinks:
                continue
            if len(pending) > MAX_LINES_PER_FLUSH:
                skipped = len(pending) - MAX_LINES_PER_FLUSH
                pending = pending[:MAX_LINES_PER_FLUSH] + [
                    f"[serverlog] {skipped} more lines in this burst were not streamed; see server_log()"
                ]
            for fn in sinks:
                try:
                    fn(pending)
                except Exception:  # noqa: BLE001 - excluded from the ring, so this cannot feed back
                    _self_log.exception("server-log sink failed")


_handler: RingLogHandler | None = None
_install_lock = threading.Lock()


def install(include_access: bool = False, loggers: tuple[str, ...] = DEFAULT_LOGGERS) -> RingLogHandler:
    """Attach the (process-wide, singleton) ring handler; idempotent. Returns it."""
    global _handler
    with _install_lock:
        if _handler is None:
            _handler = RingLogHandler(include_access=include_access)
        _handler.include_access = _handler.include_access or include_access
        names = loggers + ((ACCESS_LOGGER,) if include_access else ())
        for name in names:
            lg = logging.getLogger(name)
            if _handler not in lg.handlers:
                lg.addHandler(_handler)
        # Our own info lines (GPU guard, model swaps) must pass the logger level to reach the handler.
        app_logger = logging.getLogger("imageatelier")
        if app_logger.level == logging.NOTSET:
            app_logger.setLevel(logging.INFO)
        return _handler


def handler() -> RingLogHandler | None:
    return _handler
