export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function api(method, path, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch("/api" + path, opts);
  } catch {
    throw new ApiError(0, "Can't reach the server");
  }
  if (res.status === 204) return null;
  let data = null;
  try {
    data = await res.json();
  } catch {}
  if (!res.ok) {
    let d = data && data.detail;
    if (Array.isArray(d)) d = d.map((x) => x.msg).join("; ");
    throw new ApiError(res.status, d || res.statusText || `HTTP ${res.status}`);
  }
  return data;
}

// EventSource wrapper with our own reconnect (native retry stops on HTTP errors).
// handlers: {job, run, download, open(reconnected), status(online)}
export function connectEvents(handlers) {
  let delay = 1000;
  let everOpened = false;
  function connect() {
    const es = new EventSource("/api/events");
    es.onopen = () => {
      delay = 1000;
      handlers.status?.(true);
      handlers.open?.(everOpened);
      everOpened = true;
    };
    for (const kind of ["job", "run", "download", "model", "log", "refs", "enhanced", "chain", "style"]) {
      es.addEventListener(kind, (ev) => {
        try {
          handlers[kind]?.(JSON.parse(ev.data));
        } catch (err) {
          console.error("bad event", kind, err);
        }
      });
    }
    es.onerror = () => {
      es.close();
      handlers.status?.(false);
      setTimeout(connect, delay);
      delay = Math.min(delay * 1.7, 8000);
    };
  }
  connect();
}

// A second SSE connection that also receives `serverlog` lines (opt-in per connection, see API.md). Returns a stop function.
export function connectServerLog(onLine) {
  let es, closed = false, delay = 1000;
  function connect() {
    es = new EventSource("/api/events?serverlog=1");
    es.onopen = () => (delay = 1000);
    es.addEventListener("serverlog", (ev) => {
      try {
        onLine(JSON.parse(ev.data).line);
      } catch {}
    });
    es.onerror = () => {
      es.close();
      if (!closed) setTimeout(connect, (delay = Math.min(delay * 1.7, 8000)));
    };
  }
  connect();
  return () => {
    closed = true;
    es?.close();
  };
}
