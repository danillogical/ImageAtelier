// Service worker: app shell is precached (stale-while-revalidate); /api/* is network-first; the SSE stream is never touched.
const VERSION = "v18";
const SHELL_CACHE = `imageatelier-shell-${VERSION}`;
const API_CACHE = `imageatelier-api-${VERSION}`;
const SHELL = [
  "/", "/style.css", "/manifest.webmanifest",
  "/js/app.js", "/js/refs.js", "/js/cropper.js", "/js/gestures.js", "/js/jobs.js", "/js/imagemodels.js", "/js/api.js", "/js/util.js", "/js/icons.js", "/js/lightbox.js", "/js/settings.js", "/js/state.js", "/js/styles.js",
  "/icons/icon-192.png", "/icons/icon-512.png", "/icons/apple-touch-icon.png", "/icons/favicon-32.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => ![SHELL_CACHE, API_CACHE].includes(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function networkFirst(req) {
  try {
    const res = await fetch(req);
    const type = res.headers.get("content-type") || "";
    // Keep JSON and thumbnails for offline browsing; full-size PNGs are too big to hoard.
    if (res.ok && (type.includes("json") || type.includes("webp"))) {
      (await caches.open(API_CACHE)).put(req, res.clone());
    }
    return res;
  } catch (err) {
    const hit = await caches.match(req, { cacheName: API_CACHE });
    if (hit) return hit;
    throw err;
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(SHELL_CACHE);
  const hit = await cache.match(req, { ignoreSearch: true });
  const fresh = fetch(req).then((res) => {
    if (res.ok) cache.put(req, res.clone());
    return res;
  });
  if (hit) {
    fresh.catch(() => {});
    return hit;
  }
  return fresh;
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  if (url.pathname === "/api/events" || req.headers.get("accept") === "text/event-stream") return;
  if (url.pathname.startsWith("/api/")) {
    e.respondWith(networkFirst(req));
  } else if (req.mode === "navigate") {
    e.respondWith(staleWhileRevalidate(new Request("/")));
  } else {
    e.respondWith(staleWhileRevalidate(req));
  }
});
