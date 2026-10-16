// Shared pinch / pan / wheel / double-tap handling for the lightbox and the crop editor.
//
// The image is centred in `stage`; state is {s, x, y}: a scale (>= 1, relative to the "fit" size the caller lays out) and a pan
// offset in px from the stage centre. The caller supplies limits() = {imgW, imgH, boxW, boxH} (the unscaled image size and the box it
// must cover or fit), and pan is clamped so the scaled image always covers the box in any dimension where it is larger.
// For the crop editor the box is the crop frame and the base size already covers it, so the frame can never show padding.

const SWIPE_PX = 60;

export function zoomPan(stage, opts) {
  const { limits, onChange, maxScale = 8, doubleTapScale = 2.5, swipe = null } = opts;
  // swipe: {onSwipe(dir)} enables horizontal swiping between items while unzoomed (lightbox only).
  let s = 1, x = 0, y = 0, swipeX = 0;
  const pointers = new Map();
  let pinch = null, moved = false, lastTap = { t: 0, x: 0, y: 0 };

  const emit = (animate = false) => onChange({ s, x, y, swipe: swipeX }, animate);
  function clamp() {
    const l = limits();
    const mx = Math.max(0, (l.imgW * s - l.boxW) / 2);
    const my = Math.max(0, (l.imgH * s - l.boxH) / 2);
    x = Math.min(mx, Math.max(-mx, x));
    y = Math.min(my, Math.max(-my, y));
  }
  function rel(cx, cy) {
    const r = stage.getBoundingClientRect();
    return { x: cx - (r.left + r.width / 2), y: cy - (r.top + r.height / 2) };
  }
  function zoomTo(ns, p, animate = false) {
    ns = Math.min(maxScale, Math.max(1, ns));
    const k = ns / s;
    x = p.x - (p.x - x) * k;
    y = p.y - (p.y - y) * k;
    s = ns;
    clamp();
    emit(animate);
  }

  stage.addEventListener("pointerdown", (e) => {
    stage.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY });
    if (pointers.size === 1) moved = false;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, s, x, y, mid: rel((a.x + b.x) / 2, (a.y + b.y) / 2) };
      swipeX = 0;
      moved = true;
    }
  });
  stage.addEventListener("pointermove", (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      const ns = Math.min(maxScale, Math.max(1, pinch.s * (Math.hypot(a.x - b.x, a.y - b.y) / pinch.d)));
      const mid = rel((a.x + b.x) / 2, (a.y + b.y) / 2), k = ns / pinch.s;
      x = mid.x - (pinch.mid.x - pinch.x) * k;
      y = mid.y - (pinch.mid.y - pinch.y) * k;
      s = ns;
      clamp();
      emit();
      return;
    }
    if (Math.hypot(e.clientX - p.sx, e.clientY - p.sy) > 6) moved = true;
    if (!moved) return;
    if (swipe && s <= 1.01) swipeX += dx;
    else { x += dx; y += dy; clamp(); }
    emit();
  });
  const end = (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (pointers.size === 1) {
      // Pinch ended with one finger left: carry on as a pan from here, not a tap.
      pinch = null;
      moved = true;
      return;
    }
    if (pointers.size > 0) return;
    pinch = null;
    if (e.type === "pointercancel") { swipeX = 0; emit(true); return; }
    if (!moved) {
      const now = performance.now();
      if (now - lastTap.t < 320 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
        lastTap.t = 0;
        // Double tap/click: back to the fit size, or (when doubleTapScale is set) toggle a closer look.
        if (s > 1.01 || !doubleTapScale) api.reset(true);
        else zoomTo(doubleTapScale, rel(e.clientX, e.clientY), true);
      } else lastTap = { t: now, x: e.clientX, y: e.clientY };
      return;
    }
    if (swipe && s <= 1.01) {
      const d = swipeX;
      swipeX = 0;
      if (Math.abs(d) > SWIPE_PX) swipe.onSwipe(d < 0 ? 1 : -1);
      else emit(true);
    }
  };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    // Trackpad pinch arrives as ctrl+wheel with small deltas.
    zoomTo(s * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), rel(e.clientX, e.clientY));
  }, { passive: false });

  const api = {
    get state() { return { s, x, y }; },
    reset(animate = false) {
      s = 1; x = y = swipeX = 0;
      emit(animate);
    },
    snapBack() { swipeX = 0; emit(true); },
    refresh() { clamp(); emit(); }, // after the layout changed
  };
  return api;
}
