import { h } from "./util.js";
import { icon } from "./icons.js";
import { zoomPan } from "./gestures.js";

// items: [{url, title, filename}]
export function openLightbox(items, start = 0) {
  if (!items.length) return;
  let i = Math.min(Math.max(start, 0), items.length - 1);

  const img = h("img", { class: "lb-img", draggable: false, alt: "" });
  const counter = h("span", { class: "lb-count" });
  const title = h("div", { class: "lb-title" });
  const dl = h("a", { class: "icon-btn", title: "Download", "aria-label": "Download" }, icon("download"));
  const prev = h("button", { class: "icon-btn lb-nav left", "aria-label": "Previous", onclick: () => go(-1) }, icon("left", 24));
  const next = h("button", { class: "icon-btn lb-nav right", "aria-label": "Next", onclick: () => go(1) }, icon("right", 24));
  const stage = h("div", { class: "lb-stage" }, img);
  const root = h("div", { class: "lightbox", role: "dialog", "aria-modal": "true" },
    h("div", { class: "lb-top" }, counter, h("div", { class: "grow" }), dl,
      h("button", { class: "icon-btn", title: "Close", "aria-label": "Close", onclick: close }, icon("close"))),
    stage, prev, next, title);

  const zp = zoomPan(stage, {
    limits: () => ({ imgW: img.offsetWidth, imgH: img.offsetHeight, boxW: stage.clientWidth, boxH: stage.clientHeight }),
    swipe: { onSwipe: (d) => go(d) },
    onChange: ({ s, x, y, swipe }, animate) => {
      img.classList.toggle("animate", animate);
      img.style.transform = `translate(${x + swipe}px, ${y}px) scale(${s})`;
      root.classList.toggle("zoomed", s > 1.01);
    },
  });
  function show() {
    const it = items[i];
    img.classList.toggle("checker", !!it.transparent);
    img.src = it.url;
    counter.textContent = items.length > 1 ? `${i + 1} / ${items.length}` : "";
    title.textContent = it.title || "";
    dl.href = it.url;
    dl.download = it.filename || "image.png";
    prev.hidden = i === 0;
    next.hidden = i === items.length - 1;
    zp.reset();
    for (const j of [i - 1, i + 1]) if (items[j]) new Image().src = items[j].url;
  }
  function go(d) {
    const n = i + d;
    if (n < 0 || n >= items.length) {
      zp.snapBack();
      return;
    }
    i = n;
    show();
  }
  function close() {
    document.removeEventListener("keydown", onKey, true);
    root.classList.remove("in");
    setTimeout(() => root.remove(), 180);
    document.body.classList.remove("no-scroll");
  }
  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
  }

  document.body.append(root);
  document.body.classList.add("no-scroll");
  document.addEventListener("keydown", onKey, true);
  show();
  requestAnimationFrame(() => root.classList.add("in"));
}
