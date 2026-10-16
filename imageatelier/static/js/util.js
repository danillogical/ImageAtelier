// DOM helper: h("div", {class: "x", onclick: fn, dataset: {a: 1}}, child, "text", [children])
export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k in el && k !== "list" && k !== "form") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, kids);
  return el;
}
function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
export const clear = (el) => {
  el.replaceChildren();
  return el;
};
// Append children with the same flattening/null-skipping rules as h() (native append() would print "null").
export function add(el, ...kids) {
  append(el, kids);
  return el;
}
export const fill = (el, ...kids) => add(clear(el), ...kids);

let toastHost;
export function toast(message, kind = "info", ms = 4200) {
  if (!toastHost) toastHost = document.body.appendChild(h("div", { class: "toasts", role: "status", "aria-live": "polite" }));
  const t = h("div", { class: `toast ${kind}` }, message);
  toastHost.append(t);
  requestAnimationFrame(() => t.classList.add("in"));
  setTimeout(() => {
    t.classList.remove("in");
    setTimeout(() => t.remove(), 300);
  }, ms);
}

export function confirmDialog({ title, message, confirmLabel = "Delete", danger = true }) {
  return new Promise((resolve) => {
    const done = (v) => {
      back.classList.remove("in");
      document.removeEventListener("keydown", onKey, true);
      setTimeout(() => back.remove(), 200);
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        done(false);
      }
    };
    const ok = h("button", { class: `btn ${danger ? "danger" : "primary"}`, onclick: () => done(true) }, confirmLabel);
    const back = h(
      "div",
      { class: "backdrop center", onclick: (e) => e.target === back && done(false) },
      h("div", { class: "dialog", role: "alertdialog" }, h("h3", {}, title), h("p", {}, message),
        h("div", { class: "row end" }, h("button", { class: "btn", onclick: () => done(false) }, "Cancel"), ok))
    );
    document.body.append(back);
    document.addEventListener("keydown", onKey, true);
    requestAnimationFrame(() => {
      back.classList.add("in");
      ok.focus();
    });
  });
}

export const fmtSecs = (s) => (s == null ? "—" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`);
export const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
export function timeAgo(t) {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}
export const gb = (n) => `${n} GB`;
export const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);

// Clipboard write that also works on plain-HTTP origins (no secure context): falls back to a hidden textarea + execCommand.
export async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {}
  }
  const ta = h("textarea", { value: text, readOnly: true, style: { position: "fixed", top: "0", left: "0", opacity: "0" } });
  document.body.append(ta);
  ta.focus();
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {}
  ta.remove();
  return ok;
}

// Keyed list reconciliation: rows are created once and updated in place, so their animations/timers keep running.
// The container must hold only rows. create(item) -> {el, ...}; update(row, item) patches the row's DOM.
export function syncList(container, items, keyOf, create, update) {
  const rows = container.__rows || (container.__rows = new Map());
  const keep = new Set();
  items.forEach((item, i) => {
    const k = keyOf(item);
    keep.add(k);
    let row = rows.get(k);
    if (!row) {
      row = create(item);
      rows.set(k, row);
    }
    update(row, item);
    // Only move a row when it is out of place: re-inserting a node restarts its CSS animations.
    if (container.children[i] !== row.el) container.insertBefore(row.el, container.children[i] || null);
  });
  for (const [k, row] of rows) {
    if (!keep.has(k)) {
      row.el.remove();
      rows.delete(k);
    }
  }
}
export const setText = (el, text) => {
  if (el.textContent !== text) el.textContent = text;
};
export const setClass = (el, cls) => {
  if (el.className !== cls) el.className = cls;
};
// Token counts are shown exactly while small, then rounded to tens so the number isn't a blur.
export const fmtTokens = (n) => (n < 20 ? String(n) : String(Math.round(n / 10) * 10));
