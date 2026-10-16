// Reference images: the panel above the history strip and the composer hint.
// Cards are keyed by reference id and patched in place (syncList), so SSE `refs` events never rebuild what is on screen.
import { api } from "./api.js";
import { S } from "./state.js";
import { h, fill, toast, confirmDialog, syncList, setText } from "./util.js";
import { icon } from "./icons.js";
import { openLightbox } from "./lightbox.js";

const MAX_BYTES = 25 * 1024 * 1024;
let cfg = { isRO: () => false };
let panel = null, hint = null;
let open = null; // null = automatic (open on desktop when there are references, closed on mobile)

export const configure = (c) => Object.assign(cfg, c);
const isMobile = () => matchMedia("(max-width: 767.98px)").matches;
const hasCurrent = () => S.proj?.current_run != null;
// The number a reference has right now: in an edit the current image is Image 1 and references follow from 2.
export const numberOf = (r) => (hasCurrent() ? r.image_index_edit : r.image_index_new);
const enabledRefs = () => S.refs.filter((r) => r.enabled);

export async function refresh() {
  if (!S.proj) return;
  try {
    S.refs = await api("GET", `/projects/${S.proj.id}/refs`);
  } catch {}
  render();
}
export function setRefs(list) {
  S.refs = list;
  render();
}

// ---- panel ----
function note(text, kind = "info", busy = false) {
  if (!panel) return;
  panel.note.hidden = !text;
  panel.note.className = `refs-note ${kind}`;
  fill(panel.note, busy ? h("span", { class: "spinner sm" }) : null, h("span", {}, text || ""));
  clearTimeout(panel.noteTimer);
  if (text && !busy) panel.noteTimer = setTimeout(() => panel && (panel.note.hidden = true), 9000);
}

async function patch(ref, body) {
  try {
    await api("PATCH", `/projects/${S.proj.id}/refs/${ref.id}`, body);
    note("");
  } catch (e) {
    note(e.message, "err");
  }
  await refresh(); // on failure this also puts the switch/order back to the server's truth
}
function reorder(ref, to) {
  const list = [...S.refs];
  const from = list.findIndex((r) => r.id === ref.id);
  if (from < 0 || from === to) return;
  list.splice(to, 0, list.splice(from, 1)[0]);
  // optimistic local order; numbering is recomputed from the server's answer right after
  let k = 0;
  S.refs = list.map((r, i) => ({ ...r, order: i, image_index_edit: r.enabled ? ++k + 1 : null, image_index_new: r.enabled ? k : null }));
  render();
  patch(ref, { order: to });
}
async function remove(ref) {
  const ok = await confirmDialog({ title: `Delete “${ref.name}”?`, message: "This reference image is removed from the project. Runs already submitted keep their own copy." });
  if (!ok) return;
  try {
    await api("DELETE", `/projects/${S.proj.id}/refs/${ref.id}`);
    S.refs = S.refs.filter((r) => r.id !== ref.id);
    render();
  } catch (e) {
    note(e.message, "err");
  }
}
function view(ref) {
  const items = S.refs.map((r) => ({ url: r.image_url, title: `${r.name}${r.enabled ? ` · Image ${numberOf(r)}` : " · off"}`, filename: `${r.name}.png` }));
  openLightbox(items, S.refs.findIndex((r) => r.id === ref.id));
}

function makeCard(ref) {
  const img = h("img", { alt: "", draggable: false });
  const badge = h("span", { class: "rc-badge" });
  const thumb = h("div", { class: "rc-thumb", title: "Click to view; drag to reorder" }, img, badge);
  const nameText = h("span", { class: "rc-name-text", title: "Rename" });
  const nameInput = h("input", { class: "rc-name-input", maxLength: 80, hidden: true, "aria-label": "Reference name" });
  const sw = h("input", { type: "checkbox", "aria-label": "Use this reference" });
  const left = h("button", { class: "icon-btn tiny", type: "button", title: "Move earlier", "aria-label": "Move earlier" }, icon("left", 14));
  const right = h("button", { class: "icon-btn tiny", type: "button", title: "Move later", "aria-label": "Move later" }, icon("right", 14));
  const del = h("button", { class: "icon-btn tiny", type: "button", title: "Delete", "aria-label": "Delete reference" }, icon("trash", 14));
  const el = h("div", { class: "ref-card", dataset: { id: ref.id } }, thumb,
    h("div", { class: "rc-name" }, nameText, nameInput),
    h("div", { class: "rc-ctl" }, h("label", { class: "switch sm", title: "Use in generations" }, sw, h("span", { class: "knob" })), h("span", { class: "grow" }), left, right, del));
  const row = { el, img, badge, nameText, nameInput, sw, left, right, del, ref, index: 0 };

  sw.addEventListener("change", () => patch(row.ref, { enabled: sw.checked }));
  left.addEventListener("click", () => reorder(row.ref, row.index - 1));
  right.addEventListener("click", () => reorder(row.ref, row.index + 1));
  del.addEventListener("click", () => remove(row.ref));
  // rename inline
  nameText.addEventListener("click", () => {
    nameInput.value = row.ref.name;
    nameText.hidden = true;
    nameInput.hidden = false;
    nameInput.focus();
    nameInput.select();
  });
  let renaming = true;
  const finish = async (save) => {
    if (nameInput.hidden) return;
    const v = nameInput.value.trim();
    nameInput.hidden = true;
    nameText.hidden = false;
    if (save && v && v !== row.ref.name) await patch(row.ref, { name: v });
  };
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") { e.stopPropagation(); finish(false); }
  });
  nameInput.addEventListener("blur", () => finish(true));
  void renaming;

  // pointer drag to reorder (mouse and touch); a tap without movement opens the lightbox
  let drag = null;
  thumb.addEventListener("pointerdown", (e) => {
    if (e.button) return;
    thumb.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, moving: false };
  });
  thumb.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moving && Math.hypot(dx, dy) > 8) { drag.moving = true; el.classList.add("dragging"); }
    if (!drag.moving) return;
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    const target = dropTarget(e.clientX, e.clientY, el);
    for (const c of panel.cards.children) c.classList.toggle("drop", c === target);
  });
  const end = (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    el.classList.remove("dragging");
    el.style.transform = "";
    const target = d.moving ? dropTarget(e.clientX, e.clientY, el) : null;
    for (const c of panel.cards.children) c.classList.remove("drop");
    if (!d.moving) { if (e.type === "pointerup") view(row.ref); return; }
    if (target) reorder(row.ref, [...panel.cards.children].indexOf(target));
  };
  thumb.addEventListener("pointerup", end);
  thumb.addEventListener("pointercancel", end);
  return row;
}
function dropTarget(x, y, self) {
  let best = null, bd = Infinity;
  for (const c of panel.cards.children) {
    if (c === self) continue;
    const r = c.getBoundingClientRect();
    const d = Math.hypot(x - (r.left + r.width / 2), y - (r.top + r.height / 2));
    if (d < bd) { bd = d; best = c; }
  }
  return bd < 110 ? best : null;
}
function updateCard(row, ref, i, total) {
  row.ref = ref;
  row.index = i;
  const url = ref.thumb_url;
  if (row.img.getAttribute("src") !== url) row.img.src = url;
  const n = ref.enabled ? numberOf(ref) : null;
  row.badge.hidden = n == null;
  setText(row.badge, n == null ? "" : `Image ${n}`);
  row.el.classList.toggle("off", !ref.enabled);
  setText(row.nameText, ref.name);
  if (row.sw.checked !== ref.enabled) row.sw.checked = ref.enabled;
  row.left.disabled = i === 0;
  row.right.disabled = i === total - 1;
}

export function buildPanel() {
  const toggle = h("button", { class: "refs-toggle", type: "button", "aria-expanded": "true", onclick: () => { open = !isOpen(); render(); } },
    h("span", { class: "rt-title" }), icon("right", 14));
  const addTop = h("button", { class: "btn small", type: "button", onclick: () => panel.input.click() }, icon("plus", 14), "Add");
  const input = h("input", { type: "file", accept: "image/*", multiple: true, hidden: true, onchange: () => { const fs = [...input.files]; input.value = ""; addFiles(fs); } });
  const cards = h("div", { class: "ref-cards" });
  const addTile = h("button", { class: "ref-add", type: "button", onclick: () => input.click() }, icon("plus", 22), h("span", {}, "Add"));
  const grid = h("div", { class: "refs-grid" }, cards, addTile);
  const noteEl = h("div", { class: "refs-note", hidden: true, role: "status" });
  const el = h("div", { class: "refs-panel" }, h("div", { class: "refs-head" }, toggle, h("div", { class: "grow" }), addTop, input), noteEl, grid);
  panel = { el, toggle, cards, grid, note: noteEl, input, addTop, noteTimer: 0 };
  render();
  return el;
}
const isOpen = () => (open == null ? !isMobile() && S.refs.length > 0 : open);

export function render() {
  if (panel && S.proj) {
    const on = enabledRefs().length;
    setText(panel.toggle.firstChild, S.refs.length ? `References · ${on} on` : "References");
    const o = isOpen();
    panel.toggle.setAttribute("aria-expanded", o);
    panel.el.classList.toggle("open", o);
    panel.grid.hidden = !o;
    syncList(panel.cards, S.refs, (r) => r.id, makeCard, (row, r) => updateCard(row, r, S.refs.indexOf(r), S.refs.length));
  }
  renderHint();
}

async function addFiles(files) {
  if (!S.proj || !files.length) return;
  const pid = S.proj.id;
  let added = 0, lastNote = "";
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    if (f.size > MAX_BYTES) { note(`${f.name} is larger than 25 MB and was skipped.`, "err"); continue; }
    note(`Uploading ${i + 1} of ${files.length}…`, "info", true);
    const name = f.name.replace(/\.[^.]+$/, "").slice(0, 60);
    try {
      const res = await fetch(`/api/projects/${pid}/refs?${new URLSearchParams(name ? { name } : {})}`, { method: "POST", body: f });
      if (!res.ok) {
        let d = "";
        try { d = (await res.json()).detail; } catch {}
        throw new Error({ 413: `${f.name} is too large.`, 415: `${f.name} isn't a readable image.` }[res.status] || d || `Upload failed (${res.status})`);
      }
      const item = await res.json();
      added++;
      if (item.note) lastNote = item.note;
      if (S.proj?.id === pid) await refresh();
    } catch (e) {
      note(e.message, "err");
      return;
    }
  }
  if (added) note(lastNote || `Added ${added} reference${added === 1 ? "" : "s"}.`, lastNote ? "warn" : "info");
}

// ---- composer hint ----
export function buildHint() {
  const full = h("span", { class: "rh-full" });
  const pop = h("div", { class: "rh-pop", hidden: true });
  const chip = h("button", { class: "rh-chip", type: "button", onclick: () => (pop.hidden = !pop.hidden) });
  const el = h("div", { class: "refs-hint", hidden: true }, icon("image", 14), full, chip, pop);
  hint = { el, full, pop, chip };
  renderHint();
  return el;
}
function renderHint() {
  if (!hint) return;
  const on = enabledRefs();
  hint.el.hidden = !on.length || cfg.isRO();
  if (hint.el.hidden) return;
  const parts = on.map((r) => `image ${numberOf(r)} (${r.name})`).join(", ");
  const tail = hasCurrent() ? " · image 1 = current image" : "";
  const text = `Using references: ${parts}${tail}`;
  setText(hint.full, text);
  setText(hint.pop, text);
  setText(hint.chip, `+${on.length} ref${on.length === 1 ? "" : "s"}`);
}
