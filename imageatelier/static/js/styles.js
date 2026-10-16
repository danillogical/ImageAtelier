// Per-project art style: the header chip, the picker sheet, and the composer hint text.
// The server composes the final prompt; this file only chooses and shows the style.
import { api } from "./api.js";
import { S } from "./state.js";
import { h, fill, toast, setText } from "./util.js";
import { icon } from "./icons.js";

const CUSTOM_MAX = 300;
// CSS-only swatch per preset id (no sample images).
const SWATCH = {
  none: "repeating-linear-gradient(45deg, var(--panel3) 0 6px, var(--panel2) 6px 12px)",
  photoreal: "linear-gradient(160deg, #cfd8e3, #6d7f93 55%, #2b3440)",
  cinematic: "linear-gradient(90deg, #0e3a4a, #1b1b24 50%, #e08a2e)",
  anime: "linear-gradient(135deg, #ff9ad5, #8ec5ff 55%, #fff3a8)",
  "stylized-anime": "linear-gradient(135deg, #111 0 30%, #ff3b30 30% 55%, #ffd60a 55% 75%, #111 75%)",
  "hand-drawn":"linear-gradient(135deg, #f6e9c8, #e7c98c 50%, #c9a26a)",
  watercolor: "radial-gradient(circle at 25% 30%, #7fd0e8, transparent 55%), radial-gradient(circle at 75% 65%, #f5a3c7, transparent 55%), #f3efe8",
  "oil-painting": "linear-gradient(120deg, #7a3b1e, #d08a2e 45%, #3f5f3a)",
  "pixel-art": "conic-gradient(#3b6fd1 25%, #5fcf6b 0 50%, #f0c34a 0 75%, #d14b4b 0) 0 0 / 14px 14px",
  "3d-render": "radial-gradient(circle at 35% 30%, #ffffff, #8aa0ff 45%, #2a2f6b)",
  "low-poly": "linear-gradient(135deg, #3fb6a8 25%, #2a8fa0 0 50%, #6bd1a0 0 75%, #1f6f86 0)",
  "ink-line-art": "repeating-linear-gradient(-30deg, #f4f1ea 0 5px, #15151a 5px 6px)",
  "comic-book": "radial-gradient(circle, #15151a 1.6px, transparent 2px) 0 0 / 8px 8px, #ffd23f",
  custom: "linear-gradient(135deg, var(--accent), var(--accent2))",
};

export const styleOf = (p) => p?.style || { id: "none" };
export const presetOf = (id) => S.styles.find((x) => x.id === id);
// "Watercolor", or "Custom" for custom text; "None" when there is no style.
export const styleLabel = (st) => (st?.label || presetOf(st?.id)?.label || (st?.id && st.id !== "none" ? st.id : "None"));
export const hasStyle = (st) => !!st && st.id !== "none";
// Text for the composer hint: "Style: Watercolor" or "Style: Custom · <text>".
export const hintText = (st) => `Style: ${styleLabel(st)}${st.id === "custom" && st.custom ? ` · ${st.custom}` : ""}`;

export async function loadStyles() {
  try {
    S.styles = await api("GET", "/styles");
  } catch (e) {
    console.warn("styles", e);
  }
  renderChips();
}

let chips = []; // live chip label spans
let onChange = () => {}; // app.js re-renders the hint
export function setChangeHandler(fn) {
  onChange = fn;
}

export function buildChip() {
  const label = h("span", { class: "sc-label" });
  const btn = h("button", { class: "style-chip", type: "button", "aria-haspopup": "dialog", title: "Art style", onclick: openStylePicker },
    icon("palette", 16), label, h("span", { class: "sc-caret", "aria-hidden": "true" }, "▾"));
  chips = [{ btn, label }];
  renderChips();
  return btn;
}
export function renderChips() {
  const st = styleOf(S.proj);
  for (const c of chips) {
    setText(c.label, `Style: ${styleLabel(st)}`);
    c.btn.classList.toggle("set", hasStyle(st));
  }
}

async function save(style) {
  const P = S.proj;
  if (!P) return;
  try {
    const p = await api("PATCH", `/projects/${P.id}`, { style });
    if (S.proj?.id === P.id) P.style = p.style;
  } catch (e) {
    toast(e.message, "error");
  }
  renderChips();
  onChange();
  renderSheet();
}

// The server stored a new style (this client, another tab or another device).
export function onStyleEvent(ev) {
  const P = S.proj;
  if (!P || P.id !== ev.project_id) return;
  P.style = ev.style;
  customDraft = null;
  renderChips();
  onChange();
  renderSheet();
}

let root = null;
let gridBox = null;
let customDraft = null; // text in the Custom textarea while the sheet is open (null = follow the saved style)
let customOpen = false; // the Custom card is picked but not yet saved

function card(p, selected) {
  return h("div", {
    class: `model-card opt style-card selectable${selected ? " selected" : ""}`, role: "radio", "aria-checked": selected, tabindex: 0,
    onclick: () => pick(p),
    onkeydown: (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick(p);
      }
    },
  }, h("span", { class: "swatch", style: { background: SWATCH[p.id] || SWATCH.custom } }),
    h("div", { class: "sc-text" },
      h("div", { class: "rs-title" }, p.label, selected ? h("span", { class: "sc-check" }, icon("check", 14)) : null),
      h("div", { class: "rs-desc" }, p.description)));
}

function pick(p) {
  const cur = styleOf(S.proj);
  if (p.custom || p.id === "custom") {
    customOpen = true;
    if (customDraft == null) customDraft = cur.id === "custom" ? cur.custom || "" : "";
    renderSheet();
    gridBox?.querySelector("textarea")?.focus();
    return;
  }
  customOpen = false;
  if (cur.id === p.id) return renderSheet();
  save({ id: p.id });
}

function customBox() {
  const cur = styleOf(S.proj);
  const ta = h("textarea", { class: "style-custom", rows: 3, maxLength: CUSTOM_MAX, placeholder: "Describe the look, e.g. moody 1970s film still, grainy, muted colors", "aria-label": "Custom style" });
  ta.value = customDraft ?? "";
  const count = h("small", { class: "dim" }, `${ta.value.length}/${CUSTOM_MAX}`);
  const btn = h("button", { class: "btn small primary", type: "button" }, "Use this style");
  const sync = () => {
    customDraft = ta.value;
    setText(count, `${ta.value.length}/${CUSTOM_MAX}`);
    btn.disabled = !ta.value.trim() || (cur.id === "custom" && cur.custom === ta.value.trim());
  };
  ta.addEventListener("input", sync);
  btn.addEventListener("click", () => {
    const text = ta.value.trim();
    if (!text) return;
    customOpen = false;
    save({ id: "custom", custom: text });
  });
  sync();
  return h("div", { class: "style-custom-box" }, ta, h("div", { class: "row" }, count, h("div", { class: "grow" }), btn));
}

function renderSheet() {
  if (!gridBox) return;
  const cur = styleOf(S.proj);
  const active = customOpen ? "custom" : cur.id;
  const kids = [];
  for (const p of S.styles) {
    kids.push(card(p, p.id === active));
  }
  fill(gridBox, kids);
  if (customOpen || cur.id === "custom") {
    const box = customBox();
    gridBox.append(box);
  }
}

export function closeStylePicker() {
  if (!root) return;
  const r = root;
  root = gridBox = null;
  customDraft = null;
  customOpen = false;
  r.classList.remove("in");
  document.removeEventListener("keydown", onKey, true);
  setTimeout(() => r.remove(), 220);
}
function onKey(e) {
  if (e.key === "Escape") {
    e.stopPropagation();
    closeStylePicker();
  }
}

export async function openStylePicker() {
  if (root || !S.proj) return;
  gridBox = h("div", { class: "style-grid", role: "radiogroup", "aria-label": "Art style" });
  const sheet = h("div", { class: "sheet", role: "dialog", "aria-label": "Art style" },
    h("div", { class: "sheet-head" }, h("h3", {}, "Art style"), h("div", { class: "grow" }),
      h("button", { class: "icon-btn", "aria-label": "Close", onclick: closeStylePicker }, icon("close"))),
    h("div", { class: "sheet-body" }, h("p", { class: "rs-desc sc-intro" }, "Added to every prompt in this project. New images get the full style; edits are asked to keep it."), gridBox));
  root = h("div", { class: "backdrop sheet-wrap", onclick: (e) => e.target === root && closeStylePicker() }, sheet);
  document.body.append(root);
  document.addEventListener("keydown", onKey, true);
  requestAnimationFrame(() => root?.classList.add("in"));
  if (!S.styles.length) await loadStyles();
  renderSheet();
}
