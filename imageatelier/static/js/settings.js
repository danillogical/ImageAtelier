import { api } from "./api.js";
import { S } from "./state.js";
import { h, fill, toast, fmtBytes } from "./util.js";
import { icon } from "./icons.js";
import { modeOptions, optName, isCurrent, tagline, imageModels, textEncoders, currentImage, prettyLabel } from "./imagemodels.js";

let root = null; // backdrop element while the sheet is open
let secImage, cardsBox, modeBox, stepsBox, secTE, teBox, secPE, advBody;
const errors = {}; // section key -> message shown inline
let notice = ""; // message passed by the opener (e.g. after a 409)
let noticeSection = "model";

// The composer's model picker listens for this to stay in step with the sheet (and vice versa).
// kind "models" = only download status changed; anything else = settings changed.
export const announceConfig = (kind = "settings") => window.dispatchEvent(new CustomEvent("ws-config", { detail: kind }));

export async function loadSettings() {
  [S.settings, S.models] = await Promise.all([api("GET", "/settings"), api("GET", "/models")]);
  announceConfig();
}

async function put(partial, key) {
  errors[key] = "";
  try {
    S.settings = await api("PUT", "/settings", partial);
  } catch (e) {
    errors[key] = e.message;
  }
  renderAll();
  announceConfig();
}

function renderAll() {
  renderImage();
  renderTE();
  renderPE();
  renderAdv();
}
function renderModelsOnly() {
  renderImageModels();
  renderTE();
  renderPE();
}

const errBox = (key) => (errors[key] ? h("div", { class: "field-err", role: "alert" }, errors[key]) : null);
const noticeBox = (section) => (notice && noticeSection === section ? h("div", { class: "notice" }, notice) : null);

function toggleRow(title, desc, checked, onChange) {
  const input = h("input", { type: "checkbox", checked, onchange: () => onChange(input.checked) });
  return h("label", { class: "row-setting" },
    h("div", { class: "grow" }, h("div", { class: "rs-title" }, title), h("div", { class: "rs-desc" }, desc)),
    h("span", { class: "switch" }, input, h("span", { class: "knob" })));
}
const progress = (done, total) => {
  const pct = total ? Math.min(100, (100 * done) / total) : null;
  return h("div", { class: "dl" },
    h("div", { class: `bar${pct == null ? " indet" : ""}` }, h("i", { style: { width: pct == null ? "" : `${pct}%` } })),
    h("small", {}, total ? `${fmtBytes(done)} of ${fmtBytes(total)}` : "Starting download…"));
};
// A selectable option card (radio + title + description + status area).
function optionCard({ selected, disabled, onPick, title, desc, status }) {
  return h("div", {
    class: `model-card opt${selected ? " selected" : ""}${disabled ? " disabled" : " selectable"}`, role: "radio", "aria-checked": selected, "aria-disabled": disabled,
    onclick: () => !disabled && !selected && onPick(),
  }, h("span", { class: "radio" }, selected ? icon("check", 14) : null),
    h("div", { class: "grow" }, h("div", { class: "rs-title" }, title), h("div", { class: "rs-desc" }, desc), status));
}

// ---- Image model ----
// kind: "image" | "text_encoder" | "pe"
async function startDownload(kind, name) {
  const key = kind === "image" ? "model" : kind === "text_encoder" ? "te" : "pe";
  errors[key] = "";
  try {
    await api("POST", kind === "pe" ? `/models/pe/${name}/download` : `/models/${kind}/${name}/download`);
    S.models = await api("GET", "/models");
    announceConfig("models");
  } catch (e) {
    errors[key] = e.message;
  }
  renderModelsOnly();
}

// Delete downloaded parts (serially) after a confirm; the server refuses with a 409 if they're local, downloading or in use.
async function deleteModels(kind, items, label, selected) {
  const key = kind === "image" ? "model" : kind === "text_encoder" ? "te" : "pe";
  const gb = Math.round(items.reduce((n, m) => n + m.approx_gb, 0) * 10) / 10;
  if (!confirm(`Delete ${label} (${gb} GB)? The files are removed from the Hugging Face cache on the server and can be downloaded again later.${selected ? " Generating will fail until you download it again." : ""}`)) return;
  errors[key] = "";
  try {
    for (const m of items) await api("DELETE", `/models/${kind}/${m.name}`);
  } catch (e) {
    errors[key] = e.message;
  }
  try {
    S.models = await api("GET", "/models");
    announceConfig("models");
  } catch (e) {
    errors[key] ||= e.message;
  }
  renderModelsOnly();
}
const deleteBtn = (kind, items, label, selected) =>
  h("button", { class: "btn small danger-text", type: "button", onclick: (e) => { e.stopPropagation(); deleteModels(kind, items, label, selected); } }, "Delete");

function renderImageModels() {
  if (!cardsBox || !S.settings) return;
  const m = currentImage();
  let status;
  if (m?.state === "downloading") status = progress(m.bytes_done, m.bytes_total);
  else if (m && m.state !== "ready") {
    status = h("div", { class: "pe-status row" }, h("small", { class: "dim grow" }, `Not downloaded · ${m.approx_gb} GB`),
      h("button", { class: "btn small primary", type: "button", onclick: (e) => { e.stopPropagation(); startDownload("image", m.name); } }, "Download"));
  } else if (m) {
    status = h("div", { class: "pe-status row" }, h("small", { class: "ok grow" }, `Ready · ${m.approx_gb} GB`),
      m.source !== "local" ? deleteBtn("image", [m], prettyLabel(m.label), true) : null);
  }
  fill(cardsBox,
    noticeBox("model"),
    m ? optionCard({ selected: true, title: prettyLabel(m.label), desc: "The model that paints your images.", status, onPick: () => {} }) : null,
    m?.error && m.state !== "ready" ? h("small", { class: "err" }, m.error) : null,
    errBox("model"));
  // Mode: Turbo needs the turbo add-on; Standard needs only the image model.
  const ready = m?.state === "ready";
  const turboOk = ready && m.turbo_lora_ready;
  const mode = (turbo) => {
    const o = { turbo };
    const ok = turbo ? turboOk : ready;
    let status;
    if (turbo && ready && !turboOk) {
      status = h("div", { class: "pe-status row" }, h("small", { class: "dim grow" }, "Needs the turbo add-on"),
        h("button", { class: "btn small primary", type: "button", onclick: (e) => { e.stopPropagation(); startDownload("image", m.name); } }, "Get turbo"));
    }
    return optionCard({ selected: isCurrent(o), disabled: !ok, title: optName(o), desc: ok || !ready ? tagline(o) : tagline(o), status, onPick: () => put({ turbo }, "model") });
  };
  fill(modeBox, h("div", { class: "rs-title" }, "Mode"), h("div", { class: "opt-list", role: "radiogroup", "aria-label": "Mode" }, mode(true), mode(false)));
}

function renderSteps() {
  const st = S.settings;
  if (!stepsBox || !st) return;
  let control;
  if (st.turbo) {
    control = h("div", { class: "segmented", role: "group", "aria-label": "Steps" },
      [6, 8].map((n) => h("button", { class: st.steps === n ? "on" : "", type: "button", onclick: () => st.steps !== n && put({ steps: n }, "gen") }, `${n} steps`)));
  } else {
    const num = h("input", { type: "number", min: 2, max: 100, value: st.steps, class: "num", "aria-label": "Steps" });
    const range = h("input", { type: "range", min: 2, max: 100, value: st.steps, "aria-label": "Steps slider" });
    range.addEventListener("input", () => (num.value = range.value));
    num.addEventListener("input", () => (range.value = num.value));
    const commit = (el) => () => put({ steps: Number(el.value) }, "gen");
    range.addEventListener("change", commit(range));
    num.addEventListener("change", commit(num));
    control = h("div", { class: "steps-row" }, range, num);
  }
  fill(stepsBox, h("div", { class: "rs-title" }, "Steps"), control, errBox("gen"));
}
export function renderImage() {
  renderImageModels();
  renderSteps();
}

// ---- Text encoder ----
function renderTE() {
  if (!teBox || !S.settings) return;
  const st = S.settings;
  const DESC = {
    "heretic-8bit": ["Heretic (uncensored)", "Reads your prompt as written, without softening it."],
    "official-8bit": ["Official (Qwen)", "The standard Qwen text encoder."],
  };
  const cards = textEncoders().map((t) => {
    let status;
    if (t.state === "ready") {
      status = t.source === "local" ? h("small", { class: "ok" }, "Ready")
        : h("div", { class: "pe-status row" }, h("small", { class: "ok grow" }, "Ready"), deleteBtn("text_encoder", [t], t.label, st.text_encoder === t.name));
    }
    else if (t.state === "downloading") status = progress(t.bytes_done, t.bytes_total);
    else if (t.source === "local") {
      status = h("div", { class: "pe-status", title: t.hint || "" }, h("small", { class: "dim" }, "Not installed"),
        h("small", { class: "dim hint" }, "Copy its folder into models/ on the server."));
    } else {
      status = h("div", { class: "pe-status row" }, h("small", { class: "dim grow" }, `Not downloaded · ${t.approx_gb} GB`),
        h("button", { class: "btn small primary", type: "button", onclick: (e) => { e.stopPropagation(); startDownload("text_encoder", t.name); } }, "Download"));
    }
    const [title, desc] = DESC[t.name] || [t.label, ""];
    return optionCard({ selected: st.text_encoder === t.name, disabled: t.source === "local" && t.state !== "ready", title, desc, status, onPick: () => put({ text_encoder: t.name }, "te") });
  });
  const cur = textEncoders().find((t) => t.name === st.text_encoder);
  fill(teBox,
    noticeBox("te"),
    h("div", { class: "opt-list", role: "radiogroup", "aria-label": "Text encoder" }, cards),
    cur && cur.state !== "ready" && cur.state !== "downloading" ? h("div", { class: "field-err" }, "The selected text encoder isn't ready yet, so generating will fail until it is.") : null,
    errBox("te"));
}

// ---- Prompt enhancer ----
const officialPE = () => ["t2i", "i2i"].map((mode) => S.models.find((m) => m.kind === "pe" && (m.variant || "official") === "official" && m.mode === mode && m.quant === "8bit")).filter(Boolean);
async function downloadMissing(list) {
  for (const m of list) await startDownload("pe", m.name);
}

export function renderPE() {
  if (!secPE || !S.settings) return;
  const st = S.settings;
  const off = officialPE();
  const hers = S.models.filter((m) => m.kind === "pe" && m.variant === "heretic");
  const herMissing = hers.filter((m) => m.state !== "ready");
  // Qwen (official) needs both of its models; show one combined status.
  const missing = off.filter((m) => m.state !== "ready" && m.state !== "downloading");
  const busy = off.filter((m) => m.state === "downloading");
  let qwenStatus;
  if (busy.length) {
    const total = busy.every((m) => m.bytes_total) ? busy.reduce((n, m) => n + m.bytes_total, 0) : null;
    qwenStatus = progress(busy.reduce((n, m) => n + (m.bytes_done || 0), 0), total);
  } else if (missing.length) {
    const gb = Math.round(missing.reduce((n, m) => n + m.approx_gb, 0));
    qwenStatus = h("div", { class: "pe-status row" }, h("small", { class: "dim grow" }, "Not downloaded"),
      h("button", { class: "btn small primary", type: "button", onclick: (e) => { e.stopPropagation(); downloadMissing(missing); } }, `Download (${gb} GB)`));
  } else if (off.length) {
    qwenStatus = off.every((m) => m.source !== "local")
      ? h("div", { class: "pe-status row" }, h("small", { class: "ok grow" }, "Ready"), deleteBtn("pe", off, "Qwen prompt enhancer", st.pe_variant === "official"))
      : h("small", { class: "ok" }, "Ready");
  }
  const herReady = hers.length > 0 && herMissing.length === 0; // Heretic needs both of its models installed
  const half = (m) => (m.mode === "i2i" ? "edit" : "new-image");
  const herStatus = herReady ? h("small", { class: "ok" }, "Ready")
    : h("div", { class: "pe-status", title: herMissing.map((m) => m.hint || "").filter(Boolean).join("\n") },
        h("small", { class: "dim" }, herMissing.length ? `Not installed · missing the ${herMissing.map(half).join(" and ")} part` : "Not installed"),
        h("small", { class: "dim hint" }, "Copy the Heretic model folders into models/ on the server."));
  fill(secPE,
    h("h4", {}, "Prompt enhancer"),
    noticeBox("pe"),
    h("div", { class: "opt-list", role: "radiogroup", "aria-label": "Prompt enhancer" },
      optionCard({ selected: st.pe_variant === "official", title: "Qwen (official)", desc: "Rewrites short prompts into detailed ones.", status: qwenStatus, onPick: () => put({ pe_variant: "official" }, "pe") }),
      optionCard({ selected: st.pe_variant === "heretic", disabled: !herReady, title: "Heretic (uncensored)", desc: "Same, with fewer refusals.", status: herStatus, onPick: () => put({ pe_variant: "heretic" }, "pe") })),
    st.pe_variant === "heretic" && !herReady ? h("div", { class: "field-err" }, "Heretic is selected but isn't installed. Choose Qwen, or copy the Heretic folder into models/.") : null,
    errBox("pe"));
}

// ---- Advanced ----
export function renderAdv() {
  const st = S.settings;
  if (!advBody || !st) return;
  const tokens = h("input", { type: "number", class: "num wide", min: 256, max: 16384, step: 256, value: st.pe_max_new_tokens, "aria-label": "Max tokens" });
  tokens.addEventListener("change", () => put({ pe_max_new_tokens: Number(tokens.value) }, "adv"));
  const maxRefs = h("input", { type: "number", class: "num", min: 1, max: 9, value: st.max_refs, "aria-label": "Max references" });
  maxRefs.addEventListener("change", () => put({ max_refs: Math.min(9, Math.max(1, Number(maxRefs.value) || 1)) }, "adv"));
  fill(advBody,
    h("h5", {}, "Prompt enhancer"),
    toggleRow("Thinking", "Better rewrites, ~4× slower.", st.pe_thinking, (v) => put({ pe_thinking: v }, "adv")),
    toggleRow("English output", "Always write the enhanced prompt in English. Turn off to follow the language of your prompt.", st.pe_language === "english", (v) => put({ pe_language: v ? "english" : "auto" }, "adv")),
    h("div", { class: "row-setting" },
      h("div", { class: "grow" }, h("div", { class: "rs-title" }, "Max tokens"), h("div", { class: "rs-desc" }, "Upper limit on thinking plus the enhanced prompt (256 to 16384). Raise it if it runs out of room.")),
      tokens),
    h("h5", {}, "Reference images"),
    h("div", { class: "row-setting col" },
      h("div", { class: "rs-title" }, "Reference detail"),
      h("div", { class: "segmented", role: "group", "aria-label": "Reference detail" },
        [["auto", "Auto"], ["high", "High"], ["standard", "Standard"], ["low", "Low"]].map(([v, t]) =>
          h("button", { type: "button", class: st.ref_detail === v ? "on" : "", onclick: () => st.ref_detail !== v && put({ ref_detail: v }, "adv") }, t))),
      h("div", { class: "rs-desc" }, "Auto: 1024 for one reference, 768 for two or three. High 1024, Standard 768, Low 512.")),
    h("div", { class: "row-setting" },
      h("div", { class: "grow" }, h("div", { class: "rs-title" }, "Max references"),
        h("div", { class: "rs-desc" }, "How many enabled references a generation may use (1 to 9)."),
        st.max_refs > 3 ? h("div", { class: "field-err", role: "status" }, "Turbo was trained on up to 3.") : null),
      maxRefs),
    h("h5", {}, "Memory"),
    toggleRow("Low memory", "Frees the text encoder after each prompt; saves about 5 GB for a fraction of a second.", st.low_memory, (v) => put({ low_memory: v }, "adv")),
    toggleRow("VAE tiling", "Decodes in tiles to save about 2 GB; a few seconds slower and pixels change slightly.", st.vae_tiling, (v) => put({ vae_tiling: v }, "adv")),
    errBox("adv"));
}

// Called from SSE `download` events: refresh only the model status so inputs and sliders aren't disturbed.
export function onDownloadStatus(list) {
  S.models = S.models.map((m) => {
    const n = list.find((x) => (x.kind || "image") === (m.kind || "image") && x.name === m.name);
    return n ? { ...m, ...n } : m;
  });
  announceConfig("models");
}

window.addEventListener("ws-config", (e) => {
  if (!root) return;
  e.detail === "models" ? renderModelsOnly() : renderAll();
});

export function closeSettings() {
  if (!root) return;
  const r = root;
  root = secImage = cardsBox = modeBox = stepsBox = secTE = teBox = secPE = advBody = null;
  r.classList.remove("in");
  document.removeEventListener("keydown", onKey, true);
  setTimeout(() => r.remove(), 220);
}
function onKey(e) {
  if (e.key === "Escape") {
    e.stopPropagation();
    closeSettings();
  }
}

let advOpen = false; // the Advanced disclosure starts collapsed and keeps its state while the sheet is open
export async function openSettings({ message = "", focus = "" } = {}) {
  if (root) return;
  notice = message;
  noticeSection = focus === "pe" ? "pe" : focus === "te" ? "te" : "model";
  advOpen = false;
  for (const k of Object.keys(errors)) errors[k] = "";
  cardsBox = h("div", { class: "opt-list" });
  modeBox = h("div", { class: "mode-box" });
  stepsBox = h("div", { class: "row-setting col steps-box" });
  secImage = h("section", { class: "sec" }, h("h4", {}, "Image model"), cardsBox, modeBox, stepsBox);
  teBox = h("div", {});
  secTE = h("section", { class: "sec" }, h("h4", {}, "Text encoder"), teBox);
  secPE = h("section", { class: "sec" });
  advBody = h("div", { class: "adv-body" });
  const adv = h("details", { class: "adv sec" }, h("summary", {}, "Advanced"), advBody);
  adv.addEventListener("toggle", () => (advOpen = adv.open));
  const sheet = h("div", { class: "sheet", role: "dialog", "aria-label": "Settings" },
    h("div", { class: "sheet-head" }, h("h3", {}, "Settings"), h("div", { class: "grow" }),
      h("button", { class: "icon-btn", "aria-label": "Close", onclick: closeSettings }, icon("close"))),
    h("div", { class: "sheet-body" }, secImage, secTE, secPE, adv));
  root = h("div", { class: "backdrop sheet-wrap", onclick: (e) => e.target === root && closeSettings() }, sheet);
  document.body.append(root);
  document.addEventListener("keydown", onKey, true);
  requestAnimationFrame(() => root?.classList.add("in"));
  try {
    if (!S.settings) await loadSettings();
    else S.models = await api("GET", "/models");
    announceConfig("models");
  } catch (e) {
    toast(e.message, "error");
  }
  renderAll();
  if (focus === "model") secImage?.scrollIntoView({ block: "start" });
  if (focus === "te") secTE?.scrollIntoView({ block: "start" });
  if (focus === "pe") secPE?.scrollIntoView({ block: "start" });
}
