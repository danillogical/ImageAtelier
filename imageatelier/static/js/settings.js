import { api } from "./api.js";
import { S } from "./state.js";
import { h, fill, toast, fmtBytes } from "./util.js";
import { icon } from "./icons.js";

let root = null;
let content = null;
let notice = "";
let error = "";
const pending = new Set();
export const announceConfig = (kind = "settings") => window.dispatchEvent(new CustomEvent("ws-config", { detail: kind }));

export async function loadSettings() {
  [S.settings, S.models] = await Promise.all([api("GET", "/settings"), api("GET", "/models")]);
  announceConfig();
}

async function selectSuite(name) {
  error = "";
  try {
    S.settings = await api("PUT", "/settings", { model_suite: name });
    announceConfig();
  } catch (e) {
    error = e.message;
  }
  render();
}

async function prepareSuite(name, action) {
  if (pending.has(name)) return;
  pending.add(name);
  error = "";
  render();
  try {
    await api("POST", `/models/${name}/${action}`);
    S.models = await api("GET", "/models");
    announceConfig("models");
  } catch (e) {
    error = e.message;
  } finally {
    pending.delete(name);
    render();
  }
}

function render() {
  if (!content || !S.settings) return;
  const converting = S.models.some((m) => m.state === "converting");
  const rows = S.models.filter((m) => m.kind === "suite").map((suite) => {
    const ready = suite.state === "ready";
    const selected = ready && S.settings.model_suite === suite.name;
    const downloading = ["downloading", "retrying"].includes(suite.state);
    const busy = downloading || suite.state === "converting" || pending.has(suite.name);
    const blocked = pending.has(suite.name) || (suite.name !== "bf16" && (busy || !suite.can_convert || converting));
    let status = ready ? "Ready" : suite.name === "bf16" ? "70.8 GB download" : "Download BF16 first";
    if (suite.can_convert && !ready) status = "Convert the complete suite";
    if (downloading || suite.state === "paused") status = suite.bytes_total
      ? `${fmtBytes(suite.bytes_done || 0)} of ${fmtBytes(suite.bytes_total)}` : "Starting download…";
    if (suite.state === "paused") status = `Paused · ${suite.bytes_total ? status : "Progress saved"}`;
    if (suite.state === "retrying") status = `${status} · Reconnecting in ${suite.retry_in || 0}s…`;
    if (suite.state === "converting") status = suite.phase || "Converting…";
    if (pending.has(suite.name)) status = "Updating…";
    const action = suite.name !== "bf16" ? "convert" : downloading ? "pause"
      : ["paused", "failed"].includes(suite.state) ? "resume" : "download";
    const label = action === "pause" ? "Pause" : action === "resume" ? (suite.state === "failed" ? "Retry" : "Resume")
      : action === "convert" ? "Convert" : "Download";
    const button = ready ? h("small", { class: "ok" }, "Ready") : h("button", {
      class: "btn small primary", type: "button", disabled: blocked,
      "aria-label": `${label} ${suite.label}`,
      onclick: (e) => { e.stopPropagation(); prepareSuite(suite.name, action); },
    }, pending.has(suite.name) ? "Updating…" : suite.state === "converting" ? "Converting…" : label);
    const pick = () => ready && !selected && selectSuite(suite.name);
    return h("div", { class: `model-card opt${selected ? " selected" : ""}${ready ? " selectable" : ""}` },
      h("button", { class: "suite-choice", type: "button", disabled: !ready, role: "radio",
        "aria-label": suite.label, "aria-checked": String(selected), onclick: pick },
        h("span", { class: "radio" }, selected ? icon("check", 14) : null),
        h("span", { class: "grow" }, h("span", { class: "rs-title" }, suite.label, suite.recommended ? " · Recommended" : ""),
          h("small", { class: "dim" }, status))),
      button,
      suite.error ? h("small", { class: "err suite-error", role: "alert" }, suite.error) : null);
  });
  fill(content,
    notice ? h("div", { class: "notice" }, notice) : null,
    h("p", { class: "dim" }, "Each suite includes the image model, text and vision encoder, VAE, and both prompt enhancers."),
    h("p", { class: "dim" }, "Downloads save their progress and reconnect automatically if interrupted. Paused downloads wait for you to resume."),
    h("div", { class: "opt-list", role: "radiogroup", "aria-label": "Qwen-Image-2.1 suite" }, rows),
    error ? h("div", { class: "field-err", role: "alert" }, error) : null);
}

export function onDownloadStatus(list) {
  S.models = list;
  announceConfig("models");
}
window.addEventListener("ws-config", render);

function onKey(e) {
  if (e.key === "Escape") { e.stopPropagation(); closeSettings(); }
}
export function closeSettings() {
  if (!root) return;
  const old = root;
  root = content = null;
  old.classList.remove("in");
  document.removeEventListener("keydown", onKey, true);
  setTimeout(() => old.remove(), 220);
}
export async function openSettings({ message = "" } = {}) {
  if (root) return;
  notice = message;
  error = "";
  content = h("div", {});
  const sheet = h("div", { class: "sheet", role: "dialog", "aria-label": "Settings" },
    h("div", { class: "sheet-head" }, h("h3", {}, "Settings"), h("div", { class: "grow" }),
      h("button", { class: "icon-btn", "aria-label": "Close", onclick: closeSettings }, icon("close"))),
    h("div", { class: "sheet-body" }, h("section", { class: "sec" }, h("h4", {}, "Qwen-Image-2.1"), content)));
  root = h("div", { class: "backdrop sheet-wrap", onclick: (e) => e.target === root && closeSettings() }, sheet);
  document.body.append(root);
  document.addEventListener("keydown", onKey, true);
  requestAnimationFrame(() => root?.classList.add("in"));
  render();
  try { await loadSettings(); } catch (e) { toast(e.message, "error"); }
  render();
}
