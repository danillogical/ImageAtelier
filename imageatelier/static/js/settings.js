import { api } from "./api.js";
import { S } from "./state.js";
import { h, fill, toast, fmtBytes, confirmDialog, copyText } from "./util.js";
import { icon } from "./icons.js";

let root = null;
let content = null;
let notice = "";
let error = "";
let confirming = false;
let customTimer = null;
const pending = new Set();
export const announceConfig = (kind = "settings") => window.dispatchEvent(new CustomEvent("ws-config", { detail: kind }));

export async function loadSettings() {
  [S.settings, S.models, S.customModels, S.networkInfo] = await Promise.all([
    api("GET", "/settings"), api("GET", "/models"), api("GET", "/custom-models"),
    api("GET", "/network").catch(() => null),
  ]);
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
  if (pending.size) return;
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

async function deleteSuite(suite) {
  if (pending.size || confirming) return;
  const consequence = suite.name === "bf16"
    ? "Existing conversions will stay available. Download BF16 again before making new conversions."
    : "You can convert it again after downloading BF16.";
  confirming = true;
  let confirmed;
  try {
    confirmed = await confirmDialog({
      title: `Delete ${suite.label}?`,
      message: `Remove the entire ${suite.label} suite, including the image model, text and vision encoder, VAE, and both prompt enhancers? ${consequence}`
        + (S.settings.model_suite === suite.name ? " Select or prepare a ready suite before generating again." : ""),
      confirmLabel: "Delete suite",
    });
  } finally {
    confirming = false;
  }
  if (!confirmed) return;
  if (pending.size) return;
  pending.add(suite.name);
  error = "";
  render();
  try {
    const result = await api("DELETE", `/models/${suite.name}`);
    S.models = await api("GET", "/models");
    announceConfig("models");
    toast(`${suite.label} deleted · ${fmtBytes(result.freed_bytes)} freed`);
  } catch (e) {
    error = e.message;
  } finally {
    pending.delete(suite.name);
    render();
  }
}

async function setCustom(key, value) {
  if (pending.size) return;
  pending.add(`selection:${key}`);
  error = "";
  render();
  try {
    S.settings = await api("PUT", "/settings", { [key]: value });
    announceConfig();
  } catch (e) { error = e.message; }
  finally { pending.delete(`selection:${key}`); render(); }
}

async function deleteCustom(row) {
  if (pending.size || confirming) return;
  confirming = true;
  let confirmed;
  try {
    confirmed = await confirmDialog({ title: `Delete ${row.label}?`,
      message: "Remove this model and its prepared files? Existing projects and images will be kept.", confirmLabel: "Delete model" });
  } finally { confirming = false; }
  if (!confirmed) return;
  pending.add(row.id);
  error = "";
  render();
  try {
    await api("DELETE", `/custom-models/${row.id}`);
    await loadSettings();
    toast(`${row.label} deleted`);
  } catch (e) { error = e.message; }
  finally { pending.delete(row.id); render(); }
}

async function prepareCustom(row) {
  pending.add(row.id);
  error = "";
  render();
  try {
    await api("POST", `/custom-models/${row.id}/prepare`);
    await refreshCustom();
  } catch (e) { error = e.message; }
  finally { pending.delete(row.id); render(); }
}

async function refreshCustom() {
  S.customModels = await api("GET", "/custom-models");
  announceConfig("models");
}

function customSection(preparing) {
  const rows = Array.isArray(S.customModels) ? S.customModels : [];
  const baseReady = S.models.some((r) => r.name === S.settings.model_suite && r.state === "ready");
  const categories = [
    ["image", "custom_image", "Image Transformers"],
    ["text_encoder", "custom_text_encoder", "Text and Vision Encoders"],
    ["pe_t2i", "custom_pe_t2i", "T2I Prompt Enhancer"],
    ["pe_i2i", "custom_pe_i2i", "I2I Prompt Enhancer"],
  ];
  function card(row, key) {
    const ready = row.state === "ready";
    const selected = S.settings[key] === row.id;
    const busy = preparing || pending.size > 0;
    const status = pending.has(row.id) ? "Updating…" : ready ? "Ready"
      : row.state === "preparing" ? row.phase || "Preparing…"
      : row.state === "incomplete" ? "Downloading…" : row.state === "unsupported" ? "Unsupported" : "Not prepared";
    return h("div", { class: `model-card opt custom-model-card${selected ? " selected" : ""}${ready ? " selectable" : ""}` },
      h("button", { class: "suite-choice", type: "button", role: "radio",
        "aria-label": row.label, "aria-checked": String(selected), disabled: !ready || pending.size > 0,
        onclick: () => setCustom(key, selected ? "" : row.id) },
        h("span", { class: "radio" }, selected ? icon("check", 14) : null),
        h("span", { class: "grow" }, h("span", { class: "rs-title" }, row.label), h("small", { class: "dim" }, status))),
      h("div", { class: "custom-model-actions" },
        !ready && row.state !== "unsupported" && !row.kind.startsWith("pe_") ? h("button", {
          class: "btn small primary", type: "button", disabled: !baseReady || busy,
          "aria-label": `Prepare ${row.label}`, onclick: () => prepareCustom(row),
        }, row.state === "preparing" ? "Preparing…" : "Prepare") : null,
        h("button", { class: "btn small danger", type: "button", disabled: busy || row.state === "incomplete",
          "aria-label": `Delete ${row.label}`, onclick: () => deleteCustom(row) }, "Delete")),
      row.error ? h("small", { class: "err suite-error", role: "alert" }, row.error) : null);
  }
  return h("section", { class: "sec custom-models" },
    h("div", { class: "custom-models-heading" }, h("h4", {}, "Custom models"),
      h("button", { class: "btn small", type: "button", disabled: pending.size > 0,
        onclick: () => refreshCustom().catch((e) => toast(e.message, "error")) }, "Refresh models")),
    ...categories.map(([kind, key, title]) => h("section", { class: "custom-model-category" },
      h("h5", {}, title),
      h("div", { class: "opt-list", role: "radiogroup", "aria-label": title },
        rows.filter((r) => r.kind === kind).map((r) => card(r, key)),
        !rows.some((r) => r.kind === kind) ? h("small", { class: "dim custom-model-empty" }, "No models") : null))),
    ...rows.filter((r) => r.kind === "unknown").map((r) => h("small", { class: "err", role: "alert" }, `${r.label}: ${r.error}`)));
}

async function setNetwork(key, checked) {
  if (pending.size) return;
  pending.add(key);
  error = "";
  render();
  try {
    S.settings = await api("PUT", "/settings", { [key]: checked });
    S.networkInfo = await api("GET", "/network");
    announceConfig("network");
  } catch (e) { error = e.message; }
  finally { pending.delete(key); render(); }
}

function networkSection() {
  const info = S.networkInfo;
  if (!info || !info.api_url) return null;
  const rows = [
    ["network_api_enabled", "Worlds API", "Images stay in memory", info.api_url, info.api_listening],
    ["network_ui_enabled", "Network UI", "Saves projects and images", info.ui_url, info.ui_listening],
  ];
  return h("section", { class: "sec network-settings" }, h("h4", {}, "Network access"),
    ...rows.map(([key, label, detail, url, listening]) => h("div", { class: "network-setting" },
      h("div", { class: "grow" }, h("strong", {}, label), h("small", { class: "dim" }, detail),
        S.settings[key] ? h("code", { class: "network-url" }, `${url}${listening ? "" : " · Not listening"}`) : null),
      h("label", { class: "switch" }, h("input", { type: "checkbox", role: "switch", "aria-label": label,
        checked: Boolean(S.settings[key]), disabled: pending.size > 0,
        onchange: (e) => setNetwork(key, e.target.checked) }), h("span", { class: "knob" })))),
    h("div", { class: "network-key-row" }, h("span", { class: "dim" }, "Access key"),
      h("input", { type: "password", readOnly: true, value: info.access_key || "", "aria-label": "Network access key", autocomplete: "off" }),
      h("button", { class: "btn small", type: "button", onclick: async () => {
        toast((await copyText(info.access_key)) ? "Access key copied" : "Copy failed");
      } }, "Copy")),
    S.settings.network_ui_enabled ? h("small", { class: "dim" }, "UI username: imageatelier · Password: access key") : null,
    info.error ? h("small", { class: "err", role: "alert" }, info.error) : null);
}

function render() {
  if (!content || !S.settings) return;
  const converting = S.models.some((m) => m.state === "converting") || (Array.isArray(S.customModels) && S.customModels.some((m) => m.state === "preparing"));
  const preparing = converting || S.models.some((m) => ["downloading", "retrying"].includes(m.state));
  const rows = S.models.filter((m) => m.kind === "suite").map((suite) => {
    const ready = suite.state === "ready";
    const selected = ready && S.settings.model_suite === suite.name;
    const downloading = ["downloading", "retrying"].includes(suite.state);
    const busy = downloading || suite.state === "converting" || pending.has(suite.name);
    const blocked = pending.size > 0 || (suite.name !== "bf16" && (busy || !suite.can_convert || converting));
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
    const button = ready ? h("button", {
      class: "btn small danger", type: "button", disabled: preparing || pending.size > 0,
      "aria-label": `Delete ${suite.label}`,
      onclick: (e) => { e.stopPropagation(); deleteSuite(suite); },
    }, "Delete") : h("button", {
      class: "btn small primary", type: "button", disabled: blocked,
      "aria-label": `${label} ${suite.label}`,
      onclick: (e) => { e.stopPropagation(); prepareSuite(suite.name, action); },
    }, pending.has(suite.name) ? "Updating…" : suite.state === "converting" ? "Converting…" : label);
    const pick = () => ready && !selected && selectSuite(suite.name);
    return h("div", { class: `model-card opt${selected ? " selected" : ""}${ready ? " selectable" : ""}` },
      h("button", { class: "suite-choice", type: "button", disabled: !ready || pending.size > 0, role: "radio",
        "aria-label": suite.label, "aria-checked": String(selected), onclick: pick },
        h("span", { class: "radio" }, selected ? icon("check", 14) : null),
        h("span", { class: "grow" }, h("span", { class: "rs-title" }, suite.label, suite.recommended ? " · Recommended" : ""),
          h("small", { class: "dim" }, status))),
      button,
      suite.error ? h("small", { class: "err suite-error", role: "alert" }, suite.error) : null);
  });
  fill(content,
    notice ? h("div", { class: "notice" }, notice) : null,
    h("div", { class: "opt-list", role: "radiogroup", "aria-label": "Qwen-Image-2.1 suite" }, rows),
    customSection(preparing),
    networkSection(),
    error ? h("div", { class: "field-err", role: "alert" }, error) : null);
}

export function onDownloadStatus(list) {
  S.models = list;
  announceConfig("models");
}
window.addEventListener("ws-config", render);

function onKey(e) {
  if (e.key === "Escape" && !confirming) { e.stopPropagation(); closeSettings(); }
}
export function closeSettings() {
  if (!root) return;
  clearTimeout(customTimer);
  customTimer = null;
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
  const poll = async () => {
    if (!root) return;
    try {
      if (Array.isArray(S.customModels) && S.customModels.some((m) => ["preparing", "incomplete"].includes(m.state))) await refreshCustom();
    } catch { /* Retain the last status while offline. */ }
    if (root) customTimer = setTimeout(poll, 2500);
  };
  customTimer = setTimeout(poll, 2500);
}
