// Run with: node --experimental-vm-modules --test tests/download_settings.mjs
// Exercises the real Settings renderer and click handlers without controlling the desktop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

async function settings(state, { confirm = true, q8 = "not_converted", custom = [], networkInfo = null } = {}) {
  const calls = [];
  const dialogs = [];
  const body = { children: [], append(node) { this.children.push(node); } };
  const events = new Map();
  const context = vm.createContext({
    window: { addEventListener: (name, fn) => events.set(name, fn), dispatchEvent: (event) => events.get(event.type)?.(event) },
    document: { body, addEventListener() {}, removeEventListener() {} },
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    requestAnimationFrame: (fn) => fn(), setTimeout: (fn, ms) => { const timer = setTimeout(fn, ms); timer.unref(); return timer; }, clearTimeout,
  });
  const S = { settings: { model_suite: "q8", custom_image: "", custom_text_encoder: "" }, models: [] };
  const rows = [{ kind: "suite", name: "bf16", label: "BF16", state, bytes_done: 100, bytes_total: 200, retry_in: 5 },
    { kind: "suite", name: "q8", label: "Q8", state: q8, can_convert: state === "ready" }];
  const h = (tag, attrs, ...children) => ({ tag, attrs, children: children.flat().filter(Boolean), classList: { add() {}, remove() {} }, remove() {} });
  const api = async (method, path, payload) => {
    calls.push([method, path, payload]);
    if (method === "PUT" && path === "/settings") { Object.assign(S.settings, payload); return S.settings; }
    if (method === "DELETE" && path.startsWith("/custom-models/")) {
      const id = path.split("/").at(-1);
      const i = custom.findIndex((r) => r.id === id);
      custom.splice(i, 1);
      for (const key of ["custom_image", "custom_text_encoder", "custom_pe_t2i", "custom_pe_i2i"]) {
        if (S.settings[key] === id) S.settings[key] = "";
      }
      return { deleted: true };
    }
    if (method === "DELETE") {
      rows.find((r) => path === `/models/${r.name}`).state = "not_downloaded";
      return { deleted: true, freed_bytes: 100 };
    }
    return path === "/models" ? rows : path === "/settings" ? S.settings : path === "/custom-models" ? custom : path === "/network" ? networkInfo : {};
  };
  const mocks = {
    "./api.js": { api }, "./state.js": { S }, "./icons.js": { icon: () => null },
    "./util.js": { h, fill: (node, ...children) => { node.children = children; }, toast() {}, fmtBytes: String,
      copyText: async () => true, confirmDialog: async (options) => { dialogs.push(options); return confirm; } },
  };
  const mod = new vm.SourceTextModule(await readFile(new URL("../imageatelier/static/js/settings.js", import.meta.url), "utf8"), { context });
  await mod.link(async (name) => {
    const values = mocks[name];
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    }, { context });
  });
  await mod.evaluate();
  await mod.namespace.openSettings();
  function nodes() {
    const all = [];
    const visit = (node) => { if (node && typeof node === "object") { all.push(node); node.children?.forEach(visit); } };
    visit(body);
    return all;
  }
  return { get all() { return nodes(); }, calls, dialogs, rows, S };

}

for (const name of ["bf16", "q8"]) {
  test(`installed ${name} offers confirmed whole-suite deletion`, async () => {
    const { all, calls, dialogs, rows, S } = await settings("ready", { q8: "ready" });
    const button = all.find((n) => n.attrs?.["aria-label"] === `Delete ${name.toUpperCase()}`);
    assert.ok(button);
    assert.equal(button.attrs.disabled, false);
    button.attrs.onclick({ stopPropagation() {} });
    await new Promise(setImmediate);
    assert.equal(dialogs.length, 1);
    assert.match(dialogs[0].message, /entire.*suite/);
    assert.ok(calls.some(([method, path]) => method === "DELETE" && path === `/models/${name}`));
    assert.equal(rows.find((r) => r.name === name).state, "not_downloaded");
    assert.equal(S.settings.model_suite, "q8");
  });
}

test("cancelling confirmation never deletes files", async () => {
  const { all, calls } = await settings("ready", { confirm: false });
  all.find((n) => n.attrs?.["aria-label"] === "Delete BF16").attrs.onclick({ stopPropagation() {} });
  await new Promise(setImmediate);
  assert.ok(!calls.some(([method]) => method === "DELETE"));
});

test("active preparation disables deletion of installed suites", async () => {
  const { all } = await settings("downloading", { q8: "ready" });
  assert.equal(all.find((n) => n.attrs?.["aria-label"] === "Delete Q8").attrs.disabled, true);
});

for (const [state, label, action] of [["downloading", "Pause", "pause"], ["retrying", "Pause", "pause"],
  ["paused", "Resume", "resume"], ["failed", "Retry", "resume"], ["not_downloaded", "Download", "download"]]) {
  test(`${state} offers enabled ${label} with the correct endpoint`, async () => {
    const { all, calls } = await settings(state);
    const button = all.find((n) => n.attrs?.["aria-label"] === `${label} BF16`);
    assert.ok(button);
    assert.equal(button.attrs.disabled, false);
    assert.equal(all.find((n) => n.attrs?.["aria-label"] === "Convert Q8").attrs.disabled, true);
    button.attrs.onclick({ stopPropagation() {} });
    await new Promise(setImmediate);
    assert.ok(calls.some(([method, path]) => method === "POST" && path === `/models/bf16/${action}`));
  });
}


test("custom component selection persists independently of the base suite", async () => {
  const custom = [{ id: "custom-1111111111111111", kind: "image", label: "Anime", state: "ready" }];
  const { all, calls, S } = await settings("ready", { q8: "ready", custom });
  const select = all.find((n) => n.attrs?.["aria-label"] === "Anime");
  await select.attrs.onclick();
  assert.equal(S.settings.custom_image, custom[0].id);
  assert.equal(S.settings.model_suite, "q8");
  assert.ok(calls.some(([method, path, payload]) => method === "PUT" && path === "/settings" && payload.custom_image === custom[0].id));
});

test("custom preparation requires a ready selected base and disables suite deletion", async () => {
  const custom = [{ id: "custom-1111111111111111", kind: "image", label: "Anime", state: "not_prepared" }];
  let result = await settings("ready", { q8: "not_converted", custom });
  assert.equal(result.all.find((n) => n.attrs?.["aria-label"] === "Prepare Anime").attrs.disabled, true);
  result = await settings("ready", { q8: "ready", custom });
  assert.equal(result.all.find((n) => n.attrs?.["aria-label"] === "Prepare Anime").attrs.disabled, false);
  custom[0].state = "preparing";
  result = await settings("ready", { q8: "ready", custom });
  assert.equal(result.all.find((n) => n.attrs?.["aria-label"] === "Delete Q8").attrs.disabled, true);
});


for (const [kind, key, title] of [["image", "custom_image", "Image Transformers"],
  ["text_encoder", "custom_text_encoder", "Text and Vision Encoders"],
  ["pe_t2i", "custom_pe_t2i", "T2I Prompt Enhancer"], ["pe_i2i", "custom_pe_i2i", "I2I Prompt Enhancer"]]) {
  test(`${title} allows one choice or none`, async () => {
    const custom = [1, 2].map((n) => ({ id: `custom-${String(n).repeat(16)}`, kind, label: `Model ${n}`, state: "ready" }));
    const result = await settings("ready", { q8: "ready", custom });
    const choice = (label) => result.all.find((n) => n.attrs?.["aria-label"] === label && n.attrs?.role === "radio");
    await choice("Model 1").attrs.onclick();
    assert.equal(result.S.settings[key], custom[0].id);
    await choice("Model 2").attrs.onclick();
    assert.equal(result.S.settings[key], custom[1].id);
    assert.equal(choice("Model 1").attrs["aria-checked"], "false");
    assert.equal(choice("Model 2").attrs["aria-checked"], "true");
    await choice("Model 2").attrs.onclick();
    assert.equal(result.S.settings[key], "");
    assert.equal(choice("Model 2").attrs["aria-checked"], "false");
    assert.ok(result.all.some((n) => n.attrs?.role === "radiogroup" && n.attrs?.["aria-label"] === title));
    assert.ok(!result.all.some((n) => n.tag === "select" || n.tag === "p"));
  });
}

test("deleting a selected custom card clears its selection", async () => {
  const custom = [{ id: "custom-1111111111111111", kind: "image", label: "Anime", state: "ready" }];
  const result = await settings("ready", { q8: "ready", custom });
  await result.all.find((n) => n.attrs?.["aria-label"] === "Anime").attrs.onclick();
  await result.all.find((n) => n.attrs?.["aria-label"] === "Delete Anime").attrs.onclick();
  assert.equal(result.S.settings.custom_image, "");
  assert.equal(custom.length, 0);
  assert.equal(result.dialogs.length, 1);
});


test("network toggles independently persist API and UI listeners", async () => {
  const networkInfo = { api_url: "http://laptop.local:8401", ui_url: "http://laptop.local:8402", access_key: "test-key" };
  const result = await settings("ready", { q8: "ready", networkInfo });
  const toggle = (name) => result.all.find((n) => n.attrs?.["aria-label"] === name && n.attrs?.role === "switch");
  await toggle("Worlds API").attrs.onchange({ target: { checked: true } });
  assert.equal(result.S.settings.network_api_enabled, true);
  assert.ok(!result.S.settings.network_ui_enabled);
  await toggle("Network UI").attrs.onchange({ target: { checked: true } });
  assert.equal(result.S.settings.network_ui_enabled, true);
  await toggle("Worlds API").attrs.onchange({ target: { checked: false } });
  assert.equal(result.S.settings.network_api_enabled, false);
  assert.equal(result.S.settings.network_ui_enabled, true);
  assert.ok(result.all.some((n) => n.attrs?.type === "password" && n.attrs?.readOnly));
});
