// Run with: node --experimental-vm-modules --test tests/download_settings.mjs
// Exercises the real Settings renderer and click handlers without controlling the desktop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

async function settings(state) {
  const calls = [];
  const body = { children: [], append(node) { this.children.push(node); } };
  const events = new Map();
  const context = vm.createContext({
    window: { addEventListener: (name, fn) => events.set(name, fn), dispatchEvent: (event) => events.get(event.type)?.(event) },
    document: { body, addEventListener() {}, removeEventListener() {} },
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    requestAnimationFrame: (fn) => fn(), setTimeout,
  });
  const S = { settings: { model_suite: "q8" }, models: [] };
  const rows = [{ kind: "suite", name: "bf16", label: "BF16", state, bytes_done: 100, bytes_total: 200, retry_in: 5 },
    { kind: "suite", name: "q8", label: "Q8", state: "not_converted", can_convert: false }];
  const h = (tag, attrs, ...children) => ({ tag, attrs, children: children.flat().filter(Boolean), classList: { add() {}, remove() {} } });
  const api = async (method, path) => {
    calls.push([method, path]);
    return path === "/models" ? rows : path === "/settings" ? S.settings : {};
  };
  const mocks = {
    "./api.js": { api }, "./state.js": { S }, "./icons.js": { icon: () => null },
    "./util.js": { h, fill: (node, ...children) => { node.children = children; }, toast() {}, fmtBytes: String },
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
  const all = [];
  const visit = (node) => { if (node && typeof node === "object") { all.push(node); node.children?.forEach(visit); } };
  visit(body);
  return { all, calls };
}

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
