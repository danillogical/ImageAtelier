import { api, ApiError, connectEvents } from "./api.js";
import { S, FINAL } from "./state.js";
import { h, clear, fill, add, toast, confirmDialog, copyText, fmtSecs, fmtDate, timeAgo, syncList, setText, setClass, fmtTokens } from "./util.js";
import * as jobsUI from "./jobs.js";
import * as refsUI from "./refs.js";
import { modeOptions, optName, stepsAfter, currentEncoder, encoderName } from "./imagemodels.js";
import { icon } from "./icons.js";
import { openLightbox } from "./lightbox.js";
import { openCropper } from "./cropper.js";
import * as stylesUI from "./styles.js";
import { openSettings, loadSettings, onDownloadStatus, announceConfig } from "./settings.js";

const $app = document.getElementById("app");
let el = {}; // persistent DOM pieces
let composerText = new Map(); // draft prompt per project id
const enhEdit = new Map(); // project id -> local copy of the enhanced prompt while the user edits it (memory only, never saved)
let chain = null; // the Enhance & Generate unit in progress: {pid, chainId, enhanceJobId, step, generateJobId, run, original, e}
const dirty = new Set(); // project ids whose composer text was typed/enhanced by the user (beats the prefill)
const cleared = new Set(); // project ids whose composer was emptied (sent or cleared); stays empty until a run completes
let viewing = null; // run number shown in the stage when it is not the current run (read-only composer)
let viewingPid = null;
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // the server's limit; a 413 would reset the connection mid-send, so check first
const promptOf = (r) => r.prompt || (r.mode === "upload" ? "Uploaded image" : "");
const modeTag = (r) => (r.mode === "upload" ? "Upload" : r.mode);
refsUI.configure({ isRO: () => isRO() });
const bitLabel = (q) => String(q).replace(/^(\d+)bit$/, "$1-bit");

// ---------- shell ----------
function buildShell() {
  el.list = h("nav", { class: "proj-list", "aria-label": "Projects" });
  const panel = jobsUI.buildPanel({ residentLabel, onChange: () => renderJobs() });
  el.jobsPanel = panel.root;
  el.tabProjects = h("button", { class: "tab", role: "tab", type: "button", onclick: () => setTab("projects") }, "Projects");
  el.tabJobs = h("button", { class: "tab", role: "tab", type: "button", onclick: () => setTab("jobs") }, "Jobs", panel.badge);
  el.tabs = h("div", { class: "tabs", role: "tablist" }, el.tabProjects, el.tabJobs);
  el.navGallery = h("a", { class: "nav-item", href: "#/gallery" }, icon("gallery"), "Gallery");
  el.sidebar = h("aside", { class: "sidebar" },
    h("div", { class: "brand" }, h("img", { src: "/icons/favicon-32.png", width: 26, height: 26, alt: "" }), h("span", {}, "ImageAtelier")),
    h("button", { class: "btn primary block", onclick: newProject }, icon("plus", 18), "New project"),
    h("button", { class: "btn block from-image", type: "button", onclick: newFromImage }, icon("image", 18), "New from image…"),
    (el.file = h("input", { type: "file", accept: "image/*", hidden: true, onchange: onFilePicked })),
    el.navGallery,
    el.tabs,
    el.list,
    el.jobsPanel,
    h("div", { class: "side-foot" },
      h("span", { class: "conn", id: "conn", title: "Server connection" }),
      h("div", { class: "grow" }, chipEl()),
      h("button", { class: "icon-btn", title: "Settings", "aria-label": "Settings", onclick: () => openSettings() }, icon("settings"))));
  el.scrim = h("div", { class: "scrim", onclick: () => toggleDrawer(false) });
  el.topTitle = h("div", { class: "top-title" }, h("div", {}, "ImageAtelier"), chipEl());
  el.topbar = h("header", { class: "topbar" },
    h("button", { class: "icon-btn", "aria-label": "Projects", onclick: () => toggleDrawer(true) }, icon("menu")),
    el.topTitle,
    h("a", { class: "icon-btn", href: "#/gallery", "aria-label": "Gallery" }, icon("gallery")),
    h("button", { class: "icon-btn", "aria-label": "Settings", onclick: () => openSettings() }, icon("settings")));
  el.main = h("main", { class: "main" });
  el.offline = h("div", { class: "offline", hidden: true }, "Reconnecting to the server…");
  add($app,h("div", { class: "app" }, el.sidebar, el.scrim, h("div", { class: "pane" }, el.topbar, el.offline, el.main)));
}
function toggleDrawer(open) {
  document.body.classList.toggle("drawer-open", open);
}
function setTab(tab) {
  S.tab = tab;
  localStorage.setItem("ws-tab", tab);
  el.list.hidden = tab !== "projects";
  el.jobsPanel.hidden = tab !== "jobs";
  el.tabProjects.classList.toggle("on", tab === "projects");
  el.tabJobs.classList.toggle("on", tab === "jobs");
  el.tabProjects.setAttribute("aria-selected", tab === "projects");
  el.tabJobs.setAttribute("aria-selected", tab === "jobs");
  if (tab === "jobs") jobsUI.render();
}
function viewInJobs(id) {
  jobsUI.showJob(id);
  setTab("jobs");
  toggleDrawer(true);
}

// ---------- data ----------
let refreshTimer;
function refreshProjects(soon = false) {
  clearTimeout(refreshTimer);
  const run = async () => {
    try {
      S.projects = await api("GET", "/projects");
      renderSidebar();
      if (S.route.name === "gallery") renderGallery();
    } catch (e) {
      console.warn(e);
    }
  };
  if (soon) refreshTimer = setTimeout(run, 150);
  else return run();
}

async function loadProject(id) {
  try {
    const [p, refs] = await Promise.all([api("GET", `/projects/${id}`), api("GET", `/projects/${id}/refs`).catch(() => [])]);
    if (S.route.id !== id) return;
    S.proj = p;
    S.refs = refs;
    renderProject();
  } catch (e) {
    if (e.status === 404) {
      toast("That project no longer exists", "error");
      location.hash = "#/gallery";
    } else toast(e.message, "error");
  }
}

const projectBusy = (pid) => [...S.jobs.values()].filter((j) => j.project_id === pid && !FINAL.includes(j.state)).length;
const activeJobs = (pid) => [...S.jobs.values()].filter((j) => j.project_id === pid && j.kind !== "enhance" && !FINAL.includes(j.state));

// ---------- routing ----------
function parseRoute() {
  const m = location.hash.match(/^#\/p\/([\w-]+)/);
  if (m) return { name: "project", id: m[1] };
  return { name: "gallery" };
}
function onRoute() {
  const r = parseRoute();
  const same = r.name === S.route.name && r.id === S.route.id;
  S.route = r;
  toggleDrawer(false);
  if (same && r.name === "project") return;
  if (r.name === "gallery") {
    S.proj = null;
        renderSidebar();
    renderGallery();
    refreshProjects();
  } else {
    if (S.proj?.id !== r.id) {
      S.proj = null;
      S.refs = [];
    }
    renderSidebar();
    if (S.proj) renderProject();
    else fill(el.main,h("div", { class: "empty" }, h("div", { class: "spinner" })));
    loadProject(r.id);
  }
}

// ---- upload: start a project from a picture ----
let onFile = null; // what to do with the file chosen in the (single, hidden) picker
function pickImage(cb) {
  onFile = cb;
  el.file.value = "";
  el.file.click(); // must run inside the user's click so browsers allow the picker
}
function onFilePicked() {
  const f = el.file.files[0];
  el.file.value = "";
  if (f && onFile) onFile(f);
}
function tooBig(file) {
  if (file.size <= MAX_UPLOAD_BYTES) return false;
  toast(`That file is ${(file.size / 1048576).toFixed(0)} MB. Uploads are limited to 25 MB; choose a smaller photo.`, "error", 6000);
  return true;
}
function cropInto(pid, file, created) {
  openCropper(file, {
    pid, api,
    onUploaded: (run) => {
      onRun({ project_id: pid, run }); // the SSE `run` event does the same; this is idempotent
      refreshProjects(true);
    },
    onCancel: async () => {
      if (!created) return;
      try { await api("DELETE", `/projects/${pid}`); } catch {}
      await refreshProjects();
      if (S.route.id === pid) location.hash = "#/gallery";
    },
  });
}
function uploadHere() {
  const P = S.proj;
  if (!P || P.runs.length) return;
  pickImage((file) => !tooBig(file) && cropInto(P.id, file, false));
}
function newFromImage() {
  pickImage(async (file) => {
    if (tooBig(file)) return;
    try {
      const p = await api("POST", "/projects", {});
      await refreshProjects();
      location.hash = `#/p/${p.id}`;
      cropInto(p.id, file, true);
    } catch (e) {
      toast(e.message, "error");
    }
  });
}

async function newProject() {
  try {
    const p = await api("POST", "/projects", {});
    await refreshProjects();
    location.hash = `#/p/${p.id}`;
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------- sidebar ----------
let sideSig = "";
function renderSidebar() {
  el.navGallery.classList.toggle("active", S.route.name === "gallery");
  // Rebuilding restarts the busy dot's pulse and reloads thumbnails, so skip when nothing visible changed.
  const sig = JSON.stringify([S.route.id, S.projects.map((p) => [p.id, p.name, p.run_count, p.latest?.thumb_url, projectBusy(p.id)])]);
  if (sig === sideSig) return;
  sideSig = sig;
  clear(el.list);
  if (!S.projects.length) add(el.list,h("div", { class: "dim pad" }, "No projects yet"));
  for (const p of S.projects) {
    const busy = projectBusy(p.id);
    add(el.list,h("a", { class: `proj${S.route.id === p.id ? " active" : ""}`, href: `#/p/${p.id}` },
      p.latest ? h("img", { class: p.latest.transparent ? "checker" : "", src: p.latest.thumb_url, alt: "", loading: "lazy" }) : h("span", { class: "ph" }, icon("spark", 16)),
      h("span", { class: "pn" }, p.name),
      busy ? h("span", { class: "dot pulse", title: `${busy} active or queued job${busy === 1 ? "" : "s"}` }) : h("small", { class: "dim" }, p.run_count || "")));
  }
}

// ---------- gallery ----------
function renderGallery() {
  if (S.route.name !== "gallery") return;
  const withImg = S.projects.filter((p) => p.latest);
  fill(el.main,
    h("div", { class: "pane-head" }, h("h1", {}, "Gallery"), h("div", { class: "grow" }),
      h("button", { class: "btn", type: "button", onclick: newFromImage }, icon("image", 18), "New from image…"),
      h("button", { class: "btn primary", onclick: newProject }, icon("plus", 18), "New project")),
    S.projects.length
      ? h("div", { class: "content" }, h("div", { class: "grid" }, S.projects.map((p) => {
          const li = withImg.indexOf(p);
          return h("div", { class: "card", onclick: () => (location.hash = `#/p/${p.id}`) },
            h("div", { class: `thumb${p.latest?.transparent ? " checker" : ""}` },
              p.latest ? h("img", { src: p.latest.thumb_url, alt: p.name, loading: "lazy" }) : h("div", { class: "ph" }, icon("spark", 32)),
              p.latest?.transparent ? h("span", { class: "badge transp", title: "Transparent background" }, "Transparent") : null,
              p.latest ? h("button", {
                class: "icon-btn glass zoom", title: "View full size", "aria-label": "View full size",
                onclick: (e) => { e.stopPropagation(); openLightbox(withImg.map(lbItemProject), li); },
              }, icon("zoom", 18)) : null,
              activeJobs(p.id).length ? h("span", { class: "badge busy" }, "Generating…") : null),
            h("div", { class: "meta" }, h("div", { class: "nm" }, p.name),
              h("small", { class: "dim" }, `${p.run_count} run${p.run_count === 1 ? "" : "s"} · ${timeAgo(p.updated_at)}`)));
        })))
      : h("div", { class: "empty" }, h("div", { class: "big" }, icon("spark", 40)), h("h2", {}, "Nothing here yet"),
          h("p", { class: "dim" }, "Every image begins with a single line. Create a project and describe your first one."),
          h("button", { class: "btn primary", onclick: newProject }, icon("plus", 18), "New project")));
}
const lbItemProject = (p) => ({ transparent: p.latest.transparent, url: p.latest.image_url, title: p.name, filename: `${p.name}-${p.latest.run}.png` });

// ---------- project view ----------
let elProj = {};
function renderProject() {
  const P = S.proj;
  if (!P || S.route.id !== P.id) return;
  const cur = P.runs.find((r) => r.run === P.current_run);
  elProj.title = h("h1", { class: "title", title: "Rename", onclick: startRename }, P.name);
  elProj.stage = h("div", { class: "stage" });
  if (viewingPid !== P.id) {
    viewing = null;
    viewingPid = P.id;
  }
  elProj.jobsCount = h("small", { class: "dim" });
  elProj.jobsViewBtn = h("button", { class: "linkbtn", type: "button" }, "View in Jobs");
  elProj.jobsList = h("div", { class: "jobs-list" });
  elProj.jobs = h("div", { class: "jobs", hidden: true }, h("div", { class: "jobs-head" }, elProj.jobsCount, h("div", { class: "grow" }), elProj.jobsViewBtn), elProj.jobsList);
  elProj.strip = h("div", { class: "strip", role: "listbox", "aria-label": "History" });
  elProj.text = h("textarea", {
    class: "prompt", rows: 1, enterKeyHint: "send", autocapitalize: "sentences",
    oninput: () => {
      autosize();
      if (S.proj.enhanced && enhEdit.has(P.id)) {
        enhEdit.set(P.id, elProj.text.value); // editing a local copy of the enhanced prompt
        renderEnhUI();
        return;
      }
      composerText.set(P.id, elProj.text.value);
      if (elProj.text.value.trim()) {
        dirty.add(P.id);
        cleared.delete(P.id);
      } else {
        dirty.delete(P.id);
        cleared.add(P.id);
      }
      renderEnhUI();
    },
    onkeydown: (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submitRun();
      } else if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "Enter") {
        e.preventDefault();
        startChain();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "e") {
        e.preventDefault();
        startEnhance();
      }
    },
  });
  elProj.text.value = composerValue(P);
  elProj.ro = h("div", { class: "ro-area" });
  elProj.send = h("button", { class: "send", type: "button", "aria-label": "Generate", onclick: submitRun }, icon("send", 20));
  elProj.enhBtn = h("button", { class: "enhance-btn", type: "button", "aria-label": "Enhance prompt", title: "Enhance prompt (Cmd/Ctrl+E)", onclick: startEnhance }, icon("spark", 20));
  elProj.chainBtn = h("button", { class: "enhance-btn chain-btn", type: "button", "aria-label": "Enhance and generate", title: "Enhance & Generate (Cmd/Ctrl+Shift+Enter)", onclick: startChain }, icon("sparkgo", 20));
  elProj.enh = h("div", { class: "enh-area" });
  fill(el.main,
    h("div", { class: "pane-head" }, elProj.title, stylesUI.buildChip(), h("div", { class: "grow" }),
      h("button", { class: "icon-btn", title: "Rename", "aria-label": "Rename", onclick: startRename }, icon("pencil")),
      h("button", { class: "icon-btn", title: "Delete project", "aria-label": "Delete project", onclick: deleteProject }, icon("trash"))),
    elProj.stage, elProj.jobs, refsUI.buildPanel(), elProj.strip,
    h("form", { class: "composer", onsubmit: (e) => e.preventDefault() },
      (elProj.styleHint = h("div", { class: "refs-hint style-hint", hidden: true, role: "status" }, icon("palette", 14), h("span", { class: "sh-text" }))),
      refsUI.buildHint(),
      elProj.ro,
      elProj.enh,
      (elProj.box = h("div", { class: "composer-box" }, elProj.text, (elProj.pick = h("div", { class: "model-pick" })), (elProj.transpBtn = h("button", { class: "transp-btn", type: "button", onclick: toggleTransp }, icon("checker", 18))), elProj.enhBtn, elProj.chainBtn, elProj.send))));
  renderStage(shownRun(P) ?? cur);
  renderStrip();
  renderJobs();
  updatePlaceholder();
  autosize();
  renderEnhUI();
  renderRO();
  renderStyleHint();
  pickSig = "";
  renderPicker();
}

// ---- transparent background toggle ----
// Remembered per project; until the user chooses, it follows the current image (on for edits of a cut-out).
const transpKey = (pid) => `ws-transp-${pid}`;
function transpOn(P) {
  const v = localStorage.getItem(transpKey(P.id));
  return v == null ? curRun(P)?.transparent === true : v === "on";
}
function toggleTransp() {
  const P = S.proj;
  if (!P || isRO()) return;
  localStorage.setItem(transpKey(P.id), transpOn(P) ? "off" : "on");
  renderTransp();
}
function renderTransp() {
  const b = elProj.transpBtn;
  if (!b || !S.proj) return;
  const on = transpOn(S.proj), ro = isRO();
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", on);
  b.disabled = ro;
  b.title = ro ? "Transparent background (read-only while viewing an older image)"
    : on ? "Transparent background: ON. The subject is cut out; turn off to get a normal background."
    : "Transparent background: OFF. Turn on to cut out the subject (PNG with transparency).";
}

// ---- image model picker in the composer ----
// The dropdown picks the mode (Turbo / Standard) for the image model and text encoder chosen in Settings.
let pickSig = "";
let pickOpen = false;
const optShort = (o) => (o.turbo ? "⚡" : "Std");
const isResident = (o) => S.resident.state === "loaded" && S.resident.key?.[0] === "image" && S.resident.key[1] === S.settings.image_model && S.resident.key[2] === S.settings.text_encoder && !!S.resident.key[3] === o.turbo;
function closePickMenu() {
  pickOpen = false;
  document.querySelector(".pick-menu")?.remove();
  elProj.pickBtn?.setAttribute("aria-expanded", "false");
}
async function choose(o) {
  closePickMenu();
  if (o.turbo === S.settings.turbo) return;
  try {
    S.settings = await api("PUT", "/settings", { turbo: o.turbo });
  } catch (e) {
    toast(e.message, "error");
  }
  announceConfig();
}
function openPickMenu(opts) {
  closePickMenu();
  pickOpen = true;
  const menu = h("div", { class: "pick-menu", role: "listbox" }, opts.map((o) => {
    const on = o.turbo === S.settings.turbo;
    return h("button", { class: `pick-item${on ? " on" : ""}`, type: "button", role: "option", "aria-selected": on, onclick: () => choose(o) },
      h("span", { class: "pi-check" }, on ? icon("check", 14) : null),
      h("span", { class: "pi-main" }, h("b", {}, optName(o)), h("small", { class: "dim" }, `${stepsAfter(o)} steps`)),
      isResident(o) ? h("span", { class: "pi-loaded", title: "Currently loaded in memory" }, h("i", {}), "loaded") : null);
  }));
  document.body.append(menu);
  const r = elProj.pickBtn.getBoundingClientRect();
  const w = menu.offsetWidth;
  menu.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left))}px`;
  menu.style.bottom = `${innerHeight - r.top + 8}px`;
  elProj.pickBtn.setAttribute("aria-expanded", "true");
}
function renderPicker() {
  if (!elProj.pick || !S.proj || !S.settings) return;
  const ro = isRO();
  const opts = modeOptions();
  const cur = { turbo: S.settings.turbo };
  const te = currentEncoder();
  const teOk = te?.state === "ready";
  const sig = JSON.stringify([opts, cur, S.settings.steps, S.settings.image_model, S.settings.text_encoder, ro, S.models.length, teOk, opts.map(isResident), pickOpen]);
  if (sig === pickSig) return;
  pickSig = sig;
  const curAvail = opts.some((o) => o.turbo === cur.turbo);
  elProj.pick.classList.toggle("ro", ro);
  if (!S.models.length) return clear(elProj.pick); // not loaded yet: show nothing rather than a wrong "no model"
  if (!opts.length) {
    return fill(elProj.pick, h("button", { class: "pick-btn none", type: "button", title: "No image model is downloaded", onclick: () => openSettings({ focus: "model", message: "The image model isn't downloaded yet. Download it below." }) },
      h("span", { class: "pick-long" }, "No model – open Settings"), h("span", { class: "pick-short" }, "No model")));
  }
  const label = `${optName(cur)} · ${S.settings.steps} steps${teOk ? "" : " (text encoder not ready)"}`;
  if (opts.length === 1 && curAvail && teOk) {
    return fill(elProj.pick, h("span", { class: "pick-static", title: label }, h("span", { class: "pick-long" }, optName(cur)), h("span", { class: "pick-short" }, optShort(cur))));
  }
  if (pickOpen && !ro) requestAnimationFrame(() => pickOpen && openPickMenu(opts)); // keep an open menu in step with changes
  fill(elProj.pick, h("button", {
    class: `pick-btn${curAvail && teOk ? "" : " warn"}`, type: "button", disabled: ro, "aria-haspopup": "listbox", "aria-expanded": "false",
    title: !teOk ? "The selected text encoder isn't ready; open Settings" : curAvail ? `${label}${ro ? " (read-only while viewing an older image)" : ""}` : `${optName(cur)} isn't available; pick another mode`,
    onclick: (e) => { e.stopPropagation(); pickOpen ? closePickMenu() : openPickMenu(opts); },
  }, h("span", { class: "pick-long" }, optName(cur)), h("span", { class: "pick-short" }, optShort(cur)), h("span", { class: "pick-caret" }, "▾")));
  elProj.pickBtn = elProj.pick.firstChild;
  if (ro) closePickMenu();
}
window.addEventListener("ws-config", () => renderPicker());
document.addEventListener("pointerdown", (e) => pickOpen && !e.target.closest(".pick-menu, .pick-btn") && closePickMenu());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && pickOpen) {
    e.preventDefault();
    closePickMenu();
  }
}, true);

// ---- composer content: prefill, draft, read-only history view ----
const curRun = (P) => P.runs.find((r) => r.run === P.current_run) ?? null;
const viewedRun = (P) => (viewing != null && viewingPid === P.id && viewing !== P.current_run ? P.runs.find((r) => r.run === viewing) ?? null : null);
const shownRun = (P) => viewedRun(P) ?? curRun(P);
const isRO = () => !!(S.proj && viewedRun(S.proj));
function composerValue(P) {
  const v = viewedRun(P);
  if (v) return v.prompt;
  if (S.enh?.pid === P.id) return S.enh.stream ?? S.enh.original; // read-only while the enhancer works
  if (chain?.pid === P.id) return chain.original;
  if (P.enhanced) return enhEdit.has(P.id) ? enhEdit.get(P.id) : P.enhanced.text; // the server's enhanced prompt
  if (dirty.has(P.id)) return composerText.get(P.id) ?? "";
  if (cleared.has(P.id)) return "";
  return curRun(P)?.prompt ?? "";
}
function syncComposer() {
  if (!elProj.text || !S.proj) return;
  const v = composerValue(S.proj);
  if (elProj.text.value !== v) elProj.text.value = v;
  autosize();
  renderEnhUI();
  renderRO();
  updatePlaceholder();
  renderStyleHint();
  refsUI.render();
}
// The composer hint under the box: shown only when the project has a style.
function renderStyleHint() {
  const st = stylesUI.styleOf(S.proj);
  if (!elProj.styleHint) return;
  elProj.styleHint.hidden = !stylesUI.hasStyle(st) || isRO();
  if (!elProj.styleHint.hidden) setText(elProj.styleHint.querySelector(".sh-text"), stylesUI.hintText(st));
}
function renderRO() {
  if (!elProj.ro || !S.proj) return;
  const v = viewedRun(S.proj);
  elProj.box?.classList.toggle("ro", !!v);
  if (!v) {
    clear(elProj.ro);
    return;
  }
  fill(elProj.ro, h("div", { class: "ro-bar" },
    h("span", { class: "grow" }, h("b", {}, `Viewing #${v.run} · read-only`)),
    h("button", { class: "btn small", type: "button", disabled: !v.prompt, title: v.prompt ? "" : "An uploaded image has no prompt", onclick: async () => toast((await copyText(v.prompt)) ? "Prompt copied" : "Couldn't copy", "info") }, "Copy prompt"),
    h("button", { class: "btn small primary", type: "button", onclick: () => branchFrom(v.run) }, "Branch from here"),
    h("button", { class: "btn small", type: "button", onclick: () => selectRun(S.proj.current_run) }, "Back to latest")));
}
function selectRun(n) {
  const P = S.proj;
  if (!P) return;
  viewing = n === P.current_run ? null : n;
  viewingPid = P.id;
  renderStage(shownRun(P));
  renderStrip();
  syncComposer();
}
async function branchFrom(n) {
  const P = S.proj;
  try {
    const p = await api("PATCH", `/projects/${P.id}`, { current_run: n });
    if (S.proj?.id !== P.id) return;
    P.current_run = n;
    P.updated_at = p.updated_at;
    viewing = null;
    dirty.delete(P.id);
    cleared.delete(P.id);
    composerText.delete(P.id);
    renderStage(curRun(P));
    renderStrip();
    updatePlaceholder();
    syncComposer();
    refreshProjects(true);
    elProj.text.focus();
  } catch (e) {
    toast(e.message, "error");
  }
}
const updatePlaceholder = () => {
  if (!elProj.text || !S.proj) return;
  const v = viewedRun(S.proj);
  elProj.text.placeholder = v?.mode === "upload" ? "Uploaded image" : S.proj.current_run == null ? "Describe an image…" : "Describe an edit…";
};
function autosize() {
  const t = elProj.text;
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 160) + "px";
  elProj.box?.classList.toggle("tall", t.scrollHeight > 66); // long text: controls drop to their own row instead of squeezing the box
}

function renderStage(cur) {
  const P = S.proj;
  clear(elProj.stage);
  if (!cur) {
    add(elProj.stage,h("div", { class: "empty" }, h("div", { class: "big" }, icon("spark", 40)),
      h("h2", {}, "A blank canvas"), h("p", { class: "dim" }, "Describe an image below. Every later prompt edits the current image."),
      h("button", { class: "btn upload-btn", type: "button", onclick: uploadHere }, icon("image", 18), "Upload image")));
    return;
  }
  const img = h("img", { class: `hero${cur.transparent ? " checker" : ""}`, src: cur.image_url, alt: promptOf(cur) || "Image", draggable: false,
    onload: () => img.classList.add("in"), onclick: openHistoryLightbox });
  add(elProj.stage,img,
    h("button", { class: "icon-btn glass info", "aria-label": "Run details", "aria-pressed": S.showDetails, title: "Details",
      onclick: () => { S.showDetails = !S.showDetails; renderStage(cur); } }, icon("info")),
    S.showDetails ? detailsCard(cur) : null);
}
function detailsCard(r) {
  const row = (k, v) => [h("dt", {}, k), h("dd", {}, v)];
  if (r.mode === "upload") {
    const from = r.source_size ? ` (from ${r.source_size[0]}×${r.source_size[1]})` : "";
    return h("div", { class: "details" },
      h("div", { class: "d-prompt" }, `Uploaded · ${r.width}×${r.height}${r.preset ? ` · ${r.preset}` : ""}${from}`),
      h("dl", {}, row("Run", `#${r.run} · upload`), row("Created", fmtDate(r.created_at))));
  }
  return h("div", { class: "details" },
    h("div", { class: "d-prompt" }, r.prompt),
    h("dl", {},
      row("Run", `#${r.run} · ${r.mode}${r.parent_run ? ` of #${r.parent_run}` : ""}`),
      row("Seed", r.seed), r.transparent ? row("Background", "Transparent") : null, row("Mode", `${r.turbo ? "Turbo" : "Standard"} · ${r.steps} steps`), r.text_encoder ? row("Text encoder", encoderName(r.text_encoder)) : null,
      row("Size", `${r.width}×${r.height}`), row("Generation", fmtSecs(r.gen_s)), row("Model load", fmtSecs(r.load_s)),
      row("Created", fmtDate(r.created_at))),
    stylesUI.hasStyle(r.style) ? h("div", { class: "d-style" }, h("b", {}, "Style: "), stylesUI.styleLabel(r.style), r.style.id === "custom" && r.style.custom ? ` · ${r.style.custom}` : "") : null,
    (r.transparent || stylesUI.hasStyle(r.style)) && r.model_prompt ? h("details", { class: "d-model" }, h("summary", {}, "Sent to model"), h("div", {}, r.model_prompt)) : null,
    r.refs?.length ? refsDetail(r) : null);
}
function refsDetail(r) {
  const label = { auto: "Auto", high: "High", standard: "Standard", low: "Low" }[r.ref_detail] || r.ref_detail;
  const items = r.refs.map((x) => ({ url: x.image_url, title: `Image ${x.image_index} · ${x.name}`, filename: `${x.name}.png` }));
  return h("div", { class: "d-refs" },
    h("div", { class: "dr-title" }, "References used", r.reference_resolution ? h("span", { class: "dim" }, ` · Reference detail: ${label} · ${r.reference_resolution}`) : null),
    h("div", { class: "dr-row" }, r.refs.map((x, i) => h("button", { class: "dr-item", type: "button", title: `Image ${x.image_index} · ${x.name}`, onclick: () => openLightbox(items, i) },
      h("img", { src: x.image_url, alt: x.name, loading: "lazy", draggable: false }), h("span", {}, `Image ${x.image_index}`), h("small", { class: "dim" }, x.name)))));
}
function openHistoryLightbox() {
  const P = S.proj;
  const items = P.runs.map((r) => ({ transparent: r.transparent, url: r.image_url, title: `#${r.run} · ${promptOf(r)}`, filename: `${P.name}-${r.run}.png` }));
  openLightbox(items, Math.max(0, P.runs.findIndex((r) => r.run === shownRun(P)?.run)));
}

function renderStrip() {
  const P = S.proj;
  clear(elProj.strip);
  const shown = shownRun(P)?.run;
  for (const r of P.runs) {
    const sel = r.run === shown;
    const cell = h("button", {
      class: `tile${sel ? " current" : ""}${r.transparent ? " checker" : ""}`, role: "option", "aria-selected": sel,
      title: promptOf(r), dataset: { run: r.run },
    }, h("img", { src: r.thumb_url, alt: "", loading: "lazy", draggable: false }),
      h("span", { class: `tag ${r.mode}` }, modeTag(r)), h("span", { class: "rn" }, r.run),
      r.transparent ? h("span", { class: "transp-tag", title: "Transparent background", "aria-label": "Transparent" }, icon("checker", 11)) : null,
      r.run === P.current_run && shown !== r.run ? h("span", { class: "base-tag", title: "The next prompt edits this image" }, "base") : null);
    attachTile(cell, r);
    add(elProj.strip, cell);
  }
  elProj.strip.hidden = !P.runs.length;
  elProj.strip.querySelector(".current")?.scrollIntoView({ inline: "center", block: "nearest" });
}
let pop;
function hidePop() {
  pop?.remove();
  pop = null;
}
function attachTile(cell, r) {
  let timer, longPressed = false;
  const cancel = () => clearTimeout(timer);
  cell.addEventListener("pointerdown", (e) => {
    longPressed = false;
    if (e.pointerType === "mouse") return;
    timer = setTimeout(() => {
      longPressed = true;
      hidePop();
      const b = cell.getBoundingClientRect();
      pop = h("div", { class: "pop" }, h("b", {}, `#${r.run} · ${modeTag(r)}`), h("div", {}, promptOf(r)));
      document.body.append(pop);
      const w = pop.offsetWidth;
      pop.style.left = Math.max(8, Math.min(innerWidth - w - 8, b.left + b.width / 2 - w / 2)) + "px";
      pop.style.top = b.top - pop.offsetHeight - 8 + "px";
    }, 450);
  });
  for (const ev of ["pointerup", "pointerleave", "pointercancel", "pointermove"]) cell.addEventListener(ev, (e) => {
    cancel();
    if (ev === "pointerup" || ev === "pointercancel") setTimeout(hidePop, 1500);
  });
  cell.addEventListener("contextmenu", (e) => e.preventDefault());
  cell.addEventListener("click", () => {
    if (longPressed) return;
    selectRun(r.run);
  });
}
document.addEventListener("pointerdown", (e) => pop && !e.target.closest(".tile") && hidePop());

// ---------- jobs ----------
function renderJobs() {
  if (!elProj.jobs || !S.proj) return;
  const pid = S.proj.id;
  const rank = (j) => (["loading", "running", "paused", "cancelling"].includes(j.state) ? 0 : 1);
  const jobs = [...S.jobs.values()]
    .filter((j) => j.project_id === pid && !FINAL.includes(j.state) && j.id !== S.enh?.jobId && !(chain && j.chain_id === chain.chainId)) // the enhance/chain in the composer has its own status
    .sort((a, b) => rank(a) - rank(b) || (a.queue_position || 0) - (b.queue_position || 0) || a.created_at - b.created_at);
  elProj.jobs.hidden = !jobs.length;
  setText(elProj.jobsCount, `${jobs.length} job${jobs.length === 1 ? "" : "s"} in this project`);
  elProj.jobsViewBtn.onclick = () => jobs[0] && viewInJobs(jobs[0].id);
  // Rows are created once and patched in place, so bars and animations are never restarted by progress events.
  syncList(elProj.jobsList, jobs, (j) => j.id,
    () => {
      const pill = h("span", { class: "pill" }), jp = h("span", { class: "jp" }), slot = jobsUI.ctlSlot(), bar = jobsUI.makeBar();
      const el = h("div", { class: "job new", onanimationend: (e) => e.target === el && el.classList.remove("new") },
        h("div", { class: "job-top" }, pill, jp, slot.el), bar.el);
      return { el, pill, jp, slot, bar };
    },
    (row, j) => {
      setClass(row.pill, `pill ${j.state}`);
      setText(row.pill, jobsUI.stateLabel(j));
      setText(row.jp, `${j.kind === "enhance" ? "Enhance · " : `#${j.run} · `}${j.prompt}`);
      row.jp.title = j.prompt;
      row.slot.set(j);
      jobsUI.updateBar(row.bar, j);
    });
}

async function submitRun() {
  const P = S.proj;
  if (!P || elProj.send.disabled || S.enh || chain || isRO()) return;
  const enhanced = P.enhanced;
  // With an enhanced prompt the server's text is sent (or the local edit), and use_enhanced lets the server clear it.
  const prompt = (enhanced ? (enhEdit.has(P.id) ? enhEdit.get(P.id) : enhanced.text) : elProj.text.value).trim();
  if (!prompt) return;
  elProj.send.disabled = true;
  try {
    const body = { prompt, transparent: transpOn(P) };
    if (enhanced) body.use_enhanced = true;
    const r = await api("POST", `/projects/${P.id}/runs`, body);
    composerText.delete(P.id);
    dirty.delete(P.id);
    cleared.add(P.id); // stays empty until this run completes and prefills its prompt
    enhEdit.delete(P.id);
    if (enhanced) P.enhanced = null; // the server clears it too (and tells other clients)
    syncComposer();
    // Show the job right away; the SSE snapshot replaces it.
    S.jobs.set(r.job_id, { id: r.job_id, kind: "generate", mode: r.mode, project_id: P.id, project_name: P.name, run: r.run_reserved, prompt, transparent: body.transparent, state: "queued", step: 0, total_steps: 0, created_at: Date.now() / 1000 });
    renderJobs();
    renderSidebar();
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      openSettings({ focus: /text encoder/i.test(e.message) ? "te" : "model", message: e.message });
    } else toast(e.message, "error");
  } finally {
    if (elProj.send) elProj.send.disabled = false;
    elProj.text?.focus({ preventScroll: true });
  }
}

// ---------- rename / delete ----------
function startRename() {
  const P = S.proj;
  const input = h("input", { class: "title-input", value: P.name, maxLength: 200, "aria-label": "Project name" });
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (save && name && name !== P.name) {
      try {
        const p = await api("PATCH", `/projects/${P.id}`, { name });
        P.name = p.name;
        refreshProjects(true);
      } catch (e) {
        toast(e.message, "error");
      }
    }
    renderProject();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
  elProj.title.replaceWith(input);
  input.focus();
  input.select();
}
async function deleteProject() {
  const P = S.proj;
  const ok = await confirmDialog({ title: `Delete “${P.name}”?`, message: `This permanently removes the project and its ${P.runs.length} image${P.runs.length === 1 ? "" : "s"}.` });
  if (!ok) return;
  try {
    await api("DELETE", `/projects/${P.id}`);
    composerText.delete(P.id);
    await refreshProjects();
    location.hash = "#/gallery";
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------- events ----------
function onJob(snap) {
  jobsUI.track(snap);
  if (snap.chain_id && snap.kind !== "enhance") {
    onChainJobGenerate(snap); // keep the composer status in step, then handle it like any run job below
  }
  if (snap.kind === "enhance") {
    onEnhanceJob(snap);
    return;
  }
  const prevState = S.jobs.get(snap.id)?.state;
  if (FINAL.includes(snap.state)) {
    S.jobs.delete(snap.id);
    if (snap.state === "failed") toast(`Generation failed: ${snap.error || "unknown error"}`, "error", 7000);
    else if (snap.state === "cancelled" && prevState) toast("Generation cancelled");
    refreshProjects(true);
  } else {
    S.jobs.set(snap.id, snap);
  }
  if (S.route.name === "project" && snap.project_id === S.route.id) renderJobs();
  if (snap.state !== prevState) renderSidebar();
}
function onRun({ project_id, run }) {
  refreshProjects(true);
  const P = S.proj;
  if (P && P.id === project_id) {
    if (!P.runs.some((r) => r.run === run.run)) P.runs.push(run);
    P.current_run = run.run;
    cleared.delete(P.id);
    if (chain && chain.pid === P.id && run.run === chain.run) {
      composerText.delete(P.id);
      dirty.delete(P.id); // the original text has been used; show the rewrite the run was generated from
    }
    renderStage(shownRun(P));
    renderStrip();
    updatePlaceholder();
    syncComposer(); // prefill with the new run's prompt unless the user typed a draft meanwhile
    refsUI.render(); // the project now has a current image: reference numbers shift to Image 2..
  }
}
function applyJobs(list) {
  S.jobs = new Map(list.filter((j) => !FINAL.includes(j.state)).map((j) => [j.id, j]));
  S.recent = list.filter((j) => FINAL.includes(j.state)).reverse().slice(0, 10);
  for (const j of S.jobs.values()) jobsUI.track(j);
}
async function resync() {
  try {
    applyJobs(await api("GET", "/jobs"));
    api("GET", "/model").then((m) => onModel(m)).catch(() => {});
    reattachEnhance();
    await refreshProjects();
    if (S.route.name === "project") await loadProject(S.route.id);
  } catch (e) {
    console.warn(e);
  }
}


// ---------- resident-model chip ----------
const chips = [];
function chipEl() {
  const c = h("span", { class: "res-chip", title: "Model currently in memory" }, h("i", {}), h("span", {}));
  chips.push(c);
  renderChips();
  return c;
}
function residentLabel() {
  const { state, key } = S.resident;
  if (state === "loading") return ["loading", key?.[0] === "image" ? "Loading image model…" : "Loading prompt enhancer…"];
  if (state !== "loaded" || !key) return ["idle", "Idle"];
  if (key[0] === "image") return ["loaded", `Image · ${key[3] ? "Turbo" : "Standard"} · ${encoderName(key[2])} encoder`];
  return ["loaded", `Prompt enhancer · ${String(key[1]).startsWith("heretic") ? "Heretic" : "Qwen"}`];
}
function renderChips() {
  const [cls, text] = residentLabel();
  for (const c of chips) {
    setClass(c, `res-chip ${cls}`);
    setText(c.lastChild, text);
  }
}
function onModel(ev) {
  S.resident = { state: ev.state, key: ev.key ?? null };
  renderChips();
  jobsUI.refresh();
  renderPicker();
  renderEnhUI();
}

// ---------- prompt enhancer ----------
const activeEnhance = (pid) => [...S.jobs.values()].find((j) => j.kind === "enhance" && !j.chain_id && j.project_id === pid && !FINAL.includes(j.state));

function setDraft(pid, text) {
  composerText.set(pid, text);
  if (text.trim()) {
    dirty.add(pid);
    cleared.delete(pid);
  } else {
    dirty.delete(pid);
    cleared.add(pid);
  }
}
const rewriterLabel = (r) => (!r ? "" : `Prompt enhancer · ${r.startsWith("heretic") ? "Heretic" : "Qwen"}`);
function enhModeText(e) {
  return e.mode === "i2i" ? `Enhancing as an edit of #${e.refRun ?? S.proj?.current_run ?? "?"}` : "Enhancing for a new image";
}
function enhStateText(e) {
  if (e.cancelling || e.state === "cancelling") return "Cancelling…";
  if (e.state === "queued") return "Queued · waiting for previous job";
  if (e.phase === "thinking") return `Thinking… ${fmtTokens(e.tokens)} token${e.tokens === 1 ? "" : "s"}`;
  if (e.phase === "writing") return "Writing…";
  if (S.resident.state === "loading") return S.resident.key?.[0] === "image" ? "Switching models…" : "Loading prompt enhancer…";
  return "Preparing the inks…";
}
// What the one busy-status line should say right now (an enhance session or an Enhance & Generate unit), or null.
function currentStatus(pid) {
  const e = S.enh && S.enh.pid === pid ? S.enh : null;
  if (e) {
    return { key: `e:${e.jobId}`, main: enhModeText(e), rest: `${e.rewriter ? ` · ${rewriterLabel(e.rewriter)}` : ""} · ${enhStateText(e)}`, cancel: cancelEnhance, cancelling: !!(e.cancelling || e.state === "cancelling") };
  }
  const c = chain && chain.pid === pid ? chain : null;
  if (!c) return null;
  if (c.step === "enhance") {
    return { key: `c:${c.chainId}`, main: "1/2 · Enhancing", rest: `${c.e.rewriter ? ` · ${rewriterLabel(c.e.rewriter)}` : ""} · ${enhStateText({ ...c.e, cancelling: c.cancelling })}`, cancel: cancelChain, cancelling: !!c.cancelling || c.e.state === "cancelling" };
  }
  const g = S.jobs.get(c.generateJobId);
  return { key: `c:${c.chainId}`, main: "2/2 · Generating", rest: ` · ${c.cancelling || g?.state === "cancelling" ? "Cancelling…" : g ? jobsUI.stateLabel(g) : "Queued"}`, cancel: cancelChain, cancelling: !!c.cancelling || g?.state === "cancelling" };
}
let enhView = null; // the persistent busy-status DOM: the spinner is created once per enhance/chain and only its text is patched
const enhancerRunning = () => !!S.enh || !!chain;
function renderEnhUI() {
  if (!elProj.enh || !S.proj) return;
  const P = S.proj, pid = P.id;
  const status = currentStatus(pid);
  const ro = isRO();
  const enhanced = P.enhanced, editing = !!enhanced && enhEdit.has(pid);
  if (elProj.text) elProj.text.readOnly = !!status || ro || (!!enhanced && !editing);
  elProj.box?.classList.toggle("enhanced", !!enhanced && !editing && !status && !ro);
  elProj.box?.classList.toggle("editing", editing && !ro);
  const hasText = !!(elProj.text?.value || "").trim() && !enhanced;
  const canEnhance = !enhancerRunning() && !ro && !enhanced && hasText;
  elProj.enhBtn?.toggleAttribute("disabled", !canEnhance);
  elProj.chainBtn?.toggleAttribute("disabled", !canEnhance);
  if (elProj.enhBtn) elProj.enhBtn.title = enhanced ? "Discard the enhanced prompt first" : "Enhance prompt (Cmd/Ctrl+E)";
  elProj.send?.toggleAttribute("disabled", !!status || ro);
  renderPicker();
  renderTransp();
  elProj.enh.classList.toggle("busy", !!status);
  if (status) {
    if (!(enhView && enhView.key === status.key && enhView.root.parentNode === elProj.enh)) {
      const main = h("b", {}), rest = h("span", { class: "dim" }), cancel = h("button", { class: "btn small", type: "button" }, "Cancel");
      const root = h("div", { class: "enh-status", role: "status" }, h("span", { class: "spinner sm" }), h("span", { class: "grow" }, main, rest), cancel);
      enhView = { key: status.key, root, main, rest, cancel };
      fill(elProj.enh, root);
    }
    enhView.cancel.onclick = status.cancel;
    setText(enhView.main, status.main);
    setText(enhView.rest, status.rest);
    enhView.cancel.disabled = status.cancelling;
    return;
  }
  enhView = null;
  if (enhanced && !ro) {
    const label = `Enhanced prompt · ${enhanced.rewriter && String(enhanced.rewriter).startsWith("heretic") ? "Heretic" : "Qwen"}`;
    fill(elProj.enh, h("div", { class: `enh-status enhanced-bar${enhanced.fell_back ? " warn" : ""}`, role: "status" },
      icon("spark", 16),
      h("span", { class: "grow" }, h("b", {}, label), enhanced.fell_back ? h("span", { class: "dim" }, " · it ran out of thinking room, so your prompt was kept") : null,
        editing ? h("button", { class: "linkbtn", type: "button", onclick: revertEnhanced }, "Revert to enhanced") : null),
      editing ? null : h("button", { class: "btn small", type: "button", onclick: editEnhanced }, "Edit"),
      h("button", { class: "btn small", type: "button", onclick: discardEnhanced }, "Discard")));
  } else {
    clear(elProj.enh);
  }
}

function editEnhanced() {
  const P = S.proj;
  if (!P?.enhanced) return;
  enhEdit.set(P.id, P.enhanced.text);
  syncComposer();
  elProj.text.focus();
}
function revertEnhanced() {
  const P = S.proj;
  if (!P) return;
  enhEdit.delete(P.id);
  syncComposer();
}
async function discardEnhanced() {
  const P = S.proj;
  if (!P?.enhanced) return;
  const original = P.enhanced.original;
  try {
    await api("DELETE", `/projects/${P.id}/enhanced`);
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  P.enhanced = null;
  enhEdit.delete(P.id);
  setDraft(P.id, original); // the user's own words come back into a normal, editable box
  syncComposer();
  elProj.text.focus();
}
// The server stored, replaced or cleared the project's enhanced prompt (this client, another tab or another device).
function onEnhancedEvent(ev) {
  const P = S.proj;
  if (!P || P.id !== ev.project_id) return;
  P.enhanced = ev.enhanced;
  enhEdit.delete(P.id); // a new server value replaces any local edit
  if (S.enh?.pid === P.id && ev.enhanced) S.enh.stream = undefined;
  syncComposer();
}

async function startEnhance() {
  const P = S.proj;
  if (!P || enhancerRunning() || isRO() || P.enhanced) return;
  const text = elProj.text.value;
  if (!text.trim()) {
    toast("Type a prompt first, then press Enhance");
    elProj.text.focus();
    return;
  }
  const refRun = P.current_run;
  try {
    const r = await api("POST", `/projects/${P.id}/enhance`, { prompt: text });
    const prior = S.enh?.jobId === r.job_id ? S.enh : {};
    const rewriter = S.settings?.pe_variant === "heretic" ? "heretic-8bit" : `official-${S.settings?.pe_quant || "8bit"}`;
    S.enh = { ...prior, rewriter: prior.rewriter ?? rewriter, pid: P.id, jobId: r.job_id, original: text, mode: r.mode, refRun, phase: prior.phase ?? null, tokens: prior.tokens ?? 0, state: prior.state ?? "queued" };
    syncComposer();
  } catch (e) {
    enhanceError(e);
  }
}
function enhanceError(e) {
  if (e instanceof ApiError && e.status === 409) {
    if (/prompt enhancer|heretic/i.test(e.message)) openSettings({ focus: "pe", message: /heretic/i.test(e.message) ? "Heretic isn't installed on the server. Choose Qwen, or copy the Heretic model folders into models/." : "The prompt enhancer isn't downloaded yet. Download it below." });
    else openSettings({ focus: /text encoder/i.test(e.message) ? "te" : "model", message: e.message });
  } else toast(e.message, "error");
}
async function cancelEnhance() {
  if (!S.enh || S.enh.cancelling || S.enh.state === "cancelling") return;
  S.enh.cancelling = true;
  renderEnhUI();
  try {
    const r = await api("POST", `/jobs/${S.enh.jobId}/cancel`);
    if (!r.cancelled) finishEnhance(); // it had already finished; the done event will have been handled
  } catch (e) {
    if (S.enh) S.enh.cancelling = false;
    renderEnhUI();
    toast(e.message, "error");
  }
}
// Drop the session; the box goes back to showing the draft / the enhanced prompt from the server.
function finishEnhance() {
  S.enh = null;
  syncComposer();
}

function onEnhanceJob(snap) {
  if (snap.chain_id) return onChainJob(snap);
  const final = FINAL.includes(snap.state);
  if (final) S.jobs.delete(snap.id);
  else S.jobs.set(snap.id, snap);
  let e = S.enh;
  if (!e || e.jobId !== snap.id) {
    // A job we didn't start here (page reload, another tab or device): attach to it while it's active.
    if (final || e || chain || !snap.project_id) return;
    e = S.enh = { pid: snap.project_id, jobId: snap.id, original: snap.prompt, mode: snap.mode, refRun: null, phase: null, tokens: 0, state: snap.state };
  }
  if (S.route.name === "project" && snap.project_id === S.route.id) renderJobs();
  e.state = snap.state;
  e.rewriter = snap.rewriter ?? e.rewriter;
  e.phase = snap.phase ?? e.phase;
  e.tokens = snap.tokens ?? e.tokens;
  if (!final) {
    // The rewrite streams into the (read-only) box while it is written.
    if (snap.phase === "writing" && snap.text && snap.state !== "cancelling") {
      e.stream = snap.text;
      if (S.proj?.id === e.pid && elProj.text && !isRO()) {
        elProj.text.value = snap.text;
        autosize();
        elProj.text.scrollTop = elProj.text.scrollHeight;
      }
    }
    renderEnhUI();
    return;
  }
  const pid = e.pid;
  finishEnhance();
  if (snap.state === "failed") toast(`Enhance failed: ${snap.error || "unknown error"}`, "error", 7000);
  else if (snap.state === "cancelled") toast("Enhance cancelled");
  else if (snap.state === "done" && S.proj?.id === pid && !S.proj.enhanced) {
    // The `enhanced` event normally arrives first; fetch it if that was missed.
    api("GET", `/projects/${pid}`).then((p) => { if (S.proj?.id === pid) { S.proj.enhanced = p.enhanced; syncComposer(); } }).catch(() => {});
  }
  if (S.proj?.id === pid && elProj.text) elProj.text.scrollTop = 0;
}

// ---- Enhance & Generate ----
async function startChain() {
  const P = S.proj;
  if (!P || enhancerRunning() || isRO() || P.enhanced) return;
  const text = elProj.text.value;
  if (!text.trim()) {
    toast("Type a prompt first, then press Enhance & Generate");
    elProj.text.focus();
    return;
  }
  try {
    const r = await api("POST", `/projects/${P.id}/enhance-and-generate`, { prompt: text, transparent: transpOn(P) });
    const prior = chain?.chainId === r.chain_id ? chain : {};
    const rewriter = S.settings?.pe_variant === "heretic" ? "heretic-8bit" : `official-${S.settings?.pe_quant || "8bit"}`;
    chain = { step: "enhance", generateJobId: null, e: { state: "queued", phase: null, tokens: 0, rewriter }, ...prior, pid: P.id, chainId: r.chain_id, enhanceJobId: r.enhance_job_id, run: r.run_reserved, original: text };
    syncComposer();
  } catch (e) {
    enhanceError(e);
  }
}
async function cancelChain() {
  const c = chain;
  if (!c || c.cancelling) return;
  const jobId = c.step === "generate" ? c.generateJobId : c.enhanceJobId;
  if (!jobId) return;
  c.cancelling = true;
  renderEnhUI();
  try {
    await api("POST", `/jobs/${jobId}/cancel`);
  } catch (e) {
    c.cancelling = false;
    renderEnhUI();
    toast(e.message, "error");
  }
}
function endChain(message, kind = "info") {
  if (!chain) return;
  chain = null;
  if (message) toast(message, kind, kind === "error" ? 7000 : 4200);
  syncComposer();
  if (S.route.name === "project") renderJobs();
}
// Job snapshots that belong to a chain (either step).
function onChainJob(snap) {
  const final = FINAL.includes(snap.state);
  if (final) S.jobs.delete(snap.id);
  else S.jobs.set(snap.id, snap);
  if (!chain || chain.chainId !== snap.chain_id) {
    if (final || chain || S.enh || !snap.project_id) return; // attach to a chain started elsewhere / before a reload
    chain = { pid: snap.project_id, chainId: snap.chain_id, enhanceJobId: snap.chain_step === "enhance" ? snap.id : null, generateJobId: snap.chain_step === "generate" ? snap.id : null,
      step: snap.chain_step, run: snap.chain_step === "generate" ? snap.run : null, original: snap.prompt, e: { state: snap.state, phase: null, tokens: 0, rewriter: snap.rewriter } };
  }
  if (S.route.name === "project" && snap.project_id === S.route.id) renderJobs();
  if (snap.chain_step === "enhance") {
    chain.enhanceJobId = snap.id;
    chain.e = { state: snap.state, phase: snap.phase ?? chain.e.phase, tokens: snap.tokens ?? chain.e.tokens, rewriter: snap.rewriter ?? chain.e.rewriter };
    if (final && snap.state === "failed") return endChain(`Enhance failed: ${snap.error || "unknown error"}`, "error");
    if (final && snap.state === "cancelled") return endChain("Cancelled");
    renderEnhUI();
    return;
  }
  chain.step = "generate";
  chain.generateJobId = snap.id;
  chain.run = snap.run ?? chain.run;
  if (final) return endChain(snap.state === "failed" ? `Generation failed: ${snap.error || "unknown error"}` : snap.state === "cancelled" ? "Cancelled" : "", snap.state === "failed" ? "error" : "info");
  renderEnhUI();
}
function onChainJobGenerate(snap) {
  if (!chain || chain.chainId !== snap.chain_id) {
    if (FINAL.includes(snap.state) || chain || S.enh || !snap.project_id) return;
    chain = { pid: snap.project_id, chainId: snap.chain_id, enhanceJobId: null, generateJobId: snap.id, step: "generate", run: snap.run, original: snap.prompt, e: { state: "done", phase: null, tokens: 0, rewriter: null } };
  }
  chain.step = "generate";
  chain.generateJobId = snap.id;
  chain.run = snap.run ?? chain.run;
  if (FINAL.includes(snap.state)) endChain(""); // failures/cancels are toasted by the generic job handling
  else renderEnhUI();
}
function onChainEvent(ev) {
  if (!chain || chain.chainId !== ev.chain_id) return;
  if (ev.state === "generating") {
    chain.step = "generate";
    chain.generateJobId = ev.job_id;
    chain.run = ev.run ?? chain.run;
    if (ev.fell_back) toast("The prompt enhancer ran out of thinking room; it kept your prompt");
    renderEnhUI();
  } else if (ev.state === "cancelled") endChain("Cancelled");
  else if (ev.state === "failed") endChain(`Couldn't start the generation: ${ev.error || "unknown error"}`, "error");
  else endChain();
}
// After a reload/reconnect: pick up an enhance or a chain that is still active.
function reattachEnhance() {
  if (S.enh || chain) return;
  const active = [...S.jobs.values()].filter((x) => !FINAL.includes(x.state));
  const c = active.find((x) => x.chain_id);
  if (c) return onChainJob(c);
  const j = active.find((x) => x.kind === "enhance");
  if (j) onEnhanceJob(j);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && (S.enh || chain) && !e.defaultPrevented) {
    e.preventDefault();
    S.enh ? cancelEnhance() : cancelChain();
  }
});

// ---------- boot ----------
async function boot() {
  if (new URLSearchParams(location.search).has("still")) document.documentElement.classList.add("still");
  buildShell();
  stylesUI.setChangeHandler(renderStyleHint);
  window.addEventListener("hashchange", onRoute);
  connectEvents({
    job: onJob,
    run: onRun,
    download: onDownloadStatus,
    model: onModel,
    log: jobsUI.onLog,
    refs: (ev) => ev.project_id === S.proj?.id && refsUI.setRefs(ev.refs),
    enhanced: onEnhancedEvent,
    style: stylesUI.onStyleEvent,
    chain: onChainEvent,
    open: (reconnected) => reconnected && resync(),
    status: (ok) => {
      S.online = ok;
      el.offline.hidden = ok;
      document.getElementById("conn").classList.toggle("off", !ok);
    },
  });
  try {
    const [jobs] = await Promise.all([api("GET", "/jobs"), refreshProjects(), loadSettings().catch(() => {}), stylesUI.loadStyles()]);
    applyJobs(jobs);
  } catch (e) {
    toast(e.message, "error");
  }
  reattachEnhance();
  setTab(S.tab);
  onRoute();
  if (new URLSearchParams(location.search).has("settings")) openSettings();
}
boot();
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("/sw.js").catch((e) => console.warn("sw", e));
}
