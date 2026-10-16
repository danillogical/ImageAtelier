// Jobs tab (current job, queue, recent, job log, server log) and the controls shared with the project page.
import { api, connectServerLog } from "./api.js";
import { S, FINAL } from "./state.js";
import { h, clear, fill, add, toast, confirmDialog, copyText, fmtSecs, syncList, setText, setClass, fmtTokens } from "./util.js";

const RUNNING = ["loading", "running", "paused", "cancelling"];
const WAITING = ["queued", "held"];
const MAX_LOG_LINES = 1500;
const seen = new Map(); // job id -> {t0, pausedAt}: client-side timers (the snapshot only has created_at)
const logs = new Map(); // job id -> lines received via SSE / fetched
const expanded = new Set(); // job ids whose prompt is expanded
let opts = {};
let ui = {};
let frame = 0;

const now = () => Date.now() / 1000;
export const isActive = (j) => !FINAL.includes(j.state);
export const activeCount = () => [...S.jobs.values()].filter(isActive).length;

export function kindLabel(j) {
  if (j.kind === "enhance") return `Enhance prompt · ${j.mode === "i2i" ? "for an edit" : "for a new image"}`;
  return j.mode === "edit" ? "Edit image" : "New image";
}
export const chainText = (j) => `${j.chain_step === "enhance" ? "1/2" : "2/2"} Enhance → Generate`;
export function stateLabel(j) {
  switch (j.state) {
    case "queued": return j.queue_position ? `Queued · #${j.queue_position}` : "Queued";
    case "held": return "Held";
    case "paused": return "Paused";
    case "cancelling": return "Cancelling…";
    case "loading": return "Loading model…";
    case "running":
      if (j.kind === "enhance") return j.phase === "thinking" ? `Thinking · ${fmtTokens(j.tokens)} tokens` : j.phase === "writing" ? `Writing · ${fmtTokens(j.tokens)} tokens` : "Starting…";
      return j.total_steps ? `Step ${j.step}/${j.total_steps}` : "Starting…";
    default: return j.state;
  }
}
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function times(j) {
  const t = seen.get(j.id) || {};
  const start = t.t0 ?? j.created_at;
  const paused = (j.paused_s || 0) + (t.pausedAt ? now() - t.pausedAt : 0);
  return { elapsed: Math.max(0, now() - start), paused };
}
const timeText = (j) => {
  const t = times(j);
  return `Elapsed ${mmss(t.elapsed)}${t.paused > 0.5 ? ` · paused ${mmss(t.paused)}` : ""}`;
};
setInterval(() => {
  for (const e of document.querySelectorAll("[data-elapsed]")) {
    const j = S.jobs.get(e.dataset.elapsed);
    if (j) e.textContent = timeText(j);
  }
}, 1000);

// Called for every job event: keeps timers and the recent list.
export function track(snap) {
  const t = seen.get(snap.id) || {};
  if (!t.t0 && !WAITING.includes(snap.state) && !FINAL.includes(snap.state)) t.t0 = now();
  if (snap.state === "paused" && !t.pausedAt) t.pausedAt = now();
  else if (snap.state !== "paused") t.pausedAt = null;
  seen.set(snap.id, t);
  if (FINAL.includes(snap.state)) S.recent = [snap, ...S.recent.filter((x) => x.id !== snap.id)].slice(0, 10);
  schedule();
}

// ---- actions ----
export async function jobAction(j, action) {
  j = S.jobs.get(j.id) || j;
  try {
    const r = await api("POST", `/jobs/${j.id}/${action}`);
    if (action === "cancel") return r;
    S.jobs.set(j.id, { ...j, ...r });
    schedule();
    opts.onChange?.();
    return r;
  } catch (e) {
    toast(e.message, "error");
  }
}
async function unqueue(j) {
  const ok = await confirmDialog({ title: "Unqueue this job?", message: j.prompt.length > 120 ? `${j.prompt.slice(0, 120)}…` : j.prompt, confirmLabel: "Unqueue" });
  if (ok) await jobAction(j, "cancel");
}

// Buttons appropriate to the job's state; shared by the Jobs tab and the project page.
export function controls(j) {
  const btn = (label, fn, extra = {}) => h("button", { class: `btn small${extra.danger ? " danger-text" : ""}`, type: "button", disabled: extra.disabled, title: extra.title, onclick: fn }, label);
  if (RUNNING.includes(j.state)) {
    const cancelling = j.state === "cancelling";
    let pause;
    if (j.state === "paused") pause = btn("Resume", () => jobAction(j, "resume"));
    else if (j.pause_requested) pause = btn("Pause requested…", null, { disabled: true, title: "Takes effect at the next step" });
    else pause = btn("Pause", () => jobAction(j, "pause"), { disabled: cancelling });
    return h("span", { class: "jctl" }, pause, btn("Cancel", () => jobAction(j, "cancel"), { disabled: cancelling }));
  }
  if (j.state === "held") return h("span", { class: "jctl" }, btn("Release", () => jobAction(j, "release")), btn("Unqueue", () => unqueue(j)));
  if (j.state === "queued") return h("span", { class: "jctl" }, btn("Hold", () => jobAction(j, "hold")), btn("Unqueue", () => unqueue(j)));
  return null;
}

// A progress bar built once and patched in place (the fill animates via a CSS width transition).
export function makeBar() {
  const fillEl = h("i", {});
  return { el: h("div", { class: "bar" }, fillEl), fill: fillEl };
}
export function updateBar(bar, j) {
  const stepping = j.kind !== "enhance" && j.total_steps && ["running", "paused"].includes(j.state);
  const indet = !stepping && j.state !== "paused" && !WAITING.includes(j.state);
  const pct = stepping ? (100 * j.step) / j.total_steps : j.state === "paused" ? 100 : 0;
  setClass(bar.el, `bar${indet ? " indet" : ""}${j.state === "paused" ? " paused" : ""}`);
  const w = indet ? "" : `${pct}%`;
  if (bar.fill.style.width !== w) bar.fill.style.width = w;
}
// Holds the state-dependent buttons; they are only replaced when the job's control state changes.
export function ctlSlot() {
  const el = h("span", { class: "jctl-slot" });
  let sig = null;
  return {
    el,
    set(j) {
      const s = `${j.state}|${!!j.pause_requested}`;
      if (s === sig) return;
      sig = s;
      fill(el, controls(j));
    },
  };
}

// ---- log views ----
class LogView {
  constructor(cls) {
    this.stick = true;
    this.pre = h("div", { class: `logview ${cls}`, tabIndex: 0, onscroll: () => {
      this.stick = this.atBottom();
      if (this.stick) this.chip.hidden = true;
    } });
    this.chip = h("button", { class: "jump", type: "button", hidden: true, onclick: () => this.toBottom() }, "Jump to latest");
    this.el = h("div", { class: "logwrap" }, this.pre, this.chip);
  }
  atBottom() { return this.pre.scrollHeight - this.pre.scrollTop - this.pre.clientHeight < 24; }
  toBottom() { this.stick = true; this.chip.hidden = true; this.pre.scrollTop = this.pre.scrollHeight; }
  line(text) {
    const cls = /\b(ERROR|Traceback|failed)\b/.test(text) ? "err" : /\bWARNING\b/.test(text) ? "warn" : "";
    return h("div", { class: cls }, text);
  }
  set(lines) {
    const frag = document.createDocumentFragment();
    for (const l of lines.slice(-MAX_LOG_LINES)) frag.append(this.line(l));
    this.pre.replaceChildren(frag);
    this.toBottom();
  }
  add(text) {
    this.pre.append(this.line(text));
    while (this.pre.childElementCount > MAX_LOG_LINES) this.pre.firstChild.remove();
    if (this.stick) this.pre.scrollTop = this.pre.scrollHeight;
    else this.chip.hidden = false;
  }
  text() { return [...this.pre.children].map((c) => c.textContent).join("\n"); }
}

// Job log lines from SSE `log` events.
export function onLog({ job_id, line }) {
  let arr = logs.get(job_id);
  if (!arr) {
    arr = [];
    logs.set(job_id, arr);
    if (logs.size > 30) logs.delete(logs.keys().next().value);
  }
  arr.push(line);
  if (arr.length > MAX_LOG_LINES) arr.shift();
  if (ui.logId === job_id && !ui.logLoading) ui.jlog.add(line);
}
async function syncLog() {
  const id = wantedLogId();
  if (id === ui.logId) return;
  ui.logId = id;
  ui.jlog.set([]);
  renderLogHead();
  if (!id) return;
  ui.logLoading = true;
  let lines = [];
  try {
    lines = (await api("GET", `/jobs/${id}/log`)).lines;
  } catch {}
  if (ui.logId !== id) return;
  const have = new Set(lines);
  lines = [...lines, ...(logs.get(id) || []).filter((l) => !have.has(l))]; // lines that arrived while fetching
  logs.set(id, lines.slice(-MAX_LOG_LINES));
  ui.logLoading = false;
  ui.jlog.set(lines);
}
const currentJob = () => [...S.jobs.values()].find((j) => RUNNING.includes(j.state));
const wantedLogId = () => S.logPin ?? currentJob()?.id ?? S.recent[0]?.id ?? null;
function findJob(id) { return S.jobs.get(id) || S.recent.find((j) => j.id === id); }
function renderLogHead() {
  if (!ui.logHead) return;
  const j = ui.logId && findJob(ui.logId);
  fill(ui.logHead,
    h("span", { class: "grow lh-title" }, j ? `Job log · ${j.kind === "enhance" ? "enhance" : `run #${j.run}`} · ${j.state}` : "Job log"),
    S.logPin ? h("button", { class: "btn small", type: "button", onclick: () => { S.logPin = null; schedule(); } }, "Follow current") : null,
    h("button", { class: "btn small", type: "button", onclick: async () => toast((await copyText(ui.jlog.text())) ? "Log copied" : "Copy failed") }, "Copy"));
}

// ---- rendering ----
function schedule() {
  renderBadge();
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    if (S.tab === "jobs" && ui.root) render();
  });
}
function renderBadge() {
  const n = activeCount();
  if (ui.badge) {
    ui.badge.textContent = n;
    ui.badge.hidden = n === 0;
  }
}
// Current job card: built once, patched in place on every update.
function makeCurView() {
  const kind = h("span", { class: "jkind" }), pill = h("span", { class: "pill" }), chainBadge = h("span", { class: "chain-badge", hidden: true });
  const prompt = h("div", { class: "jprompt", title: "Click to expand" });
  const link = h("a", { class: "jproj", title: "Open project" }), run = h("span", { class: "dim" });
  const bar = makeBar(), elapsed = h("small", { class: "dim" }), slot = ctlSlot();
  const view = { id: null };
  prompt.addEventListener("click", () => {
    const open = prompt.classList.toggle("open");
    open ? expanded.add(view.id) : expanded.delete(view.id);
  });
  view.card = h("div", { class: "jcard" },
    h("div", { class: "jc-top" }, h("span", { class: "jk" }, kind, chainBadge), pill), prompt, h("div", { class: "jmeta" }, link, run), bar.el,
    h("div", { class: "jfoot" }, elapsed, h("div", { class: "grow" }), slot.el));
  view.idleLabel = h("div", { class: "dim" });
  view.idle = h("div", { class: "jcard idle" }, h("div", { class: "jc-top" }, h("span", { class: "pill" }, "Idle")), view.idleLabel);
  view.update = (j) => {
    view.id = j.id;
    setText(kind, kindLabel(j));
    chainBadge.hidden = !j.chain_id;
    if (j.chain_id) setText(chainBadge, chainText(j));
    setClass(pill, `pill ${j.state}`);
    setText(pill, stateLabel(j));
    setText(prompt, j.prompt);
    prompt.classList.toggle("open", expanded.has(j.id));
    link.hidden = !j.project_id;
    if (j.project_id) {
      link.href = `#/p/${j.project_id}`;
      setText(link, j.project_name || "project");
    }
    const runText = j.kind !== "enhance" && j.run ? ` · run #${j.run}` : "";
    setText(run, runText);
    updateBar(bar, j);
    elapsed.dataset.elapsed = j.id;
    setText(elapsed, timeText(j));
    slot.set(j);
  };
  return view;
}
function renderCurrent() {
  const j = currentJob();
  const want = j ? ui.curView.card : ui.curView.idle;
  if (ui.cur.firstChild !== want) ui.cur.replaceChildren(want);
  if (j) ui.curView.update(j);
  else setText(ui.curView.idleLabel, `Model in memory: ${opts.residentLabel()[1] === "Idle" ? "none" : opts.residentLabel()[1]}`);
}

function renderQueue() {
  const q = [...S.jobs.values()].filter((j) => WAITING.includes(j.state)).sort((a, b) => (a.queue_position || 0) - (b.queue_position || 0) || a.created_at - b.created_at);
  ui.queueEmpty.hidden = q.length > 0;
  syncList(ui.queueList, q, (j) => j.id,
    (j) => {
      const pos = h("span", { class: "qpos" }), kind = h("span", { class: "jkind" }), held = h("span", { class: "pill held" }, "Held");
      const prompt = h("div", { class: "qprompt" }), link = h("a", { class: "jproj", title: "Open project" }), slot = ctlSlot();
      const chainBadge = h("span", { class: "chain-badge", hidden: true });
      const el = h("div", { class: "qrow" }, pos, h("div", { class: "qmain" }, h("div", { class: "qtop" }, kind, chainBadge, held), prompt, h("div", { class: "jmeta" }, link)), slot.el);
      return { el, pos, kind, held, prompt, link, slot, chainBadge };
    },
    (row, j) => {
      row.el.classList.toggle("held", j.state === "held");
      row.held.hidden = j.state !== "held";
      setText(row.pos, String(j.queue_position || ""));
      setText(row.kind, kindLabel(j));
      row.chainBadge.hidden = !j.chain_id;
      if (j.chain_id) setText(row.chainBadge, chainText(j));
      setText(row.prompt, j.prompt);
      row.prompt.title = j.prompt;
      row.link.hidden = !j.project_id;
      if (j.project_id) {
        row.link.href = `#/p/${j.project_id}`;
        setText(row.link, j.project_name || "project");
      }
      row.slot.set(j);
    });
}
function renderRecent() {
  ui.recentEmpty.hidden = S.recent.length > 0;
  syncList(ui.recentList, S.recent, (j) => j.id,
    (j) => {
      const pill = h("span", { class: "pill" }), prompt = h("span", { class: "qprompt" }), sub = h("small", { class: "dim" }), dur = h("small", { class: "dim" });
      const el = h("button", { class: "rrow", type: "button", title: "Show this job's log", onclick: () => { S.logPin = j.id; schedule(); render(); } },
        pill, h("span", { class: "rmain" }, prompt, sub), dur);
      return { el, pill, prompt, sub, dur };
    },
    (row, j) => {
      const r = j.result || {};
      const d = r.gen_s != null ? r.gen_s + (r.load_s || 0) : null;
      row.el.classList.toggle("on", S.logPin === j.id);
      setClass(row.pill, `pill ${j.state}`);
      setText(row.pill, j.state);
      setText(row.prompt, j.prompt);
      setText(row.sub, `${j.kind === "enhance" ? "Enhance" : "Generate"}${j.project_name ? ` · ${j.project_name}` : ""}`);
      setText(row.dur, d != null ? fmtSecs(d) : "—");
    });
}
export function render() {
  if (!ui.root) return;
  renderBadge();
  renderCurrent();
  renderQueue();
  renderRecent();
  syncLog();
  renderLogHead();
}

// Server log: collapsible, opens its own SSE connection with ?serverlog=1 only while expanded.
function buildServerLog() {
  const view = new LogView("slog");
  let stop = null;
  const details = h("details", { class: "slogbox" });
  const summary = h("summary", {}, "Server log");
  const copy = h("button", { class: "btn small", type: "button", onclick: async (e) => { e.preventDefault(); toast((await copyText(view.text())) ? "Server log copied" : "Copy failed"); } }, "Copy");
  summary.append(copy);
  details.append(summary, view.el);
  details.addEventListener("toggle", async () => {
    if (details.open) {
      let backlog = [];
      try {
        backlog = (await api("GET", "/logs/server?tail=500")).lines;
      } catch (e) {
        toast(e.message, "error");
      }
      if (!details.open) return;
      view.set(backlog);
      stop?.();
      stop = connectServerLog((l) => view.add(l));
    } else {
      stop?.();
      stop = null;
    }
  });
  return details;
}

export function buildPanel(options) {
  opts = options;
  ui.badge = h("span", { class: "tab-badge", hidden: true });
  ui.cur = h("div", { class: "jsec" });
  ui.curView = makeCurView();
  ui.queueList = h("div", { class: "jsec" });
  ui.queueEmpty = h("div", { class: "dim jempty" }, "Nothing queued");
  ui.queue = h("div", {}, ui.queueEmpty, ui.queueList);
  ui.recentList = h("div", { class: "jsec" });
  ui.recentEmpty = h("div", { class: "dim jempty" }, "No finished jobs yet");
  ui.recent = h("div", {}, ui.recentEmpty, ui.recentList);
  ui.jlog = new LogView("jlog");
  ui.logHead = h("div", { class: "jhead row" });
  ui.root = h("div", { class: "jobs-panel" },
    h("div", { class: "jlabel" }, "Current"), ui.cur,
    h("div", { class: "jhead" }, h("span", { class: "jlabel" }, "Queue")), ui.queue,
    ui.logHead, ui.jlog.el,
    h("div", { class: "jlabel" }, "Recent"), ui.recent,
    buildServerLog());
  return { root: ui.root, badge: ui.badge };
}
export const refresh = schedule;
export function showJob(id) { S.logPin = id; }
