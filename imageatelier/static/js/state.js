// Shared mutable app state.
export const S = {
  projects: [], // GET /api/projects (gallery entries)
  proj: null, // open project: project + runs
  route: { name: "gallery" },
  jobs: new Map(), // active (non-final) job snapshots by id
  settings: null,
  models: [],
  customModels: [],
  networkInfo: null,
  styles: [], // GET /api/styles (art style presets)
  online: true,
  showDetails: false,
  resident: { state: "unloaded", key: null }, // the one model in memory
  tab: localStorage.getItem("ws-tab") === "jobs" ? "jobs" : "projects", // sidebar tab
  refs: [], // reference images of the open project (GET /api/projects/{pid}/refs)
  recent: [], // last finished jobs, newest first
  logPin: null, // job id whose log is pinned in the Jobs tab (null = follow the current job)
  enh: null, // the running prompt-enhance session {pid, jobId, original, mode, refRun, phase, tokens, state}
};
export const FINAL = ["done", "failed", "cancelled"];
if (typeof window !== "undefined") window.__ws = S; // debugging aid
