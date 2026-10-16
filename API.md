# ImageAtelier HTTP API

A PROJECT is one image lineage. The first prompt in a project is text-to-image; every later prompt EDITS the
project's *current* image. Every run saves its prompt, settings and image.

## Contract

```
GET    /api/projects                     -> [{id,name,created_at,updated_at,current_run,run_count,latest:{run,thumb_url,image_url,transparent}|null}] sorted by updated_at desc  (this IS the gallery)
POST   /api/projects {name?}             -> project (name defaults to "Untitled <n>")
GET    /api/projects/{pid}               -> project + runs:[run...] (ascending) ; 404 if missing
PATCH  /api/projects/{pid} {name?, current_run?, style?} -> project ; current_run must be an existing run (400 otherwise); style see "Art styles"
GET    /api/styles                       -> [{id, label, description, snippet, keep, custom?}] the art style presets (see "Art styles")
DELETE /api/projects/{pid}               -> 204 (removes folder)
POST   /api/projects/{pid}/runs {prompt, seed?, steps?, turbo?, transparent?, use_enhanced?, image_model?, text_encoder?, enhanced_from?}
       -> 202 {job_id, run_reserved: N, mode}
       mode = "t2i" if the project has no current_run else "edit" with ref = current_run's png; parent_run = current_run.
       Unspecified steps/turbo/image_model/text_encoder come from settings. 400 on validation error (message from engine/settings; a legacy `variant` field is 400 "variant was replaced by image_model/text_encoder"). 409 if a part isn't ready (plain-words message naming it; see below).
       On job done: write runs/NNNN.json(+png already written by the engine to output_path)+thumb, set project.current_run=N, updated_at=now.
       On failed/cancelled: no run files remain (delete any partial png); the reserved number may be skipped.
GET    /api/jobs                         -> active+recent jobs (snapshots)
POST   /api/jobs/{id}/cancel             -> {cancelled: bool}
POST   /api/jobs/{id}/pause|resume|hold|release
       -> 200 {...job snapshot (as in /api/jobs), changed: bool}; 409 {detail} on an invalid transition; 404 if unknown
GET    /api/jobs/{id}/log                -> {lines: [str]}   (that job's log, oldest first; 404 if unknown)
GET    /api/logs/server?tail=N           -> {lines: [str]}   (default 500, clamped to 1..2000)
GET    /api/projects/{pid}/refs          -> ordered [{id, name, enabled, order, width, height, created_at, image_url, thumb_url, image_index_edit, image_index_new}]
POST   /api/projects/{pid}/refs?name=    (body: raw image bytes) -> 201 the ref (enabled unless max_refs would be exceeded: then enabled:false + note)
PATCH  /api/projects/{pid}/refs/{rid} {enabled?, name?, order?} -> the ref; 409 "max N references enabled; disable one first"
DELETE /api/projects/{pid}/refs/{rid}    -> 204
GET    /api/projects/{pid}/refs/{rid}/image | /thumb ; GET /api/projects/{pid}/runs/{n}/refs/{i} (the pinned copy, Image i of run n)
GET    /api/sizes?w=&h=                  -> {resolution, presets:[{label, ratio, width, height}], closest: label|null}
POST   /api/projects/{pid}/upload?preset=&crop=  (body: the raw image bytes)
       -> 201 run json (mode "upload", run 1, becomes current_run); 409 if the project has runs or jobs; 413 / 415 / 400 see below
DELETE /api/projects/{pid}/enhanced      -> 204 (Discard the stored enhanced prompt)
POST   /api/projects/{pid}/enhance-and-generate {prompt, transparent?, seed?, thinking?, language?}
       -> 202 {chain_id, enhance_job_id, mode: "t2i"|"edit", run_reserved}
GET    /api/events                       -> text/event-stream; events: `job` (engine snapshot incl. state/step/total_steps/project_id/run), `run` (new run json once saved), `download` (models status while downloading, ~1/s), `log` ({job_id, line}), `style` ({project_id, style}; see "Art styles"), `serverlog` ({line}; only with `?serverlog=1`). Send a comment ping every 15 s.
GET    /api/settings                     -> settings
PUT    /api/settings {partial}           -> settings (validated as above; 400 on error)
GET    /api/models                       -> [{kind:"image", name, label, approx_gb, state, turbo_lora_ready, ...}, {kind:"text_encoder", name, label, approx_gb, source:"hf"|"local", state, hint?, ...}, {kind:"pe", name, mode, quant, variant, source, approx_gb, state, ...}]
POST   /api/models/image/{name}/download         -> 202 {started: bool}   (image part + turbo add-on; 404 if unknown)
POST   /api/models/text_encoder/{name}/download  -> 202 {started: bool}   (404 if unknown; 409 with the copy hint for a local encoder)
DELETE /api/models/image/{name}          -> {deleted: true, freed_bytes}   (image part + turbo LoRA from the HF cache; a text encoder sharing the repo is kept)
DELETE /api/models/text_encoder/{name}   -> {deleted: true, freed_bytes}
DELETE /api/models/pe/{name}             -> {deleted: true, freed_bytes}   (the <quant>/ weights + system_prompt.txt)
       freed_bytes is 0 if nothing was downloaded. 404 if unknown. 409 (message says why) for a local part, while a
       download touching the same repo runs, or while any job is queued/held/running/paused/cancelling. A resident
       model is unloaded first (a "model" unloaded event); afterwards a "download" event carries the refreshed list.
GET    /api/projects/{pid}/runs/{n}/image -> PNG ; /thumb -> webp
POST   /api/projects/{pid}/enhance {prompt, thinking?, quant?, language?}
       -> 202 {job_id, mode}   (prompt enhancer; NOT a run: nothing is saved to the project)
       mode = "t2i" if the project has no current_run else "i2i" with ref = current_run's png (pinned at submit).
       Unspecified options come from settings (pe_*). 400 on an empty prompt / bad option. 409 if the enhancer
       for that mode+quant isn't downloaded (message names it, e.g. "prompt enhancer 't2i-8bit' is not downloaded").
       POST /runs additionally accepts an optional `enhanced_from` (the user's original text); stored in the run json (null otherwise).
GET    /api/model                        -> {state: "loading"|"loaded"|"unloaded", key: [...]|null}   (the one resident model)
POST   /api/models/pe/{name}/download    -> 202 {started: bool}   (name e.g. "t2i-8bit"; 404 if unknown)
GET    /                                 -> static index.html ; other static paths served from imageatelier/static
```

Image URLs inside JSON use these routes.

## Decisions and details beyond the contract

- **Queued edits are pinned at submit time.** A run's reference image, `mode` and `parent_run` are taken from
  the project's `current_run` *when the POST is accepted*, not when the job starts. Consequence: prompts
  submitted back-to-back on a fresh project are all `t2i` (current_run is still null for each), and prompts
  submitted while run N is still generating edit the previous current run, not N. Runs may be queued across
  projects; the engine is FIFO.
- **Validation errors are 400**, including malformed JSON bodies (FastAPI's default 422 is remapped). Error
  body: `{"detail": "<message>"}`. Unknown project, run, or model variant in a path is 404.
- **Timestamps** (`created_at`, `updated_at`) are Unix epoch seconds (float).
- **Project** JSON: `{id, name, created_at, updated_at, current_run, enhanced, style}` (`style` defaults to `{id: "none"}`; see "Art styles"). `GET /api/projects/{pid}` adds `runs`.
- **Run** JSON (on disk): `{run, project_id, prompt, mode, parent_run, seed, steps, turbo, image_model, text_encoder, width,
  height, created_at, gen_s, load_s, job_id, enhanced_from, style}` (`style` see "Art styles"; `enhanced_from` is null unless the run was submitted with it). In API responses and `run` events, `image_url` and `thumb_url`
  are added (`/api/projects/{pid}/runs/{n}/image` and `/thumb`). `seed` and `steps` are the resolved values.
- **Gallery `latest`** describes the project's `current_run` (the image the next prompt will edit), or `null`
  if the project has no runs. `run_count` is the number of saved runs.
- **Settings** keys and defaults: `image_model "official-8bit"`, `text_encoder "heretic-8bit"`, `turbo true`, `steps 6`, `low_memory true`,
  `vae_tiling false`, `resolution 1024` (int, 256..2048, **a multiple of 32**: `resolution must be a multiple of 32` is a 400), plus the `pe_*` keys below. Rules: `image_model` and `text_encoder` must be known names (`text_encoder` is `"heretic-8bit"` or `"official-8bit"`); `variant` is gone (a PUT with it is 400 "variant was replaced by image_model/text_encoder"); turbo
  requires `steps` in {6, 8}; non-turbo requires 2 <= `steps` <= 100. If a PUT changes `turbo` without
  `steps`, `steps` resets to 6 (turbo) or 40 (non-turbo). Unknown keys are a 400. A failed PUT changes nothing.
- **Run request defaults:** omitted `turbo`, `image_model` and `text_encoder` come from settings. Omitted `steps` comes from
  settings if the effective `turbo` equals the settings' `turbo`, otherwise the mode default (6 / 40).
  `prompt` must be non-empty. A part that is not ready (not downloaded, not installed, or still downloading) is 409, with a plain message such as "The text encoder (Official, 8-bit) isn't installed. Copy the folder ..."; turbo without the turbo add-on is 409 too.
- **Run numbers** start at 1 and increase per project; `run_reserved` is returned immediately. Numbers of
  failed, cancelled, or rejected submissions may be skipped. A run exists once its `.json` exists.
- **Events** (`text/event-stream`, `event: <name>` / `data: <json>`): `job` on every engine state or
  progress change (the engine snapshot plus `project_id` and `run`); `run` once per saved run,
  `{project_id, run: {...}}`. **Ordering:** for a successful job the `run` event is sent *before* the terminal
  `job` event (`state: done`), and the same holds for failed/cancelled jobs once their partial files are
  removed. A client never sees a terminal state for a job before its run is saved and `current_run` advanced;
  until then `GET /api/jobs` reports the job as `running`, so `done` there implies the run is saved;
  `download` carries the list of `/api/models` statuses roughly once a second while a download runs (plus one
  trailing event). Keep-alive comments (`: ping`) every 15 s. Events are not replayed; reconnect and refetch.
- **Submitting right after a job finishes:** `POST .../runs` first waits (up to 30 s) for any job of that project
  that the engine has finished but the app has not yet saved, so the new run's `parent_run` is the just-finished
  run. A job that is still running when the request arrives does not count: the edit is pinned to the
  `current_run` at that moment.
- **Prompt enhancer.** `POST /projects/{pid}/enhance` queues an `enhance` job on the same FIFO engine as runs.
  Its job snapshot (in `/api/jobs` and `job` events, with `project_id`, `run: null`) has `kind: "enhance"`,
  `mode` (`t2i`|`i2i`), `quant`, `thinking`, `max_new_tokens`, `language`, `phase` (`thinking`|`writing`),
  `tokens` and `text` (the live rewrite so far). The terminal `done` snapshot carries
  `result: {rewritten_prompt, fell_back, finish_reason, thinking_tokens, answer_tokens, tok_s, gen_s, load_s,
  peak_mem_bytes, mode, quant, thinking, language}`. Run jobs keep their existing shape plus `kind: "generate"`.
  Enhance jobs have no `run` event and no settle step: `done` means the result is final. They can be cancelled
  with `POST /api/jobs/{id}/cancel` like runs.
- **One model at a time.** The engine keeps exactly one model in memory. An enhance job loads the rewriter and
  *leaves it loaded*, so repeated enhancing is fast. The image model is loaded only when a run executes (which
  unloads the rewriter), and the next enhance swaps back. Each swap emits `model` events (below).
- **`model` SSE event**: `{type:"model", state:"loading"|"loaded"|"unloaded", key:[...]|null, load_s, ...}` with keys
  `["image", image_model, text_encoder, turbo]`, `["pe_t2i", quant]` (or `["pe_t2i", "heretic-8bit"]`) or `["pe_i2i", quant]`. Unloads add `active_bytes`; a failed
  load is an `unloaded` event with `error`. The first event on every `/api/events` connection is a `model` event
  with the current resident state, and `GET /api/model` returns the same.
- **Models list and downloads.** `GET /api/models` returns the image parts (`kind: "image"`), the text encoders (`kind: "text_encoder"`),
  then the official enhancers `t2i-8bit` and `i2i-8bit` (plus the local Heretic entry below)
  (`kind: "pe"`; about 10.5 GB each). Status fields are `state`
  (`ready`|`downloading`|`not_downloaded`), `bytes_done`/`bytes_total` while downloading, and `error`. The
  `download` SSE event carries this whole list (both kinds, keyed by `kind` + `name`).
- **Enhancer settings** (same validation rules as the others; unknown values are a 400): `pe_quant` is **fixed at
  `"8bit"`** (kept as a key for compatibility; 4-bit was removed, and `"4bit"` is now a 400); `pe_thinking` boolean, default `true` (slower, usually better rewrites); `pe_max_new_tokens`
  int 256..16384, default 8192 (edit rewrites can think ~4k tokens); `pe_language` `"english"` (default) | `"auto"`; `pe_variant` `"official"` (default) | `"heretic"`.
- **Job states and strict serialization.** A job's `state` is one of `queued`, `loading` (model load), `running`,
  `cancelling`, then exactly one terminal state: `done`, `failed` or `cancelled`. The engine runs one job at a time
  and reports a terminal state only after that job's GPU work has been synchronized and its cleanup has run; the
  next queued job is dequeued only after that, so jobs never overlap and a terminal state means the GPU is free.
  Job results gain `teardown_s` (seconds spent in that sync and cleanup) and `gpu_util_at_start` (GPU utilisation
  sampled just before the job began, useful for spotting a previous job that had not settled).
- **Cancel semantics.** `POST /api/jobs/{id}/cancel` on a *queued* job removes it and it becomes `cancelled`
  immediately. On a *running* (or loading) job it sets `cancelling` immediately and returns `{cancelled: true}`;
  `cancelled` follows only after the job stops and tears down, so a client should keep showing the job (as
  "Cancelling") until the terminal event. Generation stops at the next denoising step or token; a cancel issued
  during a model load or an mlx-vlm prefill cannot interrupt it and only takes effect once that finishes. A second
  cancel of a `cancelling` job returns `true`; a finished or unknown job returns `false`. Clients must treat every
  non-terminal state (including `cancelling`) as active.
- **State machine.** `queued <-> held` (hold / release; a held job keeps its queue position but is skipped when the
  engine picks the next job), `queued | held -> cancelled` (instant), `loading | running <-> paused`
  (pause / resume), `loading | running | paused -> cancelling -> cancelled`. `done` and `failed` come from
  `running`. `held` and `paused` are non-terminal ("active"): clients, the settle logic and project deletion all
  treat them like `queued`/`running`.
- **Job controls.** `POST /api/jobs/{id}/pause|resume|hold|release` return `200` with the enriched job snapshot plus
  `changed`. `changed: false` means a no-op (already paused / already held / already queued), not an error. An action
  that is not valid for the job's current state is `409` with the engine's message (for example "cannot pause a
  queued job (use hold)"); an unknown id is `404`. `pause` sets `pause_requested: true` first; `state` becomes
  `paused` only once the job is actually blocked with its GPU work synchronized, and `resume` also withdraws a
  pause request that has not taken effect yet. `cancel` additionally works on `held` jobs (removed at once) and
  `paused` jobs (-> `cancelling`).
- **Pause boundaries.** A generate job pauses after a denoising step, an enhance job between tokens. Model load,
  prompt/reference encoding, the rewriter's prefill, VAE decode and saving cannot pause; a request made then takes
  effect at the next boundary and is dropped if the job finishes first. **A paused job blocks the queue, by
  design** (strict serialization: nothing else runs until it is resumed or cancelled). Time spent paused is
  reported as `paused_s` on the snapshot and in the result, and excluded from timings.
- **Snapshots.** Every job snapshot now also has `pause_requested`, `paused_s`, `queue_position` (1-based among
  queued + held jobs, else null) and, from the API layer, `project_name`. Queued and held jobs get a fresh `job`
  event when their `queue_position` changes.
- **`GET /api/jobs` order:** running / loading / paused / cancelling jobs first, then queued and held jobs by
  `queue_position`, then the most recent ~20 finished jobs (oldest first).
- **Logs.** `GET /api/jobs/{id}/log` is the per-job log (model loads, pauses, timings, errors);
  `GET /api/logs/server?tail=N` is the last N lines of the server log ring (uvicorn, app and engine logging;
  access logs excluded). Live lines arrive as SSE events: `log` `{job_id, line}` goes to every client; `serverlog`
  `{line}` goes only to connections opened with `GET /api/events?serverlog=1` (opt-in per connection, so ordinary
  clients never receive the chatty stream). Lines are rate-limited at the source, and the app never logs while
  forwarding them.
- **Heretic rewriters (local models).** `GET /api/models` also lists `t2i-heretic-8bit` and `i2i-heretic-8bit`:
  `{kind:"pe", name:"i2i-heretic-8bit", mode:"i2i", quant:"8bit", variant:"heretic", source:"local", approx_gb:10.4, state, ...}`
  (official entries carry `variant:"official"`, `source:"hf"`). Each is an on-disk mlx-vlm folder,
  `Qwen-Image-2.1-PE-{T2I,I2I}-Heretic-MLX-8bit/` under `$IMAGEATELIER_MODELS_DIR` (default `<repo>/models`), with
  its own `system_prompt.txt`. `state` is `ready` when the folder has `config.json`, every shard named in
  `model.safetensors.index.json`, and `system_prompt.txt`; otherwise `not_installed` (with a `hint`). They cannot be
  downloaded: `POST /api/models/pe/{name}/download` is `409` with the copy instructions.
- **`pe_variant` setting** (`"official"` default | `"heretic"`), applying to **both** new-image (`t2i`) and edit (`i2i`)
  enhances. When `"heretic"`, an enhance uses `<mode>-heretic-8bit` regardless of `pe_quant` (and of a `quant` in the
  request); when `"official"`, `<mode>-8bit`. `POST /projects/{pid}/enhance` returns `409` ("... is not installed;
  copy ...", naming the half it needs) if Heretic is selected but the needed half isn't installed. The engine's manager
  keys are `["pe_t2i", "heretic-8bit"]` / `["pe_i2i", "heretic-8bit"]` (official: `["pe_t2i", "8bit"]` / `["pe_i2i", "8bit"]`), so
  switching official <-> Heretic (or mode) reloads the rewriter and consecutive enhances with the same key do not.
  **Migration:** a stored `pe_t2i_variant` becomes `pe_variant` with the same value on load (the old key is dropped, the file
  rewritten, one info line logged). A `PUT` with `pe_t2i_variant` is `400` "pe_t2i_variant was replaced by pe_variant".
- **Which rewriter ran.** Enhance job snapshots carry `rewriter` (`"official-8bit"`, `"heretic-8bit"`;
  set once the job starts running) and the terminal `result` includes `rewriter` too.
- **Deleting a project** cancels its jobs in every non-terminal state (queued, held, loading, running, paused) and removes its folder.
- **Data dir:** `$IMAGEATELIER_DATA`, default `<repo>/data`: `settings.json`, `projects/<pid>/project.json`,
  `projects/<pid>/runs/<NNNN>.png|.json|.thumb.webp` (thumbnail max 384 px).

## 4-bit removal and settings migration

4-bit was removed everywhere: the only image variant is `8bit`, and the official enhancers are `t2i-8bit` and
`i2i-8bit` (plus the local Heretic `t2i-heretic-8bit` and `i2i-heretic-8bit`). `variant` defaults to `"8bit"` and `"4bit"` is rejected with a 400 in `PUT /api/settings`
and in `POST .../runs`. A stored `settings.json` that still says `variant: "4bit"` or `pe_quant: "4bit"` is **coerced to
`"8bit"` on load** (one info line in the server log, and the file is rewritten) instead of failing validation.

## Model parts and composed folders

A generation needs three pieces, kept as separate downloads so they can be mixed:
- **Image model** (`kind:"image"`, setting `image_model`, now only `official-8bit`): the transformer, VAE and processor,
  plus the optional turbo add-on (`turbo_lora_ready`). Downloaded from the Hub (`POST /api/models/image/{name}/download`).
- **Text encoder** (`kind:"text_encoder"`, setting `text_encoder`): `heretic-8bit` (default; from the Hub, downloadable) or
  `official-8bit` (a converted local folder under `$IMAGEATELIER_MODELS_DIR`, `source:"local"`, state `not_installed` with a `hint`
  until copied; it cannot be downloaded, so its download endpoint is 409).
- **Turbo** is a per-run mode (`turbo` setting / request field), not a model.

When a job runs, the engine *composes* a folder that symlinks the chosen image part and text encoder together (under
`$IMAGEATELIER_COMPOSE_DIR`) and loads that, so switching the text encoder only re-links and reloads; nothing is copied.
The resident model key is `["image", image_model, text_encoder, turbo]`, and results and snapshots carry `image_model`,
`text_encoder` and `turbo`; run JSON records them too. Pre-parts files (`variant`) are migrated on load: `variant` is dropped, and
`image_model`/`text_encoder` default to `official-8bit`/`heretic-8bit` (the pair the old single download contained).
The old `POST /api/models/{variant}/download` was **removed** (no alias). `4bit` stays removed.

## Image upload (starting a project from a picture)

`POST /api/projects/{pid}/upload` turns an uploaded image into **run 1** of an **empty** project, so the first prompt after it is an
edit of that image (the usual rule: `mode: "edit"`, `parent_run: 1`). Uploads never add to a project that already has anything.

- **Format:** the request body is the **raw image bytes** (any `Content-Type`; PIL sniffs JPEG/PNG/WebP/GIF/...). I chose a raw body over
  multipart so the size cap can be enforced while streaming (bytes are counted as they arrive, never trusting `Content-Length`) and the
  browser can send `fetch(url, {method:"POST", body: file})`. Options go in the query string:
  `preset` (a label from `GET /api/sizes`, e.g. `3:4`), `crop` (`x,y,w,h` in **source pixels, after EXIF orientation**, from the
  client's cropper) and `src` (`WxH`, the original oriented size). All are optional. **If the uploaded bytes are already exactly the preset's
  output size, they are stored unchanged and `crop`/`src` are only recorded as provenance** (the web client crops locally, renders the exact
  size, then sends that PNG with `preset`, `crop` and `src`). For any other size the server applies `crop` to the uploaded image itself.
- **Success:** `201` with the run json (the same shape as any run, plus the upload fields below); a `run` SSE event is broadcast and
  the thumbnail is made, exactly like a generated run.
- **Errors:** `404` unknown project; `409` the project has any run or any non-terminal job (or another upload is in flight);
  `400` empty body, unknown `preset`, malformed `crop`; `413` more than **25 MB** (the server may close the connection while the
  client is still sending, so browsers can see a network error instead of the 413: check the size client-side too) or more than **80 megapixels**;
  `415` the bytes are not a readable image.
- **Normalisation** (never any padding; the result is always an opaque RGB PNG of exactly the output size):
  1. EXIF orientation is applied, then transparency is flattened over white (an alpha channel would act as a mask in the edit path).
  2. If the image is already exactly a preset's output size at the current `resolution`, its pixels are kept as is (a client-side crop was already applied); any `preset`/`crop`/`src` given are recorded only as provenance.
  3. Otherwise the preset is `preset` or the **closest** ratio (`closest(w, h)`, smallest |log ratio difference|). The crop box is the client's
     `crop` (clamped to the image, then shrunk around its centre to the exact ratio) or the maximal centred box of that ratio
     (**cover fit**: scale to fill and crop the overflow). The box is resized with Lanczos to `dimensions(resolution, ratio)`.
  The crop uses the exact output ratio (`width/height`), so the resize never stretches.
- **Output size:** `dimensions(res, ratio)` is the engine's own formula: `w = sqrt(res*res*ratio)`, `W = max(32, round(w/32)*32)`,
  `H = max(32, round(w/ratio/32)*32)`. **Every edit keeps its parent's exact size:** the engine passes the reference image's width and
  height to the model (both are multiples of 32), instead of letting it re-derive a size from the ratio, which is not a fixed point at
  some resolutions (e.g. 16:9 at 320: a 416x224 parent would re-derive 448x224). At `resolution` 1024 the presets are 1:1 1024x1024, 4:3 1184x896,
  3:4 896x1184, 3:2 1248x832, 2:3 832x1248, 16:9 1376x768, 9:16 768x1376 (the labels and ratios come from `GET /api/sizes`, which also
  returns `closest` for `?w=&h=`).
- **Run json for an upload:** `{run: 1, mode: "upload", prompt: "", parent_run: null, seed/steps/turbo/image_model/text_encoder/gen_s/load_s/job_id/enhanced_from: null,
  width, height, created_at, source_size: [w, h], crop: [x, y, w, h], preset: "3:4"}`. `source_size` is the oriented source size and `crop` the box
  actually used. Changing `resolution` later does not resize existing images (new generations follow the reference image's size).

## Reference images (per project)

A project keeps a small library of reference images (a person's jacket, an object, a style ...) that generations can use alongside the
current image. Only **generation** uses them (prompt enhancement ignores them).

- **Storage:** `projects/<pid>/refs/<rid>.png` + `<rid>.thumb.webp` + `refs.json`. Uploading is `POST .../refs?name=` with the **raw
  image bytes** (same caps and errors as the project upload: 25 MB `413`, 80 MP `413`, not an image `415`, empty `400`; at most 30 per
  project, `409`). A reference is EXIF-oriented, flattened to opaque RGB and downscaled so the long side is <= 2048, **keeping its aspect
  (never cropped or padded)**. `name` defaults to "Reference N". Deleting a project deletes its references.
- **Enabled set and limits:** only `enabled` references are used, in `order`. The limit is the `max_refs` setting (0..9, default 3; the
  turbo LoRA was trained on at most three). A new reference is enabled if that stays within the limit, otherwise it is added **disabled**
  (the response says so in `note`). `PATCH {enabled: true}` beyond the limit is `409` "max N references enabled; disable one first".
  `PATCH {order: k}` moves it to position k (0-based); `PATCH {name}` renames. Changes are announced to all clients with an SSE
  **`refs`** event `{project_id, refs: [...]}` (the same list as `GET .../refs`).
- **Numbering (what the prompt means by "image 2"):** in an **edit** the current image is **Image 1** and the enabled references are
  **Image 2, 3, ...** in order; for a first (text-to-image) generation the references are **Image 1, 2, ...**. `GET .../refs` returns both
  numberings for every enabled reference: `image_index_edit` (2..) and `image_index_new` (1..); disabled ones have `null`.
- **Pinning at submit:** `POST .../runs` takes the enabled references *at that moment* and hard-links each (copy if linking fails) into
  `runs/<NNNN>.refs/<image number>.png`, passing those paths to the engine. Deleting, disabling or reordering a reference later therefore
  cannot affect a queued or running job. A failed or cancelled run deletes its `.refs` folder with its other partial files. If the enabled
  count exceeds `max_refs` the run is `409` "max N references enabled (got M)".
- **Run json:** `refs: [{id, name, image_index, width, height, image_url}]` (`image_url` is `/api/projects/{pid}/runs/{n}/refs/{i}`, the pinned
  copy), plus `reference_resolution` and `ref_detail` from the result. Upload runs have `refs: []` and both null.
- **Settings:** `max_refs` (int 0..9) and `ref_detail`: `auto` (default), `high` (1024), `standard` (768), `low` (512). **Auto** picks the
  reference resolution from the number of extra references: 0 extra keeps the normal resolution, 1 extra uses 1024, 2 to 3 use 768 (never 512;
  larger counts stay at 768). Fixed values apply to every count.

## Transparent backgrounds

`POST /api/projects/{pid}/runs` accepts `transparent: bool` (default `false`). When true, the model receives this wording (the model card's
trigger) instead of the bare prompt: `This is an RGBA image with transparency. <prompt>. The image has alpha channel and the background is
transparent.` where `<prompt>` is the user's text with **one trailing period removed** (so there is no `..`). The prompt enhancer is unchanged:
wrapping happens at generation time, so enhanced text is wrapped exactly like typed text.

- **Run json:** `prompt` is still the user's text; `transparent` (bool) and `model_prompt` (the exact text sent to the model; equal to `prompt`
  when not transparent) are new. Job snapshots carry `transparent`, and show the user's `prompt` (plus `model_prompt`), not the wrapped text.
- **Alpha rules (engine, at save time):** saved PNGs are always **RGBA**. A **normal** image is forced fully opaque (alpha 255 everywhere): the decoder
  used to leave alpha at 252..255, which made images faintly translucent, and this also stops an edited image's alpha acting as a mask when it is reused
  as a reference. A **transparent** image is cleaned up: **alpha < 16 becomes 0** (the MLX decode leaves the background at 1..15) and **alpha >= 240 becomes 255**
  (the subject decodes at 240..254, which looked faintly see-through); **16..239 is left as decoded** (soft edges, fur). Thumbnails are WebP and keep the alpha.
- **Edits:** a transparent image stays a cut-out only if the edit is sent with `transparent: true` as well; an edit with `transparent: false` of a cut-out
  fills the background (and the result is opaque). Clients should default the checkbox to the current image's `transparent` value.
  To keep that fill sensible, a `transparent: false` edit whose parent has any alpha < 255 is given a **flattened copy of the parent composited over white**
  (opaque RGB, same width and height, so the output size is unchanged), written at submit to `runs/<NNNN>.parent.png` next to the run (removed with
  the run's other files if it fails or is cancelled, kept otherwise as provenance). The model otherwise reads the empty RGBA area as noise.
  A `transparent: true` edit is given the RGBA original.
- `result.transparent` reports what was applied.

## Persisted enhanced prompt and Enhance & Generate

**`project.enhanced`** (in `GET /api/projects/{pid}`; `null` when there is none) is `{text, original, mode: "t2i"|"i2i", rewriter, fell_back, job_id, created_at, style}` (`style` is the project style the enhance used, see "Art styles"):
the last successful stand-alone `POST .../enhance`, so the rewrite survives a reload or restart and shows on every client.
- **Set/replace:** when a stand-alone enhance job reaches `done` (if it fell back, `text` is the original and `fell_back` is true). A newer `done` replaces it.
  A cancelled or failed enhance leaves the previous value untouched. Storing it does not change the project's `updated_at`.
- **Clear:** `DELETE .../enhanced` (204, also when there was none), or `POST .../runs` with `use_enhanced: true` once the run is **accepted** (then `enhanced_from`
  defaults to `enhanced.original` if the client did not send one). A rejected run clears nothing.
- **Events:** SSE `enhanced` `{project_id, enhanced}` on every change (`enhanced: null` when cleared). For an enhance job the `enhanced` event is sent before its terminal
  `job` event, and until it is stored `/api/jobs` reports the job as `running` (same rule as runs).

**`POST /api/projects/{pid}/enhance-and-generate`** runs an enhance and then a generation as one unit (`chain_id`). It accepts the run options that make sense
(`prompt`, `transparent`, `seed`, plus enhance `thinking`/`language`; everything else comes from settings) and returns `202 {chain_id, enhance_job_id, mode, run_reserved}`.
- **Pinned at submit, exactly like `POST /runs`:** the parent (`current_run`, or text-to-image), the enabled references (hard-linked into `runs/<NNNN>.refs/`), the parent
  flattening for a plain edit of a cut-out, the `transparent` flag, the model parts and steps, and the **run number** (reserved immediately; later runs submitted while the enhance
  is still waiting get later numbers, so the chain's run can finish after them). The enhance job itself sees the same parent image.
- **When the enhance is `done`:** the generate job is submitted with `prompt` = the rewrite (the original if it fell back), `enhanced_from` = the original text,
  and the pinned context. It is enqueued only then, so it runs behind anything queued in the meantime. The stand-alone `project.enhanced` is **not** set by a chained enhance.
- **If the enhance is cancelled or fails** (or the project was deleted): no run is created, the reserved number and all pinned files are released, and nothing is enqueued.
  If the generate cannot be submitted (for example settings changed to something invalid) the same cleanup happens.
- **Fields and events:** job snapshots (both jobs) carry `chain_id` and `chain_step` (`"enhance"` | `"generate"`; `null` for ordinary jobs). SSE `chain`
  `{chain_id, project_id, state: "generating"|"cancelled"|"failed", job_id?, run?, fell_back?, error?}` is sent right after the enhance's terminal `job` event (deleting the project cancels its waiting enhance, so a chain dropped that way reports "cancelled"). The
  generate job appears in `GET /api/jobs` as soon as it is submitted.
- Server restarts drop in-flight chains; leftover pinned files of unsaved runs are swept at startup.

## Art styles

A project has an art style: a curated text snippet the server adds to what is sent to the model. It is stored apart from the typed prompt, like the transparency wrap.

- **`GET /api/styles`** -> the presets in display order, each `{id, label, description, snippet, keep}`; the custom entry also has `custom: true` (its `snippet` and `keep` are `""`).
  Ids: `none` (default; adds nothing), `photoreal`, `cinematic`, `anime` (label "Anime / cel-shaded"), `stylized-anime`, `hand-drawn`, `watercolor`, `oil-painting`, `pixel-art`, `3d-render`, `low-poly`, `ink-line-art`, `comic-book`, `custom`.
  `snippet` is the phrase added to a new image (e.g. `in a cel-shaded anime illustration style, clean line art, flat colors`); `keep` is the note added to an edit (e.g. `Keep the cel-shaded anime illustration style.`).
  For `custom` the user's text is the snippet and the keep note is `Keep the same art style: <text>`.
- **`project.style`** is `{id}` or, for custom, `{id: "custom", custom: "<text>"}`. It is in `GET /api/projects/{pid}` and in every item of `GET /api/projects`. Projects created before the field existed read as `{id: "none"}`.
- **Set:** `PATCH /api/projects/{pid} {style: {id, custom?}}` -> the project (200). The whole value is replaced. It bumps `updated_at` like a rename. Errors are 400 `{detail}`:
  `style` null or not an object; a key other than `id`/`custom`; an `id` that is not a preset; `id: "custom"` with `custom` missing, not a string, or empty after trimming; custom text longer than 300 characters (counted after trimming);
  `custom` present and not null with any other id (it is rejected, not dropped). Stored custom text is trimmed. A failed PATCH changes nothing (including `name`/`current_run` sent in the same body).
- **Event:** SSE `style` `{project_id, style}` is sent after a PATCH that **changed** the style (comparing id and custom text; re-sending the current style sends nothing).
- **Composition** (one place, at submit; `prompt` is the text sent in `POST /runs`, or the rewrite for a chain). Whitespace around the prompt is trimmed and **one trailing period** is dropped (the rule of the transparency wrap), then:
  - new image (`mode "t2i"`): `<prompt>. <snippet>`
  - edit (`mode "edit"`): `<prompt>. <keep>`
  - style `none`: the prompt is sent unchanged (not trimmed).
  - **Transparency wraps the already-styled text:** `This is an RGBA image with transparency. <prompt>. <snippet>. The image has alpha channel and the background is transparent.`
  The style is read from the project when the run is prepared (`POST /runs`, `POST /enhance-and-generate`); changing it later does not affect runs already accepted (except a chain, below).
- **Run json:** `style` is `{id, label, custom?}` for the style in force (`label` is the preset label, "Custom" for custom; a run with no style is `{id: "none", label: "None"}`), and `model_prompt` is the final text including the style. `prompt` is still the user's text.
  Runs saved before this feature, and uploads, have `style: null`. Job snapshots of runs carry `style` too.
- **Enhancer:** `POST /enhance` and the enhance step of `enhance-and-generate` append a hint to the text given to the rewriter, using the project style at that moment (before the engine adds its English-output suffix):
  new image (`mode "t2i"`): `<prompt>\n\nArt style: <snippet>`; edit (`mode "i2i"`): `<prompt>\n\n<keep>`; style `none`: nothing. The hint is only in what the rewriter sees:
  enhance job snapshots and `enhanced.original`/`enhanced_from` show the user's own text.
- **`enhanced.style`** is `{id, custom?}` of the project style used for the enhance (`{id: "none"}` for an enhanced value stored before this feature).
- **No double-add:** the rewrite was written in the style, so it is not styled again when the style is unchanged.
  - `POST /runs` with `use_enhanced: true` and a stored `enhanced`: if `enhanced.style` equals the project's current style (same id and custom text), the prompt gets **no** snippet/keep; otherwise the current style is added as usual.
  - `enhance-and-generate`: when the enhance is done, the chain compares the style it was started with to the project's style *at that moment*: equal -> not added; different -> the current style is added, and that current style is what the run records.
  - Either way the run json still records `style`; only `model_prompt` differs. With `use_enhanced: false` the style is always added.
