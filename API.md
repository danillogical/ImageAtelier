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
POST   /api/projects/{pid}/runs {prompt, seed?, steps?, model_suite?, transparent?, use_enhanced?, enhanced_from?}
       -> 202 {job_id, run_reserved: N, mode}
       mode = "t2i" if the project has no current_run else "edit" with ref = current_run's png; parent_run = current_run.
       Omitted steps/model_suite come from settings. 400 on invalid input or independent part overrides; 409 unless the whole selected suite is ready.
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
GET    /api/models                       -> [{kind:"suite", name:"bf16"|"q8"|"q6"|"q4", label, state, recommended, ...}]
       Recommended: BF16 at >=48 GiB physical RAM, Q8 at 32–47 GiB, none below 32 GiB (6 GB overhead policy).
POST   /api/models/bf16/download         -> 202 {started: bool}; downloads all three pinned official source snapshots
POST   /api/models/bf16/pause            -> 202 {paused: bool}; saves progress and disables automatic continuation
POST   /api/models/bf16/resume           -> 202 {started: bool}; resumes saved progress (also retries a failed download)
POST   /api/models/{name}/convert        -> 202 {started: bool}; 409 until the full BF16 suite is ready or another conversion runs
       Per-part download/deletion endpoints return 410. Suite conversion is local, offline and atomic.
GET    /api/model                        -> {state: "loading"|"loaded"|"unloaded", key: [...]|null}   (the one resident model)
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
- **Settings:** `model_suite` (`bf16`, `q8`, `q6`, `q4`; default `q8`), `steps` (2–100; default 40),
  `low_memory true`, `vae_tiling false`, `resolution 1024` (256–2048, multiple of 32), and prompt/reference preferences.
  A suite must be ready before switching to it. `image_model`, `text_encoder`, `pe_quant`, `pe_variant` and `turbo` in
  GET responses are derived read-only compatibility metadata. Independent choices in PUT are rejected.
- **Run defaults:** steps and suite come from settings. `model_suite` may select another complete ready suite for one run.
  Turbo and independent component overrides are rejected. Missing any of the five suite components returns 409.
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
  `["image", image_model, text_encoder, false]`, `["pe_t2i", quant]` or `["pe_i2i", quant]`. Unloads add `active_bytes`; a failed
  load is an `unloaded` event with `error`. The first event on every `/api/events` connection is a `model` event
  with the current resident state, and `GET /api/model` returns the same.
- **Suite list and events:** `GET /api/models` and `download` SSE events carry the four whole-suite rows.
  BF16 states are `not_downloaded`, `downloading`, `retrying`, `paused`, `failed`, `ready`; conversions use `not_converted`, `converting`, `ready`.
  Downloads report `bytes_done`, `bytes_total`, and `retry_in` (seconds during retry backoff).
  After 120 seconds without progress the worker restarts; transient failures retry with 5–60 second backoff.
  Running intent and partial files survive app restarts; user-paused downloads never auto-resume.
  Download progress uses `bytes_done`/`bytes_total`. Conversion progress uses `phase`; failures use `error`.
  Quantized rows include `can_convert`, false until the complete BF16 suite is ready.
- **Enhancer preferences:** `pe_thinking`, `pe_language` and `pe_max_new_tokens` remain configurable through the API.
  The enhancer precision is derived from the chosen suite; per-request `quant` is rejected.
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
## Whole-suite models and migration

The only supported sources are `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-PE-T2I` and
`Qwen/Qwen-Image-2.1-PE-I2I`, at revisions pinned in `imageatelier/suites.py`.
BF16 contains the transformer, complete text/vision encoder, VAE and both enhancers. Q8, Q6 and Q4 are local conversions
of that entire suite. The convolutional reference VAE stays at supported full precision in converted suites.
Converted suites live at `$IMAGEATELIER_MODELS_DIR/Qwen-Image-2.1/<variant>/`; a verified `suite.json` manifest is required.
Conversion acquires engine maintenance and a shared filesystem lock. Its subprocess loads only local pinned snapshots,
converts components serially, checks image-component round trips and enhancer reloads, and publishes the final directory atomically.

Old independent and uncensored selections migrate to official Q8. Old Turbo 6/8-step settings migrate to 40 steps.
Historical run metadata is retained. The current Settings UI exposes only whole-suite preparation and selection.

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
- **Enabled set and limits:** only `enabled` references are used, in `order`. The limit is the `max_refs` setting (0..9, default 3). A new reference is enabled if that stays within the limit, otherwise it is added **disabled**
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

- **`GET /api/styles`** -> the presets in display order, each `{id, label, description, snippet, keep, suffix}`; the custom entry also has `custom: true` (its `snippet` and `keep` are `""`).
  Ids: `none` (default; adds nothing), `photoreal`, `cinematic`, `anime` (label "Anime / cel-shaded"), `stylized-anime`, `hand-drawn`, `watercolor`, `oil-painting`, `pixel-art`, `3d-render`, `low-poly`, `ink-line-art`, `comic-book`, `custom`.
  `snippet` leads a new-image prompt; `keep` leads an edit. Both anime snippets are `An anime illustration of `, immediately followed by the subject, then the preset-specific `suffix` (cel shading or stylized ink/shadows).
  For `custom` the user's text is the snippet and the keep note is `Keep the same art style: <text>`.
- **`project.style`** is `{id}` or, for custom, `{id: "custom", custom: "<text>"}`. It is in `GET /api/projects/{pid}` and in every item of `GET /api/projects`. Projects created before the field existed read as `{id: "none"}`.
- **Set:** `PATCH /api/projects/{pid} {style: {id, custom?}}` -> the project (200). The whole value is replaced. It bumps `updated_at` like a rename. Errors are 400 `{detail}`:
  `style` null or not an object; a key other than `id`/`custom`; an `id` that is not a preset; `id: "custom"` with `custom` missing, not a string, or empty after trimming; custom text longer than 300 characters (counted after trimming);
  `custom` present and not null with any other id (it is rejected, not dropped). Stored custom text is trimmed. A failed PATCH changes nothing (including `name`/`current_run` sent in the same body).
- **Event:** SSE `style` `{project_id, style}` is sent after a PATCH that **changed** the style (comparing id and custom text; re-sending the current style sends nothing).
- **Composition** (at submit; `prompt` is the user text or a rewrite): style text leads the prompt.
  - New image (`mode "t2i"`): `<snippet>. <prompt>`. Anime presets use `An anime illustration of <prompt>. <suffix>`.
  - Edit (`mode "edit"`): `<keep>. <prompt>` (without repeating the keep note's final period).
  - Style `none`: the prompt is unchanged, including whitespace.
  - Styled prompts trim surrounding whitespace and preserve user punctuation.
  - Transparency follows the style prefix and precedes the subject: `<style>. This is an RGBA image with transparency. <prompt>. The image has alpha channel and the background is transparent.` Anime new images use `An anime illustration of an RGBA image with transparency: <prompt>. <suffix> The image has alpha channel and the background is transparent.` With style `none`, the RGBA sentence leads the prompt as before. One trailing period is removed before joining the alpha reminder.
  The style is pinned when the run is prepared; changing it later does not affect accepted jobs (except chains, below).
- **Run json:** `style` is `{id, label, custom?}` for the style in force (`label` is the preset label, "Custom" for custom; a run with no style is `{id: "none", label: "None"}`), and `model_prompt` is the final text including the style. `prompt` is still the user's text.
  Runs saved before this feature, and uploads, have `style: null`. Job snapshots of runs carry `style` too.
- **Enhancer:** both enhancement routes receive the same style-first composition as generation, before the engine adds its English-output suffix. Snapshots and stored originals still show the user's text.
- **`enhanced.style`** records the style requested for enhancement.
- **Rewrites:** generation always ensures the current style is at the beginning, even if the rewriter omitted or moved it. An existing matching prefix is kept once. Anime-specific details are restored if missing.
- **Chains:** when enhancement finishes, the project's current style is applied before submitting generation. Already accepted generation jobs remain unchanged.



## Custom image transformers and text encoders

`GET /api/custom-models` scans single-file `.safetensors` checkpoints and local prompt-enhancer folders directly under the configured models directory.
Rows contain `id`, `kind` (`image`, `text_encoder`, `pe_t2i`, `pe_i2i`, or `unknown`), `label`, `filename`, `state`
(`not_prepared`, `preparing`, `ready`, `incomplete`, or `unsupported`), `phase`, and `error`. Valid checkpoints also report `bytes` and `path`.
The API accepts discovered IDs, never file paths.

`POST /api/custom-models/{id}/prepare` returns 202 `{started: boolean}`. It uses the currently selected complete base
suite's configuration and runs offline under model maintenance. Unsupported IDs or formats return 400; an incomplete
base suite or conflicting preparation returns 409. Poll `GET /api/custom-models` for progress and errors.

`PUT /api/settings` accepts `custom_image`, `custom_text_encoder`, `custom_pe_t2i`, and `custom_pe_i2i`: an empty string uses the base suite; a prepared
custom ID selects an override. Unprepared checkpoints return 409; wrong component types and invalid IDs return 400.
`model_suite` continues to choose the base VAE, processor, and prompt enhancers. Generation metadata records custom
components as `{custom-id}@{base-suite}` so queued jobs retain their selected base.


Complete local Hugging Face/MLX enhancer directories named with `-PE-T2I-` or `-PE-I2I-` are discovered alongside
single-file checkpoints and load directly. Partial folders have state `incomplete` and cannot be selected or deleted
through Settings while downloading. Selected enhancer IDs are pinned in enhancement jobs as `rewriter`.

`DELETE /api/custom-models/{id}` removes that model's source file/folder and any prepared output under engine
maintenance. Active preparation, downloads, or generation return 409; unknown IDs return 404. On success it returns
`{deleted: true, freed_bytes: number}` and clears any setting that selected the deleted model. Other category selections,
projects, and generated images are preserved.


## Independent LAN listeners and memory API

Local-only `GET /api/network` (never cached) returns `network_api_enabled`, `network_ui_enabled`, `api_listening`,
`ui_listening`, `api_url`, `ui_url`, `ui_username`, `access_key`, and `error`. Keys are absent from `/api/settings`.
`PUT /api/settings {network_api_enabled: boolean, network_ui_enabled: boolean}` controls the listeners immediately;
both default to false. Remote UI clients and foreign browser origins cannot read the key or modify these switches (403).
Bind failures return 409 and preserve the previous settings/listeners. Disabling either listener cancels its connections;
UI job processing continues in the shared engine. Configured listeners are restored on startup.

### Memory worker (`:8001`)

Every route requires `Authorization: Bearer <access_key>`. No UI, project, filesystem, or history routes are exposed.

- `GET /health` reports `{service:"imageatelier", mode:"memory", stores_images:false, provider:"qwen-image-2.1", active_requests,
  actions:["enhance","generate"], max_reference_images:10, polling:true, cancellation_status:true, result_ttl_seconds:600}`.
  Feature-detect `cancellation_status` before relying on retained cancellation.
- `POST /enhance` accepts JSON and **returns text, without generating an image**:
  - `prompt`: nonempty string, at most 16000 characters. No built-in project style is added.
  - `mode`: `t2i` or `i2i`. Omit it to choose T2I without references or I2I with references.
    Explicit `i2i` requires an image; explicit `t2i` rejects images instead of ignoring them.
  - `references`: zero to ten PNG/JPEG/WebP base64 data URLs, in image-number order. In I2I,
    Image 1 is the source image; Images 2–10 are additional references. All images are supplied to the vision model.
  - `thinking`: optional boolean; `language`: optional `english` or `auto`;
    `max_new_tokens`: optional integer 256–16384. Omitted options use current Settings.
  - `wait`: optional boolean, default true. Set false to get a request token immediately and poll for progress.
  - The selected T2I/I2I prompt enhancer in Settings is used (custom models included). Model and
    option selections are pinned when accepted, so later Settings changes cannot alter a queued request.

  Success is JSON with `rewritten_prompt`, `fell_back`, `mode`, `rewriter`, and model/timing metadata,
  with `Cache-Control: no-store`. For example:

  ```json
  {"rewritten_prompt":"A quiet library lit by warm reading lamps...","fell_back":false,"mode":"t2i","rewriter":"official-bf16"}
  ```

  Check `fell_back`: if the enhancer could not produce a valid rewrite, the returned text is the original prompt
  and `fell_back` is true. Review or modify `rewritten_prompt`, then make a separate `/generate` request.
  Calling `/enhance` alone creates no image, project, history entry or persisted enhanced-prompt record.
- `POST /generate` accepts JSON:
  - `prompt`: nonempty string, at most 16000 characters, sent verbatim (Worlds supplies its own art/style instructions).
  - `width`, `height`: both omitted, or both multiples of 32 between 256 and 2048.
  - `steps`: optional integer 2–100; defaults to the selected UI settings. `seed`: optional nonnegative integer.
  - `mode`: `txt2img` (default), `t2i`, or `edit`. Edit requires at least one reference; the first is the current image.
  - `references`: PNG/JPEG/WebP base64 data URLs, in conditioning order; no paths or fetched URLs. Per-image limits are
    8 MiB encoded file bytes and 16 million pixels. Up to ten images total, including an edit's source image,
    are accepted. This API limit is independent of the normal UI's maximum-extra-reference setting.
  - `transparent`: optional boolean (default false), applies the normal alpha finalization. The caller supplies any
    transparency wording in its prompt.
  - `wait`: optional boolean, default true. Set false for the polling workflow below.

By default, both POST routes keep the **same HTTP request open until completion**. `/enhance` returns JSON;
`/generate` returns the finished PNG bytes. With `wait: false`, either route returns 202 and a polling token instead.
There is no SSE stream on this API. The server remains
responsive while requests wait on the asynchronous GPU queue; model work is serialized with normal UI jobs.

The request body limit is 48 MiB, including base64 expansion of all images. Reference lists accept multiple images on
both POST routes, up to ten. The same 8 MiB file-byte / 16-million-pixel limit applies to each enhancement reference.
At most two memory requests can be outstanding **across both actions**, including queued work and unclaimed polling
results. A full queue returns 429
with Retry-After. Malformed requests return 400, oversized bodies 413, unavailable models 409, unavailable/cancelled jobs
503, and timeout 504. Client disconnect cancels the job (499 if a response can still be delivered). Each action has a
30-minute timeout, including queue time; configure the client's read timeout and any reverse proxy accordingly.
An explicit retry is needed after an HTTP error; there is no hidden provider fallback.

Success returns raw `image/png`, `Cache-Control: no-store`, and headers `X-Image-Provider`, `X-Image-Width`,
`X-Image-Height`, and `X-Image-Seed`. No local output file, project, pinned reference or image history record is created.
The response buffer is released after sending (also on a failed send); engine request data is removed. Models remain
resident, and the GPU worker is shared with regular UI jobs. No prompt rewriting, project styles or disk-backed upscaling
are applied automatically.

#### Agent/client example: enhance, revise, then edit with references

Use the memory listener on port 8001, not the project's `/api/.../enhance` routes on the disk-backed UI. Read the access
key from an environment variable; do not commit or log it. The example's input and output files live on the **client**
(webstudio), while ImageAtelier receives images inline. Use `mode: "t2i"` and omit `references` for text-only enhancement.

```python
import base64
import json
import os
from pathlib import Path
import urllib.request

base = os.environ["IMAGEATELIER_URL"].rstrip("/")  # http://<laptop>.local:8001
key = os.environ["IMAGEATELIER_ACCESS_KEY"]
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def post(action, body):
    request = urllib.request.Request(base + action, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
    })
    return http.open(request, timeout=1800)

def png_reference(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()

references = [png_reference("source.png"), png_reference("outfit.png"), png_reference("pose.png")]
with post("/enhance", {"prompt": "Use Image 2's outfit and Image 3's pose for Image 1's character.",
                       "mode": "i2i", "references": references, "thinking": False}) as response:
    enhanced = json.load(response)
if enhanced["fell_back"]:
    raise RuntimeError("Enhancer returned the original prompt; review it before generating")

final_prompt = enhanced["rewritten_prompt"]  # Worlds/another agent can inspect and change this text here.
with post("/generate", {"prompt": final_prompt, "mode": "edit", "references": references,
                        "width": 832, "height": 1248, "steps": 40, "seed": 42}) as response:
    Path("edited.png").write_bytes(response.read())
```

`/generate` always uses the supplied final prompt verbatim. There is no `enhance_prompt` boolean: enhancement is an
explicit, separate action so the client controls the prompt handed to the image model.

#### Optional polling and progress updates

Submit either action with `"wait": false`. The response is **202 Accepted**, with `Location` pointing to the status
endpoint and `Retry-After: 1`. Example:

```json
{"request_id":"<opaque-token>","status_url":"/requests/<opaque-token>","result_url":"/requests/<opaque-token>/result","cancel_url":"/requests/<opaque-token>"}
```

These URLs are relative to the memory API base URL. The request token identifies the transient job; it does **not**
replace Bearer authentication. Send the same access-key header on every status, result and cancellation call.

| Method and route | Behavior |
| --- | --- |
| `GET /requests/{request_id}` | Returns state, queue position and progress. Poll every 1–2 seconds. |
| `GET /requests/{request_id}/result` | Returns the PNG or enhancement JSON when finished; 409 with Retry-After while pending. |
| `DELETE /requests/{request_id}` | Cancels queued/running work or discards a retained result, frees the token and its slot, returns 204. |
| `DELETE /requests/{request_id}?retain_status=true` | Requests cancellation but keeps the request, watcher and slot; returns 204 for a known token. |

Plain DELETE cancels and discards: the token and its capacity slot are freed and later calls return 404. With
`retain_status=true`, queued or held work becomes `cancelled` immediately; running, loading or paused work reports
`cancelling` with `expires_in: null` and no `result_url` (its result returns 409) until the worker finishes teardown,
even past the 30-minute request timeout, and then reports `cancelled`. A request that already finished keeps its
`done` or `failed` outcome, and a `done` result remains retrievable. Repeating the retained call is harmless and does
not extend the TTL. **A 204, a 404, or a change in `active_requests` does not confirm teardown; only a polled terminal
`cancelled` state does.** A `failed` "interrupted" state is an actionable failure, not a confirmed cancellation.

Example status response:

```json
{"request_id":"<opaque-token>","kind":"generate","state":"running","queue_position":null,"progress":{"phase":null,"step":12,"total_steps":40,"tokens":null},"error":null,"result_url":null,"expires_in":null}
```

States are `queued`, `held`, `loading`, `running`, `paused`, `cancelling`, `done`, `failed` or `cancelled`.
`queue_position` is one-based while queued and null otherwise. Generation exposes `step` and `total_steps`;
enhancement exposes `phase` (`thinking` or `writing`) and generated `tokens`. Inapplicable or not-yet-known values are
null. Progress does not expose thinking text. Show phase names, steps or token counts; steps describe denoising,
not an overall percentage that also covers loading, encoding and decoding.

When done, status supplies `result_url` and `expires_in` (seconds). Fetch the result once and save it on the client.
**Result retrieval consumes the request**, including a terminal error response; subsequent status/result calls return
404. The PNG buffer is released after sending, including failed sends. A failed/interrupted download must be regenerated.
Status requests alone do not consume or extend the result's lifetime. For `failed`/`cancelled`, inspect `error` and
fetch the error result or delete the request to free its slot.

Unclaimed completed results/errors expire after **10 minutes**. A retained cancellation or error starts that
10-minute clock when its terminal state is observed. Fetching the result consumes the request (`cancelled` returns
503), and a plain DELETE discards it. Retained requests count toward the two-request limit, so a third request returns
429 with `Retry-After`. Unfinished jobs time out after **30 minutes** and their inputs are released after worker
teardown, except that a retained cancellation keeps waiting for the real terminal state. Disconnecting after a 202
does not cancel asynchronous work: use DELETE. Disabling the API or shutting down ImageAtelier cancels and clears
outstanding requests. Tokens and images are memory-only; after consumption, plain DELETE, expiry or app restart the
token returns 404.

Using the `post()` helper above, a polling enhancement looks like this:

```python
import time

headers = {"Authorization": "Bearer " + key}
with post("/enhance", {"prompt": "a quiet library", "wait": False}) as response:
    ticket = json.load(response)
try:
    while True:
        with http.open(urllib.request.Request(base + ticket["status_url"], headers=headers), timeout=30) as response:
            status = json.load(response)
        print(status["state"], status["progress"])
        if status["state"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    with http.open(urllib.request.Request(base + ticket["result_url"], headers=headers), timeout=1800) as response:
        enhanced = json.load(response)  # /generate's result is PNG bytes instead.
finally:
    # Best-effort cancellation/discard if interrupted. A consumed/expired token already returns 404.
    try:
        http.open(urllib.request.Request(base + ticket["cancel_url"], headers=headers, method="DELETE"), timeout=10).close()
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
```

### Network UI (`:8002`)

The normal app is available with HTTP Basic authentication: username `imageatelier`, password equal to the access key.
It uses normal disk-backed projects and generation. Its network/key controls are unavailable remotely. Legacy LAN-bound
main/HTTP listeners also enforce the UI switch and authentication, so they cannot bypass the off setting.
