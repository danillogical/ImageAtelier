# ImageAtelier

A home-network web app for creating and editing images with official **Qwen-Image-2.1** on Apple Silicon (MLX).
A project keeps an image lineage: the first prompt creates an image, and later prompts edit its current image.
Reference images, transparent output, prompt enhancement and generation history are supported.

## Requirements

- Apple Silicon Mac, macOS 14 or newer; 48 GB RAM or more recommended for BF16, or 32 GB for Q8 inference
- About **70.8 GB** for the complete original BF16 suite, plus space for each conversion
- [uv](https://docs.astral.sh/uv/) and [Homebrew](https://brew.sh), used to install mkcert

BF16 inference and initial conversion may need substantially more memory than quantized inference. A RAM recommendation
is a starting point for choosing a converted suite, not a guarantee that the entire BF16 conversion fits in memory.

## Quickstart

```sh
./setup.sh                    # dependencies, local CA and HTTPS certificate
./setup.sh --model bf16       # optional: download the complete source suite now
./local.sh                   # local Mac window, its own data in ./data-local
./local.sh --no-gui           # local server in the terminal
./run.sh                     # home-network server: HTTPS :443 and HTTP :80
./run.sh --http              # debugging server on HTTP :8080
```

`./generate-imageatelier-app.sh [folder]` builds a double-clickable launcher for `local.sh` in this checkout.
Running `local.sh` again restarts the existing local window/server on the selected `PORT` (8000 by default).
Closing the local window shows “Shutting down background server…” until the server has exited, then closes the app.
Data lives in `./data` for the network server or `./data-local` for the local window; override with `IMAGEATELIER_DATA`.
To start the network server at login, use `./install-launchd.sh`; remove it with `./install-launchd.sh --uninstall`.

## Model suites

The interface follows your system appearance: cream parchment in light mode and dark brown parchment in dark mode,
with matching pen icons, favicons and install assets. The native Dock icon uses the appearance at launch.
Regenerate both icon sets with `.venv/bin/python tools/make_icons.py`; SVG ornaments use the current theme's gold.
An already installed home-screen icon may need to be re-added for the operating system to refresh it.

Settings has one section, **Qwen-Image-2.1**, with four rows:

| Suite | Action |
|---|---|
| BF16 | Download |
| Q8 | Convert |
| Q6 | Convert |
| Q4 | Convert |

Every suite contains the image transformer, full text/vision encoder, VAE, T2I prompt enhancer and I2I prompt enhancer.
Only complete suites can be selected. Individual components cannot be downloaded, selected or mixed independently.
Each installed suite has a Delete button with confirmation. Deletion unloads resident weights and is blocked while jobs,
downloads or conversions are active. Deleting BF16 removes the pinned source checkpoints from the Hugging Face cache;
existing conversions stay available, but new conversions require downloading BF16 again.
BF16 downloads have Pause and Resume controls. Partial files are saved and resumed from their last downloaded byte.
After two minutes without progress, downloads reconnect automatically, retrying with a delay of 5–60 seconds.
Running downloads continue after restarting the app; an explicitly paused download stays paused until you resume it.
Turbo and uncensored variants are no longer supported. Old choices migrate to the official Q8 suite and standard 40-step generation;
old images and history remain on disk.

The BF16 download uses pinned revisions of three official repositories:

- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1): image transformer, text/vision encoder, VAE and processor
- [Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I): new-image prompt enhancer
- [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I): image-aware editing prompt enhancer

Convert buttons stay disabled until all BF16 components are complete. Conversion is local and offline, runs one component
at a time, and includes both prompt enhancers and the vision weights. Eligible transformer, text/vision and enhancer layers
are quantized. The reference VAE is exported with the suite in its supported full precision. A converted suite becomes ready
only after verification and atomic publication; a failed conversion leaves no selectable partial suite and can be retried.
Generation is paused during conversion. Restarting the app stops an active conversion; its partial output is not treated as ready.

With 6 GB reserved for macOS and other apps, the UI recommends BF16 on Macs with at least 48 GB RAM,
and Q8 on Macs with 32–47 GB. No suite is marked recommended below 32 GB. Selecting a suite is always explicit.
At 1024×1024 with text-encoder offload and VAE tiling disabled, measured peak model allocations were
32.4 GB for BF16 and 22.9 GB for Q8. Q6 is estimated at 21.2 GB and Q4 at 19.4 GB, giving totals of
27.2 GB and 25.4 GB with that overhead; both therefore also have a conservative 32 GB RAM recommendation.
Those lower-bit estimates keep the full-precision VAE and activation memory fixed; they are not measured runs.
VAE tiling may make Q6 or Q4 suitable for a conservative 24 GB budget, but that combination has not been benchmarked.
Source downloads use the Hugging Face cache (`IMAGEATELIER_HF_CACHE` overrides it); converted suites live under
`models/Qwen-Image-2.1/` (`IMAGEATELIER_MODELS_DIR` overrides `models/`). Existing unrelated model files are not deleted.
Only one inference model is held in memory at a time. The model and enhancers inherit the Qwen Research License.

## Custom image models and text encoders

Put single-file Qwen-Image-2.1 `.safetensors` checkpoints directly in `models/` (or
`IMAGEATELIER_MODELS_DIR`). Open **Settings → Custom models → Refresh models**. The app inspects the
weights to distinguish image transformers from text and vision encoders.

Download or convert and select a complete official base suite, then click **Prepare** for each custom
checkpoint. Choose its card in **Image Transformers** or **Text and Vision Encoders**. Click the selected card again to remove its override. The base suite supplies the VAE, processor, and both prompt enhancers.
Generation pins the selected custom components and base suite in each job.

Preparation runs offline and verifies parameter names and shapes against Qwen's architecture. It supports dense
BF16/F16/F32 transformers, including fused `img_mlp.gate_up` weights, and dense or ComfyUI `int8_tensorwise`
text encoders, including ConvRot. Encoders must include their vision tower. INT8 encoders expand to BF16,
using more disk and RAM than the source file. Prepared copies live under `models/.custom/`; originals stay in place.
Replacing a source file invalidates its preparation. The existing `models/` Git ignore rule covers all of these files.
Custom Hugging Face/MLX prompt-enhancer folders are listed in **T2I Prompt Enhancer** and **I2I Prompt Enhancer**.
Name their folders with `-PE-T2I-` or `-PE-I2I-` and include all indexed safetensors shards, configuration,
tokenizer, processor, and `system_prompt.txt`. Complete folders load directly through mlx-vlm without another copy.
Within each category, clicking a card selects it and clears the previous selection; clicking it again selects none.
An empty category selection uses the official base suite's component. The **Delete** action removes the source and
its prepared copy after confirmation, unloads resident weights, and clears its selection without changing projects.
GGUF and LoRA-only checkpoints are not supported by this import flow.

## Network access for Worlds and other computers

Open **Settings → Network access** in the local app. The two switches are independent, off by default, take effect
immediately, and persist across restarts:

- **Worlds API** listens on `http://<laptop>.local:8001`. `/enhance` returns a rewritten prompt as JSON for you to review
  or change; `/generate` separately returns PNG bytes for generation or image edits. Both accept multiple reference images,
  up to ten total (including an edit's source). It never creates projects, saves output images, writes reference temp files,
  or keeps image-job history.
- **Network UI** listens on `http://<laptop>.local:8002`. It serves the normal app with the same projects, images, model
  settings, and disk storage as the local window. Sign in as `imageatelier`, using the access key as the password.

Copy the **Access key** from local Settings. API clients send it as `Authorization: Bearer <key>`. Network controls and
key display are local-only. The key is persisted separately in the data folder with owner-only permissions. These are
trusted-LAN HTTP listeners; use a trusted network or an HTTPS reverse proxy for encrypted traffic. Closing the app stops
both listeners. `IMAGEATELIER_API_PORT` and `IMAGEATELIER_UI_PORT` override their default ports.

Both listeners share the existing single GPU worker and selected image model/text encoder; there is no second model
process. Both actions wait for completion by default. Set `wait: false` to receive a token immediately, poll for queue/step/token
progress, then fetch the result. DELETE cancels and discards; `DELETE ...?retain_status=true` (advertised by `/health`
`cancellation_status`) keeps the request pollable as `cancelling` until worker teardown finishes, then `cancelled`, until it is
consumed, discarded or expires. `/health` `request_lookup` advertises exact, memory-only recovery of an optional
correlated `wait: false` request; see API.md. Unclaimed completed results expire after ten minutes. The memory API allows
two outstanding requests across both actions (including retained results and cancellations) and returns 429 when full. Reference inputs are image data URLs,
never local paths or remote URLs. Per-request references, output bytes and job records are released after response,
disconnect, timeout or cancellation. Model weights remain resident for reuse. This is application-level memory-only image
handling; macOS may still use virtual-memory swap. Settings, model weights and normal UI data retain their usual storage.

From **webstudio.local**, call the laptop URL displayed in Settings; webstudio decides where to save the response:

```sh
curl --fail -H "Authorization: Bearer $IMAGEATELIER_ACCESS_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An anime illustration of a quiet village.","width":832,"height":1248,"steps":40,"seed":42}' \
  "http://<laptop>.local:8001/generate" --output generated.png
```

This provides the remote worker endpoint. Worlds' image-provider adapter must call this endpoint and commit the returned
PNG on webstudio; pointing Worlds at it requires that adapter rather than moving the game or asset store to this laptop.
See [API.md](API.md#memory-worker-8001) for the agent/client contract, request limits, errors and a complete
enhance → revise → generate example with multiple reference images.

## Phone setup

1. Open `http://<host>.local/ca` and follow the certificate installation instructions. On iOS, enable full trust in
   Settings → General → About → Certificate Trust Settings after installing the profile.
2. Open `https://<host>.local` and choose Share → Add to Home Screen.

Both network addresses share projects. HTTP supports the app except the offline PWA installation, which needs HTTPS.
`HTTP_MODE=ca ./run.sh` restricts the HTTP listener to certificate setup and HTTPS redirects.
The CA private key is never served. `./setup.sh --trust-local` also trusts the CA on this Mac.

## Troubleshooting

- Certificate name mismatch: re-run `./setup.sh` after changing host names or IP addresses.
- Stale UI: reload twice after an app update; the service worker refreshes the cached shell.
- Port occupied: stop the previous server or change `PORT`/`CA_PORT`.
- Interrupted download: automatic retries preserve progress. Use Pause to stop retries and Resume to continue.
- Failed download: fix the reported access or disk-space problem, then use Retry; completed files are kept.
- Failed conversion: the row shows the error. Resolve it, then retry Convert.

## Development

After dependency changes, use `./relock.sh` so the lock file retains public PyPI URLs.

```sh
uv sync --inexact --extra dev
.venv/bin/python -m pytest
```

The API is documented in [API.md](API.md).
