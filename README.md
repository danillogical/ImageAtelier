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

Settings has one section, **Qwen-Image-2.1**, with four rows:

| Suite | Action |
|---|---|
| BF16 | Download |
| Q8 | Convert |
| Q6 | Convert |
| Q4 | Convert |

Every suite contains the image transformer, full text/vision encoder, VAE, T2I prompt enhancer and I2I prompt enhancer.
Only complete suites can be selected. Individual components cannot be downloaded, selected or mixed independently.
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
