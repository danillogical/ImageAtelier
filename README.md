# ImageAtelier

A small home-network web app for generating and editing images with **Qwen-Image-2.1-Heretic** on Apple Silicon (MLX).
A *project* is one image lineage: the first prompt is text-to-image, every later prompt edits the project's current image.
Every run keeps its prompt, settings and image. It installs to a phone's Home Screen as a PWA.

## Requirements

- Apple Silicon Mac, macOS 14 or newer, 32 GB RAM or more recommended
- About 18 GB of disk for the 8-bit image model (plus a little for the turbo LoRA), and about 10.5 GB for each prompt rewriter you use
- [uv](https://docs.astral.sh/uv/) (`brew install uv`) and [Homebrew](https://brew.sh) (used to install `mkcert`)

## Quickstart

Copy or rsync this folder to the Mac that will run it (the `certs/ca/` folder carries the CA with it), then:

```sh
./setup.sh                 # Python env, local CA, HTTPS certificate
./setup.sh --model 8bit    # optional: also download a model now (the UI can download later)
./run.sh                   # the app on HTTPS :443 and plain HTTP :80
```

`setup.sh` is safe to re-run. It regenerates the certificate when the host name or IP addresses change. Pass `--trust-local`
to also add the CA to *this* Mac's keychain (it prompts for your password; nothing touches the keychain otherwise).

`./run.sh --http` serves everything over plain HTTP on :8080 for debugging. Set `PORT` / `CA_PORT` to change ports.

For use on this Mac only, `./local.sh` opens ImageAtelier in its own window on `localhost:8000`, with its own data in `./data-local`; quitting the window stops the server (`./local.sh --no-gui` serves in the terminal instead). `./generate-imageatelier-app.sh [folder]` builds `ImageAtelier.app`, a double-clickable launcher for `local.sh` in this checkout (drag it to Applications; regenerate it if the checkout moves).
Data (projects, images, settings) lives in `./data` (override with `IMAGEATELIER_DATA`).

To start at login: `./install-launchd.sh` (logs in `data/logs/`); remove with `./install-launchd.sh --uninstall`.

## Phone setup

1. On the phone, open `http://<host>.local/ca` (for example `http://my-mac.local/ca`) and follow the steps
   for your OS. On iOS: download the certificate, then *Settings, Profile Downloaded, Install*, then
   *Settings, General, About, Certificate Trust Settings*, and switch on full trust.
2. Open `https://<host>.local`, then *Share, Add to Home Screen*.

The app is also served over plain HTTP on :80 (`http://<host>.local`), with no CA setup needed. Everything works there except the PWA install: browsers only enable service workers over HTTPS, so on HTTP *Add to Home Screen* gives a full-screen shortcut, not an offline-capable app, and traffic is unencrypted on your Wi-Fi. Both addresses share the same projects. Run `HTTP_MODE=ca ./run.sh` to restrict :80 to `/ca`, `/ca/rootCA.crt`, and a redirect to HTTPS.
The CA private key never leaves `certs/ca/` and is not served.

## Prompt enhancer

The composer has an **Enhance** action that rewrites your prompt with Qwen's own prompt-enhancer models before you generate:
text-to-image for an empty project, and an image-aware rewrite (I2I) once the project has an image to edit. Nothing is saved
until you generate.

| Model | 8-bit |
|---|---|
| T2I enhancer (`prithivMLmods/Qwen-Image-2.1-PE-T2I-MLX`) | 10.5 GB |
| I2I enhancer (`prithivMLmods/Qwen-Image-2.1-PE-I2I-MLX`) | 10.5 GB |

Download them from Settings (or `POST /api/models/pe/<name>/download`). Only **one model is in memory at a time**: enhancing
loads the rewriter and keeps it, running an image swaps it for the image model, and the next enhance swaps back (a few seconds).
*Thinking* trades speed for quality: about 8 s per rewrite with it off versus about 30 s with it on, on an M5 (on by default;
change `pe_thinking` in Settings).

### Local models (Heretic prompt enhancer and text encoder)

Some models are not on the Hugging Face Hub and are installed by copying their folders into `models/` (or into the folder named
by `IMAGEATELIER_MODELS_DIR`). The Heretic prompt enhancer is two of them: copy `Qwen-Image-2.1-PE-T2I-Heretic-MLX-8bit/` (new
images) and `Qwen-Image-2.1-PE-I2I-Heretic-MLX-8bit/` (edits), each an mlx-vlm 8-bit conversion that includes its own
`system_prompt.txt`, into `models/`, then choose **Settings, Prompt enhancer, Heretic (uncensored)**. Each is about 10.4 GB. It
is abliterated (fewer refusals, slightly less steerable on style). Settings shows "not installed" and disables the choice until both
folders are complete. The official text encoder (`Qwen-Image-2.1-TextEncoder-MLX-8bit/`) is installed the same way.
`models/` is not part of git and is copied along with the project folder.

## Memory

Peak memory for image generation at 1024 px is about **20 GB** with *Low memory* on (the default; it frees the text encoder after
each prompt for about 0.2 s per job); leave it on with 32 GB. A prompt rewriter needs about 12 to 13 GB while it is loaded.
*VAE tiling* saves roughly another 2 GB but is a few seconds slower and changes pixels slightly. Only one generation runs at a time;
other prompts queue.

## Models and licence

| Variant | Repo | Size |
|---|---|---|
| 8-bit | `vanch007/Qwen-Image-2.1-Heretic-mlx-8bit` | about 18 GB |

That download is used as two parts: the **image model** (the official Qwen-Image 2.1 transformer and VAE) and the **Heretic text encoder**.
They are chosen separately in Settings, and an *official* text encoder can be installed locally by copying
`Qwen-Image-2.1-TextEncoder-MLX-8bit/` into `models/` (see API.md, "Model parts and composed folders").

Turbo mode uses the `Viggle/Qwen-Image-2.1-viggle-turbo` LoRA (6 or 8 steps). Revisions are pinned in `imageatelier/models.py`.
Qwen-Image weights are under the Qwen research licence: check its terms before any use beyond personal and research use.
The "Heretic" text encoder is **abliterated** (refusal behaviour removed), so it will not decline prompts; the people who can reach
this server are responsible for what it generates. Keep it on a trusted network.

## Troubleshooting

- **Certificate name mismatch after your IP or host name changed:** re-run `./setup.sh`, then restart `./run.sh`. Devices keep
  trusting the same CA, so phones need no new setup.
- **A UI update does not show on the phone:** the service worker serves the app shell from cache. Bump `VERSION` in
  `imageatelier/static/sw.js` whenever static files change, then reload twice.
- **The `.local` name does not resolve:** use the IP address shown by `./run.sh` (the certificate includes it), or check that the
  phone is on the same Wi-Fi network as the Mac.
- **Port already in use:** another server (or a previous run) holds 443 or 80; set `PORT=` / `CA_PORT=` (e.g. `PORT=8443 CA_PORT=8080 ./run.sh`). macOS lets a normal user bind ports below 1024 only on all interfaces, so `BIND=127.0.0.1` with port 443/80 needs a high port instead.
- **Service worker and "Add to Home Screen" need HTTPS**, so a plain-HTTP address will not give the full app experience.

## Changing dependencies

After changing dependencies in `pyproject.toml`, run `./relock.sh` instead of `uv lock` / `uv add`. If your shell sets
`UV_INDEX_URL` / `UV_DEFAULT_INDEX` to an internal index, uv writes that index into `uv.lock`, which other Macs cannot reach;
`relock.sh` re-locks against public PyPI.

## Tests

```sh
uv sync --extra dev && .venv/bin/python -m pytest
```

(`./setup.sh` runs a plain `uv sync`, which removes the dev extras again; run the command above whenever you want to test.)
The API is documented in `API.md`.
