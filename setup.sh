#!/usr/bin/env bash
# One-time (and re-runnable) setup: Python env, local CA, HTTPS certificate, optional model download.
#   ./setup.sh [--trust-local] [--model 8bit|none]
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lan.sh
source ./lan.sh

TRUST_LOCAL=0
MODEL=none
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trust-local) TRUST_LOCAL=1 ;;
    --model) MODEL="${2:-}"; shift ;;
    --model=*) MODEL="${1#--model=}" ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$MODEL" in 8bit|none) ;; *) echo "error: --model must be 8bit or none" >&2; exit 2 ;; esac

step() { printf '\n==> %s\n' "$*"; }

step "Checking platform"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "error: ImageAtelier needs macOS on Apple Silicon (arm64); this is $(uname -s)/$(uname -m)." >&2
  exit 1
fi
echo "macOS $(sw_vers -productVersion), arm64: ok"

step "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not installed. Install it with one of:" >&2
  echo "    brew install uv" >&2
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh     (review the script first)" >&2
  echo "then re-run ./setup.sh" >&2
  exit 1
fi
uv --version

step "Installing Python dependencies (uv sync)"
uv sync

step "Checking mkcert"
if ! command -v mkcert >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "error: mkcert is missing and Homebrew is not installed." >&2
    echo "Install Homebrew from https://brew.sh, or install mkcert yourself" >&2
    echo "(https://github.com/FiloSottile/mkcert), then re-run ./setup.sh" >&2
    exit 1
  fi
  echo "mkcert not found; running: brew install mkcert"
  brew install mkcert
fi
echo "mkcert: $(command -v mkcert)"

step "Creating the HTTPS certificate"
# The CA lives inside the project so it can be copied along with the app.
export CAROOT="$PWD/certs/ca"
mkdir -p "$CAROOT"
HOST="$(lan_host)"
SANS=("$HOST" localhost 127.0.0.1)
# Plain-DNS names too (e.g. a router that registers DHCP hostnames): the bare name and name.<DHCP domain>.
SHORT="${HOST%.local}"
SANS+=("$SHORT")
DOMAIN="$(ipconfig getoption en0 domain_name 2>/dev/null || ipconfig getoption en1 domain_name 2>/dev/null || true)"
[[ -n "$DOMAIN" && "$DOMAIN" != local ]] && SANS+=("${SHORT}.${DOMAIN}")
# EXTRA_SANS="studio.home other.name" adds any custom names (e.g. a DNS record on the router).
for extra in ${EXTRA_SANS:-}; do SANS+=("$extra"); done
while IFS= read -r ip; do [[ -n "$ip" ]] && SANS+=("$ip"); done < <(lan_ips)
WANT="$(printf '%s\n' "${SANS[@]}")"
HAVE=""
[[ -f certs/sans.txt ]] && HAVE="$(cat certs/sans.txt)"
if [[ -f certs/server.pem && -f certs/server-key.pem && "$WANT" == "$HAVE" ]]; then
  echo "certificate already covers: ${SANS[*]}"
else
  [[ -n "$HAVE" ]] && echo "names/addresses changed; regenerating the certificate"
  mkcert -cert-file certs/server.pem -key-file certs/server-key.pem "${SANS[@]}"
  printf '%s\n' "${SANS[@]}" > certs/sans.txt
  chmod 600 certs/server-key.pem certs/ca/rootCA-key.pem 2>/dev/null || true
  echo "certificate covers: ${SANS[*]}"
fi
if [[ "$TRUST_LOCAL" == 1 ]]; then
  echo "Installing the CA into this Mac's trust store (--trust-local; may prompt for your password)"
  mkcert -install
else
  echo "CA not added to this Mac's keychain (pass --trust-local to do that)"
fi

if [[ "$MODEL" != none ]]; then
  step "Downloading model: $MODEL (this can take a while)"
  .venv/bin/python - "$MODEL" <<'PY'
import time
from imageatelier import models

parts = [("image", "official-8bit", models.image_status, models.start_image_download, models.is_image_downloading),
         ("text encoder", "heretic-8bit", models.te_status, models.start_te_download, models.is_te_downloading)]
for label, name, status, start, busy in parts:
    if status(name)["state"] == "ready":
        print(f"{label} {name} already downloaded"); continue
    start(name)
    while busy(name):
        s = status(name)
        done, total = s.get("bytes_done", 0), s.get("bytes_total")
        print(f"  {label}: {done/1e9:.1f} / {total/1e9:.1f} GB" if total else f"  {label}: {done/1e9:.1f} GB", flush=True)
        time.sleep(10)
    s = status(name)
    if s["state"] != "ready" or s.get("error"):
        print(f"{label} download failed:", s.get("error") or s["state"]); raise SystemExit(1)
    print(f"{label} {name} ready")
PY
fi

step "Done"
HTTPS_PORT="${PORT:-443}"; CA_PORT="${CA_PORT:-80}"
HS=""; [[ "$HTTPS_PORT" != 443 ]] && HS=":${HTTPS_PORT}"
CS=""; [[ "$CA_PORT" != 80 ]] && CS=":${CA_PORT}"
echo "Start the server with ./run.sh, then open:"
echo "  https://${HOST}${HS}"
while IFS= read -r ip; do [[ -n "$ip" ]] && echo "  https://${ip}${HS}"; done < <(lan_ips)
echo "Phones: first open  http://${HOST}${CS}/ca  to install and trust the CA."
