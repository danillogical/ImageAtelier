#!/usr/bin/env bash
# Local-only WebStudio on this Mac: http://localhost:${PORT:-8000}, bound to 127.0.0.1, own data dir.
# First run seeds settings with the official Qwen parts (official text encoder + official prompt enhancer, no Heretic).
#   ./local.sh            open the window (quit it to stop the server)
#   ./local.sh --no-gui   serve in the terminal (Ctrl+C to stop)
#   PORT=8010 ./local.sh  another port
set -euo pipefail
cd "$(dirname "$0")"

export WEBSTUDIO_DATA="${WEBSTUDIO_DATA:-$PWD/data-local}"
mkdir -p "$WEBSTUDIO_DATA"
if [[ ! -f "$WEBSTUDIO_DATA/settings.json" ]]; then
  # Only the keys that differ from the defaults; the app fills in the rest and validates on load.
  printf '%s\n' '{"text_encoder": "official-8bit", "pe_variant": "official"}' > "$WEBSTUDIO_DATA/settings.json"
  echo "seeded $WEBSTUDIO_DATA/settings.json (official text encoder + official prompt enhancer)"
fi

export BIND=127.0.0.1 PORT="${PORT:-8000}"
if [[ "${1:-}" == "--no-gui" ]]; then
  exec ./run.sh --http
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv is missing; run ./setup.sh first" >&2; exit 1
fi
if ! .venv/bin/python -c 'import webview' 2>/dev/null; then
  uv sync --inexact --extra gui
fi
exec .venv/bin/python -m webstudio.gui
