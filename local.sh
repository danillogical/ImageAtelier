#!/usr/bin/env bash
# Local-only ImageAtelier on this Mac: http://localhost:${PORT:-8000}, bound to 127.0.0.1, own data dir.
# First run seeds settings with the complete official Qwen Q8 suite.
# Restarts an existing local ImageAtelier on the selected port.
#   ./local.sh            open the window (quit it to stop the server)
#   ./local.sh --no-gui   serve in the terminal (Ctrl+C to stop)
#   PORT=8010 ./local.sh  another port
set -euo pipefail
cd "$(dirname "$0")"

export IMAGEATELIER_DATA="${IMAGEATELIER_DATA:-$PWD/data-local}"
mkdir -p "$IMAGEATELIER_DATA"
if [[ ! -f "$IMAGEATELIER_DATA/settings.json" ]]; then
  # Only the keys that differ from the defaults; the app fills in the rest and validates on load.
  printf '%s\n' '{"model_suite": "q8"}' > "$IMAGEATELIER_DATA/settings.json"
  echo "seeded $IMAGEATELIER_DATA/settings.json (official Q8 suite)"
fi

export BIND=127.0.0.1 PORT="${PORT:-8000}"
if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv is missing; run ./setup.sh first" >&2; exit 1
fi
if [[ "${1:-}" != "--no-gui" ]] && ! .venv/bin/python -c 'import webview' 2>/dev/null; then
  uv sync --inexact --extra gui
fi
.venv/bin/python -m imageatelier.local_restart
if [[ "${1:-}" == "--no-gui" ]]; then
  exec ./run.sh --http
fi
exec .venv/bin/python -m imageatelier.gui
