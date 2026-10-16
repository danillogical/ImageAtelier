#!/usr/bin/env bash
# Start WebStudio.   ./run.sh          HTTPS on $PORT (default 443) + plain HTTP on $CA_PORT (default 80)
#                    HTTP_MODE=ca ./run.sh  plain HTTP serves only the CA page + a redirect to HTTPS
#                    ./run.sh --http   plain HTTP on $PORT (default 8080), for debugging
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lan.sh
source ./lan.sh

MODE=https
[[ "${1:-}" == "--http" ]] && MODE=http
export TQDM_DISABLE=1
export WEBSTUDIO_DATA="${WEBSTUDIO_DATA:-$PWD/data}"
mkdir -p "$WEBSTUDIO_DATA"
if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv is missing; run ./setup.sh first" >&2; exit 1
fi

HOST="$(lan_host)"
if [[ "$MODE" == http ]]; then
  PORT="${PORT:-8080}"
  echo "WebStudio (plain HTTP, no CA page listener): http://${HOST}:${PORT}  http://localhost:${PORT}"
  exec .venv/bin/python -m uvicorn webstudio.app:app --host "${BIND:-0.0.0.0}" --port "$PORT" --timeout-graceful-shutdown 3
fi

PORT="${PORT:-443}"
if [[ ! -f certs/server.pem || ! -f certs/server-key.pem ]]; then
  echo "error: certificates missing; run ./setup.sh first (or ./run.sh --http)" >&2; exit 1
fi
export WEBSTUDIO_CA_PORT="${CA_PORT:-80}"
export WEBSTUDIO_HTTP_MODE="${HTTP_MODE:-app}"
export WEBSTUDIO_BIND="${BIND:-0.0.0.0}"
export WEBSTUDIO_HTTPS_PORT="$PORT"
HS=""; [[ "$PORT" != 443 ]] && HS=":${PORT}"
CS=""; [[ "$WEBSTUDIO_CA_PORT" != 80 ]] && CS=":${WEBSTUDIO_CA_PORT}"
echo "WebStudio:  https://${HOST}${HS}"
while IFS= read -r ip; do [[ -n "$ip" ]] && echo "            https://${ip}${HS}"; done < <(lan_ips)
[[ "$WEBSTUDIO_HTTP_MODE" == app ]] && echo "            http://${HOST}${CS}  (plain HTTP: works everywhere, but no PWA install)"
echo "Trust the CA on a new device: http://${HOST}${CS}/ca"
exec .venv/bin/python -m uvicorn webstudio.app:app --host "${BIND:-0.0.0.0}" --port "$PORT" \
  --ssl-certfile certs/server.pem --ssl-keyfile certs/server-key.pem --timeout-graceful-shutdown 3
