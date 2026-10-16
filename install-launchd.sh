#!/usr/bin/env bash
# Keep ImageAtelier running at login via launchd.
#   ./install-launchd.sh             install (or reinstall) and start
#   ./install-launchd.sh --uninstall stop and remove
#   ./install-launchd.sh --dry-run   render the plist to a temp file and lint it; changes nothing
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
LABEL=com.imageatelier.server
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

render() {
  sed -e "s|__ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" launchd/$LABEL.plist.template > "$1"
  plutil -lint "$1"
}

case "${1:-}" in
  --uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null && echo "stopped $LABEL" || echo "$LABEL was not loaded"
    rm -f "$DEST"; echo "removed $DEST"
    # Also clean up the pre-rename WebStudio agent so an upgrade leaves nothing behind.
    launchctl bootout "$DOMAIN/com.webstudio.server" 2>/dev/null && echo "stopped com.webstudio.server" || true
    rm -f "$HOME/Library/LaunchAgents/com.webstudio.server.plist"
    ;;
  --dry-run)
    out="$(mktemp -t imageatelier-plist).plist"
    render "$out"; echo "rendered to $out"
    ;;
  "")
    [[ -f certs/server.pem ]] || { echo "error: run ./setup.sh first" >&2; exit 1; }
    mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/logs"
    render "$DEST"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$DEST"
    launchctl enable "$DOMAIN/$LABEL"
    echo "installed and started $LABEL; logs in $ROOT/data/logs/"
    ;;
  *) echo "usage: $0 [--uninstall|--dry-run]" >&2; exit 2 ;;
esac
