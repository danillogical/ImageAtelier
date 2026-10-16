#!/usr/bin/env bash
# Re-lock dependencies against public PyPI so uv.lock works on any machine.
# (A corporate UV_INDEX_URL / UV_DEFAULT_INDEX would otherwise be written into uv.lock.)
set -euo pipefail
cd "$(dirname "$0")"
env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX=https://pypi.org/simple uv lock "$@"
echo "uv.lock refreshed against https://pypi.org/simple"
