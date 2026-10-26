#!/usr/bin/env bash
# Privacy retention report by default. Never override the legal hold here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for arg in "$@"; do
  case "$arg" in --apply) ;; *) echo "Usage: $0 [--apply]" >&2; exit 2 ;; esac
done
mkdir -p "$ROOT/logs"
{
  if docker compose ps --status running --services 2>/dev/null | grep -qx backend; then
    docker compose exec -T backend python script/purge_chat_records.py "$@"
  else
    python3 "$ROOT/script/purge_chat_records.py" "$@"
  fi
} >> "$ROOT/logs/privacy_retention.log" 2>&1
