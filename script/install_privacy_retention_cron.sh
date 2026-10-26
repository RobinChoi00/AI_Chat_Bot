#!/usr/bin/env bash
# Weekly candidate report. Explicit --apply opt-in still respects the legal hold.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=""
case "${1:-}" in "") ;; --apply) MODE=" --apply" ;; *) echo "Usage: $0 [--apply]" >&2; exit 2 ;; esac
MARKER="# AI_Chat_Bot privacy retention"
chmod +x "$ROOT/script/purge_chat_records.sh" "$ROOT/script/purge_chat_records.py"
mkdir -p "$ROOT/logs"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
crontab -l 2>/dev/null | grep -vF "$MARKER" > "$TMP" || true
echo "30 4 * * 0 \"$ROOT/script/purge_chat_records.sh\"$MODE $MARKER" >> "$TMP"
crontab "$TMP"
echo "Installed weekly privacy retention${MODE:- dry-run}; legal hold remains enforced."
