#!/usr/bin/env bash
# Weekly warranty evidence purge (Sundays 4:00 AM server time).
#
# Usage (on EC2, from project root):
#   chmod +x script/install_evidence_purge_cron.sh
#   ./script/install_evidence_purge_cron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PURGE_SH="$ROOT/script/purge_warranty_evidence.sh"
MARKER="# AI_Chat_Bot warranty evidence purge"
CRON_LINE="0 4 * * 0 ${PURGE_SH} --apply ${MARKER}"

chmod +x "$PURGE_SH" "$ROOT/script/purge_warranty_evidence.py"
mkdir -p "$ROOT/logs"

# Drop any previous evidence-purge cron lines (old host-python form included).
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -vF "$MARKER" > "$TMP" || true
echo "$CRON_LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"

echo "Installed weekly evidence purge:"
crontab -l | grep -F "$MARKER"
