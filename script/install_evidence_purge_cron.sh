#!/usr/bin/env bash
# Weekly warranty evidence purge (Sundays 4:00 AM server time).
#
# Usage (on EC2, from project root):
#   chmod +x script/install_evidence_purge_cron.sh
#   ./script/install_evidence_purge_cron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PURGE="$ROOT/script/purge_warranty_evidence.py"
MARKER="# AI_Chat_Bot warranty evidence purge"
CRON_LINE="0 4 * * 0 cd ${ROOT} && python3 ${PURGE} --apply >> ${ROOT}/logs/evidence_purge.log 2>&1 ${MARKER}"

chmod +x "$PURGE"
mkdir -p "$ROOT/logs"

if crontab -l 2>/dev/null | grep -Fq "$MARKER"; then
  echo "Evidence purge cron already installed."
  crontab -l | grep -F "$MARKER" || true
  exit 0
fi

( crontab -l 2>/dev/null || true; echo "$CRON_LINE" ) | crontab -
echo "Installed weekly evidence purge:"
crontab -l | grep -F "$MARKER"
