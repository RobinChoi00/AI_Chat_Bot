#!/usr/bin/env bash
# Install weekly Freshdesk ticket sync on EC2 (Sundays 3:00 AM server time).
#
# Usage (on EC2, from project root):
#   chmod +x script/install_freshdesk_cron.sh
#   ./script/install_freshdesk_cron.sh
#
# To include KB sync each week:
#   FRESHDESK_SYNC_KB=1 ./script/install_freshdesk_cron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC="$ROOT/script/sync_freshdesk.sh"
MARKER="# AI_Chat_Bot Freshdesk weekly sync"
CRON_LINE="0 3 * * 0 ${SYNC} ${MARKER}"

chmod +x "$SYNC"

if crontab -l 2>/dev/null | grep -Fq "$MARKER"; then
  echo "Freshdesk cron already installed."
  crontab -l | grep -F "$MARKER" || true
  exit 0
fi

( crontab -l 2>/dev/null || true; echo "$CRON_LINE" ) | crontab -
echo "Installed weekly Freshdesk sync:"
crontab -l | grep -F "$MARKER"
