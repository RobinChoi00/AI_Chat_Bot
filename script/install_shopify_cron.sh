#!/usr/bin/env bash
# Install weekly OsakiUSA Shopify catalog sync on EC2 (Sundays 3:30 AM server time).
#
# Usage (on EC2, from project root):
#   chmod +x script/install_shopify_cron.sh
#   ./script/install_shopify_cron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC="$ROOT/script/sync_shopify.sh"
MARKER="# AI_Chat_Bot Shopify weekly sync"
CRON_LINE="30 3 * * 0 ${SYNC} ${MARKER}"

chmod +x "$SYNC"

if crontab -l 2>/dev/null | grep -Fq "$MARKER"; then
  echo "Shopify cron already installed."
  crontab -l | grep -F "$MARKER" || true
  exit 0
fi

( crontab -l 2>/dev/null || true; echo "$CRON_LINE" ) | crontab -
echo "Installed weekly Shopify sync:"
crontab -l | grep -F "$MARKER"
