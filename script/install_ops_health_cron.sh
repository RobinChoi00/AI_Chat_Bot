#!/usr/bin/env bash
# Daily RC IVR readiness check (Mon–Fri 9:00 AM server time).
#
# Usage (on EC2, from project root):
#   chmod +x script/install_ops_health_cron.sh
#   ./script/install_ops_health_cron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECK="$ROOT/script/check_rc_ivr_readiness.py"
MARKER="# AI_Chat_Bot ops health (RC IVR)"
CRON_LINE="0 9 * * 1-5 cd ${ROOT} && python3 ${CHECK} --notify >> ${ROOT}/logs/ops_health.log 2>&1 ${MARKER}"

chmod +x "$CHECK" "$ROOT/script/notify_ops_alert.py"

if crontab -l 2>/dev/null | grep -Fq "$MARKER"; then
  echo "Ops health cron already installed."
  crontab -l | grep -F "$MARKER" || true
  exit 0
fi

( crontab -l 2>/dev/null || true; echo "$CRON_LINE" ) | crontab -
echo "Installed daily RC IVR health check:"
crontab -l | grep -F "$MARKER"
