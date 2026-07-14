#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB="$ROOT/script/backup_db.sh"
MARKER="# AI_Chat_Bot daily verified backup"
CRON_LINE="15 3 * * * $JOB >> $ROOT/logs/backup.log 2>&1 $MARKER"

mkdir -p "$ROOT/logs"
chmod +x "$JOB"

(crontab -l 2>/dev/null | grep -Fv "$MARKER" || true; echo "$CRON_LINE") | crontab -
echo "Installed daily backup cron. Verify with: crontab -l"
