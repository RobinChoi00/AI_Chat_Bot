#!/usr/bin/env bash
# Weekly SQLite backup for warranty + chat history.
# Usage:
#   ./script/backup_db.sh
# Cron (Sundays 3 AM server time):
#   0 3 * * 0 /home/ubuntu/AI_Chat_Bot/script/backup_db.sh >> /var/log/warranty_db_backup.log 2>&1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/db_data/chat_history.db"
DEST_DIR="$ROOT/db_data/backups"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DEST_DIR/chat_history_${STAMP}.db"

if [[ ! -f "$SRC" ]]; then
  echo "Missing database: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
echo "Backup written: $DEST"

# Keep last 8 weekly copies (~2 months).
ls -1t "$DEST_DIR"/chat_history_*.db 2>/dev/null | tail -n +9 | xargs -r rm -f
