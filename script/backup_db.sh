#!/usr/bin/env bash
# Consistent SQLite + warranty evidence backup with optional encrypted S3 copy.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$ROOT/backups/$STAMP"

if [[ -z "${BACKUP_S3_URI:-}" && -f "$ROOT/.env" ]]; then
  BACKUP_S3_URI="$(awk -F= '$1 == "BACKUP_S3_URI" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT/.env")"
fi

mkdir -p "$ROOT/backups"
python3 "$ROOT/script/backup_runtime.py" --root "$ROOT" --output "$DEST"

if [[ -n "${BACKUP_S3_URI:-}" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "BACKUP_S3_URI is set but the AWS CLI is unavailable." >&2
    exit 1
  fi
  aws s3 cp \
    "$DEST" \
    "${BACKUP_S3_URI%/}/$STAMP/" \
    --recursive \
    --only-show-errors \
    --sse "${BACKUP_S3_SSE:-AES256}"
  echo "Encrypted off-site backup uploaded: ${BACKUP_S3_URI%/}/$STAMP/"
else
  echo "WARNING: BACKUP_S3_URI is empty; backup is local-only." >&2
fi

# Keep 8 local restore points. S3 lifecycle policy controls off-site retention.
find "$ROOT/backups" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z -r \
  | tail -z -n +9 \
  | xargs -0 -r rm -rf

echo "Backup completed: $DEST"
