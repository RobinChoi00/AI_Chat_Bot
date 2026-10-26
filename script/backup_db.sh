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

# A preservation hold covers backups too. Read only the non-secret flag; never
# source .env as shell code. Missing/invalid values preserve existing backups.
HOLD_FLAG="${PRIVACY_RETENTION_LEGAL_HOLD:-}"
if [[ -z "$HOLD_FLAG" && -f "$ROOT/.env" ]]; then
  HOLD_FLAG="$(awk -F= '$1 ~ /^[[:space:]]*PRIVACY_RETENTION_LEGAL_HOLD[[:space:]]*$/ {sub(/^[^=]*=/, ""); gsub(/[[:space:]\047\042]/, ""); print tolower($0); exit}' "$ROOT/.env")"
fi
case "${HOLD_FLAG:-1}" in
  0|false|no|off)
    # Keep 8 local restore points when a hold has explicitly been released.
    find "$ROOT/backups" -mindepth 1 -maxdepth 1 -type d -name '????????T??????Z' -print \
      | sort -r \
      | tail -n +9 \
      | while IFS= read -r backup_path; do
          if [[ "${backup_path##*/}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
            rm -rf -- "$backup_path"
          fi
        done
    ;;
  *) echo "Preservation hold active; existing backup restore points retained." ;;
esac

echo "Backup completed: $DEST"
