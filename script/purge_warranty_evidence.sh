#!/usr/bin/env bash
# Purge old warranty evidence (run from project root on EC2).
#
# Prefer docker exec so host Python need not have SQLAlchemy.
# Cron example (Sundays 4:00 AM):
#   0 4 * * 0 /home/ubuntu/AI_Chat_Bot/script/purge_warranty_evidence.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APPLY_FLAG=()
DAYS_ENV=()
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY_FLAG=(--apply) ;;
    --days=*) DAYS_ENV=(-e "WARRANTY_EVIDENCE_RETENTION_DAYS=${arg#--days=}") ;;
  esac
done

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/evidence_purge.log"

{
  echo "$(date -Is) Starting warranty evidence purge apply=${APPLY_FLAG[*]:-dry-run}"
  if docker compose ps --status running --services 2>/dev/null | grep -qx backend; then
    docker compose exec -T "${DAYS_ENV[@]}" backend \
      python script/purge_warranty_evidence.py "${APPLY_FLAG[@]}"
  else
    # Fallback for local/dev without docker
    PYTHONPATH="${ROOT}/app:${ROOT}:${PYTHONPATH:-}" \
      python3 "$ROOT/script/purge_warranty_evidence.py" "${APPLY_FLAG[@]}"
  fi
  echo "$(date -Is) Evidence purge finished"
} >> "$LOG" 2>&1
