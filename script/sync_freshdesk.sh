#!/usr/bin/env bash
# Weekly Freshdesk → warranty knowledge sync (run from project root on EC2).
#
# Uses Search API: Resolved/Closed tickets only (see freshdesk_sync.py).
# Defaults: 30 search pages (~900 resolved cap) × up to 12 calendar months.
#
# Cron example (Sundays 3:00 AM):
#   0 3 * * 0 /home/ubuntu/AI_Chat_Bot/script/sync_freshdesk.sh
#
# Override via env:
#   FRESHDESK_SYNC_MAX_PAGES=20 FRESHDESK_SYNC_MONTHS_BACK=6 ./script/sync_freshdesk.sh
#   FRESHDESK_SYNC_KB=1 ./script/sync_freshdesk.sh   # also sync Solutions/KB
#   FRESHDESK_NO_REBUILD_FAISS=1 ./script/sync_freshdesk.sh  # skip FAISS rebuild
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_PAGES="${FRESHDESK_SYNC_MAX_PAGES:-30}"
MONTHS_BACK="${FRESHDESK_SYNC_MONTHS_BACK:-12}"
SYNC_KB="${FRESHDESK_SYNC_KB:-0}"
NO_REBUILD_FAISS="${FRESHDESK_NO_REBUILD_FAISS:-0}"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/freshdesk_sync.log"

KB_ARGS=()
if [[ "${SYNC_KB}" == "1" || "${SYNC_KB}" == "true" ]]; then
  KB_ARGS=(--sync-kb)
fi

FAISS_ARGS=()
if [[ "${NO_REBUILD_FAISS}" == "1" || "${NO_REBUILD_FAISS}" == "true" ]]; then
  FAISS_ARGS=(--no-rebuild-faiss)
fi

{
  echo "$(date -Is) Starting Freshdesk sync (max_pages=${MAX_PAGES}, months_back=${MONTHS_BACK}, sync_kb=${SYNC_KB}, rebuild_faiss=$((1-NO_REBUILD_FAISS)))"
  docker compose exec -T backend python script/freshdesk_extractor.py \
    --max-pages "${MAX_PAGES}" \
    --months-back "${MONTHS_BACK}" \
    "${KB_ARGS[@]}" \
    "${FAISS_ARGS[@]}"
  echo "$(date -Is) Freshdesk sync finished"
} >> "$LOG" 2>&1
