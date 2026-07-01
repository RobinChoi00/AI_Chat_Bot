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
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_PAGES="${FRESHDESK_SYNC_MAX_PAGES:-30}"
MONTHS_BACK="${FRESHDESK_SYNC_MONTHS_BACK:-12}"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/freshdesk_sync.log"

{
  echo "$(date -Is) Starting Freshdesk sync (max_pages=${MAX_PAGES}, months_back=${MONTHS_BACK})"
  docker compose exec -T backend python script/freshdesk_extractor.py \
    --max-pages "${MAX_PAGES}" \
    --months-back "${MONTHS_BACK}"
  echo "$(date -Is) Freshdesk sync finished"
} >> "$LOG" 2>&1
