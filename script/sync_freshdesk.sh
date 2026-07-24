#!/usr/bin/env bash
# Weekly Freshdesk → warranty knowledge sync (run from project root on EC2).
#
# Calls sync helpers inside the backend container, then invalidates caches.
# On failure, sends an ops alert email (same path as Shopify sync).
#
# Cron example (Sundays 2:30 AM, before Shopify):
#   30 2 * * 0 /home/ubuntu/AI_Chat_Bot/script/sync_freshdesk.sh
#
# Override via env:
#   FRESHDESK_SYNC_NO_ALERT=1 ./script/sync_freshdesk.sh
#   FRESHDESK_SYNC_MONTHS=6 ./script/sync_freshdesk.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NO_ALERT="${FRESHDESK_SYNC_NO_ALERT:-0}"
MONTHS="${FRESHDESK_SYNC_MONTHS:-6}"
PAGES="${FRESHDESK_SYNC_PAGES:-20}"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/freshdesk_sync.log"

_notify_failure() {
  local exit_code=$?
  if [[ "${NO_ALERT}" == "1" || "${NO_ALERT}" == "true" ]]; then
    return "${exit_code}"
  fi
  python3 "$ROOT/script/notify_ops_alert.py" \
    --subject "[AI Chat Bot] Freshdesk knowledge sync failed" \
    --body "Freshdesk weekly sync exited with code ${exit_code} on $(hostname)." \
    --body-file "$LOG" \
    --tail 80 || true
  return "${exit_code}"
}

{
  echo "$(date -Is) Starting Freshdesk knowledge sync (months=${MONTHS}, pages=${PAGES})"

  FRESHDESK_SYNC_MONTHS="${MONTHS}" FRESHDESK_SYNC_PAGES="${PAGES}" \
    docker compose exec -T \
      -e "FRESHDESK_SYNC_MONTHS=${MONTHS}" \
      -e "FRESHDESK_SYNC_PAGES=${PAGES}" \
      backend python - <<'PY'
import os
import sys

sys.path.insert(0, "/app/app")
sys.path.insert(0, "/app")

from freshdesk_sync import sync_freshdesk_knowledge, sync_freshdesk_solutions
from freshdesk_knowledge_refresh import (
    build_knowledge_yield_stats,
    invalidate_warranty_knowledge_caches,
    log_kb_sync_yield,
    log_ticket_sync_yield,
)

months = int(os.environ.get("FRESHDESK_SYNC_MONTHS", "6"))
pages = int(os.environ.get("FRESHDESK_SYNC_PAGES", "20"))

ticket_result = sync_freshdesk_knowledge(max_pages=pages, months_back=months)
print("tickets:", ticket_result)
ticket_ok = bool(ticket_result.get("ok", True))
ticket_count = int(
    ticket_result.get("ticket_count")
    or ticket_result.get("usable_qa_pairs")
    or 0
)
resolved_scanned = int(ticket_result.get("resolved_scanned") or 0)
invalidate_warranty_knowledge_caches()
stats = build_knowledge_yield_stats(
    synced_ticket_rows=ticket_count,
    resolved_scanned=resolved_scanned,
)
log_ticket_sync_yield(
    ok=ticket_ok,
    ticket_count=ticket_count,
    resolved_scanned=resolved_scanned,
    stats=stats,
)

kb_result = sync_freshdesk_solutions(max_articles=500)
print("kb:", kb_result)
kb_ok = bool(kb_result.get("ok", True))
article_count = int(kb_result.get("article_count") or 0)
invalidate_warranty_knowledge_caches()
kb_stats = build_knowledge_yield_stats(synced_kb_articles=article_count)
log_kb_sync_yield(ok=kb_ok, article_count=article_count, stats=kb_stats)

if not ticket_ok:
    raise SystemExit(f"ticket sync failed: {ticket_result}")
if not kb_ok:
    raise SystemExit(f"kb sync failed: {kb_result}")
print("knowledge caches invalidated")
PY

  echo "$(date -Is) Freshdesk knowledge sync finished"
} >> "$LOG" 2>&1 || _notify_failure
