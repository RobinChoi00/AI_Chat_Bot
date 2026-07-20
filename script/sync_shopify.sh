#!/usr/bin/env bash
# Weekly OsakiUSA Shopify → products_export.csv sync (run from project root on EC2).
#
# CSV is written on the host because raw_data is mounted read-only inside the backend container.
#
# Cron example (Sundays 3:30 AM, after Freshdesk sync):
#   30 3 * * 0 /home/ubuntu/AI_Chat_Bot/script/sync_shopify.sh
#
# Override via env:
#   SHOPIFY_SYNC_DRY_RUN=1 ./script/sync_shopify.sh
#   SHOPIFY_NO_REBUILD_FAISS=1 ./script/sync_shopify.sh
#   SHOPIFY_NO_RESTART=1 ./script/sync_shopify.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN="${SHOPIFY_SYNC_DRY_RUN:-0}"
NO_REBUILD_FAISS="${SHOPIFY_NO_REBUILD_FAISS:-0}"
NO_RESTART="${SHOPIFY_NO_RESTART:-0}"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/shopify_sync.log"

DRY_ARGS=()
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  DRY_ARGS=(--dry-run)
fi

{
  echo "$(date -Is) Starting Shopify sync (dry_run=${DRY_RUN}, rebuild_faiss=$((1-NO_REBUILD_FAISS)), restart=$((1-NO_RESTART)))"
  python3 "$ROOT/script/sync_shopify_products.py" "${DRY_ARGS[@]}"

  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    if python3 "$ROOT/script/clean_shopify_data.py"; then
      echo "$(date -Is) cleaned_osaki_products.csv rebuilt"
    else
      echo "$(date -Is) WARN: clean_shopify_data.py failed or missing deps; continuing"
    fi
  fi

  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${NO_REBUILD_FAISS}" != "1" && "${NO_REBUILD_FAISS}" != "true" ]]; then
    docker compose exec -T backend python script/master_ingester.py --only osaki_products
  fi

  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${NO_RESTART}" != "1" && "${NO_RESTART}" != "true" ]]; then
    docker compose restart backend
  fi

  echo "$(date -Is) Shopify sync finished"
} >> "$LOG" 2>&1
