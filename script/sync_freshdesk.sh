#!/usr/bin/env bash
# Weekly Freshdesk → warranty knowledge sync (run from project root on EC2).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"

docker compose exec -T backend python script/freshdesk_extractor.py --max-pages 5 \
  >> "$ROOT/logs/freshdesk_sync.log" 2>&1

echo "$(date -Is) Freshdesk sync finished" >> "$ROOT/logs/freshdesk_sync.log"
