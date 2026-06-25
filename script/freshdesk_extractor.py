"""
CLI wrapper for Freshdesk → freshdesk_tickets.json sync.

Requires .env:
  FRESHDESK_DOMAIN, FRESHDESK_API_KEY

Run (EC2 Docker):
  docker compose exec backend python script/freshdesk_extractor.py
  docker compose exec backend python script/freshdesk_extractor.py --test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from freshdesk_sync import FreshdeskETL, sync_freshdesk_knowledge  # noqa: E402
from warranty_knowledge import clear_knowledge_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Freshdesk tickets to JSON")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only verify FRESHDESK_DOMAIN + API key (no export)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max Freshdesk API pages to fetch (default 5)",
    )
    args = parser.parse_args()

    if args.test:
        etl = FreshdeskETL()
        raise SystemExit(0 if etl.verify_connection() else 1)

    result = sync_freshdesk_knowledge(max_pages=args.max_pages)
    clear_knowledge_cache()
    print(result)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
