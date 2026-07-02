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

from freshdesk_sync import (  # noqa: E402
    FreshdeskETL,
    _OUTPUT_PATH,
    sync_freshdesk_knowledge,
    sync_freshdesk_solutions,
)
from freshdesk_ticket_summarizer import (  # noqa: E402
    is_enabled as summarizer_enabled,
    summarize_missing_tickets,
)
from warranty_knowledge import clear_knowledge_cache  # noqa: E402
from freshdesk_knowledge_refresh import invalidate_warranty_knowledge_caches  # noqa: E402

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
        default=30,
        help="Max Freshdesk Search pages to fetch (default 30, up to 60)",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=12,
        help="Calendar months of Resolved/Closed tickets to scan (default 12)",
    )
    parser.add_argument(
        "--no-llm-rescue",
        action="store_true",
        help="Skip the LLM rescue pass (regex-only steps).",
    )
    parser.add_argument(
        "--sync-kb",
        action="store_true",
        help="Also sync Freshdesk Solutions/KB to freshdesk_solutions.json.",
    )
    parser.add_argument(
        "--no-rebuild-faiss",
        action="store_true",
        help="Skip freshdesk_qa FAISS rebuild after a successful sync.",
    )
    args = parser.parse_args()

    if args.test:
        etl = FreshdeskETL()
        raise SystemExit(0 if etl.verify_connection() else 1)

    result = sync_freshdesk_knowledge(
        max_pages=args.max_pages,
        months_back=args.months_back,
    )

    if result.get("ok") and not args.no_llm_rescue and summarizer_enabled():
        import json

        try:
            with open(_OUTPUT_PATH, encoding="utf-8") as handle:
                raw_tickets = json.load(handle)
        except (OSError, ValueError):
            raw_tickets = []
        if raw_tickets:
            def _progress(idx: int, total: int) -> None:
                if idx == 1 or idx == total or idx % 20 == 0:
                    print(f"  LLM rescue {idx}/{total}")

            stats = summarize_missing_tickets(raw_tickets, progress=_progress)
            result["llm_rescue_stats"] = stats
            print(f"LLM rescue stats: {stats}")

    if args.sync_kb:
        kb_result = sync_freshdesk_solutions()
        print(f"KB sync: {kb_result}")
        if not kb_result.get("ok"):
            raise SystemExit(1)

    invalidate_warranty_knowledge_caches()

    if result.get("ok") and not args.no_rebuild_faiss:
        try:
            from warranty_faiss_rebuilder import rebuild_freshdesk_qa_index  # noqa: E402

            faiss_result = rebuild_freshdesk_qa_index()
            result["faiss_rebuild"] = faiss_result
            print(f"FAISS rebuild: {faiss_result}")
        except Exception as exc:
            result["faiss_rebuild_error"] = str(exc)
            print(f"FAISS rebuild failed: {exc}")

    print(result)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
