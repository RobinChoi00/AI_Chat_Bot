"""
Freshdesk → data/freshdesk_tickets.json sync for warranty self-help knowledge.

Env:
  FRESHDESK_DOMAIN, FRESHDESK_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "freshdesk_tickets.json"
_SOLUTIONS_PATH = _PROJECT_ROOT / "data" / "freshdesk_solutions.json"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MERGED_ANSWER_RE = re.compile(
    r"^(this ticket is closed|merged into ticket|expedite shipping)",
    re.I,
)

# Freshdesk Search API hard-limits pagination to 10 pages × 30 rows.
_SEARCH_MAX_PAGES_PER_QUERY = 10
_SEARCH_PER_PAGE = 30
_DEFAULT_MONTHS_BACK = 12
_DEFAULT_MAX_SEARCH_PAGES = 30


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", value)).strip()


def ticket_question(ticket: dict) -> str:
    text = (ticket.get("description_text") or "").strip()
    if text:
        return text
    return strip_html(ticket.get("description"))


def conversation_agent_answer(conversations: list[dict]) -> str:
    replies: list[str] = []
    for conv in conversations:
        if conv.get("incoming") is not False:
            continue
        if conv.get("private"):
            continue
        body = (conv.get("body_text") or "").strip() or strip_html(conv.get("body"))
        if body:
            replies.append(body)
    return "\n".join(replies)


def is_usable_qa_pair(question: str, answer: str) -> bool:
    if not question.strip() or not answer.strip():
        return False
    if _MERGED_ANSWER_RE.match(answer.strip()):
        return False
    if "merged into ticket" in answer.lower():
        return False
    return True


def normalize_freshdesk_domain(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path

    value = value.split("/")[0].strip()

    if "." not in value:
        return f"{value}.freshdesk.com"

    if value.endswith(".freshdesk.com"):
        return value

    return value


def iter_month_windows(months_back: int) -> Iterator[tuple[str, str]]:
    """
    Yield ``(start_iso, end_iso)`` calendar-month windows, newest first.

    Freshdesk Search caps at 300 hits per query; monthly windows keep each
    chunk under that limit for busy inboxes.
    """
    months_back = max(1, int(months_back))
    today = datetime.now(timezone.utc).date()
    year, month = today.year, today.month

    for _ in range(months_back):
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, month + 1, 1)
        yield start.isoformat(), end.isoformat()

        month -= 1
        if month == 0:
            month = 12
            year -= 1


def build_resolved_search_query(start: str, end: str) -> str:
    """Freshdesk Search query for Resolved(4) + Closed(5) in a date window."""
    return (
        f'"(status:4 OR status:5) AND created_at:>\'{start}\' '
        f"AND created_at:<'{end}'\""
    )


class FreshdeskETL:
    def __init__(self) -> None:
        raw_domain = os.environ.get("FRESHDESK_DOMAIN", "")
        api_key = os.environ.get("FRESHDESK_API_KEY", "")
        if not raw_domain:
            raise EnvironmentError("FRESHDESK_DOMAIN is not set.")
        if not api_key:
            raise EnvironmentError("FRESHDESK_API_KEY is not set.")

        self.raw_domain = raw_domain.strip()
        self.domain = normalize_freshdesk_domain(raw_domain)
        self.api_key = api_key.strip()
        self.base_url = f"https://{self.domain}/api/v2"
        self.auth: tuple[str, str] = (self.api_key, "X")
        self.headers = {"Content-Type": "application/json"}

    def verify_connection(self) -> bool:
        url = f"{self.base_url}/tickets"
        try:
            response = requests.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params={"per_page": 1},
                timeout=15,
            )
        except requests.exceptions.RequestException as exc:
            logger.error("Freshdesk connection failed: %s", exc)
            return False

        if response.status_code == 200:
            return True
        if response.status_code == 401:
            logger.error("Freshdesk 401 — invalid API key.")
        elif response.status_code == 404:
            logger.error("Freshdesk 404 — check FRESHDESK_DOMAIN (%s).", self.domain)
        else:
            logger.error("Freshdesk HTTP %s: %s", response.status_code, (response.text or "")[:300])
        return False

    def fetch_conversations(self, ticket_id: int) -> str:
        url = f"{self.base_url}/tickets/{ticket_id}/conversations"
        try:
            response = requests.get(url, auth=self.auth, headers=self.headers, timeout=15)
            if response.status_code == 200:
                return conversation_agent_answer(response.json())
        except Exception:
            pass
        return ""

    def _fetch_ticket_question(self, ticket_id: int) -> str:
        """Search hits may omit description — fetch the ticket body on demand."""
        url = f"{self.base_url}/tickets/{ticket_id}"
        try:
            response = requests.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params={"include": "description"},
                timeout=15,
            )
            if response.status_code == 200:
                return ticket_question(response.json())
        except Exception:
            pass
        return ""

    def _search_resolved_page(
        self, query: str, page: int
    ) -> tuple[list[dict[str, Any]], int]:
        url = f"{self.base_url}/search/tickets"
        try:
            response = requests.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params={"query": query, "page": page},
                timeout=30,
            )
            if response.status_code == 400 and page > _SEARCH_MAX_PAGES_PER_QUERY:
                return [], 0
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Freshdesk search failed (page %s): %s", page, exc)
            return [], 0

        payload = response.json()
        results = payload.get("results") or []
        total = int(payload.get("total") or 0)
        return results, total

    def fetch_resolved_tickets(
        self,
        *,
        months_back: int = _DEFAULT_MONTHS_BACK,
        max_pages: int = _DEFAULT_MAX_SEARCH_PAGES,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Pull Resolved(4) + Closed(5) tickets via the Search API.

        Unlike ``GET /tickets`` (recent mixed-status inbox), Search returns
        only matching tickets so we do not waste API calls on open/pending
        rows.  Monthly date windows avoid the Search cap of 300 hits/query.

        Parameters
        ----------
        months_back : calendar months to walk back (newest first).
        max_pages : total Search pages to fetch across all month windows
            (each page = 30 tickets; Freshdesk allows ≤10 pages/query).
        """
        stats: dict[str, Any] = {
            "resolved_scanned": 0,
            "usable_qa_pairs": 0,
            "search_pages_fetched": 0,
            "month_windows_scanned": 0,
            "fetch_mode": "search",
        }

        if not self.verify_connection():
            return [], stats

        pages_budget = max(1, min(int(max_pages), 60))
        seen_ids: set[int] = set()
        extracted: list[dict[str, Any]] = []

        for start, end in iter_month_windows(months_back):
            if pages_budget <= 0:
                break

            stats["month_windows_scanned"] += 1
            query = build_resolved_search_query(start, end)
            page = 1

            while page <= _SEARCH_MAX_PAGES_PER_QUERY and pages_budget > 0:
                results, _total = self._search_resolved_page(query, page)
                stats["search_pages_fetched"] += 1
                pages_budget -= 1

                if not results:
                    break

                for ticket in results:
                    ticket_id = int(ticket.get("id") or 0)
                    if not ticket_id or ticket_id in seen_ids:
                        continue
                    seen_ids.add(ticket_id)
                    stats["resolved_scanned"] += 1

                    question = ticket_question(ticket)
                    if not question.strip():
                        question = self._fetch_ticket_question(ticket_id)

                    answer = self.fetch_conversations(ticket_id)
                    time.sleep(0.15)

                    if not is_usable_qa_pair(question, answer):
                        continue

                    stats["usable_qa_pairs"] += 1
                    extracted.append(
                        {
                            "ticket_id": ticket_id,
                            "subject": ticket.get("subject", ""),
                            "question": question,
                            "answer": answer,
                        }
                    )

                if len(results) < _SEARCH_PER_PAGE:
                    break
                page += 1

            # Empty month window — keep walking back; busy months may fill
            # all 10 pages and still have more rows (handled by next months).

        return extracted, stats

    # ------------------------------------------------------------------
    # Solutions (Knowledge Base) API — categories / folders / articles.
    # These are Freshdesk's curated help-desk articles, distinct from tickets.
    # ------------------------------------------------------------------

    def _get_json(self, url: str, *, timeout: int = 20) -> Any:
        response = requests.get(
            url, auth=self.auth, headers=self.headers, timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    def probe_solutions(self) -> dict[str, Any]:
        """
        Cheap read-only probe: returns category count and total published
        article count without downloading article bodies. Useful for admins
        to decide whether investing in KB ingest is worthwhile.
        """
        if not self.verify_connection():
            return {"reachable": False, "categories": 0, "articles": 0}

        try:
            categories = self._get_json(f"{self.base_url}/solutions/categories")
        except requests.exceptions.RequestException as exc:
            logger.warning("Freshdesk Solutions probe failed: %s", exc)
            return {"reachable": True, "categories": 0, "articles": 0, "error": str(exc)}

        cat_count = len(categories or [])
        article_total = 0
        folder_total = 0
        for cat in categories or []:
            try:
                folders = self._get_json(
                    f"{self.base_url}/solutions/categories/{cat['id']}/folders"
                )
            except requests.exceptions.RequestException:
                continue
            folder_total += len(folders or [])
            for folder in folders or []:
                article_total += int(folder.get("articles_count") or 0)
                time.sleep(0.05)
            time.sleep(0.1)

        return {
            "reachable": True,
            "categories": cat_count,
            "folders": folder_total,
            "articles": article_total,
        }

    def iter_solution_articles(
        self, *, max_articles: int = 500
    ) -> Iterator[dict[str, Any]]:
        """
        Yield every published article, walking categories → folders → articles.
        Rate-limited with a small sleep between calls.
        """
        if not self.verify_connection():
            return

        try:
            categories = self._get_json(f"{self.base_url}/solutions/categories")
        except requests.exceptions.RequestException as exc:
            logger.error("Freshdesk Solutions categories fetch failed: %s", exc)
            return

        yielded = 0
        for cat in categories or []:
            cat_id = cat.get("id")
            cat_name = str(cat.get("name") or "").strip()
            if not cat_id:
                continue
            try:
                folders = self._get_json(
                    f"{self.base_url}/solutions/categories/{cat_id}/folders"
                )
            except requests.exceptions.RequestException as exc:
                logger.warning("Freshdesk folders fetch failed (%s): %s", cat_id, exc)
                continue

            for folder in folders or []:
                folder_id = folder.get("id")
                folder_name = str(folder.get("name") or "").strip()
                if not folder_id:
                    continue
                try:
                    articles = self._get_json(
                        f"{self.base_url}/solutions/folders/{folder_id}/articles"
                    )
                except requests.exceptions.RequestException as exc:
                    logger.warning(
                        "Freshdesk articles fetch failed (%s): %s", folder_id, exc
                    )
                    continue

                for article in articles or []:
                    # Skip drafts (status != 2 == published in Freshdesk).
                    if int(article.get("status") or 0) != 2:
                        continue
                    yield {
                        "article_id": article.get("id"),
                        "category": cat_name,
                        "folder": folder_name,
                        "title": str(article.get("title") or "").strip(),
                        "description_text": (
                            str(article.get("description_text") or "").strip()
                            or strip_html(article.get("description"))
                        ),
                        "tags": list(article.get("tags") or []),
                        "updated_at": article.get("updated_at"),
                    }
                    yielded += 1
                    if yielded >= max_articles:
                        return
                    time.sleep(0.1)
                time.sleep(0.1)


def sync_freshdesk_solutions(*, max_articles: int = 500) -> dict[str, Any]:
    """
    Pull all published Freshdesk KB articles to ``data/freshdesk_solutions.json``.

    Never overwrites the file with an empty payload so a transient auth failure
    doesn't wipe the existing knowledge base.
    """
    etl = FreshdeskETL()
    articles: list[dict[str, Any]] = []
    try:
        for article in etl.iter_solution_articles(max_articles=max_articles):
            if not article.get("description_text"):
                continue
            articles.append(article)
    except requests.exceptions.RequestException as exc:
        logger.error("Freshdesk Solutions sync failed: %s", exc)
        return {
            "ok": False,
            "article_count": 0,
            "output_path": str(_SOLUTIONS_PATH),
            "message": f"Freshdesk API error: {exc}",
        }

    if not articles:
        return {
            "ok": False,
            "article_count": 0,
            "output_path": str(_SOLUTIONS_PATH),
            "message": (
                "No published KB articles found. Either the Freshdesk account "
                "has no Solutions content, or the API key lacks permission."
            ),
        }

    _SOLUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SOLUTIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(articles, handle, ensure_ascii=False, indent=2)

    return {
        "ok": True,
        "article_count": len(articles),
        "output_path": str(_SOLUTIONS_PATH),
        "domain": etl.domain,
        "message": f"Saved {len(articles)} Freshdesk KB articles.",
    }


def probe_freshdesk_solutions() -> dict[str, Any]:
    etl = FreshdeskETL()
    return etl.probe_solutions()


def sync_freshdesk_knowledge(
    *,
    max_pages: int = _DEFAULT_MAX_SEARCH_PAGES,
    months_back: int = _DEFAULT_MONTHS_BACK,
) -> dict[str, Any]:
    """
    Pull Freshdesk Q&A via Search (Resolved + Closed only) and write
    ``data/freshdesk_tickets.json``.

    Returns a stats dict. Does not overwrite the JSON file when zero rows
    are extracted (keeps the previous snapshot).
    """
    etl = FreshdeskETL()
    extracted, fetch_stats = etl.fetch_resolved_tickets(
        months_back=months_back,
        max_pages=max_pages,
    )
    out_path = _OUTPUT_PATH

    if not extracted:
        return {
            "ok": False,
            "ticket_count": 0,
            "output_path": str(out_path),
            "domain": etl.domain,
            **fetch_stats,
            "message": (
                "No Q&A rows extracted. Check FRESHDESK_DOMAIN/API key, "
                "or no resolved tickets with public agent replies in the "
                f"last {months_back} month(s)."
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(extracted, handle, ensure_ascii=False, indent=4)

    logger.info(
        "Freshdesk sync saved %s Q&A rows (scanned %s resolved) to %s",
        len(extracted),
        fetch_stats.get("resolved_scanned"),
        out_path,
    )
    return {
        "ok": True,
        "ticket_count": len(extracted),
        "output_path": str(out_path),
        "domain": etl.domain,
        **fetch_stats,
        "message": f"Saved {len(extracted)} Freshdesk Q&A entries.",
    }
