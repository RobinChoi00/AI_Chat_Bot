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
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "freshdesk_tickets.json"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MERGED_ANSWER_RE = re.compile(
    r"^(this ticket is closed|merged into ticket|expedite shipping)",
    re.I,
)


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

    def fetch_resolved_tickets(self, max_pages: int = 5) -> list[dict]:
        if not self.verify_connection():
            return []

        tickets: list[dict] = []
        page = 1

        while page <= max_pages:
            url = f"{self.base_url}/tickets"
            params = {
                "updated_since": "2023-01-01T00:00:00Z",
                "include": "description",
                "page": page,
                "per_page": 30,
            }

            try:
                response = requests.get(
                    url, auth=self.auth, headers=self.headers, params=params, timeout=30
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                logger.error("Freshdesk fetch failed on page %s: %s", page, exc)
                break

            results = response.json()
            if not results:
                break

            for ticket in results:
                if ticket.get("status") not in (4, 5):
                    continue
                ticket_id = ticket["id"]
                question = ticket_question(ticket)
                answer = self.fetch_conversations(ticket_id)
                time.sleep(0.3)

                if is_usable_qa_pair(question, answer):
                    tickets.append(
                        {
                            "ticket_id": ticket_id,
                            "subject": ticket.get("subject", ""),
                            "question": question,
                            "answer": answer,
                        }
                    )

            page += 1

        return tickets


def sync_freshdesk_knowledge(*, max_pages: int = 5) -> dict[str, Any]:
    """
    Pull Freshdesk Q&A and write data/freshdesk_tickets.json.

    Returns a stats dict. Does not overwrite the JSON file when zero rows
    are extracted (keeps the previous snapshot).
    """
    etl = FreshdeskETL()
    extracted = etl.fetch_resolved_tickets(max_pages=max_pages)
    out_path = _OUTPUT_PATH

    if not extracted:
        return {
            "ok": False,
            "ticket_count": 0,
            "output_path": str(out_path),
            "domain": etl.domain,
            "message": (
                "No Q&A rows extracted. Check FRESHDESK_DOMAIN/API key, "
                "or no resolved tickets with public agent replies."
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(extracted, handle, ensure_ascii=False, indent=4)

    logger.info("Freshdesk sync saved %s Q&A rows to %s", len(extracted), out_path)
    return {
        "ok": True,
        "ticket_count": len(extracted),
        "output_path": str(out_path),
        "domain": etl.domain,
        "message": f"Saved {len(extracted)} Freshdesk Q&A entries.",
    }
