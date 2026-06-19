"""
Pull resolved Freshdesk tickets into data/freshdesk_tickets.json for warranty diagnosis.

Requires .env (or environment):
  FRESHDESK_DOMAIN  — subdomain only (osakititan) OR full host (osakititan.freshdesk.com)
  FRESHDESK_API_KEY — Profile Settings → API Key

Run on EC2 (inside Docker):
  docker compose exec backend python script/freshdesk_extractor.py
  docker compose exec backend python script/freshdesk_extractor.py --test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", value)).strip()


def ticket_question(ticket: dict) -> str:
    """List tickets omit description unless include=description is set."""
    text = (ticket.get("description_text") or "").strip()
    if text:
        return text
    return strip_html(ticket.get("description"))


def conversation_agent_answer(conversations: list[dict]) -> str:
    replies: list[str] = []
    for conv in conversations:
        if conv.get("incoming") is not False:
            continue
        body = (conv.get("body_text") or "").strip() or strip_html(conv.get("body"))
        if body:
            replies.append(body)
    return "\n".join(replies)


def normalize_freshdesk_domain(raw: str) -> str:
    """
    Accept several .env formats and return the API host (no scheme/path).

    Examples:
      osakititan              -> osakititan.freshdesk.com
      osakititan.freshdesk.com -> osakititan.freshdesk.com
      https://osakititan.freshdesk.com/ -> osakititan.freshdesk.com
    """
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

    # Custom portal host — API usually still needs *.freshdesk.com (see Freshdesk docs).
    return value


class FreshdeskETL:
    def __init__(self) -> None:
        raw_domain = os.environ.get("FRESHDESK_DOMAIN", "")
        api_key = os.environ.get("FRESHDESK_API_KEY", "")
        if not raw_domain:
            raise EnvironmentError("FRESHDESK_DOMAIN 환경 변수가 설정되지 않았습니다.")
        if not api_key:
            raise EnvironmentError("FRESHDESK_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.raw_domain = raw_domain.strip()
        self.domain = normalize_freshdesk_domain(raw_domain)
        self.api_key = api_key.strip()
        self.base_url = f"https://{self.domain}/api/v2"
        self.auth: tuple[str, str] = (self.api_key, "X")
        self.headers = {"Content-Type": "application/json"}

        logger.info("Freshdesk host: %s (from FRESHDESK_DOMAIN=%r)", self.domain, self.raw_domain)

    def verify_connection(self) -> bool:
        """Quick API probe — helps distinguish wrong domain (404) vs bad key (401)."""
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
            logger.error("🚨 Connection failed: %s", exc)
            return False

        if response.status_code == 200:
            logger.info("✅ Freshdesk API OK — authentication and domain look correct.")
            return True

        if response.status_code == 401:
            logger.error(
                "🚨 401 Unauthorized — API key is invalid or disabled. "
                "Freshdesk → Profile Settings → View API Key."
            )
            return False

        if response.status_code == 404:
            logger.error(
                "🚨 404 Not Found for %s\n"
                "   This usually means FRESHDESK_DOMAIN is wrong.\n"
                "   In Freshdesk Admin → Helpdesk → Helpdesk Settings, find the URL that ends in "
                "**.freshdesk.com** (custom support URLs often do NOT work for API).\n"
                "   Set .env to the subdomain only, e.g. FRESHDESK_DOMAIN=yourcompany\n"
                "   Test manually:\n"
                "   curl -u 'YOUR_API_KEY:X' 'https://YOUR-SUBDOMAIN.freshdesk.com/api/v2/tickets?per_page=1'",
                url,
            )
            return False

        logger.error(
            "🚨 Freshdesk API returned HTTP %s: %s",
            response.status_code,
            (response.text or "")[:300],
        )
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
            logger.info("📄 Fetching Freshdesk Tickets - Page %s...", page)
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
            except requests.exceptions.HTTPError as exc:
                logger.error("🚨 API Request Failed: %s", exc)
                body = getattr(exc.response, "text", "") or ""
                if body:
                    logger.error("Response body: %s", body[:500])
                break
            except requests.exceptions.RequestException as exc:
                logger.error("🚨 API Request Failed: %s", exc)
                break

            results = response.json()
            if not results:
                break

            logger.info(
                "🔍 Page %s: %s tickets returned — filtering resolved Q&A…",
                page,
                len(results),
            )

            resolved_count = 0
            valid_qa_count = 0

            for ticket in results:
                if ticket.get("status") not in (4, 5):
                    continue
                resolved_count += 1
                ticket_id = ticket["id"]
                question = ticket_question(ticket)
                answer = self.fetch_conversations(ticket_id)
                time.sleep(0.3)

                if question.strip() and answer.strip():
                    valid_qa_count += 1
                    tickets.append(
                        {
                            "ticket_id": ticket_id,
                            "subject": ticket.get("subject", ""),
                            "question": question,
                            "answer": answer,
                        }
                    )

            logger.info(
                "📊 Page %s — resolved: %s | valid Q&A: %s",
                page,
                resolved_count,
                valid_qa_count,
            )
            page += 1

        return tickets

    def execute_pipeline(self) -> None:
        logger.info("🚀 Freshdesk Data Extraction 파이프라인 가동을 시작합니다.")
        extracted_data = self.fetch_resolved_tickets(max_pages=5)
        logger.info("✅ 총 %s개의 유의미한 Q&A 세트를 추출했습니다.", len(extracted_data))

        if extracted_data:
            os.makedirs("data", exist_ok=True)
            out_path = os.path.join("data", "freshdesk_tickets.json")
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(extracted_data, handle, ensure_ascii=False, indent=4)
            logger.info("✅ Saved to %s", out_path)
        else:
            logger.warning(
                "⚠️ No Q&A rows saved. Fix domain/API key, or no resolved tickets with agent replies."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Freshdesk tickets to JSON")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only verify FRESHDESK_DOMAIN + API key (no export)",
    )
    args = parser.parse_args()

    etl = FreshdeskETL()
    if args.test:
        raise SystemExit(0 if etl.verify_connection() else 1)
    etl.execute_pipeline()


if __name__ == "__main__":
    main()
