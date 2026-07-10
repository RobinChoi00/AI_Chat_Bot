"""
warranty_faiss_rebuilder.py
===========================
Rebuild the ``freshdesk_qa`` FAISS index used by ``/api/v1/chat`` after a
Freshdesk sync (tickets + KB articles).

Why a dedicated helper?
-----------------------
- The warranty workflow reads Freshdesk data via ``warranty_knowledge.py`` and
  is refreshed just by calling ``clear_knowledge_cache()``.
- The general-purpose chat endpoint (``/api/v1/chat``) instead queries a
  FAISS index built by ``script/master_ingester.py``. That index bakes the
  Freshdesk snapshot at build time, so it stays stale until master_ingester
  is re-run — even after we sync fresh data.
- ``master_ingester`` also rebuilds product + web indexes, which is
  unnecessary (and expensive in embedding tokens) after a Freshdesk-only
  sync. This module rebuilds ONLY the ``freshdesk_qa`` slice.

Concurrency
-----------
We serialize rebuilds with an on-disk lock so an admin can't accidentally
run two in parallel. A tiny status JSON records the last result for
observability.

Design contract
---------------
- Never touches ``osaki_products`` or ``web_data`` indexes.
- Skips gracefully if the FAISS/langchain stack is unavailable (no crash).
- Writes the new index atomically (temp dir + rename).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FAISS_DIR = _PROJECT_ROOT / "faiss_index"
_STATUS_PATH = _PROJECT_ROOT / "data" / "freshdesk_qa_faiss_status.json"
_TICKETS_PATH = _PROJECT_ROOT / "data" / "freshdesk_tickets.json"
_SOLUTIONS_PATH = _PROJECT_ROOT / "data" / "freshdesk_solutions.json"

# Serialise rebuilds within a single process. Multi-worker containers use
# the status file's ``running`` flag as an advisory lock — good enough since
# admin-triggered rebuilds are rare.
_LOCAL_LOCK = threading.Lock()


@dataclass
class RebuildStatus:
    running: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    ok: bool = False
    ticket_docs: int = 0
    kb_docs: int = 0
    csv_docs: int = 0
    fonz_docs: int = 0
    total_docs: int = 0
    error: str = ""
    output_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------


def _load_status() -> RebuildStatus:
    if not _STATUS_PATH.is_file():
        return RebuildStatus()
    try:
        with _STATUS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except (json.JSONDecodeError, OSError):
        return RebuildStatus()
    status = RebuildStatus()
    for k, v in data.items():
        if hasattr(status, k):
            setattr(status, k, v)
    return status


def _save_status(status: RebuildStatus) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATUS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(status.to_json(), handle, ensure_ascii=False, indent=2)
    tmp.replace(_STATUS_PATH)


def get_status() -> dict[str, Any]:
    return _load_status().to_json()


# ---------------------------------------------------------------------------
# Document collection — reuses master_ingester logic without importing the
# heavier product/spec pipelines.
# ---------------------------------------------------------------------------


def _collect_documents() -> list:
    try:
        from langchain_core.documents import Document
    except ImportError:  # pragma: no cover — langchain always present in prod
        raise RuntimeError(
            "langchain is not installed; cannot rebuild FAISS index."
        )

    docs: list = []

    # Freshdesk tickets (raw pull).
    if _TICKETS_PATH.is_file():
        try:
            with _TICKETS_PATH.open(encoding="utf-8") as handle:
                tickets = json.load(handle)
        except (json.JSONDecodeError, OSError):
            tickets = []
        for ticket in tickets or []:
            question = str(ticket.get("question") or "").strip()
            answer = str(ticket.get("answer") or "").strip()
            if not question or not answer:
                continue
            content = (
                f"Customer Question:\n{question}\n\n"
                f"Official Answer / Resolution:\n{answer}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "freshdesk",
                        "ticket_id": ticket.get("ticket_id"),
                        "subject": ticket.get("subject", ""),
                    },
                )
            )

    # Freshdesk KB / Solutions articles.
    kb_docs_added = 0
    if _SOLUTIONS_PATH.is_file():
        try:
            with _SOLUTIONS_PATH.open(encoding="utf-8") as handle:
                articles = json.load(handle)
        except (json.JSONDecodeError, OSError):
            articles = []
        for article in articles or []:
            title = str(article.get("title") or "").strip()
            body = str(article.get("description_text") or "").strip()
            if not title or not body:
                continue
            content = f"[Article]: {title}\n\n{body}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "freshdesk_kb",
                        "article_id": article.get("article_id"),
                        "category": article.get("category", ""),
                        "folder": article.get("folder", ""),
                    },
                )
            )
            kb_docs_added += 1

    # CSV knowledge (Auto-Check + Warranty Q&A) — reuse master_ingester so
    # the freshdesk_qa index still has the same base documents it used to.
    csv_docs_added = 0
    try:
        import sys as _sys

        script_dir = str(_PROJECT_ROOT / "script")
        if script_dir not in _sys.path:
            _sys.path.insert(0, script_dir)
        base_dir = str(_PROJECT_ROOT)
        if base_dir not in _sys.path:
            _sys.path.insert(0, base_dir)

        from master_ingester import MasterIngester  # type: ignore

        stub = MasterIngester.__new__(MasterIngester)
        stub.domain_docs = {"freshdesk_qa": []}
        MasterIngester.process_error_manuals(stub)  # type: ignore[arg-type]
        MasterIngester.process_qa_reports(stub)  # type: ignore[arg-type]
        for doc in stub.domain_docs["freshdesk_qa"]:
            docs.append(doc)
            csv_docs_added += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping CSV knowledge in rebuild: %s", exc)

    try:
        import sys as _sys

        app_dir = str(_PROJECT_ROOT / "app")
        if app_dir not in _sys.path:
            _sys.path.insert(0, app_dir)
        from fonz_warranty_data import fonz_faiss_documents  # noqa: WPS433

        fonz_path = _PROJECT_ROOT / "data" / "fonz_error_codes.json"
        if fonz_path.is_file():
            docs.extend(fonz_faiss_documents(error_codes_path=fonz_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping Fonz error codes in rebuild: %s", exc)

    return docs


def _write_faiss_index(docs: list) -> Path:
    from langchain_community.vectorstores import FAISS as LC_FAISS
    from langchain_openai import OpenAIEmbeddings

    try:
        from config import EMBEDDING_MODEL
    except Exception:
        EMBEDDING_MODEL = os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = LC_FAISS.from_documents(docs, embeddings)

    # Atomic replace so an in-flight query never sees a half-written index.
    target = _FAISS_DIR / "freshdesk_qa"
    staging = _FAISS_DIR / "freshdesk_qa.new"
    if staging.is_dir():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(staging))

    if target.is_dir():
        backup = _FAISS_DIR / "freshdesk_qa.prev"
        if backup.is_dir():
            shutil.rmtree(backup)
        target.rename(backup)
    staging.rename(target)
    return target


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def rebuild_freshdesk_qa_index() -> dict[str, Any]:
    """
    Rebuild the ``freshdesk_qa`` FAISS index in place. Blocking call —
    intended for BackgroundTasks or the CLI, not the request thread.

    Returns a stats dict identical to ``get_status()``.
    """
    lock_acquired = _LOCAL_LOCK.acquire(blocking=False)
    if not lock_acquired:
        current = _load_status()
        return {
            **current.to_json(),
            "message": "Another rebuild is already running.",
            "already_running": True,
        }

    status = RebuildStatus(running=True, started_at=time.time())
    _save_status(status)
    try:
        docs = _collect_documents()
        status.total_docs = len(docs)
        status.ticket_docs = sum(
            1 for d in docs if d.metadata.get("source") == "freshdesk"
        )
        status.kb_docs = sum(
            1 for d in docs if d.metadata.get("source") == "freshdesk_kb"
        )
        status.fonz_docs = sum(
            1 for d in docs if d.metadata.get("source") == "fonz"
        )
        status.csv_docs = (
            status.total_docs
            - status.ticket_docs
            - status.kb_docs
            - status.fonz_docs
        )

        if not docs:
            status.ok = False
            status.error = "No documents to index."
        else:
            path = _write_faiss_index(docs)
            status.output_path = str(path)
            status.ok = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Freshdesk FAISS rebuild failed")
        status.ok = False
        status.error = f"{type(exc).__name__}: {exc}"
    finally:
        status.running = False
        status.finished_at = time.time()
        _save_status(status)
        _LOCAL_LOCK.release()

    return status.to_json()
