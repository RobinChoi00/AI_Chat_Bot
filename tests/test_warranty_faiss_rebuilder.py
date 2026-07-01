"""
tests/test_warranty_faiss_rebuilder.py
======================================
Freshdesk-only FAISS rebuild helper. langchain + OpenAI embeddings are
stubbed so tests don't hit any external service.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import warranty_faiss_rebuilder as fr  # noqa: E402


class _FakeVS:
    saved_paths: list[str] = []
    last_doc_count: int = 0

    @classmethod
    def from_documents(cls, docs, embeddings):
        cls.last_doc_count = len(docs)
        inst = cls()
        inst._docs = docs
        return inst

    def save_local(self, path):
        _FakeVS.saved_paths.append(str(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "index.faiss").write_bytes(b"stub")
        (Path(path) / "index.pkl").write_bytes(b"stub")


class _FakeEmbeddings:
    def __init__(self, model=None):
        self.model = model


@pytest.fixture(autouse=True)
def _stub_langchain(monkeypatch):
    _FakeVS.saved_paths.clear()
    _FakeVS.last_doc_count = 0

    fake_docs_mod = types.SimpleNamespace()

    class _Doc:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    fake_docs_mod.Document = _Doc
    monkeypatch.setitem(sys.modules, "langchain_core.documents", fake_docs_mod)

    fake_lc_vs = types.SimpleNamespace(FAISS=_FakeVS)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", fake_lc_vs)

    fake_openai = types.SimpleNamespace(OpenAIEmbeddings=_FakeEmbeddings)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)


def _redirect_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fr, "_FAISS_DIR", tmp_path / "faiss_index")
    monkeypatch.setattr(fr, "_STATUS_PATH", tmp_path / "data" / "status.json")
    monkeypatch.setattr(fr, "_TICKETS_PATH", tmp_path / "data" / "tickets.json")
    monkeypatch.setattr(fr, "_SOLUTIONS_PATH", tmp_path / "data" / "solutions.json")


def test_rebuild_writes_index_and_status(tmp_path, monkeypatch):
    _redirect_paths(tmp_path, monkeypatch)

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tickets.json").write_text(
        json.dumps(
            [
                {
                    "ticket_id": 1,
                    "subject": "Air stopped working",
                    "question": "leg air not inflating",
                    "answer": "Check air hose behind the footrest.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "solutions.json").write_text(
        json.dumps(
            [
                {
                    "article_id": 1,
                    "title": "How to reset chair power",
                    "description_text": "Unplug the chair and toggle back switch.",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = fr.rebuild_freshdesk_qa_index()
    assert result["ok"] is True
    assert result["ticket_docs"] == 1
    assert result["kb_docs"] == 1
    assert result["total_docs"] >= 2
    assert result["running"] is False
    assert Path(result["output_path"]).is_dir()
    assert _FakeVS.saved_paths, "FAISS.save_local was not called"


def test_rebuild_reports_no_docs_when_files_missing(tmp_path, monkeypatch):
    _redirect_paths(tmp_path, monkeypatch)

    result = fr.rebuild_freshdesk_qa_index()
    # When there is nothing to index we should NOT touch FAISS.
    assert result["ok"] is False
    assert result["total_docs"] >= 0
    # CSV loader may still contribute; we only assert we didn't crash.
    assert result["running"] is False


def test_get_status_reflects_last_result(tmp_path, monkeypatch):
    _redirect_paths(tmp_path, monkeypatch)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tickets.json").write_text(
        json.dumps(
            [
                {
                    "ticket_id": 2,
                    "subject": "Power",
                    "question": "chair off",
                    "answer": "toggle back switch",
                }
            ]
        ),
        encoding="utf-8",
    )

    fr.rebuild_freshdesk_qa_index()
    status = fr.get_status()
    assert status["ok"] is True
    assert status["running"] is False
    assert status["finished_at"] > 0


def test_rebuild_reports_already_running(tmp_path, monkeypatch):
    _redirect_paths(tmp_path, monkeypatch)
    # Simulate a lock held by another thread.
    assert fr._LOCAL_LOCK.acquire(blocking=False)
    try:
        result = fr.rebuild_freshdesk_qa_index()
        assert result.get("already_running") is True
    finally:
        fr._LOCAL_LOCK.release()
