"""Tests for warranty terminal message enrichment."""

from __future__ import annotations

from warranty_terminal_enrichment import build_terminal_enrichment


class _Ticket:
    ticket_id = "t1"
    issue_type = "installation"
    model_name = "OS-4000T"


class _Engine:
    def get_turns(self, ticket_id: str):
        return []


def test_install_terminal_includes_video_link():
    node = {
        "node_id": "install_send_video",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Here is your installation guide.",
        "evidence_required": [],
    }
    result = build_terminal_enrichment(_Engine(), _Ticket(), node)
    assert result is not None
    assert "Watch installation video" in result["message"]
    assert result["defer_email"] is True
    assert result["show_contact_form"] is False
    assert result["install_video"]["url"]
