"""Tests for warranty terminal message enrichment."""

from __future__ import annotations

from warranty_self_help import soften_terminal_prompt
from warranty_terminal_enrichment import build_terminal_enrichment


class _Turn:
    def __init__(self, answer_key: str = "", customer_answer: str = "", node_prompt: str = "", node_id: str = ""):
        self.answer_key = answer_key
        self.customer_answer = customer_answer
        self.node_prompt = node_prompt
        self.node_id = node_id


class _TicketInstall:
    ticket_id = "t1"
    issue_type = "installation"
    model_name = "OS-4000T"


class _TicketDefect:
    ticket_id = "t2"
    issue_type = "defect"
    model_name = "OS-4000T"


class _EngineInstall:
    def get_turns(self, ticket_id: str):
        return []


class _EngineDefect:
    def get_turns(self, ticket_id: str):
        return [
            _Turn("warranty"),
            _Turn("defect"),
            _Turn("power"),
            _Turn("remote_on"),
            _Turn("back_switch_sound", "Turned on the back switch and heard something from the chair"),
        ]


def test_install_terminal_includes_video_link():
    node = {
        "node_id": "install_send_video",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Here is your installation guide.",
        "evidence_required": [],
    }
    result = build_terminal_enrichment(_EngineInstall(), _TicketInstall(), node)
    assert result is not None
    assert "Watch installation video" in result["message"]
    assert result["defer_email"] is True
    assert result["show_contact_form"] is False


def test_defect_terminal_self_help_first_and_deferred_email():
    node = {
        "node_id": "defect_power_main_pcb_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review and arrange the appropriate PCB repair or replacement.",
        "evidence_required": ["video_of_issue"],
    }
    result = build_terminal_enrichment(_EngineDefect(), _TicketDefect(), node)
    assert result is not None
    assert result["defer_email"] is True
    assert result["show_contact_form"] is False
    assert "PCB repair or replacement" not in result["message"]
    assert "I still need help" in result["message"]
    assert "leave your email below" not in result["message"].lower()
    assert result["message"].index("Based on similar cases") < result["message"].index("warranty team will review")


def test_soften_terminal_prompt():
    raw = "Our team will arrange a replacement remote for you."
    softened = soften_terminal_prompt(raw)
    assert "replacement remote" not in softened.lower()
    assert "review" in softened.lower()
