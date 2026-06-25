"""Tests for warranty terminal message enrichment."""

from __future__ import annotations

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


def test_install_terminal_includes_video_and_help_offer():
    node = {
        "node_id": "install_send_video",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Here is your installation guide.",
        "evidence_required": [],
    }
    result = build_terminal_enrichment(_EngineInstall(), _TicketInstall(), node)
    assert result is not None
    assert "Watch —" in result["message"]
    assert "footrest and base" in result["message"].lower()
    assert result["defer_email"] is True
    assert result["phase"] == "awaiting_help_consent"
    assert len(result["help_offer_options"]) == 2


def test_install_air_hose_terminal_includes_diy_steps_and_video():
    node = {
        "node_id": "install_air_hose_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Footrest or air installation help.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("installation"),
                _Turn("model_name", "OS-4000T"),
                _Turn("footrest_or_no_air"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketInstall(), node)
    assert result is not None
    assert "footrest-to-base air hose" in result["message"].lower()
    assert "What you can try" in result["message"]
    assert "Watch —" in result["message"]
    assert result["diagnosis"]["steps"]
    assert result["phase"] == "awaiting_help_consent"


def test_defect_terminal_diagnosis_and_help_offer():
    node = {
        "node_id": "defect_power_main_pcb_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review and arrange the appropriate PCB repair or replacement.",
        "evidence_required": ["video_of_issue"],
    }
    result = build_terminal_enrichment(_EngineDefect(), _TicketDefect(), node)
    assert result is not None
    assert result["phase"] == "awaiting_help_consent"
    assert result["defer_email"] is True
    assert "PCB repair or replacement" not in result["message"]
    assert "Would you like our warranty team" in result["message"]
    assert "What you can try" in result["message"]
    assert result["diagnosis"]["steps"]
