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
    assert result["help_offer_options"][0]["answer_key"] == "no_self_help"
    assert result["interaction_mode"] == "troubleshooting"


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


def test_voice_not_working_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_voice_not_working_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Voice help.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("voice"),
                _Turn("voice_no_response"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert "voice" in result["message"].lower()
    assert result["diagnosis"]["steps"]
    assert result["phase"] == "awaiting_help_consent"


def test_rolling_noise_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_rolling_noise_massage_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review this noise issue.",
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("rolling"),
                _Turn("noise_massaging"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert "strap" in result["message"].lower() or "back pad" in result["message"].lower()
    assert "Our team will review this noise issue" not in result["message"]
    assert result["diagnosis"]["steps"]
    assert result["phase"] == "awaiting_help_consent"


def test_remote_connection_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_remote_connection_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Please try unplugging the chair's cable connection and plugging it back in firmly.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("remote"),
                _Turn("no_power"),
                _Turn("bad_connection"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert "cable" in result["message"].lower()
    assert result["diagnosis"]["steps"]
    assert result["phase"] == "awaiting_help_consent"


def test_power_no_click_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_power_no_click_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review and arrange a Power PCB replacement.",
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("power"),
                _Turn("remote_off"),
                _Turn("no_clicking"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert "Power PCB replacement" not in result["message"]
    assert any(
        "fuse" in step.lower() or "outlet" in step.lower() or "switch" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
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
    assert "Try the steps above first" in result["message"]
    assert "What you can try" in result["message"]
    assert result["diagnosis"]["steps"]


def test_air_pump_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_air_pump_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review your case and arrange the necessary service.",
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("air"),
                _Turn("feet_calves"),
                _Turn("never_worked"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert any("hose" in step.lower() for step in result["diagnosis"]["steps"])
    assert "air pump" not in result["message"].lower()
    assert result["phase"] == "awaiting_help_consent"
    assert result["interaction_mode"] == "troubleshooting"


def test_footrest_extend_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_footrest_extend_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review the leg extension issue.",
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("footrest"),
                _Turn("legrest_not_extend"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert any(
        "power cycle" in step.lower() or "side panel" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
    assert result["phase"] == "awaiting_help_consent"
    assert result["interaction_mode"] == "troubleshooting"


def test_cosmetic_wg_terminal_includes_photo_guidance():
    node = {
        "node_id": "defect_cosmetic_wg_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Please send photos of the damage.",
        "evidence_required": ["damage_photos", "box_photos"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("cosmetic"),
                _Turn("other"),
                _Turn("visible_at_unboxing"),
                _Turn("yes_white_glove"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What to prepare" in result["message"]
    assert any("photo" in step.lower() for step in result["diagnosis"]["steps"])
    assert "white glove" in result["message"].lower()
    assert result["phase"] == "awaiting_help_consent"
    assert result["interaction_mode"] == "preparation"


def test_cosmetic_side_fixed_terminal_uses_self_help():
    node = {
        "node_id": "defect_cosmetic_side_fixed_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Please check if the issue is resolved.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("cosmetic"),
                _Turn("side_panel"),
                _Turn("panels_fixed"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "side panel" in result["message"].lower()
    assert result["phase"] == "awaiting_help_consent"


def test_recline_actuator_terminal_includes_diy_steps():
    node = {
        "node_id": "defect_recline_actuator_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will review and arrange an actuator replacement.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("recline"),
                _Turn("backrest"),
                _Turn("multiple_not_working"),
                _Turn("moves_on_off"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert "actuator replacement" not in result["message"]
    assert any(
        "power" in step.lower() or "side panel" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
    assert result["phase"] == "awaiting_help_consent"


def test_recline_main_pcb_terminal_uses_none_working_steps():
    node = {
        "node_id": "defect_recline_main_pcb_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will arrange the Main PCB review.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("recline"),
                _Turn("backrest"),
                _Turn("none_working"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "Main PCB review" not in result["message"]
    assert "What you can try" in result["message"]
    assert any(
        "side panel" in step.lower() or "power cycle" in step.lower()
        for step in result["diagnosis"]["steps"]
    )


def test_heating_not_heating_terminal_includes_warmup_steps():
    node = {
        "node_id": "defect_heating_not_heating_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "Let's confirm heat is enabled.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("heat"),
                _Turn("not_heating"),
                _Turn("will_try_warmup"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" in result["message"]
    assert any(
        "10 minute" in step.lower() or "warm" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
    assert result["phase"] == "awaiting_help_consent"


def test_heating_too_hot_terminal_includes_safety_steps():
    node = {
        "node_id": "defect_heating_too_hot_terminal",
        "type": "terminal",
        "action": "send_info",
        "prompt": "If heat feels too hot, stop using it.",
        "evidence_required": [],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [_Turn("defect"), _Turn("heat"), _Turn("too_hot")]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert any(
        "cool" in step.lower() or "stop using" in step.lower()
        for step in result["diagnosis"]["steps"]
    )


def test_delivery_replace_claim_terminal_includes_diy_prep():
    node = {
        "node_id": "delivery_replace_claim_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Please send photos of both the damaged box and the damage to the chair.",
        "evidence_required": ["damage_photos", "box_photos", "signed_delivery_receipt"],
    }

    class _TicketDelivery:
        ticket_id = "td1"
        issue_type = "delivery"
        model_name = "OS-4000T"

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("delivery"),
                _Turn("yes_box_damage"),
                _Turn("signed_damaged"),
                _Turn("visible_at_unboxing"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDelivery(), node)
    assert result is not None
    assert "What to prepare" in result["message"]
    assert any(
        "receipt" in step.lower() or "photo" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
    assert result["phase"] == "awaiting_help_consent"
    assert result["interaction_mode"] == "preparation"


def test_delivery_signed_cleared_terminal_warns_compensation_difficulty():
    node = {
        "node_id": "delivery_signed_cleared_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Since signed cleared, harder to claim.",
        "evidence_required": ["damage_photos"],
    }

    class _TicketDelivery:
        ticket_id = "td2"
        issue_type = "delivery"
        model_name = "OS-4000T"

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("delivery"),
                _Turn("yes_box_damage"),
                _Turn("signed_cleared"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDelivery(), node)
    assert result is not None
    assert "cleared" in result["message"].lower()
    assert result["diagnosis"]["steps"]


def test_unmapped_rolling_no_movement_terminal_uses_flowchart_prompt_not_generic_diy():
    node = {
        "node_id": "defect_rolling_no_movement_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": (
            "Our team will review and arrange the necessary service for the massage mechanism."
        ),
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("rolling"),
                _Turn("no_movement"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "massage mechanism" in result["message"].lower()
    assert "What you can try" not in result["message"]
    assert "What to prepare" in result["message"]
    assert "video" in result["message"].lower()
    assert result["phase"] == "awaiting_help_consent"


def test_unmapped_rolling_worked_terminal_softens_repair_language():
    node = {
        "node_id": "defect_rolling_worked_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": "Our team will diagnose and arrange the appropriate repair.",
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("rolling"),
                _Turn("worked_before_stopped"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "appropriate repair" not in result["message"].lower()
    assert "review your case" in result["message"].lower()
    assert "What to prepare" in result["message"]
    assert result["diagnosis"]["steps"]


def test_unmapped_rolling_power_no_move_terminal_includes_category_prep_hints():
    node = {
        "node_id": "defect_rolling_power_no_move_terminal",
        "type": "terminal",
        "action": "awaiting_admin",
        "prompt": (
            "Our team will assess and arrange the necessary repair for the massage mechanism."
        ),
        "evidence_required": ["video_of_issue"],
    }

    class _Engine:
        def get_turns(self, ticket_id: str):
            return [
                _Turn("defect"),
                _Turn("rolling"),
                _Turn("power_but_no_move"),
            ]

    result = build_terminal_enrichment(_Engine(), _TicketDefect(), node)
    assert result is not None
    assert "What you can try" not in result["message"]
    assert any(
        "air" in step.lower() or "video" in step.lower()
        for step in result["diagnosis"]["steps"]
    )
