"""Tests for defect self-help / diagnosis formatting."""

from __future__ import annotations

from install_videos import lookup_install_video
from warranty_self_help import (
    build_workflow_diagnosis,
    format_diagnosis_message,
    infer_defect_category_from_turns,
)


class _Turn:
    def __init__(
        self,
        answer_key: str = "",
        customer_answer: str = "",
        node_prompt: str = "",
        node_id: str = "",
    ):
        self.answer_key = answer_key
        self.customer_answer = customer_answer
        self.node_prompt = node_prompt
        self.node_id = node_id


def test_lookup_install_video_default():
    result = lookup_install_video("")
    assert result["url"]
    assert result["match"] == "default"


def test_build_workflow_diagnosis_power_back_switch():
    turns = [
        _Turn("power"),
        _Turn("back_switch_sound", "Turned on the back switch and heard something from the chair"),
    ]
    diagnosis = build_workflow_diagnosis(
        defect_category="power",
        path_text="back switch heard something power remote",
        node_id="defect_power_main_pcb_terminal",
        turns=turns,
        model_name="OS-4000T",
    )
    message = format_diagnosis_message(diagnosis)
    assert diagnosis["steps"]
    assert "What you can try" in message
    assert "Would you like our warranty team" in message
    assert "2816860269" not in message
    assert "HOUSTON" not in message
    assert "REOPENED" not in message


def test_infer_defect_category_from_turns():
    turns = [
        _Turn(answer_key="defect"),
        _Turn(answer_key="power"),
    ]
    assert infer_defect_category_from_turns(turns) == "power"
