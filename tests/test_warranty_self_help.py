"""Tests for defect self-help formatting."""

from __future__ import annotations

from install_videos import lookup_install_video
from warranty_self_help import find_defect_self_help, infer_defect_category_from_turns


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


def test_find_defect_self_help_power_back_switch():
    turns = [
        _Turn("power"),
        _Turn("back_switch_sound", "Turned on the back switch and heard something from the chair"),
    ]
    text = find_defect_self_help(
        defect_category="power",
        path_text="back switch heard something power remote",
        node_id="defect_power_main_pcb_terminal",
        turns=turns,
    )
    assert text is not None
    assert "similar cases" in text.lower()
    assert "power cord" in text.lower() or "fuse" in text.lower()


def test_infer_defect_category_from_turns():
    turns = [
        _Turn(answer_key="defect"),
        _Turn(answer_key="power"),
    ]
    assert infer_defect_category_from_turns(turns) == "power"
