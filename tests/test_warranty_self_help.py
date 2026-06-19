"""Tests for install video lookup and defect self-help enrichment."""

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


def test_lookup_install_video_series():
    result = lookup_install_video("OS-4000T")
    assert result["url"]
    assert result["match"] in {"series", "model", "default"}


def test_find_defect_self_help_power():
    text = find_defect_self_help(
        defect_category="power",
        path_text="fuse blown chair won't turn on clicking remote",
        model_name="OS-4000T",
    )
    assert text is not None
    assert "try" in text.lower() or "check" in text.lower()


def test_infer_defect_category_from_turns():
    turns = [
        _Turn(answer_key="defect"),
        _Turn(answer_key="power"),
    ]
    assert infer_defect_category_from_turns(turns) == "power"
