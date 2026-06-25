"""Tests for defect self-help / diagnosis formatting."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import install_videos  # noqa: E402
from install_videos import lookup_install_video  # noqa: E402
from warranty_self_help import (  # noqa: E402
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


def test_lookup_install_video_updated_urls():
    install_videos._load_catalog.cache_clear()
    pano = lookup_install_video("4D Panorama")
    assert pano["match"] == "model"
    assert pano["url"] == "https://youtu.be/kjgBL9EilYo"

    pinnacle = lookup_install_video("Pinnacle 5D Duoflex AI")
    assert pinnacle["url"] == "https://youtu.be/UtiWbrLrNHo"

    dreamer = lookup_install_video("Osaki 3D Dreamer V2")
    assert dreamer["url"] == "https://youtu.be/7BmcooshWLs"


def test_lookup_install_video_multi_clip_escape_duo():
    install_videos._load_catalog.cache_clear()
    result = lookup_install_video("Osaki Platinum - Escape Duo 4D")
    assert result["match"] == "model"
    assert len(result["videos"]) == 2
    assert result["videos"][0]["url"] == "https://youtu.be/amqSdBFdDAg"
    assert result["videos"][1]["url"] == "https://youtu.be/87cYF1dvv_E"


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


def test_build_install_air_hose_diagnosis_includes_core_steps():
    from warranty_self_help import build_install_air_hose_diagnosis

    diagnosis = build_install_air_hose_diagnosis(
        path_text="installation footrest no air",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("footrest" in step.lower() for step in diagnosis["steps"])
    assert "footrest-to-base" in diagnosis["summary"].lower()


def test_build_voice_diagnosis_includes_core_steps():
    from warranty_self_help import build_voice_diagnosis

    diagnosis = build_voice_diagnosis(
        symptom="voice_no_response",
        path_text="voice control does not work commands",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("command" in step.lower() for step in diagnosis["steps"])

    ghost = build_voice_diagnosis(
        symptom="false_triggers",
        path_text="voice ghost random tv",
        model_name="OS-4000T",
    )
    assert any("tv" in step.lower() or "unplug" in step.lower() for step in ghost["steps"])
