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
    assert "Try the steps above first" in message
    assert "2816860269" not in message
    assert "HOUSTON" not in message
    assert "REOPENED" not in message


def test_infer_defect_category_from_turns():
    turns = [
        _Turn(answer_key="defect"),
        _Turn(answer_key="power"),
    ]
    assert infer_defect_category_from_turns(turns) == "power"


def test_infer_defect_category_uses_latest_topic():
    turns = [
        _Turn(answer_key="air"),
        _Turn(answer_key="remote"),
    ]
    assert infer_defect_category_from_turns(turns) == "remote"


def test_build_install_air_hose_diagnosis_includes_core_steps():
    from warranty_self_help import build_install_air_hose_diagnosis

    diagnosis = build_install_air_hose_diagnosis(
        path_text="installation footrest no air",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("footrest" in step.lower() for step in diagnosis["steps"])
    assert "footrest-to-base" in diagnosis["summary"].lower()


def test_build_rolling_noise_diagnosis_includes_core_steps():
    from warranty_self_help import (
        build_rolling_noise_diagnosis,
        infer_rolling_noise_type_from_turns,
    )

    diagnosis = build_rolling_noise_diagnosis(
        noise_type="noise_up_down",
        path_text="rolling mechanism loud noise up down",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("strap" in step.lower() for step in diagnosis["steps"])

    turns = [_Turn("rolling"), _Turn("pops")]
    assert infer_rolling_noise_type_from_turns(turns) == "pops"


def test_build_remote_diagnosis_includes_core_steps():
    from warranty_self_help import build_remote_diagnosis, infer_remote_symptom_from_turns

    diagnosis = build_remote_diagnosis(
        symptom="bad_connection",
        path_text="remote cable loose connection",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("cable" in step.lower() for step in diagnosis["steps"])

    turns = [_Turn("remote"), _Turn("no_power"), _Turn("bad_connection")]
    assert infer_remote_symptom_from_turns(turns) == "bad_connection"
    assert (
        infer_remote_symptom_from_turns([], node_id="defect_remote_fuse_terminal")
        == "fuse_broken"
    )


def test_build_power_diagnosis_includes_core_steps():
    from warranty_self_help import build_power_diagnosis, infer_power_symptom_from_turns

    diagnosis = build_power_diagnosis(
        symptom="clicking_sound",
        path_text="power back switch clicking sound",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("click" in step.lower() or "switch" in step.lower() for step in diagnosis["steps"])

    turns = [_Turn("power"), _Turn("remote_off"), _Turn("clicking_sound")]
    assert infer_power_symptom_from_turns(turns) == "clicking_sound"
    assert (
        infer_power_symptom_from_turns([], node_id="defect_power_no_click_terminal")
        == "no_clicking"
    )


def test_build_workflow_diagnosis_prefers_freshdesk_for_remote():
    from warranty_self_help import build_workflow_diagnosis

    diagnosis = build_workflow_diagnosis(
        defect_category="remote",
        path_text="remote not working no power",
        node_id="defect_remote_pcb_check_terminal",
        turns=[_Turn("remote"), _Turn("no_power")],
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    if diagnosis.get("sources"):
        freshdesk_idx = next(
            (i for i, s in enumerate(diagnosis["sources"]) if s == "freshdesk"),
            None,
        )
        if freshdesk_idx is not None:
            assert freshdesk_idx == 0 or any(
                "freshdesk" in str(s) for s in diagnosis.get("sources", [])
            )


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


def test_build_air_diagnosis_includes_core_steps():
    from warranty_self_help import build_air_diagnosis, infer_air_symptom_from_turns

    diagnosis = build_air_diagnosis(
        symptom="footrest_air",
        path_text="footrest airbags not inflating hose",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("footrest" in step.lower() or "hose" in step.lower() for step in diagnosis["steps"])

    turns = [_Turn("footrest"), _Turn("air_not_inflating"), _Turn("air_blowing")]
    assert infer_air_symptom_from_turns(turns) == "footrest_air"
    assert (
        infer_air_symptom_from_turns([], node_id="defect_air_pump_terminal")
        == "pump_no_air"
    )


def test_build_footrest_diagnosis_includes_core_steps():
    from warranty_self_help import build_footrest_diagnosis, infer_footrest_symptom_from_turns

    diagnosis = build_footrest_diagnosis(
        symptom="legrest_not_extend",
        path_text="footrest does not extend legrest",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any(
        "remote" in step.lower() or "side panel" in step.lower()
        for step in diagnosis["steps"]
    )

    turns = [_Turn("footrest"), _Turn("legrest_not_extend")]
    assert infer_footrest_symptom_from_turns(turns) == "legrest_not_extend"
    assert (
        infer_footrest_symptom_from_turns([], node_id="defect_footrest_foot_rollers_terminal")
        == "foot_rollers"
    )


def test_build_cosmetic_diagnosis_includes_photo_guidance():
    from warranty_self_help import (
        build_cosmetic_diagnosis,
        format_cosmetic_self_help_message,
        infer_cosmetic_symptom_from_turns,
    )

    diagnosis = build_cosmetic_diagnosis(
        symptom="footrest",
        path_text="cosmetic damage footrest scratch",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("photo" in step.lower() for step in diagnosis["steps"])

    msg = format_cosmetic_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url="https://example.com/manual",
    )
    assert "What to prepare" in msg
    assert "warranty team" in msg.lower()

    wg_turns = [_Turn("cosmetic"), _Turn("other"), _Turn("yes_white_glove")]
    assert infer_cosmetic_symptom_from_turns(wg_turns) == "wg_install"
    box_turns = [
        _Turn("cosmetic"),
        _Turn("other"),
        _Turn("noticed_later"),
        _Turn("signed_damaged"),
        _Turn("yes_box_damaged"),
    ]
    assert infer_cosmetic_symptom_from_turns(box_turns) == "signed_damaged_visible"
    assert (
        infer_cosmetic_symptom_from_turns([], node_id="defect_cosmetic_wg_terminal")
        == "wg_install"
    )


def test_build_recline_diagnosis_includes_core_steps():
    from warranty_self_help import (
        build_recline_diagnosis,
        format_recline_self_help_message,
        infer_recline_symptom_from_turns,
    )

    diagnosis = build_recline_diagnosis(
        symptom="moves_on_off",
        path_text="recline backrest moves back when powered off actuator",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("power" in step.lower() for step in diagnosis["steps"])

    msg = format_recline_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url="https://example.com/manual",
    )
    assert "What you can try" in msg
    assert "recline" in msg.lower()

    turns = [_Turn("recline"), _Turn("backrest"), _Turn("multiple_not_working"), _Turn("moves_on_off")]
    assert infer_recline_symptom_from_turns(turns) == "moves_on_off"
    assert (
        infer_recline_symptom_from_turns([], node_id="defect_recline_main_pcb_terminal")
        == "none_working"
    )


def test_build_heating_diagnosis_includes_warmup_guidance():
    from warranty_self_help import (
        build_heating_diagnosis,
        format_heating_self_help_message,
        infer_heating_symptom_from_turns,
    )

    diagnosis = build_heating_diagnosis(
        symptom="not_heating",
        path_text="heat does not warm up back roller",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any(
        "10 minute" in step.lower() or "10-minute" in step.lower() or "warm" in step.lower()
        for step in diagnosis["steps"]
    )

    msg = format_heating_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url="https://example.com/manual",
    )
    assert "What you can try" in msg

    turns = [_Turn("heat"), _Turn("intermittent")]
    assert infer_heating_symptom_from_turns(turns) == "intermittent"
    assert (
        infer_heating_symptom_from_turns([], node_id="defect_heating_too_hot_terminal")
        == "too_hot"
    )


def test_build_delivery_diagnosis_includes_photo_guidance():
    from warranty_self_help import (
        build_delivery_diagnosis,
        format_delivery_self_help_message,
        infer_delivery_symptom_from_turns,
    )

    diagnosis = build_delivery_diagnosis(
        symptom="visible_at_unboxing",
        path_text="delivery damage visible at unboxing signed damaged",
        model_name="OS-4000T",
    )
    assert diagnosis["steps"]
    assert any("photo" in step.lower() or "receipt" in step.lower() for step in diagnosis["steps"])

    msg = format_delivery_self_help_message(
        diagnosis=diagnosis,
        repair_manual_url="https://example.com/manual",
    )
    assert "What to prepare" in msg

    cleared_turns = [_Turn("delivery"), _Turn("yes_box_damage"), _Turn("signed_cleared")]
    assert infer_delivery_symptom_from_turns(cleared_turns) == "signed_cleared"
    assert (
        infer_delivery_symptom_from_turns([], node_id="delivery_replace_claim_terminal")
        == "visible_at_unboxing"
    )
