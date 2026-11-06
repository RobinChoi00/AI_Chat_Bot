"""Deterministic warranty common-question answers."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from warranty_common_questions import (  # noqa: E402
    looks_like_factual_common_question,
    try_answer_common_question,
    try_confusion_common_question,
    try_factual_common_question,
)
from warranty_side_questions import (  # noqa: E402
    _looks_like_valid_workflow_answer,
    try_answer_side_question,
)


_MENU = {
    "node_id": "issue_type",
    "type": "question",
    "prompt": "Which area do you need help with: setup, delivery, or a chair malfunction?",
    "options": [
        {"answer_key": "installation", "label": "Installation Issue"},
        {"answer_key": "delivery", "label": "Delivery Issue"},
        {"answer_key": "defect", "label": "Defect / Malfunction"},
    ],
}

_INTENT = {
    "node_id": "delivery_intent_q",
    "type": "question",
    "prompt": "How can we help with your delivery today?",
    "options": [
        {"answer_key": "status_check", "label": "Check delivery / tracking status"},
        {"answer_key": "damage_issue", "label": "Damage or a problem with the delivery"},
    ],
}

_INSTRUCTION = {
    "node_id": "defect_power_recline_move_q",
    "type": "instruction",
    "prompt": "When you turn the chair OFF, does the non-functioning recline part move back?",
    "options": [
        {
            "answer_key": "moves_on_off",
            "label": "Yes, it moves back to its default position when powered off",
        },
        {"answer_key": "stays_stuck", "label": "No, it stays stuck in its current position"},
    ],
}

_HEATING = {
    "node_id": "defect_heating_which",
    "type": "question",
    "prompt": "Which heating issue best describes your situation?",
    "options": [
        {"answer_key": "not_heating", "label": "Heating does not warm up at all"},
        {"answer_key": "too_hot", "label": "Heat feels too hot or uneven"},
    ],
}

_OTHER = {
    "node_id": "delivery_other_problem_q",
    "type": "question_text",
    "prompt": "Please briefly describe the delivery problem so our team can help.",
    "answer_key": "other_delivery_problem_description",
}


class _Ticket:
    domain = "osakiusa.com"

    def get_collected(self):
        return {}


@pytest.mark.parametrize("text", [
    "how long is the warranty?",
    "is this covered under warranty?",
    "will you replace my chair?",
    "I want a refund",
    "can I return it?",
    "how much does a repair cost?",
    "do I need to pay for shipping parts?",
    "when will a technician come?",
    "what is your phone number?",
    "where do I find the model number?",
    "can you just send me a new remote?",
    "how long will this take?",
    "what's the status of my case?",
])
def test_factual_questions_are_answered_without_generic_fallback(text):
    msg = try_answer_side_question(
        node=_INTENT,
        answer=text,
        model_name="OS-4000T",
        issue_type="delivery",
        turns=[],
        ticket=_Ticket(),
    )
    assert msg is not None
    assert "don't have a specific answer" not in msg
    assert "wasn't fully sure" not in msg
    assert "How can we help with your delivery today?" in msg


def test_warranty_length_cites_published_page_and_does_not_approve():
    msg = try_factual_common_question(
        _MENU, "how long is the warranty?", ticket=_Ticket(),
    )
    assert msg is not None
    assert "osakiusa.com/pages/warranty" in msg
    assert "cannot confirm or approve" in msg
    assert "888-848-2630" in msg


def test_replace_and_send_part_are_not_approved():
    for text in ("will you replace my chair?", "can you just send me a new remote?"):
        msg = try_factual_common_question(_MENU, text, ticket=_Ticket())
        assert msg is not None
        assert "cannot approve" in msg
        assert "888-848-2630" in msg


def test_refund_points_to_sales_and_published_return_policy():
    msg = try_factual_common_question(_MENU, "I want a refund", ticket=_Ticket())
    assert msg is not None
    assert "ext. 2" in msg
    assert "pages/sales-policy" in msg
    assert "cannot start a return" in msg


def test_technician_timing_does_not_schedule():
    msg = try_factual_common_question(_INTENT, "when will a technician come?", ticket=_Ticket())
    assert msg is not None
    assert "Nothing has been scheduled" in msg
    assert not looks_like_factual_common_question("when will my chair arrive")


def test_instruction_option_label_is_a_workflow_answer():
    label = "Yes, it moves back to its default position when powered off"
    assert _looks_like_valid_workflow_answer(_INSTRUCTION, label) is True
    assert try_answer_side_question(
        node=_INSTRUCTION,
        answer=label,
        model_name="OS-4000T",
        issue_type="defect",
        turns=[],
    ) is None


def test_heat_label_is_a_workflow_answer():
    label = "Heat feels too hot or uneven"
    assert _looks_like_valid_workflow_answer(_HEATING, label) is True
    assert try_answer_side_question(
        node=_HEATING,
        answer=label,
        model_name="OS-4000T",
        issue_type="defect",
        turns=[],
    ) is None


def test_confusion_at_free_text_node_does_not_look_like_an_answer():
    msg = try_confusion_common_question(_OTHER, "I don't understand")
    assert msg is not None
    assert "briefly describe the delivery problem" in msg
    assert try_answer_side_question(
        node=_OTHER,
        answer="I don't understand",
        issue_type="delivery",
        turns=[],
    ) is not None


def test_korean_at_menu_stays_on_the_question():
    msg = try_answer_common_question(_MENU, "의자가 안 켜져요")
    assert msg is not None
    assert "영어" in msg
    assert "888-848-2630" in msg


def test_korean_description_of_a_delivery_problem_is_left_to_submit():
    assert try_confusion_common_question(_OTHER, "의자가 안 켜져요") is None
    assert try_answer_side_question(
        node=_OTHER,
        answer="의자가 안 켜져요",
        issue_type="delivery",
        turns=[],
    ) is None


def test_already_tried_is_acknowledged():
    msg = try_answer_side_question(
        node=_MENU,
        answer="I already tried that",
        issue_type="defect",
        turns=[],
    )
    assert msg is not None
    assert "already tried" in msg.lower()
    assert "wasn't fully sure" not in msg


def test_send_part_at_problem_type_is_not_treated_as_remote_option():
    node = {
        "node_id": "defect_problem_type",
        "type": "question",
        "prompt": "What type of problem are you experiencing?",
        "options": [
            {"answer_key": "remote", "label": "Remote / controller issue"},
            {"answer_key": "power", "label": "Power issue"},
        ],
    }
    assert _looks_like_valid_workflow_answer(node, "can you just send me a new remote?") is False
    msg = try_answer_side_question(
        node=node,
        answer="can you just send me a new remote?",
        issue_type="defect",
        turns=[],
    )
    assert msg is not None
    assert "cannot approve" in msg
    assert "Remote / controller" not in msg or "cannot approve" in msg
