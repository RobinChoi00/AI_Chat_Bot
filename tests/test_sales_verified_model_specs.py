"""Deidentified regressions for narrow questions from negatively rated answers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sales_catalog import load_product_index  # noqa: E402
from sales_verified_model_specs import verified_model_answer  # noqa: E402


def model(handle):
    return next(product for product in load_product_index() if product.handle == handle)


def answer(handle, question):
    return verified_model_answer(model(handle), question)


def test_hamilton_answers_box_count_and_published_box_dimensions():
    handle = "osaki-os-3d-hamilton-le"
    count = answer(handle, "How many shipping boxes are there for the Hamilton LE?")
    assert "**2 shipping boxes**" in count
    assert "70.5 lb" not in count
    assert "https://osakimassagechair.com/products/osaki-os-3d-hamilton-le" in count
    detail = answer(handle, "What are the carton dimensions and package weight?")
    assert "**Box 1:** 30.0 × 60.0 × 37.6 in (W × L × H), 269.0 lb" in detail
    assert "**Box 2:** 55.9 × 28.0 × 15.0 in (W × L × H), 70.5 lb" in detail


def test_bravo_flex_explains_both_features_without_leaking_to_standard_bravo():
    question = "What is walking technology and what are the convenient side panels?"
    detail = answer("osaki-bravo-duo-flex-chair", question)
    assert "walking-like pattern" in detail
    assert "while you stay seated" in detail
    assert "adjust the chair position or start a massage program" in detail
    assert detail.count("https://") == 1
    assert answer("osaki-4d-bravo-duo", question) is None
    assert answer("osaki-bravo-duo-flex-chair", "How to remove the side panels?") is None


def test_cura_roller_motor_is_unverified_not_inferred_from_4d_mechanism():
    detail = answer("titan-pro-cura-4d", "Does its roller use a brushless motor?")
    assert "can't verify" in detail
    assert "doesn't specify the motor type" in detail
    assert "https://titanchair.com/products/titan-pro-cura-4d" in detail


@pytest.mark.parametrize("question", ["Does it have casters?", "Does the Vito have wheels for moving?"])
def test_vito_does_not_confuse_transport_wheels_with_massage_rollers(question):
    detail = answer("os-3d-ai-vito-massage-chair", question)
    assert "can't confirm" in detail
    assert "not transport" in detail
    assert "dolly" in detail


def test_orion_describes_exact_touchscreen_controls_with_limits():
    detail = answer("osaki-4d-3d-orion-duo-mech", "Can I adjust the speed and pressure on this chair?")
    assert "**6 levels of 4D roller depth**" in detail
    assert "**5 levels each for Kneading, Rapping (tapping), and Airbags**" in detail
    assert "doesn't give a pressure measurement" in detail
    assert "Manufacturer instructions (p. 22)" in detail
    assert answer("osaki-manhattan-duo-massage-chair", "Can I adjust speed and pressure?") is None


@pytest.mark.parametrize("question", [
    "What is the minimum user height?", "What is the min person heighth?", "What's the recommended height range?",
])
def test_orion_does_not_substitute_the_standard_line_height_range(question):
    detail = answer("osaki-4d-3d-orion-duo-mech", question)
    assert "4'9\" to 6'4\"" in detail
    assert "Manufacturer instructions (p. 32)" in detail
    assert "5'0\"" not in detail


@pytest.mark.parametrize("height", ["6 ft 2", "6ft2", "6'2\""])
def test_orion_measured_fit_reports_both_limits_without_promising_comfort(height):
    detail = answer("osaki-4d-3d-orion-duo-mech", f"Will the Orion Duo 4D and 3D fit me at {height} and 220 lb?")
    assert "**6'2\" and 220 lb** is within" in detail
    assert "**280 lb maximum user weight**" in detail
    assert "does not guarantee comfort" in detail


def test_orion_out_of_range_does_not_recommend_this_chair():
    detail = answer("osaki-4d-3d-orion-duo-mech", "Will the chair fit a 6 ft 6 person weighing 305 pounds?")
    assert "outside at least one" in detail
    assert "would not recommend" in detail


def test_orion_near_weight_cap_does_not_hide_inconsistent_troubleshooting_figure():
    detail = answer("osaki-4d-3d-orion-duo-mech", "Will this chair fit a user who weighs 270 lb?")
    assert "280 lb maximum user weight" in detail
    assert "verify the limit" in detail
    assert "is within" not in detail


def test_orion_body_fit_does_not_claim_a_delivery_path_or_other_chair_fits():
    assert answer("osaki-4d-3d-orion-duo-mech", "Will I fit this chair through a 6 ft 2 doorway?") is None
    assert answer("osaki-4d-3d-orion-duo-mech", "Compare the Orion and the Vito for a user who is 6 ft 2 and 220 lb") is None
