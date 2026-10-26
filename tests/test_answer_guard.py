"""Tests for answer_guard — hallucination blocking."""

import sys
import json
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from answer_guard import sanitize_agent_response

def test_warranty_tool_uses_engine_customer_message_not_llm_rewrite():
    tool = (
        "WARRANTY_CONTINUE\n"
        "TICKET_ID: WR-1\n"
        "PROMPT: When you toggle the back switch, do you hear a click?\n"
        "CUSTOMER_MESSAGE:\n"
        "Based on what you've told us about your **OS-4000T**, here is a quick note.\n\n"
        "**What you can try:**\n"
        "1. Toggle the back power switch OFF, wait 10 seconds, then ON.\n\n"
        "When you toggle the back switch, do you hear a click?\n"
        "INSTRUCTION: Deliver CUSTOMER_MESSAGE to the customer verbatim."
    )
    out = sanitize_agent_response(
        "Sounds like a compressor failure — we'll send a technician tomorrow.",
        tools_called=["answer_warranty_question"],
        user_query="no click",
        tool_results=[tool],
    )
    assert "technician" not in out.lower()
    assert "Toggle the back power switch" in out
    assert out.rstrip().endswith("do you hear a click?")


def test_blocks_price_without_catalog_tool():
    out = sanitize_agent_response(
        "The Osaki Solo Flex is $4,999 today.",
        tools_called=[],
        user_query="how much is solo flex",
    )
    assert "$4,999" not in out
    assert "catalog" in out.lower() or "model" in out.lower()


def test_allows_price_from_tool_result():
    out = sanitize_agent_response(
        "The Osaki Solo Flex is $4,999.",
        tools_called=["search_chair_specs"],
        user_query="price for solo flex",
        tool_results=["BASE PRICE (USD): $4,999.00\nOther specs..."],
    )
    assert "$4,999" in out


def test_blocks_numbered_repair_steps_without_tool():
    out = sanitize_agent_response(
        "1. Turn off the chair.\n2. Remove the back panel.\n3. Check the fuse.",
        tools_called=[],
        user_query="how do I fix my chair",
    )
    assert "Remove the back panel" not in out
    assert "support" in out.lower()


def test_strips_discount_narrative():
    out = sanitize_agent_response(
        "Originally $7,999, now only $4,999!",
        tools_called=["search_chair_specs"],
        user_query="price",
        tool_results=["BASE PRICE (USD): $4,999.00"],
    )
    assert "Originally" not in out


def test_blocks_ungrounded_spec_numbers_with_authoritative_tool():
    tool_blob = (
        "--- Result 1: Osaki Hypnos ---\n"
        "BASE PRICE (USD): $8,999.00\n"
        "AUTHORITATIVE SPEC VALUES (use EXACTLY these numbers, do not paraphrase):\n"
        "  - Minimum Doorway: 32 inches\n"
        "  - Chair Weight: 285 lbs\n"
    )
    out = sanitize_agent_response(
        "You need a doorway clearance of at least 36 inches and the chair weighs about 300 lbs.",
        tools_called=["search_chair_specs"],
        user_query="doorway size for hypnos",
        tool_results=[tool_blob],
    )
    assert "36 inches" not in out
    assert "300 lbs" not in out
    assert "accurate information" in out.lower()


def test_allows_matching_spec_numbers_from_tool():
    tool_blob = (
        "AUTHORITATIVE SPEC VALUES (use EXACTLY these numbers, do not paraphrase):\n"
        "  - Minimum Doorway: 32 inches\n"
    )
    out = sanitize_agent_response(
        "The minimum doorway clearance is 32 inches.",
        tools_called=["search_chair_specs"],
        user_query="doorway",
        tool_results=[tool_blob],
    )
    assert "32 inches" in out


def test_spanish_repair_guard_stays_in_customer_language():
    out = sanitize_agent_response(
        "1. Retire el panel trasero.\n2. Cambie el fusible.",
        tools_called=[],
        user_query="¿Cómo reparo mi silla?",
    )
    assert "Aún no tengo pasos verificados" in out
    assert "Retire el panel" not in out


def test_conversation_language_survives_neutral_followup_and_ignores_bot_footer():
    from answer_guard import conversation_language
    history=[{'role':'user','content':'한국어로 의자를 추천해 주세요'},
             {'role':'assistant','content':'Some accidental English footer.'}]
    assert conversation_language('3000',history)=='ko'
    assert conversation_language('Solo Flex',history)=='ko'
    assert conversation_language('Please answer in English',history)=='en'
    spanish=[{'role':'user','content':'Necesito una silla con masaje suave'}]
    assert conversation_language('sí',spanish)=='es'
    assert conversation_language('3000',spanish)=='es'


def test_guard_fallback_respects_resolved_conversation_language():
    out=sanitize_agent_response('1. Remove the back panel.',tools_called=[],user_query='3000',language='ko')
    assert '확인된' in out
    assert 'Remove the back' not in out


def test_numbered_catalog_recommendations_are_not_repair_instructions():
    out=sanitize_agent_response('1. Fixture Alpha — $3,000.\n2. Fixture Beta — $2,500.',tools_called=['recommend_chairs'],
                               user_query='recommend a chair',tool_results=['BASE PRICE (USD): $3,000\nBASE PRICE (USD): $2,500'])
    assert 'Fixture Alpha' in out
    assert 'repair' not in out


def model_facts(model,price,weight,door,mechanism='4D'):
    import json
    return 'MODEL_FACTS_JSON: '+json.dumps({'model':model,'currency':'USD','price_usd':price,
         'specs':{'max_user_weight_lb':weight,'doorway_assembled_in':door,'massage_mechanism':mechanism,'foot_roller':'Yes'}})


def test_different_model_price_cannot_be_borrowed_from_same_tool_results():
    blob=model_facts('Fixture Alpha',3000,260,30)+'\n'+model_facts('Fixture Beta',6000,375,35)
    out=sanitize_agent_response('Fixture Alpha costs $6,000.',tools_called=['search_chair_specs'],user_query='price of Fixture Alpha',tool_results=[blob])
    assert '$6,000' not in out
    assert 'accurate information' in out


def test_wrong_model_capacity_is_not_justified_by_another_model_or_nearby_rounding():
    blob=model_facts('Fixture Alpha',3000,260,30)+'\n'+model_facts('Fixture Beta',6000,375,35)
    wrong=sanitize_agent_response('Fixture Alpha supports up to 375 lbs.',tools_called=['search_chair_specs'],user_query='Fixture Alpha capacity',tool_results=[blob])
    rounded=sanitize_agent_response('Fixture Alpha maximum user weight is 280 lbs.',tools_called=['search_chair_specs'],user_query='Fixture Alpha capacity',tool_results=[blob])
    assert '375 lbs' not in wrong
    assert '280 lbs' not in rounded


def test_model_fact_guard_preserves_correct_comparison_and_currency():
    blob=model_facts('Fixture Alpha',3000,260,30)+'\n'+model_facts('Fixture Beta',6000,375,35)
    correct='Fixture Alpha costs $3,000.\nFixture Beta costs $6,000.'
    assert sanitize_agent_response(correct,tools_called=['search_chair_specs'],user_query='compare',tool_results=[blob])==correct
    foreign=sanitize_agent_response('Fixture Alpha costs 3,000 EUR.',tools_called=['search_chair_specs'],user_query='price',tool_results=[blob])
    assert '3,000 EUR' not in foreign


def test_model_fact_guard_blocks_mechanism_and_foot_roller_swaps():
    blob=model_facts('Fixture Alpha',3000,260,30,'3D')
    assert '4D' not in sanitize_agent_response('Fixture Alpha has a 4D mechanism.',tools_called=['search_chair_specs'],user_query='specs',tool_results=[blob])
    assert 'no foot rollers' not in sanitize_agent_response('Fixture Alpha has no foot rollers.',tools_called=['search_chair_specs'],user_query='foot rollers',tool_results=[blob])


def test_unsupported_approval_and_delivery_guarantees_are_blocked():
    for text in ('Your replacement is approved.','We guarantee delivery tomorrow.','The chair will cure your back pain.'):
        out=sanitize_agent_response(text,tools_called=['get_warranty_or_policy'],user_query='help',tool_results=['Policy lookup'])
        assert 'cannot approve' in out


def test_explicit_language_request_in_history_survives_neutral_followup():
    from answer_guard import conversation_language
    assert conversation_language('3000',[{'role':'user','content':'Please answer in Korean'}])=='ko'
    assert conversation_language('Solo Flex',[{'role':'user','content':'Please reply in Spanish'}])=='es'


def test_missing_catalog_results_cannot_authorize_fluent_recommendation_facts():
    text='Fixture Omega is a 4D chair supporting 350 lbs and fitting a 30 inch doorway.'
    output=sanitize_agent_response(text,tools_called=['recommend_chairs'],user_query='350 lb chair',
                                   tool_results=['NO_RESULTS: No chairs match the given criteria.'])
    assert '350 lbs' not in output and '30 inch' not in output and 'Fixture Omega' not in output
    assert 'requirements' in output and 'specialist' in output
    korean=sanitize_agent_response(text,tools_called=['recommend_chairs'],user_query='350',language='ko',tool_results=['NO_RESULTS: No chairs'])
    assert '조건' in korean and 'Fixture Omega' not in korean


def test_feature_by_model_table_cannot_borrow_other_column_price_and_capacity():
    blob=model_facts('Fixture Alpha',3000,260,30)+'\n'+model_facts('Fixture Beta',6000,375,35)
    wrong='| Feature | Fixture Alpha | Fixture Beta |\n|---|---|---|\n| Price | $6,000 | $6,000 |\n| Max user weight | 375 lbs | 375 lbs |'
    output=sanitize_agent_response(wrong,tools_called=['search_chair_specs'],user_query='compare',tool_results=[blob])
    assert 'accurate information' in output
    correct='| Feature | Fixture Alpha | Fixture Beta |\n|---|---|---|\n| Price | $3,000 | $6,000 |\n| Max user weight | 260 lbs | 375 lbs |'
    assert sanitize_agent_response(correct,tools_called=['search_chair_specs'],user_query='compare',tool_results=[blob])==correct


def test_failed_tracking_cannot_leave_a_fabricated_carrier_or_eta():
    text='Current status: In transit. Carrier: ExampleCarrier. Estimated delivery: June 15.'
    output=sanitize_agent_response(text,tools_called=['lookup_order_status'],user_query='track order',
                                   tool_results=['MISSING_INPUT: Need both a valid order number and the email used at checkout.'])
    assert 'ExampleCarrier' not in output and 'June 15' not in output
    assert 'order number' in output and 'email' in output


def measurement_facts(model="Fixture Alpha", **specs):
    return "MODEL_FACTS_JSON: " + json.dumps({
        "model": model,
        "currency": "USD",
        "specs": {
            "doorway_assembled_in": 32,
            "doorway_disassembled_in": 28,
            "chair_weight_lb": 285,
            "max_user_weight_lb": 260,
            "standing_width_in": 30,
            "standing_height_in": 48,
            "reclined_width_in": 34,
            "reclined_height_in": 40,
            **specs,
        },
    })


def guarded_measurement(text, blob=None, query="Fixture Alpha specifications"):
    return sanitize_agent_response(
        text, tools_called=["search_chair_specs"], user_query=query,
        tool_results=[blob or measurement_facts()],
    )


@pytest.mark.parametrize("text", [
    "Fixture Alpha needs a 34 inch doorway and weighs 300 lbs.",
    "Fixture Alpha minimum doorway is 32.1 inches.",
    "Fixture Alpha needs a 34-inch doorway.",
    "Fixture Alpha has a 280-pound capacity.",
    "Fixture Alpha chair weight is 286 lbs.",
    "Fixture Alpha maximum user weight is 285 lbs.",
    "Fixture Alpha doorway is 30 inches.",
    "Fixture Alpha doorway is 28 inches assembled.",
    "Fixture Alpha disassembled doorway is 32 inches.",
    "Fixture Alpha standing width is 34 inches.",
    "Fixture Alpha reclined height is 48 inches.",
    "Fixture Alpha maximum user weight is 300.",
    "Fixture Alpha chair weight is 285 kg.",
])
def test_measurement_guard_rejects_nearby_values_and_field_or_unit_swaps(text):
    assert "accurate information" in guarded_measurement(text)


@pytest.mark.parametrize("text", [
    "Fixture Alpha needs a 32 inch doorway and weighs 285 lbs.",
    "Fixture Alpha minimum doorway is 32.0 inches.",
    "Fixture Alpha needs a 32-inch doorway.",
    "Fixture Alpha has a 260-pound capacity.",
    "Fixture Alpha disassembled doorway is 28 inches.",
    "Fixture Alpha is 30 inches wide and 48 inches tall.",
    "Fixture Alpha reclined width is 34 inches and reclined height is 40 inches.",
    "Fixture Alpha doorway is 81.28 cm and chair weight is 129.3 kg.",
    "Fixture Alpha doorway is 812.8 mm.",
    "Fixture Alpha doorway is 2.667 ft.",
])
def test_measurement_guard_preserves_exact_fields_and_precise_conversions(text):
    assert guarded_measurement(text) == text


@pytest.mark.parametrize("text", [
    "Fixture Alpha doorway is 81 cm.",
    "Fixture Alpha doorway is 86.4 cm.",
    "Fixture Alpha chair weight is 136.1 kg.",
])
def test_conversions_do_not_reintroduce_percentage_rounding_loophole(text):
    assert "accurate information" in guarded_measurement(text)


def test_legacy_authoritative_specs_are_field_bound_and_not_a_numeric_pool():
    blob = (
        "--- Result 1: Fixture Alpha ---\n"
        "AUTHORITATIVE SPEC VALUES:\n"
        "- Minimum Doorway (inch): 32\n"
        "- Chair Weight: 285 lbs\n"
        "- Dimension - Standing - Width (inch): 30\n"
        "- Dimension - Reclined - Width (inch): 34\n"
    )
    correct = "Fixture Alpha doorway is 32 inches, chair weight is 285 lbs, and standing width is 30 inches."
    assert guarded_measurement(correct, blob) == correct
    for wrong in ("Fixture Alpha doorway is 30 inches.", "Fixture Alpha chair weight is 300 lbs.",
                  "Fixture Alpha standing width is 34 inches."):
        assert "accurate information" in guarded_measurement(wrong, blob)


def test_structured_facts_override_conflicting_supplemental_measurements():
    blob = measurement_facts() + "\nAUTHORITATIVE SPEC VALUES:\n- Minimum Doorway: 34 inches\n"
    assert "accurate information" in guarded_measurement("Fixture Alpha doorway is 34 inches.", blob)


def test_height_ranges_check_every_endpoint_and_order():
    blob = measurement_facts(minimum_user_height_in=58, maximum_user_height_in=75)
    correct = "Fixture Alpha user height range is 58–75 inches."
    assert guarded_measurement(correct, blob) == correct
    repeated_unit = "Fixture Alpha user height range is 58 inches to 75 inches."
    assert guarded_measurement(repeated_unit, blob) == repeated_unit
    for wrong in ("Fixture Alpha user height range is 60–75 inches.",
                  "Fixture Alpha user height range is 58–77 inches.",
                  "Fixture Alpha user height range is 75–58 inches.",
                  "Fixture Alpha user height range is 75 inches to 58 inches.",
                  "Fixture Alpha user height range is 58–58 inches."):
        assert "accurate information" in guarded_measurement(wrong, blob)


def test_legacy_source_range_is_not_ignored():
    blob = "--- Result 1: Fixture Alpha ---\nAUTHORITATIVE SPEC VALUES:\n- User Height Range: 58–75 inches\n"
    correct = "Fixture Alpha user height range is 58–75 inches."
    assert guarded_measurement(correct, blob) == correct
    assert "accurate information" in guarded_measurement(correct.replace("58", "60"), blob)


def test_measurements_remain_bound_to_models_in_prose_and_query_context():
    blob = measurement_facts() + "\n" + measurement_facts("Fixture Beta", doorway_assembled_in=36)
    correct = "Fixture Alpha doorway is 32 inches, while Fixture Beta doorway is 36 inches."
    assert guarded_measurement(correct, blob) == correct
    assert "accurate information" in guarded_measurement(correct.replace("32", "36"), blob)
    assert guarded_measurement("32 inches.", blob, "Fixture Alpha doorway") == "32 inches."
    assert "accurate information" in guarded_measurement("36 inches.", blob, "Fixture Alpha doorway")


def test_feature_by_model_tables_bind_bare_numbers_to_header_units_and_fields():
    blob = measurement_facts() + "\n" + measurement_facts("Fixture Beta", doorway_assembled_in=36, chair_weight_lb=300)
    correct = (
        "| Feature | Fixture Alpha | Fixture Beta |\n|---|---|---|\n"
        "| Doorway (in) | 32 | 36 |\n| Chair weight (lb) | 285 | 300 |"
    )
    assert guarded_measurement(correct, blob) == correct
    assert "accurate information" in guarded_measurement(correct.replace("32 | 36", "36 | 32"), blob)
    assert "accurate information" in guarded_measurement(correct.replace("285 | 300", "300 | 285"), blob)


def test_model_by_feature_tables_bind_values_to_each_header():
    blob = measurement_facts() + "\n" + measurement_facts("Fixture Beta", doorway_assembled_in=36, chair_weight_lb=300)
    correct = (
        "| Model | Doorway (in) | Chair weight (lb) |\n|---|---|---|\n"
        "| Fixture Alpha | 32 | 285 |\n| Fixture Beta | 36 | 300 |"
    )
    assert guarded_measurement(correct, blob) == correct
    assert "accurate information" in guarded_measurement(correct.replace("32 | 285", "36 | 285"), blob)
    assert "accurate information" in guarded_measurement(correct.replace("36 | 300", "300 | 36"), blob)


def test_single_model_table_checks_header_units():
    correct = "| Feature | Fixture Alpha |\n|---|---|\n| Doorway (cm) | 81.28 |"
    assert guarded_measurement(correct) == correct
    assert "accurate information" in guarded_measurement(correct.replace("81.28", "86.36"))
