"""Tests for answer_guard — hallucination blocking."""

import sys
from pathlib import Path

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
