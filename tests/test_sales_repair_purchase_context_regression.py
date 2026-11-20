"""Context isolation cases through the actual Sales router, with no providers."""
from __future__ import annotations

from dataclasses import replace

import pytest

from script.quality_v4_product import SalesLog
from script.simulate_chatbot_quality import offline_api


@pytest.fixture
def conversation(monkeypatch):
    monkeypatch.setenv('WARRANTY_REQUIRE_CHAT_EMAIL', '0')
    with offline_api() as client:
        log = SalesLog(client, 'repair-purchase-context-synthetic')
        assert log.setup_ok
        yield log
        assert not client.attempted_external_delivery


def _start_mixed(log):
    reply = log.say('My Titan TP-Epic 4D is broken. My old room doorway is 24 inches wide, '
                    'and I want to buy an Osaki OS-Champ. The new chair must fit my 35 inch doorway, assembled only.')
    assert 'conversation.dual_task' in reply['tools_used'], reply['reply']
    assert 'Titan TP-Epic 4D' in reply['reply'].split('→ Warranty')[0]
    assert 'New purchase — Osaki OS-Champ → Sales' in reply['reply']
    return reply


def test_old_chair_measurement_does_not_replace_shopping_limit(conversation):
    _start_mixed(conversation)
    assert float(conversation.saved['recommend_prefs']['doorway_in']) == 35
    assert conversation.saved['recommend_prefs']['doorway_fit'] == 'assembled'

    old = conversation.say('Back to my broken chair, its installation doorway was 24 inches. '
                           'What number do I call about the repair?')
    assert old['intent'] == 'warranty_redirect'
    assert '**Titan TP-Epic 4D**' in old['reply'] and 'ext. 3' in old['reply']
    assert "haven't submitted a service ticket" in old['reply']
    assert float(conversation.saved['recommend_prefs']['doorway_in']) == 35


def test_ambiguous_cost_asks_which_and_explicit_purchase_does_not_reopen_repair(conversation):
    _start_mixed(conversation)
    old = conversation.say('Back to my broken chair, what number should I call?')
    assert old['intent'] == 'warranty_redirect'

    ambiguous = conversation.say('How much is it?')
    assert ambiguous['intent'] == 'unclear'
    assert 'repair cost for Titan TP-Epic 4D' in ambiguous['reply']
    assert 'purchase price for Osaki OS-Champ' in ambiguous['reply']
    assert '$' not in ambiguous['reply']

    purchase = conversation.say('I am not asking about the repair; for the new one, how much is it?')
    assert purchase['intent'] in {'price', 'specs'}, purchase['reply']
    assert 'Osaki OS-Champ' in purchase['reply'] and '$1,299' in purchase['reply']
    assert 'Titan TP-Epic 4D' not in purchase['reply'] and not purchase['handoff']


def test_same_model_can_be_a_distinct_new_purchase(conversation):
    reply = conversation.say('My Titan TP-Epic 4D is broken and I want to buy another of the same model. '
                             'Please keep them separate.')
    assert 'Repair — Titan TP-Epic 4D' in reply['reply']
    assert 'New purchase — Titan TP-Epic 4D → Sales' in reply['reply']
    assert not reply['handoff']

    fact = conversation.say('For the one I am buying, how wide is it assembled?')
    assert fact['intent'] == 'specs', fact['reply']
    assert 'Titan TP-Epic 4D' in fact['reply']
    assert 'doorway' in fact['reply'].casefold()


def test_older_repair_model_correction_does_not_inherit_old_component(conversation):
    conversation.say('I own a Titan TP-Epic 4D and its foot rollers stopped working.')
    correction = conversation.say('Correction: the broken chair is actually an Osaki OS-4000T. The Epic is not broken.')
    assert correction['intent'] == 'warranty_redirect', correction['reply']
    assert '**Osaki OS-4000T**' in correction['reply']
    assert 'OS-4000XT' not in correction['reply'] and 'foot rollers' not in correction['reply']


@pytest.mark.parametrize('question', [
    'I want to buy an Osaki OS-Champ. If the new one breaks, who repairs it?',
    'I am looking to buy a new Osaki OS-Champ. Who should I call if it needs a repair?',
    'If my next chair breaks, can I get technical support?',
])
def test_hypothetical_service_for_new_chair_does_not_reopen_actual_repair(conversation, question):
    initial = conversation.say('I own a Titan TP-Epic 4D and its foot rollers stopped working.')
    assert initial['intent'] == 'warranty_redirect'
    reply = conversation.say(question)
    assert reply['intent'] == 'prepurchase_policy', reply['reply']
    assert 'ext. 3' in reply['reply'] and 'osakiusa.com/pages/warranty' in reply['reply']
    assert not reply['handoff'] and 'Titan TP-Epic 4D' not in reply['reply']
    assert conversation.saved['owner_service_case']['model'] == 'Titan TP-Epic 4D'


def test_loaded_catalog_product_cannot_capture_generic_old_or_new_one(conversation, monkeypatch):
    import sales_catalog
    from sales_agent import _guess_model_from_text

    old_ref = "Going back to the one that's broken, how do I get technical support for the old chair?"
    new_ref = 'And for the new one, how wide should the doorway be?'
    products = sales_catalog.load_product_index()
    base = sales_catalog.resolve_product('Osaki OS-Champ')
    assert base
    # Unlike the tracked fixture, the live catalog has a product whose
    # distinctive name is an ordinary pronoun after dropping brand/mechanism.
    one = replace(base, handle='osaki-one-4d', title='Osaki One 4D', display_name='Osaki One 4D')
    monkeypatch.setattr(sales_catalog, 'load_product_index', lambda: (one, *products))
    assert _guess_model_from_text(old_ref) == one
    assert _guess_model_from_text(new_ref) == one

    _start_mixed(conversation)
    repair = conversation.say(old_ref)
    assert repair['intent'] == 'warranty_redirect'
    assert 'Titan TP-Epic 4D' in repair['reply'] and 'Osaki One 4D' not in repair['reply']
    assert conversation.saved['owner_service_case']['model'] == 'Titan TP-Epic 4D'

    purchase = conversation.say(new_ref)
    assert purchase['intent'] == 'specs'
    assert 'Osaki OS-Champ' in purchase['reply'] and 'Osaki One 4D' not in purchase['reply']
    assert conversation.saved['owner_purchase_model'] == 'osaki-os-champ'

    explicit = conversation.say('Correction: the broken chair is actually an Osaki One 4D.')
    assert explicit['intent'] == 'warranty_redirect' and 'Osaki One 4D' in explicit['reply']
    assert conversation.saved['owner_service_case']['model'] == 'Osaki One 4D'
