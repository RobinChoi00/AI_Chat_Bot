"""Synthetic reproductions of historical ticket-correspondence contamination."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from app.agent_tools import tool_get_repair_help, tool_lookup_order_status, tool_capture_sales_lead
from app.answer_guard import sanitize_agent_response
import warranty_knowledge as knowledge

COPIED='Since the parts are on the way, I will go ahead and close this ticket. Your chair is fixed.'


class RawTicketRetriever:
    def search(self,*args,**kwargs):
        pytest.fail('Raw support ticket index must never be queried for customer repair steps')


def entry(source='reviewed_knowledge',model='Synthetic Alpha',steps=('Inspect the visible remote cable while the chair is unplugged.',)):
    return knowledge.KnowledgeEntry(source,'remote','Internal title with ticket correspondence',COPIED,steps,applicable_models=(model,))


def test_raw_ticket_correspondence_is_not_retrieved_or_returned(monkeypatch):
    monkeypatch.setattr(knowledge,'search_knowledge',lambda **kwargs:[entry('freshdesk',steps=(COPIED,))])
    result=tool_get_repair_help(qa_retriever=RawTicketRetriever(),issue_description='My remote cable is loose',model_name='Synthetic Alpha')
    assert result.startswith('NO_RESULTS:')
    assert COPIED not in result
    output=sanitize_agent_response(COPIED,tools_called=['get_repair_help'],user_query='My remote cable is loose',tool_results=[result])
    assert 'parts are on the way' not in output
    assert 'close this ticket' not in output
    assert 'chair is fixed' not in output


def test_only_reviewed_matching_safe_steps_are_returned(monkeypatch):
    monkeypatch.setattr(knowledge,'search_knowledge',lambda **kwargs:[entry(),entry(model='Synthetic Beta'),entry(steps=(COPIED,))])
    result=tool_get_repair_help(qa_retriever=RawTicketRetriever(),issue_description='Loose remote cable',model_name='Synthetic Alpha')
    assert result.count('Inspect the visible remote cable')==1
    assert 'VERIFIED_REPAIR_GUIDANCE:' in result
    assert 'Internal title' not in result and COPIED not in result
    safe='1. Inspect the visible remote cable while the chair is unplugged.'
    assert sanitize_agent_response(safe,tools_called=['get_repair_help'],user_query='Loose remote',tool_results=[result])==safe


def test_model_unknown_never_infers_it_from_a_unique_error_code(monkeypatch):
    monkeypatch.setattr(knowledge,'search_knowledge',lambda **kwargs:pytest.fail('Need an exact model'))
    result=tool_get_repair_help(qa_retriever=RawTicketRetriever(),issue_description='Code E99',error_code='E99')
    assert result.startswith('NO_RESULTS:') and 'exact chair model' in result


def test_repair_tool_name_alone_does_not_authorize_unverified_steps():
    output=sanitize_agent_response('1. Inspect the visible cable.',tools_called=['get_repair_help'],user_query='remote',tool_results=['NO_RESULTS: No guide'])
    assert 'verified repair' in output


def test_authenticated_current_tracking_status_survives_repair_status_guard():
    result=tool_lookup_order_status(fetch_fn=lambda *args:{'status':'IN_TRANSIT'},build_response_fn=lambda *args:'Your order is in transit.',
                                    target_domain='osakiusa.com',order_id='1234',email='synthetic@example.invalid')
    assert 'VERIFIED_ORDER_STATUS:' in result
    assert sanitize_agent_response('Your order is in transit.',tools_called=['lookup_order_status'],user_query='Order1234',tool_results=[result])=='Your order is in transit.'
    wrong=sanitize_agent_response('Your order has been delivered.',tools_called=['lookup_order_status'],user_query='Order1234',tool_results=[result])
    assert 'has been delivered' not in wrong


@pytest.mark.parametrize('order,email',[('1234',''),('','synthetic@example.invalid'),("1234' OR name:*",'synthetic@example.invalid')])
def test_public_tracking_tool_requires_both_valid_identifiers(order,email):
    result=tool_lookup_order_status(fetch_fn=lambda *args:pytest.fail('Must not query'),build_response_fn=lambda *args:'',target_domain='osakiusa.com',order_id=order,email=email)
    assert result.startswith('MISSING_INPUT:') and 'both' in result


def test_background_lead_does_not_claim_delivery_or_fixed_response_time(monkeypatch):
    from app import agent_tools
    class Thread:
        def __init__(self,**kwargs): pass
        def start(self): pass
    monkeypatch.setattr(agent_tools.threading,'Thread',Thread)
    result=tool_capture_sales_lead(send_email_fn=lambda *args:None,customer_email='synthetic@example.invalid',interest_summary='Chair',target_domain='osakiusa.com')
    assert result.startswith('LEAD_SEND_QUEUED:')
    assert '24 hours' not in result and 'SUCCESS' not in result
