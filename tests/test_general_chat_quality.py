"""Exercise the real chat endpoint with mocked models and external services."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'app'))
    monkeypatch.setenv('APP_ENV','development')
    monkeypatch.setenv('OPENAI_API_KEY','test-key')
    monkeypatch.setenv('SHOPIFY_WEBHOOK_SECRET','test-webhook-secret')
    module=importlib.import_module('app.main')
    # These regression fixtures exercise the legacy Chat Completions adapter.
    # Separate tests below exercise the upgraded Responses endpoint.
    monkeypatch.setattr(module, 'AGENT_MODEL', 'gpt-5.6')
    monkeypatch.setattr(module,'vs_products',object())
    monkeypatch.setattr(module,'vs_qa',object())
    monkeypatch.setattr(module,'vs_web',object())
    monkeypatch.setattr(module,'products_retriever',object())
    monkeypatch.setattr(module,'qa_retriever',object())
    monkeypatch.setattr(module,'web_retriever',object())
    monkeypatch.setattr(module.WarrantyEngine,'get_active_session_ticket',lambda _:None)
    monkeypatch.setattr(module,'cache_get',lambda _:None)
    monkeypatch.setattr(module,'cache_set',lambda *args:None)
    monkeypatch.setattr(module,'_persist_usage_log',lambda **kwargs:None)
    logs=[]
    monkeypatch.setattr(module,'_persist_chat_log',lambda *args:logs.append(args))
    import app.request_trace as traces
    recorded=[]
    monkeypatch.setattr(traces,'persist_trace',lambda trace,engine:recorded.append(trace.to_dict()))
    import app.scope_classifier as scope
    monkeypatch.setattr(scope,'evaluate_scope',lambda *args:SimpleNamespace(is_blocked=False))
    return module,logs,recorded


def reply(text='',calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text,tool_calls=calls or []))],usage=None)


def model(module,monkeypatch,completion):
    monkeypatch.setattr(module,'openai_client',SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion))))


def response_reply(text='', *, name=None, call_id=None):
    output = []
    if name:
        output = [
            {'type': 'reasoning', 'id': 'rs_fixture', 'summary': [], 'encrypted_content': 'fixture-encrypted-reasoning'},
            {'type': 'function_call', 'id': 'fc_fixture', 'call_id': call_id,
             'name': name, 'arguments': '{"model_name":"Fixture Alpha","spec_topic":"features"}', 'status': 'completed'},
        ]
    else:
        output = [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                   'content': [{'type': 'output_text', 'text': text}]}]
    return SimpleNamespace(status='completed', error=None, output=output, model='gpt-6.1-sol', usage=None)


def test_upgraded_responses_tool_roundtrip_preserves_guards_and_provider_evidence(main_module, monkeypatch):
    module, logs, traces = main_module
    monkeypatch.setattr(module, 'AGENT_MODEL', 'gpt-6.1-sol')
    requests = []
    replies = iter([
        response_reply(name='search_chair_specs', call_id='call_provider_exact'),
        response_reply('Here are the verified model features.'),
    ])
    def create(**kwargs):
        requests.append(kwargs)
        return next(replies)
    monkeypatch.setattr(module, 'openai_client', SimpleNamespace(responses=SimpleNamespace(create=create)))
    executed = []
    monkeypatch.setattr(module, '_execute_tool', lambda name, args, domain, **kwargs: executed.append(name) or 'Verified model facts')
    with TestClient(module.app) as client:
        result = client.post('/api/v1/chat', json={'session_id': 'upgrade-response-roundtrip', 'user_query': 'Tell me the features of Fixture Alpha'})
    assert result.status_code == 200
    assert 'verified model features' in result.text
    assert executed == ['search_chair_specs']
    assert requests[0]['store'] is False
    assert requests[0]['parallel_tool_calls'] is False
    assert requests[0]['reasoning']['effort'] == 'medium'
    assert all(tool['type'] == 'function' and 'function' not in tool for tool in requests[0]['tools'])
    assert any(item.get('type') == 'reasoning' for item in requests[1]['input'])
    assert any(item.get('type') == 'function_call_output' and item['call_id'] == 'call_provider_exact' for item in requests[1]['input'])
    assert traces[-1]['model_calls_successful'] == 2
    assert traces[-1]['provider_models'] == ['gpt-6.1-sol']
    assert 'fixture-encrypted-reasoning' not in json.dumps(traces[-1])
    assert logs[-1][2] == result.text


def test_upgraded_provider_failure_is_safe_and_has_no_successful_model_evidence(main_module, monkeypatch):
    module, logs, traces = main_module
    monkeypatch.setattr(module, 'AGENT_MODEL', 'gpt-6.1-sol')
    def fail(**kwargs):
        raise RuntimeError('fixture-private-provider-details')
    monkeypatch.setattr(module, 'openai_client', SimpleNamespace(responses=SimpleNamespace(create=fail)))
    with TestClient(module.app) as client:
        result = client.post('/api/v1/chat', json={'session_id': 'upgrade-response-failure', 'user_query': 'Tell me about Fixture Alpha'})
    assert result.status_code == 200
    assert 'fixture-private-provider-details' not in result.text
    assert traces[-1]['model_calls_failed'] == 1
    assert traces[-1]['model_calls_successful'] == 0
    assert traces[-1]['provider_models'] == []
    assert logs[-1][2] == result.text


def test_neutral_followup_keeps_korean_model_context_and_department_footer(main_module,monkeypatch):
    module,logs,traces=main_module
    calls=[]
    def complete(**kwargs):
        calls.append(kwargs)
        return reply('조건을 확인한 후 적합한 제품 정보를 안내드리겠습니다.')
    model(module,monkeypatch,complete)
    with TestClient(module.app) as client:
        response=client.post('/api/v1/chat',json={'user_query':'3000','session_id':'quality-language','current_domain':'osakiusa.com',
              'chat_history':[{'role':'user','content':'한국어로 의자를 추천해 주세요'},{'role':'assistant','content':'예산을 알려주세요.'}]})
    assert response.status_code==200
    assert 'RESPONSE_LANGUAGE: ko' in calls[0]['messages'][0]['content']
    assert 'For sales inquiries' not in response.text
    assert '구매 상담' in response.text
    assert traces[-1]['final_response']==response.text
    assert logs[-1][2]==response.text


def test_model_failure_records_actual_localized_customer_reply(main_module,monkeypatch):
    module,logs,traces=main_module
    def fail(**kwargs):
        raise RuntimeError('synthetic-provider-private-details')
    model(module,monkeypatch,fail)
    with TestClient(module.app) as client:
        response=client.post('/api/v1/chat',json={'user_query':'Solo Flex','session_id':'quality-error',
            'chat_history':[{'role':'user','content':'가격을 한국어로 알려주세요.'}]})
    assert response.status_code==200
    assert '현재 조회를 완료하지 못했습니다' in response.text
    assert 'synthetic-provider-private-details' not in response.text
    assert traces[-1]['final_response']==response.text
    assert logs[-1][2]==response.text


def test_general_comparison_can_fetch_second_readonly_model_source(main_module,monkeypatch):
    module,logs,traces=main_module
    calls=[]
    executed=[]
    tool=lambda ident,name:SimpleNamespace(id=ident,function=SimpleNamespace(name='search_chair_specs',arguments=json.dumps({'model_name':name,'spec_topic':'features'})))
    responses=iter([reply(calls=[tool('one','Fixture Alpha')]),reply(calls=[tool('two','Fixture Beta')]),reply('Here is the verified comparison of the two models.')])
    def complete(**kwargs):
        calls.append(kwargs)
        return next(responses)
    model(module,monkeypatch,complete)
    monkeypatch.setattr(module,'_execute_tool',lambda name,args,domain,**kwargs:executed.append(args['model_name']) or 'Verified model facts')
    with TestClient(module.app) as client:
        response=client.post('/api/v1/chat',json={'user_query':'Compare the features of Fixture Alpha and Fixture Beta','session_id':'quality-two-lookups'})
    assert response.status_code==200
    assert executed==['Fixture Alpha','Fixture Beta']
    assert calls[1]['tools']
    assert all(item['function']['name'] not in {'capture_sales_lead','answer_warranty_question','start_warranty_workflow'} for item in calls[1]['tools'])
    assert 'verified comparison' in response.text


def test_stateful_warranty_never_replays_cached_node_or_advances_two_nodes(main_module,monkeypatch):
    module,logs,traces=main_module
    monkeypatch.setattr(module.WarrantyEngine,'get_active_session_ticket',lambda _:SimpleNamespace(ticket_id='quality-ticket',status='in_progress'))
    monkeypatch.setattr(module.WarrantyEngine,'get_current_node',lambda _: {'node_id':'power','type':'question','prompt':'Does it turn on?','options':[]})
    cache_keys=[]
    monkeypatch.setattr(module,'cache_get',lambda key:cache_keys.append(key) or ('WRONG PREVIOUS NODE' if key else None))
    calls=[]
    tool=SimpleNamespace(id='one',function=SimpleNamespace(name='answer_warranty_question',arguments=json.dumps({'ticket_id':'quality-ticket','answer_key':'yes'})))
    responses=iter([reply(calls=[tool]),reply('A rewritten response that should not replace engine copy')])
    def complete(**kwargs):
        calls.append(kwargs)
        return next(responses)
    model(module,monkeypatch,complete)
    monkeypatch.setattr(module,'_execute_tool',lambda *args,**kwargs:'WARRANTY_CONTINUE\nCUSTOMER_MESSAGE:\nDoes the remote light turn on?\nINSTRUCTION: Deliver verbatim.')
    with TestClient(module.app) as client:
        response=client.post('/api/v1/chat',json={'user_query':'hello','session_id':'quality-stateful'})
    assert cache_keys==[None]
    assert 'WRONG PREVIOUS NODE' not in response.text
    assert 'Does the remote light turn on?' in response.text
    assert 'tools' not in calls[1]
    assert 'For sales inquiries' not in response.text
    assert 'warranty team' in response.text.lower()


def test_strict_delivery_wrong_order_never_falls_back_to_another_order_by_email(main_module,monkeypatch):
    module,_,_=main_module
    monkeypatch.setattr(module,'get_store_config',lambda _: {'shop_domain':'synthetic-shop.invalid','shop_access_token':'synthetic-token'})
    searches=[]
    def search(url,headers,query,terms):
        searches.append(terms)
        if terms.startswith("email:"):
            return [{'node':{'name':'#OTHER','displayFulfillmentStatus':'UNFULFILLED'}}]
        return []
    monkeypatch.setattr(module,'_shopify_orders_search',search)
    result=module.fetch_shopify_order_status('WRONG123','synthetic@example.invalid','osakiusa.com',allow_order_only=False)
    assert 'error' in result
    assert searches and all('AND email:' in query for query in searches)
    searches.clear()
    missing=module.fetch_shopify_order_status('','synthetic@example.invalid','osakiusa.com',allow_order_only=False)
    assert 'error' in missing
    assert searches==[]
    legacy=module.fetch_shopify_order_status('','synthetic@example.invalid','osakiusa.com',allow_order_only=True)
    assert legacy['status']=='PROCESSING'
    assert searches==["email:'synthetic@example.invalid'"]


@pytest.mark.parametrize('order,email', [('1234', "x' OR email:*@example.invalid"), ("1234' OR name:*",'synthetic@example.invalid'), ('','synthetic@example.invalid'), ('1234','')])
def test_public_tracking_rejects_incomplete_or_search_injection_identifiers(main_module,monkeypatch,order,email):
    module,_,_=main_module
    searches=[]
    monkeypatch.setattr(module,'_shopify_orders_search',lambda *args:searches.append(args) or [])
    result=module.fetch_shopify_order_status(order,email,'osakiusa.com',allow_order_only=False)
    assert 'error' in result
    assert searches==[]


def test_unavailable_catalog_is_not_cached_and_recovery_gets_fresh_evidence(main_module,monkeypatch):
    module,logs,traces=main_module
    cache={}
    monkeypatch.setattr(module,'make_cache_key',lambda *args:'synthetic-faq')
    monkeypatch.setattr(module,'cache_get',lambda key:cache.get(key))
    monkeypatch.setattr(module,'cache_set',lambda key,text:cache.__setitem__(key,text))
    tool=SimpleNamespace(id='lookup',function=SimpleNamespace(name='recommend_chairs',arguments=json.dumps({'user_need':'4D chair'})))
    responses=iter([reply(calls=[tool]),reply('Fixture Omega supports up to 350 lbs.'),reply(calls=[tool]),reply('Osaki Solo Flex 4D is a catalog option.')])
    model(module,monkeypatch,lambda **kwargs:next(responses))
    facts={'model':'Osaki Solo Flex 4D','currency':'USD','price_usd':4999,'specs':{'massage_mechanism':'4D'}}
    results=iter(['NO_RESULTS: No matching catalog data.','MODEL_FACTS_JSON: '+json.dumps(facts)])
    monkeypatch.setattr(module,'_execute_tool',lambda *args,**kwargs:next(results))
    body={'user_query':'Recommend a 4D chair','session_id':'quality-recovery'}
    with TestClient(module.app) as client:
        unavailable=client.post('/api/v1/chat',json=body)
        assert 'Fixture Omega' not in unavailable.text
        assert cache=={}
        recovered=client.post('/api/v1/chat',json=body)
    assert 'Osaki Solo Flex 4D' in recovered.text
    assert len(cache)==1
    assert traces[-1]['turns'][0]['source_evidence']==[{'source_type':'catalog','model':'Osaki Solo Flex 4D','currency':'USD','available':True}]


def test_tracking_api_uses_exact_current_lookup_answer_not_model_invention(main_module,monkeypatch):
    module,logs,traces=main_module
    tool=SimpleNamespace(id='lookup',function=SimpleNamespace(name='lookup_order_status',arguments=json.dumps({'order_id':'1234','email':'synthetic@example.invalid'})))
    responses=iter([reply(calls=[tool]),reply('Tracking number: INVENTED123. Carrier: InventedCarrier. Estimated delivery: tomorrow.')])
    model(module,monkeypatch,lambda **kwargs:next(responses))
    monkeypatch.setattr(module,'fetch_shopify_order_status',lambda *args,**kwargs:{'status':'PROCESSING'})
    monkeypatch.setattr(module,'build_deterministic_tracking_response',lambda *args:'Your verified order is being prepared. No carrier tracking is available yet.')
    with TestClient(module.app) as client:
        response=client.post('/api/v1/chat',json={'user_query':'Track order 1234 with synthetic@example.invalid','session_id':'quality-tracking'})
    assert 'INVENTED' not in response.text and 'tomorrow' not in response.text
    assert 'Your verified order is being prepared' in response.text
    assert 'ORDER_LOOKUP_RESULT_JSON' not in response.text
    assert traces[-1]['turns'][0]['source_evidence']==[{'source_type':'verified_order_lookup','available':True}]
