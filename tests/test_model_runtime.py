"""Request compatibility and provider identity, with no network or persistence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import model_runtime as runtime


def response(*,model='provider-snapshot',content='{"label":"specs"}',tools=None):
    return SimpleNamespace(model=model,choices=[SimpleNamespace(message=SimpleNamespace(content=content,tool_calls=tools or []))])


class Client:
    def __init__(self,result=None,error=None):
        self.seen=[]
        self.result=result or response()
        self.error=error
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        self.seen.append(kwargs)
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize('model',['gpt-5','gpt-5.6','gpt-6-luna','gpt-6.1-sol'])
def test_reasoning_router_json_has_sufficient_budget_and_no_rejected_legacy_parameters(model):
    client=Client()
    events=[]
    runtime.create_completion(client,model=model,role='router',json_mode=True,temperature=0,
                              messages=[{'role':'user','content':'Synthetic classifier test'}],observer=lambda **kwargs:events.append(kwargs))
    request=client.seen[0]
    assert 'temperature' not in request and 'max_tokens' not in request
    assert request['max_completion_tokens']==2048
    assert request['reasoning_effort']=='low'
    assert request['response_format']=={'type':'json_object'}
    assert request['store'] is False
    assert events==[{'role':'router','requested_model':model,'provider_model':'provider-snapshot','succeeded':True,'error_code':None}]


def test_legacy_four_override_keeps_temperature_json_and_supported_token_limit():
    client=Client()
    runtime.create_completion(client,model='gpt-4.1-mini',role='router',json_mode=True,temperature=0,
                              messages=[],observer=lambda **kwargs:None)
    assert client.seen[0]['temperature']==0
    assert 'reasoning_effort' not in client.seen[0]
    assert client.seen[0]['max_completion_tokens']==2048
    assert client.seen[0]['store'] is False


@pytest.mark.parametrize('budget',[0,-1,32,float('inf'),'invalid'])
def test_invalid_or_tiny_reasoning_budgets_cannot_recreate_32_token_router_failure(budget):
    options=runtime.completion_options('gpt-6-luna',role='router',max_output_tokens=budget)
    assert 512<=options['max_completion_tokens']<=16384


def test_agent_default_and_reasoning_values_are_conservative():
    options=runtime.completion_options('gpt-6.1-sol',role='agent')
    assert options['reasoning_effort']=='medium' and options['max_completion_tokens']==3000
    assert runtime.completion_options('gpt-6.1-sol',reasoning_effort='max')['reasoning_effort']=='medium'
    for model in ('gpt-6.1-sol','gpt-6-astra'):
        assert runtime.completion_options(model,role='agent',reasoning_effort='none')['reasoning_effort']=='medium'
        assert runtime.completion_options(model,role='router',reasoning_effort='none')['reasoning_effort']=='low'


@pytest.mark.parametrize('model',['gpt-6.1-sol','gpt-6-astra'])
def test_responses_only_tools_are_rejected_before_any_chat_request(model):
    client=Client()
    with pytest.raises(ValueError,match='Responses API'):
        runtime.create_completion(client,model=model,tools=[{'type':'function'}],observer=lambda **kwargs:pytest.fail('No call was made'))
    assert client.seen==[]


@pytest.mark.parametrize('model',['gpt-6-sol','gpt-6-luna'])
def test_other_six_chat_tools_use_required_none_effort(model):
    client=Client()
    runtime.create_completion(client,model=model,tools=[{'type':'function'}],observer=lambda **kwargs:None)
    assert client.seen[0]['reasoning_effort']=='none'


def test_provider_errors_are_reraised_and_observation_contains_no_error_or_prompt_text():
    class AuthenticationError(Exception):
        status_code=401
    error=AuthenticationError('SECRET provider body and customer@example.invalid')
    client=Client(error=error)
    events=[]
    with pytest.raises(AuthenticationError):
        runtime.create_completion(client,model='gpt-6-luna',role='router',messages=[{'role':'user','content':'PRIVATE customer text'}],
                                  observer=lambda **kwargs:events.append(kwargs))
    assert events==[{'role':'router','requested_model':'gpt-6-luna','provider_model':None,'succeeded':False,'error_code':'authentication'}]
    assert 'SECRET' not in str(events) and 'PRIVATE' not in str(events)


def test_observer_failure_never_breaks_working_completion():
    client=Client()
    def fail(**kwargs):
        raise RuntimeError('Unavailable metrics database')
    assert runtime.create_completion(client,model='gpt-6-luna',role='router',observer=fail) is client.result


def test_no_provider_identity_is_invented_for_a_response_without_model():
    events=[]
    runtime.observe_model_call(role='router',requested_model='gpt-6-luna',response=response(model=None),observer=lambda **kwargs:events.append(kwargs))
    assert events[0]['provider_model'] is None


def test_invalid_provider_model_metadata_cannot_leak_customer_text_or_secrets():
    events=[]
    for invalid in ('private-customer@example.invalid','https://private.invalid/customer','sk-secret-key','model with customer text'):
        runtime.observe_model_call(role='router',requested_model='gpt-6-luna',response=response(model=invalid),observer=lambda **kwargs:events.append(kwargs))
    assert all(item['provider_model'] is None for item in events)
    assert 'private-customer' not in str(events) and 'sk-secret-key' not in str(events)


@pytest.mark.parametrize('override',[None,'gpt-4.1-mini','  '])
def test_config_defaults_and_explicit_legacy_override_have_honest_source_metadata(override):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    configuration=Path(__file__).resolve().parents[1]/'config.py'
    code="import json,runpy,sys,types; sys.modules['dotenv']=types.SimpleNamespace(load_dotenv=lambda **kwargs:None); result=runpy.run_path(sys.argv[1]); print(json.dumps(result['MODEL_CONFIGURATION']))"
    environment=dict(os.environ)
    environment.pop('OPENAI_AGENT_MODEL',None)
    environment.pop('OPENAI_ROUTER_MODEL',None)
    if override is not None:
        environment['OPENAI_ROUTER_MODEL']=override
    result=subprocess.run([sys.executable,'-I','-c',code,str(configuration)],env=environment,capture_output=True,text=True,check=True)
    metadata=json.loads(result.stdout)
    assert metadata['agent']=={'effective_model':'gpt-6.1-sol','default_model':'gpt-6.1-sol','source':'default'}
    if override and override.strip():
        assert metadata['router']=={'effective_model':'gpt-4.1-mini','default_model':'gpt-6-luna','source':'env'}
    else:
        assert metadata['router']=={'effective_model':'gpt-6-luna','default_model':'gpt-6-luna','source':'default'}


def test_responses_metadata_uses_actual_model_even_on_validation_error():
    events=[]
    raw=SimpleNamespace(model='resolved-agent',output_text='A synthetic answer')
    runtime.observe_model_call(role='agent',requested_model='gpt-6.1-sol',response=raw,error=ValueError('PRIVATE validation data'),observer=lambda **kwargs:events.append(kwargs))
    assert events==[{'role':'agent','requested_model':'gpt-6.1-sol','provider_model':'resolved-agent','succeeded':False,'error_code':'unexpected_error'}]


def test_empty_chat_content_is_observed_as_failure_but_tool_calls_are_valid():
    events=[]
    runtime.observe_model_call(role='router',requested_model='gpt-6-luna',response=response(content=''),observer=lambda **kwargs:events.append(kwargs))
    assert events[-1]['succeeded'] is False and events[-1]['error_code']=='empty_response'
    runtime.observe_model_call(role='agent',requested_model='gpt-6.1-sol',response=response(content='',tools=[object()]),observer=lambda **kwargs:events.append(kwargs))
    assert events[-1]['succeeded'] is True


def test_valid_responses_refusal_is_a_completed_call_not_a_quality_claim():
    events=[]
    raw=SimpleNamespace(model='gpt-6.1-sol',output_text='',output=[SimpleNamespace(type='message',content=[SimpleNamespace(type='refusal',refusal='I cannot provide that advice.')])])
    runtime.observe_model_call(role='agent',requested_model='gpt-6.1-sol',response=raw,observer=lambda **kwargs:events.append(kwargs))
    assert events[-1]['succeeded'] is True
    assert events[-1]['provider_model']=='gpt-6.1-sol'
    assert events[-1]['error_code'] is None
