"""The quality runner must exercise complete API conversations and fail closed."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from script import evaluate_llm_quality as ev


def case(surface="general", turns=None):
    return {"id": "synthetic-test", "surface": surface, "provenance": "synthetic", "turns": turns or [{"message": "A question", "expect": {"contains": ["answer"]}}]}


def client_for(handler, *, observe_setup=False):
    def with_consent(request):
        # Most tests below focus on evaluated turns. Setup has separate tests.
        if not observe_setup and request.url.path.endswith("/consent"):
            return httpx.Response(200, json={"ok": True})
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(with_consent))


def test_uses_actual_runtime_config_name(monkeypatch):
    import config
    monkeypatch.setattr(config, "AGENT_MODEL", "configured-model")
    assert ev._get_model() == "configured-model"


def test_legacy_cases_are_converted_with_assertions_and_provenance():
    source = json.loads(ev.CASES_PATH.read_text())
    converted = ev.legacy_conversations(source)
    assert len(converted) == 23
    assert all(item["provenance"] == "synthetic" and item["surface"] == "general" for item in converted)
    assert converted[0]["turns"][0]["expect"]["expected_tool"] == "get_showroom_info"


def test_repository_fixtures_have_real_assertions_and_multiple_turn_surfaces():
    source = json.loads(ev.CONVERSATIONS_PATH.read_text())
    assert ev.validate_conversations(source) == []
    assert {row["surface"] for row in source["cases"]} == {"general", "sales", "warranty"}
    assert sum(len(row["turns"]) > 1 for row in source["cases"]) >= 10
    assert {row["provenance"] for row in source["cases"]} == {"synthetic", "reviewed_anonymized"}
    reviewed=[row for row in source['cases'] if row['provenance']=='reviewed_anonymized']
    assert len(reviewed)>=5
    assert all(row.get('source_ref') and row.get('adaptation') for row in reviewed)


def test_fixture_validation_rejects_empty_no_assertion_and_duplicate_cases():
    assert ev.validate_conversations({"cases": []})
    source = {"cases": [case(), case(turns=[{"message": "hi", "expect": {}}])]}
    errors = ev.validate_conversations(source)
    assert any("duplicated" in error for error in errors)
    assert any("expectations" in error for error in errors)


def test_fixture_validation_rejects_vacuous_or_mistyped_assertions():
    spec = case(turns=[{"message": "hi", "expect": {"contains": "answer", "json_equals": {}, "expected_handoff": "false", "max_products": -1}}])
    assert len(ev.validate_conversations({"cases": [spec]})) == 4


@pytest.mark.parametrize("surface,sequence", [("sales", ["lookup"]), ("general", "lookup"), ("general", [""]), ("general", [None])])
def test_tool_sequence_validation_rejects_unsupported_surface_or_malformed_names(surface, sequence):
    spec = case(surface, [{"message": "A question", "expect": {"expected_tool_sequence": sequence}}])
    assert ev.validate_conversations({"cases": [spec]})


def test_tool_sequence_allows_an_explicit_no_tool_expectation():
    spec = case(turns=[{"message": "A greeting", "expect": {"expected_tool_sequence": []}}])
    assert ev.validate_conversations({"cases": [spec]}) == []
    assert ev._needs_trace(spec["turns"][0]["expect"])


@pytest.mark.parametrize("target", ["https://production.example.com", "http://127.0.0.1@evil.example", "http://localhost:8000/path", "http://user:password@localhost:8000", "file:///tmp/db"])
def test_unsafe_or_remote_targets_need_explicit_authorization(target):
    with pytest.raises(ValueError):
        ev.validate_target(target, False)


def test_remote_opt_in_preserves_validation():
    assert ev.validate_target("https://staging.example.com/", True) == "https://staging.example.com"
    with pytest.raises(ValueError):
        ev.validate_target("https://secret@staging.example.com", True)


def test_general_route_reads_stream_and_replays_full_history():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"Complete answer from tools")
    spec = case(turns=[{"message": "first", "expect": {"contains": ["Complete answer"]}}, {"message": "follow-up", "expect": {"contains": ["Complete answer"]}}])
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com")
    assert result["passed"]
    assert all(request.url.path == "/api/v1/chat" for request in requests)
    bodies = [json.loads(request.content) for request in requests]
    assert bodies[0]["session_id"].startswith("eval-")
    assert bodies[0]["session_id"] == bodies[1]["session_id"]
    assert bodies[1]["chat_history"] == [{"role": "user", "content": "first"}, {"role": "assistant", "content": "Complete answer from tools"}]
    assert "response_sha256" in result["turns"][0]
    assert "Complete answer" not in json.dumps(result)  # no raw provider/knowledge text in reports


@pytest.mark.parametrize("surface,path", [("general", "/api/v1/chat"), ("sales", "/api/v1/sales/chat")])
def test_public_chat_records_session_consent_before_first_customer_turn(surface, path):
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/consent"):
            return httpx.Response(200, json={"consent_recorded": True})
        return httpx.Response(200, json={"reply": "An answer"}) if surface == "sales" else httpx.Response(200, content=b"An answer")
    with client_for(handler, observe_setup=True) as client:
        result = ev.run_case(case(surface), client, base_url="http://localhost", domain="osakiusa.com")
    assert result["passed"]
    consent, chat = seen
    session = json.loads(chat.content)["session_id"]
    assert chat.url.path == path
    assert consent.url.path == f"/api/v1/warranty/session/{session}/consent"
    assert json.loads(consent.content) == {"domain": "osakiusa.com"}


@pytest.mark.parametrize("surface", ["general", "sales"])
def test_failed_consent_stops_before_any_customer_turn_and_is_counted(surface):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(503, text="private backend detail")
    with client_for(handler, observe_setup=True) as client:
        result = ev.run_case(case(surface), client, base_url="http://localhost", domain="osakiusa.com")
    assert len(seen) == 1 and seen[0].url.path.endswith("/consent")
    assert not result["passed"] and result["unrun_turns"] == 1
    report = ev.summarize([result])
    assert not report["quality_evaluated"]
    assert report["assertion_failure_counts"]["request_or_availability"] == 1
    assert "private backend detail" not in json.dumps(report)


def test_tool_call_alone_cannot_pass_without_final_answer():
    spec = case(turns=[{"message": "price?", "expect": {"expected_tool": "search_chair_specs"}}])
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"tools_called": ["search_chair_specs"], "model": "candidate"})
        return httpx.Response(200, content=b"")
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com", admin_key="test-key")
    assert not result["passed"]
    assert "empty final response" in result["turns"][0]["errors"]


def test_tool_checks_use_authenticated_runtime_evidence_and_model_mismatch_fails():
    seen = []
    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"tools_called": ["search_chair_specs"], "model": "old-model"})
        return httpx.Response(200, content=b"The final grounded answer")
    with client_for(handler) as client:
        result = ev.run_case(case(turns=[{"message": "price?", "expect": {"expected_tool": "search_chair_specs"}}]), client, base_url="http://localhost", domain="osakiusa.com", admin_key="test-key", expected_model="candidate")
    assert not result["passed"]
    assert seen[1].headers["X-Admin-Key"] == "test-key"
    assert seen[1].url.path.startswith("/admin/traces/eval-")
    assert "model differs" in result["turns"][0]["errors"][0]
    assert "test-key" not in json.dumps(result)


@pytest.mark.parametrize("evidence", [
    {"model": "gpt-6.1-sol"},
    {"model": "gpt-6.1-sol", "cache_hit": True, "provider_models": [], "model_calls_successful": 0},
    {"model": "gpt-6.1-sol", "requested_models": ["gpt-6.1-sol"], "provider_models": ["gpt-6.1-sol"], "model_calls_successful": 0},
    {"model": "gpt-6.1-sol", "requested_models": ["gpt-6.1-sol"], "provider_models": ["gpt-6.1-sol"], "model_calls_successful": 2, "model_calls_unverified": 1},
])
def test_expected_model_requires_actual_successful_provider_evidence(evidence):
    def handler(request):
        return httpx.Response(200, json=evidence) if request.method == "GET" else httpx.Response(200, content=b"An answer")
    with client_for(handler) as client:
        result = ev.run_case(case(), client, base_url="http://localhost", domain="osakiusa.com", admin_key="synthetic-key", expected_model="gpt-6.1-sol")
    assert not result["passed"]
    assert "successful provider model execution was not verified" in result["turns"][0]["errors"]


@pytest.mark.parametrize("provider", ["gpt-6.1-sol", "gpt-6.1-sol-2026-10-23"])
def test_expected_model_accepts_actual_exact_or_iso_dated_provider_snapshot(provider):
    evidence = {"model": "gpt-6.1-sol", "requested_models": ["gpt-6.1-sol"], "provider_models": [provider], "model_calls_successful": 1, "model_calls_unverified": 0}
    def handler(request):
        return httpx.Response(200, json=evidence) if request.method == "GET" else httpx.Response(200, content=b"An answer")
    with client_for(handler) as client:
        result = ev.run_case(case(), client, base_url="http://localhost", domain="osakiusa.com", admin_key="synthetic-key", expected_model="gpt-6.1-sol")
    assert result["passed"]
    row = result["turns"][0]
    assert row["requested_model"] == "gpt-6.1-sol"
    assert row["provider_models"] == [provider]
    assert row["observed_model"] == provider


@pytest.mark.parametrize("provider", ["gpt-6-luna", "gpt-6.1-sol-other", "gpt-6.1-sol-2026-99-99", "gpt-6.1-sol-2026-10-23-extra"])
def test_expected_model_rejects_other_models_and_arbitrary_alias_suffixes(provider):
    _, errors = ev.model_evidence({"model": "gpt-6.1-sol", "requested_models": ["gpt-6.1-sol"], "provider_models": [provider], "model_calls_successful": 1}, "gpt-6.1-sol")
    assert "provider-returned model differs from expected model" in errors


def test_sales_reuses_session_and_scores_structured_handoff_and_budget():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/api/v1/sales/chat"
        return httpx.Response(200, json={"reply": "Here is an answer", "intent": "recommend", "handoff": False, "products": [{"price_usd": 4000}]})
    spec = case("sales", [{"message": "under 5000", "expect": {"expected_intent": "recommend", "max_product_price": 5000}}, {"message": "Actually under 3000", "expect": {"max_product_price": 3000}}])
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com")
    assert seen[0]["session_id"] == seen[1]["session_id"]
    assert seen[0]["channel"] == "test"
    assert result["turns"][0]["passed"]
    assert not result["turns"][1]["passed"]
    assert not result["passed"]


def test_completed_preference_fixture_cannot_pass_without_recommendations():
    source = json.loads(ev.CONVERSATIONS_PATH.read_text())
    spec = next(row for row in source["cases"] if row["id"] == "sales-complete-fit-recommendation")
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "An answer without recommendations", "intent": "recommend", "handoff": False, "products": []})
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com")
    assert len(seen) == 3
    assert seen[-1]["payload"] == "recommend:goal:full_body"
    assert not result["passed"]
    assert "not enough recommended products" in result["turns"][-1]["errors"]


def test_warranty_runs_real_session_setup_and_uses_returned_ticket_id():
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/consent") or request.url.path.endswith("/contact-email"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"ticket": {"ticket_id": "generated-ticket", "model_name": "Solo Flex", "status": "in_progress"}})
    spec = case("warranty", [{"action": "register-model", "body": {"model": "Solo Flex"}, "expect": {"json_equals": {"ticket.model_name": "Solo Flex"}}}, {"action": "answer", "body": {"answer": "defect"}, "expect": {"json_equals": {"ticket.status": "in_progress"}}}])
    with client_for(handler, observe_setup=True) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com")
    assert result["passed"]
    assert seen[0].url.path.endswith("/consent")
    assert json.loads(seen[1].content)["customer_email"].endswith("@example.invalid")
    assert seen[2].url.path.endswith("/register-model")
    assert seen[3].url.path == "/api/v1/warranty/generated-ticket/answer"


def test_http_error_is_failure_and_response_body_is_not_leaked():
    def handler(request):
        return httpx.Response(500, text="sensitive provider token: do-not-print")
    with client_for(handler) as client:
        result = ev.run_case(case(), client, base_url="http://localhost", domain="osakiusa.com")
    assert not result["passed"]
    assert "do-not-print" not in json.dumps(result)


def test_unrun_turns_are_distinguished_from_failed_turns():
    def handler(request):
        return httpx.Response(503)
    spec = case(turns=[{"message": "first", "expect": {"contains": ["answer"]}}, {"message": "second", "expect": {"contains": ["answer"]}}])
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com")
    assert len(result["turns"]) == 1
    assert result["unrun_turns"] == 1
    assert ev.summarize([result])["unrun_turns"] == 1


def test_sse_final_text_is_concatenated():
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b'data: {"text":"Grounded "}\n\ndata: {"content":"answer"}\n\ndata: [DONE]\n', request=httpx.Request("POST", "http://localhost/api/v1/chat"))
    assert ev._read_stream(response) == "Grounded answer"


def test_actual_general_chat_raw_stream_is_not_discarded_as_empty_sse():
    text = "고객에게 보이는 정상 답변\n\nThe grounded final answer."
    response = httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"},
                              content=text.encode(), request=httpx.Request("POST", "http://localhost/api/v1/chat"))
    assert ev._read_stream(response) == text


@pytest.mark.parametrize('reply',["We could not complete this lookup. Please try again.","No pudimos completar esta consulta.","현재 조회를 완료하지 못했습니다."])
def test_localized_application_error_cannot_count_as_quality_success(reply):
    errors=ev.check_answer(reply,{}, {'not_contains':['guaranteed to cure']})
    assert 'application error fallback was returned' in errors


def test_recommendation_metadata_rejects_wrong_or_missing_fit_and_mechanism():
    checks={'min_products':1,'min_product_capacity_lb':350,'required_product_mechanism':'4D','max_product_price':5000}
    valid={'price_usd':4500,'fit':{'maximum_user_weight_lb':375},'specs':{'massage_mechanism':'4D'}}
    assert ev.check_answer('Matching chair',{'products':[valid]},checks)==[]
    for product in ({'price_usd':4500}, {**valid,'fit':{'maximum_user_weight_lb':330}}, {**valid,'specs':{'massage_mechanism':'3D'}}):
        assert ev.check_answer('Matching chair',{'products':[product]},checks)
    assert ev.check_answer('No products',{'products':[]},checks)


def test_recorded_tool_failure_cannot_satisfy_tool_execution_check():
    errors=ev.check_answer('A fluent reply',{}, {'expected_tool':'search_chair_specs'},
                           {'tools_called':['search_chair_specs'],'turns':[{'tool_name':'search_chair_specs','tool_success':False}]})
    assert 'required tool did not execute successfully' in errors


@pytest.mark.parametrize("names", [["search_chair_specs", "search_chair_specs"], ["capture_sales_lead", "search_chair_specs"], ["recommend_chairs"], []])
def test_exact_tool_sequence_rejects_duplicate_extra_wrong_and_missing_actions(names):
    trace = {"turns": [{"tool_name": name, "tool_success": True} for name in names]}
    errors = ev.check_answer("A fluent answer", {}, {"expected_tool_sequence": ["search_chair_specs"]}, trace)
    assert "executed tool sequence differs from expected sequence" in errors


@pytest.mark.parametrize("trace", [None, {}, {"turns": None}, {"turns": ["malformed"]}])
def test_missing_execution_records_cannot_prove_no_tool_was_used(trace):
    assert ev.check_answer("A fluent answer", {}, {"expected_tool_sequence": []}, trace)


def test_exact_tool_sequence_requires_successful_execution_and_authenticated_trace():
    checks = {"expected_tool_sequence": ["search_chair_specs", "get_warranty_or_policy"]}
    trace = {"turns": [{"tool_name": name, "tool_success": True} for name in checks["expected_tool_sequence"]]}
    assert ev.check_answer("A grounded answer", {}, checks, trace) == []
    trace["turns"][1]["tool_success"] = False
    assert "tool sequence contains an unsuccessful execution" in ev.check_answer("A fluent answer", {}, checks, trace)
    requested = []
    def handler(request):
        requested.append(request)
        return httpx.Response(200, json={"turns": []}) if request.method == "GET" else httpx.Response(200, content=b"A grounded answer")
    spec = case(turns=[{"message": "A greeting", "expect": {"expected_tool_sequence": []}}])
    with client_for(handler) as client:
        result = ev.run_case(spec, client, base_url="http://localhost", domain="osakiusa.com", admin_key="synthetic-key")
    assert result["passed"]
    assert requested[1].url.path.startswith("/admin/traces/eval-")
    assert requested[1].headers["X-Admin-Key"] == "synthetic-key"


def test_expected_source_requires_successful_available_exact_model_evidence():
    checks={'expected_sources':[{'source_type':'catalog','model':'Osaki Solo Flex 4D','currency':'USD'}]}
    correct={'tool_success':True,'source_evidence':[{'source_type':'catalog','model':'Osaki Solo Flex 4D','currency':'USD','available':True}]}
    assert ev.check_answer('The exact model answer',{},checks,{'turns':[correct]})==[]
    for wrong in ({**correct,'tool_success':False}, {'tool_success':True,'source_evidence':[{'source_type':'catalog','model':'Different Chair','currency':'USD','available':True}]}, {'tool_success':True,'source_evidence':[{'source_type':'catalog','model':'Osaki Solo Flex 4D','currency':'USD','available':False}]}):
        assert 'required source/model evidence not observed' in ev.check_answer('Fluent reply',{},checks,{'turns':[wrong]})
    assert ev.check_answer('Fluent reply',{},checks,None)


def test_source_only_expectation_fetches_authenticated_trace():
    requested=[]
    def handler(request):
        requested.append(request)
        if request.method=='GET':
            return httpx.Response(200,json={'turns':[{'tool_success':True,'source_evidence':[{'source_type':'catalog','model':'Osaki Solo Flex 4D','available':True}]}]})
        return httpx.Response(200,content=b'Exact source answer')
    spec=case(turns=[{'message':'A source question','expect':{'expected_sources':[{'source_type':'catalog','model':'Osaki Solo Flex 4D'}]}}])
    with client_for(handler) as client:
        result=ev.run_case(spec,client,base_url='http://localhost',domain='osakiusa.com',admin_key='synthetic-key')
    assert result['passed']
    assert requested[1].headers['X-Admin-Key']=='synthetic-key'


def test_dialogue_goal_and_loop_checks_fail_even_if_each_turn_passes():
    spec=case('sales',[{'message':'Question one','expect':{'not_contains':['guaranteed cure']}},
                       {'message':'Please answer that question','expect':{'not_contains':['guaranteed cure']}}])
    spec['dialogue_expect']={'max_identical_assistant_replies':1,'final_contains_any':['foot rollers']}
    with client_for(lambda request:httpx.Response(200,json={'reply':'What is your height?','intent':'recommend'})) as client:
        result=ev.run_case(spec,client,base_url='http://localhost',domain='osakiusa.com')
    assert all(turn['passed'] for turn in result['turns'])
    assert not result['passed']
    assert set(result['dialogue_errors'])=={'final dialogue goal was not demonstrated','identical assistant reply loop exceeded limit'}
    assert 'What is your height' not in json.dumps(result)


def test_empty_held_out_split_fails_without_network_or_quality_claim(capsys,monkeypatch):
    monkeypatch.setattr(httpx,'Client',lambda *args,**kwargs:pytest.fail('No cases means no request'))
    assert ev.main(['--dry-run','--split','held_out'])==1
    result=json.loads(capsys.readouterr().out)
    assert result['quality_evaluated'] is False
    assert 'no cases' in result['error']


def test_all_request_failures_do_not_claim_quality_was_measured():
    spec=case()
    with client_for(lambda request:httpx.Response(503)) as client:
        result=ev.run_case(spec,client,base_url='http://localhost',domain='osakiusa.com')
    summary=ev.summarize([result])
    assert summary['status']=='fail' and summary['quality_evaluated'] is False
    assert summary['evaluated_turns']==0
    assert summary['assertion_failure_counts']['request_or_availability']==1


def test_audit_capacity_failure_cannot_pass_on_fluent_wrong_recommendations():
    source=json.loads(ev.CONVERSATIONS_PATH.read_text())
    spec=next(row for row in source['cases'] if row['id']=='reviewed-sales-628-exact-capacity-and-4d')
    def handler(request):
        return httpx.Response(200,json={'reply':'Here is a matching budget chair','intent':'recommend','products':[
            {'price_usd':4500,'fit':{'maximum_user_weight_lb':330},'specs':{'massage_mechanism':'3D'}}]})
    with client_for(handler) as client:
        result=ev.run_case(spec,client,base_url='http://localhost',domain='osakiusa.com')
    assert not result['passed']
    assert all(not turn['passed'] for turn in result['turns'])


def test_summary_has_percentiles_no_test_failure_and_comparable_baseline():
    results = [{"id": "a", "passed": True, "turns": [{"latency_ms": 10}, {"latency_ms": 100}]}, {"id": "b", "passed": False, "turns": [{"latency_ms": 20}]}]
    report = ev.summarize(results, {"cases": [{"id": "a", "passed": True}, {"id": "b", "passed": True}]}, max_p95_ms=50)
    assert report["status"] == "fail"
    assert report["regressions"] == ["b"]
    assert report["latency_ms"] == {"p50": 20, "p95": 100, "samples": 3, "budget_failed": True}
    assert ev.summarize([])["status"] == "fail"
    assert ev.summarize(results, {"cases": []})["comparison_errors"]


def test_dry_run_does_not_call_http_and_is_not_quality_success(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: pytest.fail("dry-run must not create a network client"))
    assert ev.main(["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "structure_valid"
    assert report["quality_evaluated"] is False
    assert report["total_cases"] == 23 + len(json.loads(ev.CONVERSATIONS_PATH.read_text())["cases"])


def test_no_implicit_live_run():
    with pytest.raises(SystemExit) as result:
        ev.main([])
    assert result.value.code == 2


def test_incompatible_baseline_is_rejected_before_creating_network_client(tmp_path, capsys, monkeypatch):
    source = {"cases": [case("sales")]}
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(source))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"mode": "dry_run", "quality_evaluated": False}))
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: pytest.fail("invalid baseline must fail before requests"))
    assert ev.main(["--live", "--suite", "conversations", "--cases", str(cases_path), "--baseline", str(baseline)]) == 1
    assert "completed live API evaluation" in capsys.readouterr().out
