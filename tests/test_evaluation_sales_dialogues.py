"""Replay reviewed failure patterns through real APIs, with no live services.

This exercises stored session state and deterministic Sales replies. It is a
regression check, not an estimate of LLM quality or customer resolution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from script import evaluate_llm_quality as evaluation


@pytest.fixture
def isolated_sales_api(monkeypatch):
    monkeypatch.setenv('SALES_INTENT_LLM','0')
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    import requests
    import smtplib
    import sales_agent
    import sales_router
    import warranty_models
    monkeypatch.setattr(requests.sessions.Session,'request',lambda *args,**kwargs:pytest.fail('No outbound customer or stock requests'))
    monkeypatch.setattr(smtplib,'SMTP',lambda *args,**kwargs:pytest.fail('No actual emails'))
    monkeypatch.setattr(sales_agent,'fetch_live_stock',lambda *args,**kwargs:None)
    monkeypatch.setattr(sales_agent,'fetch_stock_many',lambda *args,**kwargs:{})
    database=create_engine('sqlite:///:memory:',connect_args={'check_same_thread':False},poolclass=StaticPool)
    warranty_models.Base.metadata.create_all(bind=database)
    monkeypatch.setattr(warranty_models,'_engine',database)
    monkeypatch.setattr(warranty_models,'_SessionFactory',sessionmaker(bind=database,expire_on_commit=False))
    app=FastAPI()
    app.include_router(sales_router.router)
    with TestClient(app) as client:
        yield client
    database.dispose()


@pytest.mark.parametrize('case_id',[
    'reviewed-sales-530-direct-foot-roller-no-interview',
    'reviewed-sales-625-replacement-remote-not-new-chair',
    'reviewed-sales-631-positive-purchase-feedback',
    'reviewed-sales-628-exact-capacity-and-4d',
])
def test_reviewed_sales_dialogue_contract_through_isolated_api(isolated_sales_api,case_id):
    fixtures=json.loads(evaluation.CONVERSATIONS_PATH.read_text())
    case=next(row for row in fixtures['cases'] if row['id']==case_id)
    result=evaluation.run_case(case,isolated_sales_api,base_url='http://testserver',domain='osakiusa.com')
    assert result['passed'],result
    assert result['split']=='regression'
