"""General-chat tools must honor the same exact constraints as Sales chat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from app import agent_tools as tools
from sales_catalog import ProductSpecs
from sales_spec_index import ModelFitSpec


def product(name, *, price=3000, mechanism="4D"):
    return ProductSpecs(handle=name.lower().replace(' ', '-'), title=name, display_name=name,
                        vendor='Osaki', status='active', price_usd=price, massage_mechanism=mechanism,
                        track_type='SL-Track', zero_gravity='Yes', heating='Yes', airbag='Yes',
                        foot_roller='Yes', auto_programs='8', massage_styles='5')


def doc(name, price=3000, currency="USD", body=""):
    return Document(page_content=f"BASE PRICE ({currency}): ${price}\n{body}", metadata={"title":name,"currency":currency,"source":"synthetic fixture"})


class Retriever:
    def __init__(self, docs):
        self.docs=docs
    def search(self, *args, **kwargs):
        return self.docs


@pytest.fixture
def catalog(monkeypatch):
    import product_catalog, sales_catalog, sales_spec_index
    products = {}
    fits = {}
    monkeypatch.setattr(product_catalog, 'resolve_catalog_price', lambda name: None)
    monkeypatch.setattr(sales_catalog, 'resolve_product', lambda name: products.get(name))
    monkeypatch.setattr(sales_spec_index, 'lookup_fit_spec', lambda name: fits.get(name))
    return products, fits


def test_stated_budget_never_widens_thirty_percent(catalog):
    products, fits=catalog
    products.update({name:product(name,price=price) for name,price in [('Fixture Alpha',3000),('Fixture Beta',3900)]})
    result=tools.tool_recommend_chairs(products_retriever=Retriever([doc('Fixture Alpha',3000),doc('Fixture Beta',3900)]),
                                       user_need='Recommend a chair',budget_max=3000)
    assert 'Fixture Alpha' in result
    assert 'Fixture Beta' not in result
    assert 'maximum budget $3,000' in result


def test_latest_customer_budget_correction_overrides_stale_tool_argument(catalog):
    products, fits=catalog
    products.update({'Fixture Alpha':product('Fixture Alpha',price=2900),'Fixture Beta':product('Fixture Beta',price=4900)})
    result=tools.tool_recommend_chairs(products_retriever=Retriever([doc('Fixture Alpha',2900),doc('Fixture Beta',4900)]),
                                       user_need='My budget is $5000\nActually my maximum is $3000',budget_max=5000)
    assert 'Fixture Alpha' in result
    assert 'Fixture Beta' not in result


def test_exact_weight_and_4d_requirements_exclude_known_and_unknown_mismatches(catalog):
    products, fits=catalog
    for name,capacity,mech in [('Fixture Suitable',375,'4D'),('Fixture Light',330,'4D'),('Fixture Three',375,'3D'),('Fixture Unknown',None,'4D')]:
        products[name]=product(name,mechanism=mech)
        fits[name]=ModelFitSpec(name=name,max_user_lb=capacity,door_asm_in=30)
    result=tools.tool_recommend_chairs(products_retriever=Retriever([doc(name) for name in products]),
                                       user_need='Need a chair for 350 lbs with a 4D massage system')
    assert 'Fixture Suitable' in result
    assert 'Fixture Light' not in result
    assert 'Fixture Three' not in result
    assert 'Fixture Unknown' not in result


def test_doorway_limit_requires_published_assembled_fit(catalog):
    products,fits=catalog
    for name,width in [('Fixture Small',29),('Fixture Wide',35),('Fixture Unknown',None)]:
        products[name]=product(name)
        fits[name]=ModelFitSpec(name=name,max_user_lb=375,door_asm_in=width,door_dis_in=25)
    result=tools.tool_recommend_chairs(products_retriever=Retriever([doc(name) for name in products]),
                                       user_need='I have a 30 inch doorway')
    assert 'Fixture Small' in result
    assert 'Fixture Wide' not in result
    assert 'Fixture Unknown' not in result


def test_exact_model_specs_do_not_leak_other_model_values(catalog):
    products,fits=catalog
    products['Fixture Alpha']=product('Fixture Alpha')
    products['Fixture Beta']=product('Fixture Beta')
    result=tools.tool_search_chair_specs(products_retriever=Retriever([
        doc('Fixture Beta',6000,'USD','Maximum User Weight: 375 lbs'),
        doc('Fixture Alpha',3000,'USD','Maximum User Weight: 260 lbs'),
    ]),query='maximum user weight',model_name='Fixture Alpha')
    assert 'Fixture Alpha' in result
    assert 'Fixture Beta' not in result
    assert '375 lbs' not in result
    facts=json.loads(next(line.split(': ',1)[1] for line in result.splitlines() if line.startswith('MODEL_FACTS_JSON:')))
    assert facts['model']=='Fixture Alpha'
    assert facts['currency']=='USD'


def test_spec_fallback_never_substitutes_nearby_model_or_foreign_currency(catalog):
    products,fits=catalog
    products['Fixture Alpha']=product('Fixture Alpha')
    other=tools.tool_search_chair_specs(products_retriever=Retriever([doc('Fixture Beta')]),query='price',model_name='Fixture Alpha')
    euros=tools.tool_search_chair_specs(products_retriever=Retriever([doc('Fixture Alpha',3000,'EUR')]),query='price',model_name='Fixture Alpha')
    assert other.startswith('NO_RESULTS:')
    assert euros.startswith('NO_RESULTS:')


def test_unknown_fit_does_not_get_promoted_to_a_safe_recommendation(catalog):
    products,fits=catalog
    products['Fixture Unknown']=product('Fixture Unknown')
    result=tools.tool_recommend_chairs(products_retriever=Retriever([doc('Fixture Unknown')]),user_need='350 lb, 4D, 30 inch doorway')
    assert result.startswith('NO_RESULTS:')
    assert 'Do not relax' in result
