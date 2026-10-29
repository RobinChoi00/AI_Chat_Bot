"""Product generations and published feature claims must survive shopper wording."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sales_catalog import _published_features, product_feature_answer, resolve_product
from sales_compare import lookup_shop_models, split_compare_terms


@pytest.mark.parametrize("query,handle", [
    ("What is the price of the Maestro LE 2.0?", "osaki-maestro-le-2"),
    ("Does the Maestro LE 2.0 have Bluetooth speakers?", "osaki-maestro-le-2"),
    ("No, I mean the Osaki 4D Maestro LE 2.0, not the older OS-Pro Maestro LE.", "osaki-maestro-le-2"),
    ("Please use the old OS-Pro Maestro LE, not the Maestro LE 2.0.", "osaki-os-pro-maestro-le"),
    ("What does Maestro LE 2 cost?", "osaki-maestro-le-2"),
    ("Can I try the Solo Flex on Saturday?", "osaki-solo-flex-4d"),
    ("What colors does the Solo Flex come in?", "osaki-solo-flex-4d"),
    ("What is the price of Bravo Duo Flex?", "osaki-bravo-duo-flex-chair"),
])
def test_sentence_and_correction_keep_exact_product_identity(query, handle):
    assert resolve_product(query).handle == handle


@pytest.mark.parametrize("query", [
    "Maestro LE 3.0",
    "What is the price of Maestro LE 2.1?",
    "OS-4000T or Hypnos",
    "Maestro LE 2.0 or Solo Flex",
])
def test_unknown_generation_or_alternatives_do_not_become_another_model(query):
    assert resolve_product(query) is None


def test_comparison_lookup_keeps_decimal_generations_and_short_names():
    assert lookup_shop_models("Maestro LE 2").unique.handle == "osaki-maestro-le-2"
    assert lookup_shop_models("Maestro LE 3.0").matches == ()
    assert lookup_shop_models("Maestro LE 2.1").matches == ()
    assert len(lookup_shop_models("Maestro").matches) > 1


def test_exact_generation_has_its_own_price_and_published_features():
    current = resolve_product("Maestro LE 2.0")
    older = resolve_product("OS-Pro Maestro LE")
    assert current.handle != older.handle
    assert current.as_public_dict()["catalog_handle"] == current.handle
    public = current.as_public_dict()
    assert public["specs"]["bluetooth_speakers"] == "Yes"
    assert public["specs"]["body_scanning"] == "Yes"
    assert public["feature_sources"]["body_scanning"] == "Shopify export: product description"
    reply = product_feature_answer(current, "What colors, Bluetooth speakers and body scanning are included?")
    assert current.display_name in reply
    assert all(color in reply for color in current.colors)
    assert "**Bluetooth speakers**: Yes" in reply
    assert "**Body scanning**: Yes" in reply
    assert older.display_name not in reply


def test_missing_feature_data_is_unverified_instead_of_no():
    unknown = replace(resolve_product("Solo Flex"), colors=(), bluetooth_speakers="", body_scanning="", feature_sources=())
    reply = product_feature_answer(unknown, "What colors and Bluetooth speakers and body scanning are included?")
    assert reply.count("not verified") == 3
    assert ": No" not in reply


@pytest.mark.parametrize("body,expected", [
    ("<li>Surround Sound Bluetooth Speakers</li><li>Computerized Body Scanning</li>", ("Yes", "Yes")),
    ("<p>No Bluetooth speakers</p><p>Body scanning is not supported</p>", ("No", "No")),
    ("<p>Does this model have Bluetooth speakers?</p>", ("", "")),
    ("<li>Optional Bluetooth speakers</li>", ("", "")),
    ("<p>Bluetooth speakers</p><p>No Bluetooth speakers</p>", ("", "")),
    ("<script>Bluetooth speakers</script><p>Comfortable cushioning</p>", ("", "")),
])
def test_description_features_require_an_unambiguous_published_statement(body, expected):
    bluetooth, scanning, _sources = _published_features({"Body (HTML)": body})
    assert (bluetooth, scanning) == expected


def test_conflicting_metafield_and_description_do_not_establish_a_feature():
    bluetooth, _scanning, sources = _published_features({
        "Bluetooth Speaker (product.metafields.custom.bluetooth_speaker)": "YES",
        "Body (HTML)": "<p>No Bluetooth speakers</p>",
    })
    assert bluetooth == ""
    assert "bluetooth_speakers" not in dict(sources)


@pytest.mark.parametrize("question", [
    "Does the Maestro LE 2.0 scan my body?",
    "Does it scan the body automatically?",
    "Is my body automatically scanned?",
])
def test_body_scanning_paraphrases_use_the_published_exact_model(question):
    reply = product_feature_answer(resolve_product("Maestro LE 2.0"), question)
    assert "**Body scanning**: Yes, according to the published catalog" in reply
    assert "Osaki 4D Maestro LE 2.0" in reply


@pytest.mark.parametrize("question,label,model", [
    ("Can I control the Maestro LE 2.0 with an app?", "App control", "Maestro LE 2.0"),
    ("Does it support mobile application control?", "App control", "Maestro LE 2.0"),
    ("Does the Hypnos have voice control?", "Voice control", "Hypnos"),
    ("Can I use voice commands?", "Voice control", "Hypnos"),
])
def test_unverified_control_features_receive_a_direct_verification_boundary(question, label, model):
    reply = product_feature_answer(resolve_product(model), question)
    assert f"**{label}**: not verified in this catalog" in reply
    assert f"**{label}**: Yes" not in reply
    assert f"**{label}**: No" not in reply


@pytest.mark.parametrize("message,expected", [
    ("I want to compare Osaki OS-Champ and Titan TP-Epic 4D.", ["osaki-os-champ", "titan-tp-epic-4d"]),
    ("Please compare Titan TP-Epic 4D and Osaki OS-Champ for my father.", ["titan-tp-epic-4d", "osaki-os-champ"]),
    ("The two exact models are Osaki OS-Champ and Titan TP-Epic 4D. Please compare only those.", ["osaki-os-champ", "titan-tp-epic-4d"]),
    ("We would like to compare Solo Flex and Bravo Duo Flex for our parents.", ["osaki-solo-flex-4d", "osaki-bravo-duo-flex-chair"]),
])
def test_comparison_terms_exclude_request_framing_and_recipient(message, expected):
    pair = split_compare_terms(message)
    assert pair is not None
    assert [lookup_shop_models(term).unique.handle for term in pair] == expected


@pytest.mark.parametrize("model,question,expected", [
    ("Maestro LE 2.0", "Can I control the Maestro LE 2.0 with an app?", "**App control**: not verified"),
    ("Maestro LE 2.0", "Does the Maestro LE 2.0 scan my body?", "**Body scanning**: Yes"),
    ("Hypnos", "Does the Hypnos have voice control?", "**Voice control**: not verified"),
])
def test_feature_followup_api_keeps_exact_product_and_does_not_start_an_interview(model, question, expected):
    from script.simulate_chatbot_quality import offline_api

    with offline_api() as client:
        sid = "catalog-feature-regression"
        client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": "osakiusa.com"})
        body = {"session_id": sid, "domain": "osakiusa.com", "message": f"What does {model} cost?"}
        assert client.post("/api/v1/sales/chat", json=body).status_code == 200
        body["message"] = question
        response = client.post("/api/v1/sales/chat", json=body)
        assert response.status_code == 200
        reply = response.json()
        assert expected in reply["reply"]
        assert {p["model"] for p in reply["products"]} == {resolve_product(model).display_name}
        assert "user height" not in reply["reply"].lower()
        assert not client.attempted_external_delivery


@pytest.mark.parametrize("message", [
    "I want to compare Osaki OS-Champ and Titan TP-Epic 4D.",
    "Please compare Titan TP-Epic 4D and Osaki OS-Champ for my father.",
    "The two exact models are Osaki OS-Champ and Titan TP-Epic 4D. Please compare only those.",
])
def test_conversational_comparison_api_retains_both_named_candidates(message):
    from script.simulate_chatbot_quality import offline_api

    with offline_api() as client:
        sid = "catalog-compare-regression"
        client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": "osakiusa.com"})
        response = client.post("/api/v1/sales/chat", json={"session_id": sid, "domain": "osakiusa.com", "message": message})
        assert response.status_code == 200
        reply = response.json()
        assert {p["model"] for p in reply["products"]} == {"Osaki OS-Champ", "Titan TP-Epic 4D"}
        assert not client.attempted_external_delivery
