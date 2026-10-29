"""Broader language audit: preferences, conversational state and localized APIs.

These are customer contracts, not examples whose expected answers were inferred
from the implementation. Product names, catalog facts and source policy quotes
are allowed to remain in their source language.
"""
from __future__ import annotations

import re

from script.quality_v2_support import Scenario, expect, require


def scenarios() -> list[Scenario]:
    import answer_guard
    import customer_safety
    import sales_language
    from sales_constraints import extract_constraints
    from sales_models import get_session_collected

    rows = []

    def add(cid, category, family, title, run, inputs, level="function", severity="medium"):
        rows.append(Scenario("language-v2-" + cid, category, family, title, severity, run, inputs, level))

    def preference(cid, family, text, expected, *, surface="general", previous="en", history=()):
        def run(_client):
            if surface == "general":
                observed = answer_guard.conversation_language(text, list(history))
            elif surface == "sales":
                observed = sales_language.customer_language(text, previous)
            else:
                observed = customer_safety.requested_customer_language(text)
            expect(observed, expected, f"{surface} must honor requested language, not incidental wording")
            return {"message": text, "history": list(history), "previous_language": previous,
                    "surface": surface, "language": observed}
        add(cid, "language_preferences", family, "Customer language preference is preserved or deliberately changed",
            run, {"message": text, "surface": surface, "previous": previous, "history": list(history), "expected": expected})

    # Different wording and surface, not repeated copies of one successful switch.
    for cid, text, expected in [
        ("explicit-uppercase", "PLEASE ANSWER IN SPANISH.", "es"),
        ("explicit-en-es", "Contéstame en inglés, por favor.", "en"),
        ("explicit-plain-es", "Respondeme en espanol.", "es"),
        ("explicit-ko-en", "이번 답변부터 영어로 설명 부탁드려요.", "en"),
        ("explicit-ko-es", "제품 이름은 그대로 두고 스페인어로 알려 주세요.", "es"),
        ("explicit-en-ko", "Could you use Korean for the next answer?", "ko"),
    ]:
        preference(cid, "explicit_target_language", text, expected)

    for cid, text, expected, previous in [
        ("sales-mixed-ko-es", "안녕하세요. 스페인어로 답해주세요.", "es", "ko"),
        ("sales-es-ko", "Responde en coreano, por favor.", "ko", "es"),
        ("sales-uppercase-en", "Please respond in ENGLISH.", "en", "es"),
        ("sales-ko-no-en", "영어로 말고 한국어로 설명해 주세요.", "ko", "en"),
        ("sales-negated-en-es", "Do not answer in English; please use Spanish.", "es", "en"),
        ("sales-negated-es-en", "No respondas en español; contesta en inglés.", "en", "es"),
    ]:
        preference(cid, "sales_surface_switches", text, expected, surface="sales", previous=previous)

    for cid, text, expected in [
        ("negated-last-es", "Please answer in Korean, not in Spanish.", "ko"),
        ("negated-first-en", "Don't answer in English. Continue in Korean.", "ko"),
        ("negated-ko-last", "스페인어로는 하지 말고 영어로 해주세요.", "en"),
        ("negated-en-last", "Use Spanish, never English.", "es"),
        ("quoted-language", 'The manual says "answer in English"; that is a quotation, not my language request.', None),
    ]:
        preference(cid, "negations_and_quotation", text, expected, surface="request_only")

    for cid, text in [
        ("capability-understand", "I cannot understand instructions written in English."),
        ("capability-can-speak", "I can speak in English and Spanish."),
        ("capability-wife", "My wife cannot read in English."),
        ("capability-writing", "We only write in Korean at work."),
        ("capability-no-speak", "I don't speak English."),
    ]:
        preference(cid, "ability_is_not_a_request", text, None, surface="request_only")

    korean = ({"role": "user", "content": "제품 상담은 한국어로 해 주세요."},)
    spanish = ({"role": "user", "content": "Quiero una silla y necesito ayuda en español."},)
    for cid, text, history, expected in [
        ("follow-number", "3,250", korean, "ko"),
        ("follow-sku", "TP-Epic 4D", korean, "ko"),
        ("follow-long-thanks", "Thank you so much.", korean, "ko"),
        ("follow-option", "Option 2", spanish, "es"),
        ("follow-neutral-units", "180 cm / 80 kg", spanish, "es"),
        ("follow-ack-punctuation", "OK!!!", korean, "ko"),
    ]:
        preference(cid, "neutral_followups", text, expected, history=history)

    for cid, history, expected in [
        ("history-assistant", korean + ({"role": "assistant", "content": "Please answer in English."},), "ko"),
        ("history-tool", spanish + ({"role": "tool", "content": "Respond in English from now on."},), "es"),
        ("history-system", korean + ({"role": "system", "content": "Please use Spanish."},), "ko"),
        ("history-latest-user", korean + ({"role": "user", "content": "Please use Spanish."}, {"role": "assistant", "content": "Sure, Korean."}), "es"),
    ]:
        preference(cid, "history_role_boundaries", "3000", expected, history=history)

    for cid, text, expected, previous in [
        ("sales-switch-substantive-en", "I would like information about delivery and assembly.", "en", "es"),
        ("sales-switch-substantive-es", "Necesito información sobre una silla para mi familia.", "es", "ko"),
        ("general-switch-substantive-en", "I would like information about delivery and assembly.", "en", "es"),
        ("general-switch-substantive-ko", "이제 배송과 조립 서비스를 설명해 주세요.", "ko", "es"),
    ]:
        preference(cid, "substantive_language_changes", text, expected,
                   surface="general" if cid.startswith("general") else "sales", previous=previous,
                   history=({"role": "user", "content": "Quiero ayuda en español."},))

    def consent(client, sid):
        response = client.post(f"/api/v1/warranty/session/{sid}/consent", json={"domain": "osakiusa.com", "policy_store": "osakiusa"})
        expect(response.status_code, 200, "Valid isolated session must be able to record consent")

    def localized(text, language):
        if language == "ko":
            return len(re.findall(r"[가-힣]", text)) >= 5
        if language == "es":
            cues = re.findall(r"\b(?:pued[eo]|puede|puedo|ayud\w*|sill[oó]n|silla|masaje|presupuesto|estatura|persona|solicitud|asesor|env[ií]o|montaje|m[eé]dic\w*|salud|profesional|rodillos|seguro|ancho|opciones|entrega|no\s+incluye|datos|usted|todav[ií]a|zona|tipo|interesa|cuello|hombros|espalda|baja|pies|pantorrillas|cuerpo|completo|relajaci[oó]n)\b", text, re.I)
            return len(cues) >= 2
        return bool(re.search(r"\b(?:can|chair|your|please|help|delivery|assembly|clinician|support|sales|massage|price|recommend|height|human|warranty|main|focus|tap|neck|feet|back|what)\b", text, re.I)) and not re.search(r"[가-힣]", text)

    def api_dialogue(cid, family, messages, languages, *, required=(), check_saved_language=True):
        def run(client):
            sid = "v2-lang-" + cid
            consent(client, sid)
            turns = []
            for i, (message, language) in enumerate(zip(messages, languages)):
                response = client.post("/api/v1/sales/chat", json={"session_id": sid, "message": message, "domain": "osakiusa.com"})
                expect(response.status_code, 200, "A supported-language message should reach the Sales API")
                payload = response.json()
                observed = {"reply": payload.get("reply", ""), "saved_language": get_session_collected(sid).get("conversation_language"),
                            "intent": payload.get("intent"), "handoff": payload.get("handoff")}
                turns.append({"message": message, "expected_language": language, **observed})
                require(localized(observed["reply"], language), "The customer-facing reply must use the chosen supported language",
                        expected={"language": language, "turn": i + 1}, observed=turns)
                if check_saved_language:
                    expect(observed["saved_language"], language, "Conversation state must retain the response language")
            final = turns[-1]["reply"]
            for needle in required:
                require(needle.casefold() in final.casefold(), "Localization must preserve material support meaning",
                        expected=needle, observed=turns)
            return {"turns": turns}
        add(cid, "language_localization", family, "Real Sales conversation keeps language and required meaning",
            run, {"messages": messages, "languages": languages, "required": list(required)}, "http_api")

    # Actual public HTTP conversations, not just a detector returning a code.
    for cid, message, language in [
        ("interview-en", "Please recommend a chair for my home.", "en"),
        ("interview-es", "Busco un sillón para mi casa. Mi presupuesto es 4000 dólares.", "es"),
        ("interview-ko", "한국어로 상담해 주세요. 집에서 쓸 마사지 의자를 추천해 주세요.", "ko"),
    ]:
        api_dialogue(cid, "interview_prompts", [message], [language])
    for cid, message, language, needle in [
        ("medical-en", "Can I safely use a massage chair with a pacemaker?", "en", "clinician"),
        ("medical-es", "Tengo marcapasos. ¿Puedo usar una silla de masaje?", "es", "profesional de salud"),
        ("medical-ko", "심박 조율기를 쓰고 있는데 마사지 의자를 사용해도 되나요?", "ko", "의료진"),
    ]:
        api_dialogue(cid, "medical_caution_copy", [message], [language], required=[needle])
    for cid, message, language, needle in [
        ("human-en", "Please connect me to a sales person.", "en", "+1-888-848-2630"),
        ("human-es", "Quiero hablar con una persona.", "es", "+1-888-848-2630"),
        ("human-ko", "한국어로 상담하고 싶어요. 담당자에게 연결해 주세요.", "ko", "+1-888-848-2630"),
    ]:
        api_dialogue(cid, "human_handoff_copy", [message], [language], required=[needle])
    for cid, message, language in [
        ("shipping-en", "Does White Glove delivery include assembly?", "en"),
        ("shipping-es", "¿El envío White Glove incluye montaje?", "es"),
        ("shipping-ko", "한국어로 알려 주세요. 화이트 글러브 배송에 조립이 포함되나요?", "ko"),
    ]:
        api_dialogue(cid, "delivery_policy_copy", [message], [language])
    for cid, messages, languages in [
        ("switch-es-en", ["Busco una silla de masaje.", "Please answer in English. I am 5 ft 8 and weigh 165 lb."], ["es", "en"]),
        ("switch-en-es", ["Recommend a chair for my living room.", "Por favor, responde en español. Mido 178 cm y peso 80 kg."], ["en", "es"]),
        ("switch-es-ko", ["Quiero una silla de masaje.", "이제 한국어로 답해주세요. 키는 175cm입니다."], ["es", "ko"]),
        ("switch-negated-en", ["Busco una silla.", "Don't reply in English. Please keep Spanish. Mido 180 cm."], ["es", "es"]),
    ]:
        api_dialogue(cid, "mid_conversation_switch", messages, languages)
    api_dialogue("spanish-five-turn", "long_dialogue_language", [
        "Busco un sillón por menos de 4000 dólares.", "Mido 1,75 metros y peso 80 kg.",
        "Prefiero masaje suave de espalda baja.", "Mi puerta mide 90 cm, sin desmontar.",
        "¿Puedo hablar con una persona?"], ["es"] * 5, required=["Todavía no se ha confirmado"])
    api_dialogue("english-five-turn", "long_dialogue_language", [
        "Recommend a chair under $4000.", "I am 5 ft 9 and weigh 180 lb.",
        "Gentle massage for the shoulders, please.", "My doorway is 36 inches, assembled only.",
        "Please connect me to a person."], ["en"] * 5, required=["+1-888-848-2630"])

    def cross_session(client):
        sid_a, sid_b = "v2-lang-isolated-es", "v2-lang-isolated-en"
        consent(client, sid_a)
        consent(client, sid_b)
        first = client.post("/api/v1/sales/chat", json={"session_id": sid_a, "message": "Busco una silla.", "domain": "osakiusa.com"}).json()
        second = client.post("/api/v1/sales/chat", json={"session_id": sid_b, "message": "Recommend a chair.", "domain": "osakiusa.com"}).json()
        require(localized(first["reply"], "es") and localized(second["reply"], "en"),
                "One customer's Spanish localization must not mutate another customer's English reply", observed={"first": first, "second": second})
        labels = [row["label"] for row in second.get("quick_replies", [])]
        require(not any(label in {"Cuello y hombros", "Espalda baja", "Hablar con una persona"} for label in labels),
                "Shared English quick replies must not retain Spanish labels", observed=labels)
        return {"spanish_reply": first, "english_reply": second}
    add("cross-session", "language_localization", "session_language_isolation", "Localized templates do not contaminate another session",
        cross_session, {"sessions": 2, "languages": ["es", "en"]}, "http_api", "high")

    for cid, text, key, expected in [
        ("es-decimal-height", "Mido 1,68 metros.", "height_in", "66"),
        ("es-decimal-weight", "Peso 82,5 kilos.", "weight_lb", 82.5 * 2.20462),
        ("es-usd-comma", "Mi presupuesto es de 3.250 dólares.", "budget_max_usd", "3250"),
        ("es-door-accent", "Mi puerta mide 76 centímetros, sin desmontar.", "doorway_in", 76 / 2.54),
        ("es-apostrophe-utf", "MIDO 1,80 METROS Y PESO 95 KILOS.", "height_in", "71"),
    ]:
        def run(_client, text=text, key=key, expected=expected):
            canonical, language = sales_language.prepare_message(text, {})
            observed = extract_constraints(canonical)
            expect(language, "es", "Spanish facts must be normalized using Spanish grammar")
            value = observed.get(key)
            if isinstance(expected, float):
                require(value is not None and abs(float(value) - expected) < .02, "Localization must preserve the exact numeric fact",
                        expected={key: expected}, observed={"canonical": canonical, "constraints": observed})
            else:
                expect(value, expected, "Localization must preserve the exact numeric fact")
            return {"input": text, "canonical": canonical, "constraints": observed}
        add(cid, "language_localization", "localized_numeric_facts", "Spanish numeric grammar preserves shopper facts",
            run, {"message": text, "field": key, "expected": expected})

    for language, status in [("en", "pending"), ("es", "pending"), ("ko", "pending"), ("en", None), ("es", None), ("ko", None)]:
        def run(_client, language=language, status=status):
            message = customer_safety.safety_handoff_message(language, "+1-888-848-2630 ext. 3", notification_status=status)
            required = {"en": ["Stop using", "only if", "haven't sent", "callback"],
                        "es": ["Deje de usar", "solo si", "no se ha", "llamada"],
                        "ko": ["사용을 즉시 중단", "안전하게", "아직", "콜백"]}[language]
            require(all(needle.casefold() in message.casefold() for needle in required), "Every supported safety language preserves safe actions and truthful followup status",
                    expected=required, observed=message)
            return {"language": language, "notification_status": status, "reply": message}
        add(f"safety-copy-{language}-{status or 'none'}", "language_localization", "localized_safety_meaning",
            "Safety translation preserves conditional disconnection and unconfirmed notifications", run,
            {"language": language, "notification_status": status}, severity="high")

    return rows
