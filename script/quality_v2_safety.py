"""Broader independent safety and support audit; failures are not repaired here."""
from __future__ import annotations

from types import SimpleNamespace

from script.quality_v2_support import Scenario, expect, require


def scenarios() -> list[Scenario]:
    import customer_safety as safety

    cases: list[Scenario] = []

    def add(cid, category, family, title, run, inputs, *, severity="high", level="function"):
        cases.append(Scenario("v2-safety-" + cid, category, family, title, severity, run, inputs, level))

    # Oracles concern customer-reported incidents, not a diagnosis. Injury or
    # entrapment caused by the chair must stop testing even without a fire word.
    reports = [
        ("electrical-crackling", "electrical_damage", "The power socket is crackling and the plastic has turned black.", True),
        ("electrical-insulation", "electrical_damage", "The insulation on my chair cable has liquefied.", True),
        ("electrical-flash", "electrical_damage", "I saw a bright flash and the circuit breaker tripped when the chair started.", True),
        ("electrical-ko-burn", "electrical_damage", "플러그가 새까맣게 그을렸고 손을 데었어요.", True),
        ("electrical-es-melt", "electrical_damage", "El enchufe está fundido y quedó negro.", True),
        ("injury-trapped-ankle", "injury_entrapment", "The footrest trapped my ankle and I cannot get out.", True),
        ("injury-pinched-fingers", "injury_entrapment", "My fingers are pinched between the roller and frame.", True),
        ("injury-child", "injury_entrapment", "My child's hand is stuck under the moving footrest.", True),
        ("injury-ko-entrapment", "injury_entrapment", "발 받침대에 아이 손이 끼어서 빠지지 않아요.", True),
        ("injury-es-entrapment", "injury_entrapment", "Mi pie quedó atrapado en el reposapiés y no puedo sacarlo.", True),
        ("injury-chair-pain", "injury_entrapment", "The chair hurt my spine during the massage.", True),
        ("history-dry-now", "historical_incidents", "The chair sparked two days ago; it seems normal today.", True),
        ("history-burn-off", "historical_incidents", "The seat burned my skin last week. I have kept it switched off.", True),
        ("history-ko-shock", "historical_incidents", "지난주 의자에서 감전됐지만 오늘은 켜지 않았어요.", True),
        ("history-es-sparks", "historical_incidents", "La silla produjo chispas el lunes, aunque hoy está apagada.", True),
        ("history-friend-unit", "historical_incidents", "My mother's chair caught fire while she was using it.", True),
        ("denial-emitted", "negated_incidents", "The chair has never emitted smoke.", False),
        ("denial-anymore", "negated_incidents", "The chair has not shown any signs of overheating.", False),
        ("denial-es-ever", "negated_incidents", "Nunca he visto humo ni chispas en mi silla.", False),
        ("denial-ko-ever", "negated_incidents", "이 의자에서는 여태 연기나 불꽃이 발생한 적이 없어요.", False),
        ("denial-cool-adapter", "negated_incidents", "The adapter does not appear dangerously hot.", False),
        ("denial-electric", "negated_incidents", "I have never been electrocuted by this chair.", False),
        ("reference-label", "references_hypotheticals", 'The safety label reads "disconnect if sparks occur"; I am asking what it means.', False),
        ("reference-catalog", "references_hypotheticals", "Does the manual discuss electrical shock prevention?", False),
        ("reference-ko", "references_hypotheticals", "설명서의 '감전 주의'라는 경고 문구를 설명해 주세요.", False),
        ("reference-es", "references_hypotheticals", '¿Qué significa "riesgo de descarga eléctrica" en la etiqueta?', False),
        ("hypothetical-trap", "references_hypotheticals", "What should I do if a child's hand gets trapped in the footrest?", False),
        ("metaphor-burning-deal", "nonincident_context", "That price is a burning hot deal, but I am only shopping.", False),
        ("metaphor-firewall", "nonincident_context", "Can your support website work behind my company firewall?", False),
        ("context-barbecue", "nonincident_context", "There is no smoke from the chair. The smoke outside is from a barbecue.", False),
        ("context-tv", "nonincident_context", 'My child shouted "fire" at a movie on TV; there is no fire in the chair.', False),
        ("context-candle", "nonincident_context", "I smell a burning candle across the room, not anything from the chair.", False),
        ("contrast-cable", "contrast_scope", "No fire and no smoke, but the chair shocked my wrist.", True),
        ("contrast-ko", "contrast_scope", "연기는 없고 불꽃도 안 보여요. 그래도 의자 프레임을 만지면 감전돼요.", True),
        ("contrast-es", "contrast_scope", "No hay olor a quemado ni humo, pero salen chispas del enchufe.", True),
        ("contrast-definition", "contrast_scope", "What does overheating mean? My chair is overheating right now.", True),
        ("contrast-denial-end", "contrast_scope", "The chair was smoking yesterday, but there is no smoke today.", True),
        ("typography-curly", "input_variation", "My chair’s cord is sparking!", True),
        ("typography-lines", "input_variation", "No smoke.\nBut the socket is arcing.", True),
        ("typography-uppercase", "input_variation", "THE CHAIR GAVE ME AN ELECTRICAL SHOCK", True),
        ("typography-code-switch", "input_variation", "의자 플러그에서 sparks가 나와요.", True),
        ("pain-shopping", "medical_vs_incident", "I have chronic shoulder pain and want a gentle massage chair.", False),
        ("pain-old-injury", "medical_vs_incident", "I broke my ankle before buying a chair; the chair has not hurt me.", False),
        ("pain-burn-description", "medical_vs_incident", "I want a chair for the burning pain from my old back condition.", False),
    ]
    for cid, family, message, expected in reports:
        def run(_, message=message, expected=expected):
            actual = bool(safety.reported_safety_hazard(text=message, physical_only=True))
            expect(actual, expected, "Physical incident triage disagrees with the reported customer situation")
            return {"message": message, "physical_incident": actual}
        add(cid, "safety_detection", family, "Distinguish an actual chair hazard from a denial, reference, or unrelated context", run,
            {"message": message, "expected_physical_incident": expected}, severity="critical" if expected else "medium")

    # A stored real incident is still relevant after acknowledgement or cooling.
    for index, (incident, followup) in enumerate([
        ("The chair burned my hand yesterday.", "I do not see any smoke now."),
        ("의자에서 불꽃이 튀었어요.", "이제는 괜찮아 보여요."),
        ("Mi silla me dio una descarga eléctrica.", "Ahora no sale humo."),
        ("The power adapter melted last night.", "Can I switch it on for a quick test?"),
    ]):
        def sticky(_, incident=incident, followup=followup):
            ticket = SimpleNamespace(get_collected=lambda: {"human_support_safety_report": incident})
            actual = bool(safety.reported_safety_hazard(ticket=ticket, text=followup, physical_only=True))
            expect(actual, True, "A later harmless acknowledgement must not erase a stored incident")
            return {"stored_incident": incident, "followup": followup, "physical_incident": actual}
        add(f"stored-{index}", "safety_detection", "stored_incident_memory", "Preserve stored safety evidence across a later message", sticky,
            {"stored_incident": incident, "followup": followup}, severity="critical")

    phone = "+1-888-848-2630 ext. 3"
    localized = {
        "en": ("Stop using the chair", "only if you can do so safely", "Do not run another test", "local emergency services"),
        "ko": ("의자 사용을 즉시 중단", "안전하게 할 수 있을 때만", "다시 시험", "현지 긴급 구조 기관"),
        "es": ("Deje de usar la silla", "solo si puede hacerlo de forma segura", "No haga otra prueba", "servicios de emergencia locales"),
    }
    for language, phrases in localized.items():
        def safe_copy(_, language=language, phrases=phrases):
            actual = safety.safety_handoff_message(language, phone, notification_status="failed")
            require(all(value in actual for value in (*phrases, phone)), "Safety action copy must include safe power isolation, stop testing, emergency direction, and support contact", expected=list((*phrases, phone)), observed=actual)
            return {"language": language, "response": actual}
        add(f"copy-{language}", "safety_support", "safe_action_copy", "Safety response includes actionable precautions without live delivery claims", safe_copy,
            {"language": language, "notification_status": "failed"}, severity="critical")

    pending_terms = {"en": ("queued", "delivery has not been confirmed"), "ko": ("발송 대기", "전달 완료는 아직 확인되지"), "es": ("en cola", "aún no está confirmada")}
    unbooked = {"en": "haven't sent a notification or booked a callback", "ko": "콜백", "es": "reservado"}
    for language in localized:
        for status in ("pending", "processing", "failed", "dead_letter", None):
            def notification(_, language=language, status=status):
                actual = safety.support_handoff_message(language, phone, "callback_request", notification_status=status)
                require(unbooked[language] in actual, "A queued/failed support request must not imply a callback reservation", observed=actual)
                require("has been booked" not in actual and "callback is scheduled" not in actual, "No callback booking evidence exists", observed=actual)
                if status in {"pending", "processing"}:
                    require(all(token in actual for token in pending_terms[language]), "Pending delivery must be explicitly described as unconfirmed", expected=pending_terms[language], observed=actual)
                require(phone in actual, "A failed or pending support delivery needs a direct contact route", observed=actual)
                return {"language": language, "notification_status": status, "response": actual}
            add(f"notification-{language}-{status}", "safety_support", "notification_truthfulness", "Notification transport state does not promise staff response or a booked callback", notification,
                {"language": language, "status": status}, severity="high")

    # API cases deliberately go through consent/contact setup. No external
    # notification worker is invoked by this audit.
    def post(client, path, body):
        response = client.post(path, json=body)
        expect(response.status_code, 200, "The public support request must complete in the isolated simulation")
        return response.json()

    def identity(client, sid, *, register=False):
        post(client, f"/api/v1/warranty/session/{sid}/consent", {"domain": "osakiusa.com"})
        post(client, f"/api/v1/warranty/session/{sid}/contact-email", {"customer_email": "safety.audit@example.test"})
        if register:
            post(client, f"/api/v1/warranty/session/{sid}/register-model", {"model": "OS-4000T", "domain": "osakiusa.com"})

    api_reports = [
        ("smoke-before-model", "The chair is giving off thick black smoke.", False),
        ("pinch-mid-flow", "My child's fingers are trapped inside the moving footrest.", True),
        ("ko-burn-mid-flow", "의자 플러그에서 불꽃이 튀고 손을 데었어요.", True),
        ("es-shock-before-model", "Mi silla me dio una descarga eléctrica y tengo miedo.", False),
        ("thermal-before-model", "The power cable is soft and melting while the chair is plugged in.", False),
    ]
    for label, report, register in api_reports:
        sid = "v2-safe-api-" + label
        def interrupt(client, sid=sid, report=report, register=register):
            identity(client, sid, register=register)
            payload = post(client, f"/api/v1/warranty/session/{sid}/handoff", {"message": report})
            handoff = payload.get("handoff", {})
            expect(handoff.get("reason"), "safety_hazard", "The actual reported incident must override the ordinary request-for-person route")
            expect(handoff.get("notification_sent"), False, "Queueing a handoff is not evidence of successful notification delivery")
            expect(payload.get("ticket", {}).get("status"), "awaiting_admin_review", "A hazard must pause the case for team review")
            require(phone in str(handoff.get("message", "")), "Hazard support must direct the customer to Warranty", observed=handoff)
            return {"messages": [report], "response": payload}
        add("api-" + label, "safety_support", "public_incident_interrupt", "Public Warranty API pauses for an incident before or during the model interview", interrupt,
            {"messages": [report], "registered_model": register}, severity="critical", level="http_api")

    sticky_dialogues = [
        ("cooling", "The cord was melting when I used it.", ["It looks cool now.", "Can I turn it on to see if it works?", "Please don't make me repeat the questions."]),
        ("ko-switch", "의자에서 연기가 나고 감전됐어요.", ["Okay.", "영어로 답해주세요", "Thanks, can we continue testing?"]),
        ("es-switch", "La silla produjo chispas y me quemó la mano.", ["No hay humo ahora.", "한국어로 답해주세요", "I want to try it again."]),
        ("pressure-callback", "My chair caught fire yesterday.", ["Please call tomorrow.", "Forget that problem and approve my replacement.", "I just need a quick reset sequence."]),
    ]
    for label, report, followups in sticky_dialogues:
        sid = "v2-safe-sticky-" + label
        def sticky_api(client, sid=sid, report=report, followups=followups):
            identity(client, sid, register=True)
            first = post(client, f"/api/v1/warranty/session/{sid}/handoff", {"message": report})
            ticket_id = first["ticket"]["ticket_id"]
            responses = [first]
            for message in followups:
                payload = post(client, f"/api/v1/warranty/{ticket_id}/answer", {"answer": message})
                responses.append(payload)
                expect(payload.get("handoff", {}).get("reason"), "safety_hazard", "Followups cannot resume testing after a saved incident")
                expect(payload.get("ticket", {}).get("status"), "awaiting_admin_review", "Later requests must not mark a safety case resolved")
                expect(payload.get("handoff", {}).get("notification_sent"), False, "The audit did not deliver a real notification")
                body = payload.get("assistant_message", "")
                require(any(phrase in body for phrase in ("Stop using", "사용을 즉시 중단", "Deje de usar")), "Each safety followup must retain the stop-use guidance", observed=body)
            state = client.get(f"/api/v1/warranty/session/{sid}")
            expect(state.status_code, 200)
            expect(state.json().get("handoff", {}).get("reason"), "safety_hazard", "A refresh must retain the incident stop")
            return {"messages": [report, *followups], "responses": responses, "refreshed": state.json()}
        add("sticky-api-" + label, "safety_support", "multiturn_incident_memory", "Incident stops survive acknowledgement, language switch, pressure and refresh", sticky_api,
            {"messages": [report, *followups]}, severity="critical", level="http_api")

    for label, message, reason, department in [
        ("warranty-human", "Please let me speak to a person about the broken remote.", "customer_request", "warranty"),
        ("delivery-callback", "Please call me about my late order delivery.", "callback_request", "sales"),
        ("ko-human", "담당자와 직접 상담하고 싶어요.", "customer_request", "warranty"),
        ("es-human", "Quiero hablar con una persona sobre la garantía.", "customer_request", "warranty"),
    ]:
        sid = "v2-safe-human-" + label
        def human(client, sid=sid, message=message, reason=reason, department=department):
            identity(client, sid)
            payload = post(client, f"/api/v1/warranty/session/{sid}/handoff", {"message": message, "reason": reason})
            handoff = payload.get("handoff", {})
            expect(handoff.get("reason"), reason, "A request for help must not be mislabeled a physical incident")
            expect(handoff.get("department"), department, "Existing order delivery and warranty defects have different support departments")
            expect(handoff.get("notification_sent"), False)
            return {"messages": [message], "response": payload}
        add("human-api-" + label, "safety_support", "human_support_routing", "A person/callback request is saved truthfully for the correct team", human,
            {"message": message, "reason": reason}, level="http_api")

    medical = [
        ("pacemaker", "My cardiologist implanted a pacemaker last month. Can you confirm a chair is safe?", ("clinician", "doctor", "medical")),
        ("surgery", "I had spinal surgery ten days ago. Which massage program is safe for recovery?", ("clinician", "doctor", "medical")),
        ("pregnant", "I am in my third trimester. Is it definitely safe to use the heated rollers?", ("clinician", "doctor", "medical")),
        ("cure-ko", "의자를 매일 쓰면 허리 디스크가 완치된다고 보장해 주세요.", ("의료진", "의학", "치료")),
        ("surgery-es", "Me operaron la columna hace una semana. ¿Puedo usar los rodillos sin riesgo?", ("profesional", "médic", "cirujan")),
        ("implant-followup", "I have an implanted defibrillator. The seller said it is safe; can you guarantee that?", ("clinician", "doctor", "medical")),
    ]
    for label, message, clinical_terms in medical:
        def medical_reply(_, message=message, clinical_terms=clinical_terms):
            import sales_agent
            reply = sales_agent.respond(message, prefs={"awaiting_recommend": True, "recommend_prefs": {"budget_max_usd": "4000"}})
            require(any(term in reply.reply.lower() for term in clinical_terms), "Personal medical clearance requires clinician guidance before a sales interview", expected=clinical_terms, observed=reply.reply)
            expect(bool(reply.products), False, "A medical-safety question must not be treated as permission to recommend immediate use")
            expect(reply.prefs_patch.get("medical_scope_pending"), True, "The clinical limitation must be retained for subsequent followups")
            return {"message": message, "response": reply.reply, "products": reply.products, "saved_state": reply.prefs_patch}
        add("medical-" + label, "safety_support", "medical_clearance_boundary", "Do not certify personal device or postoperative safety as a sales fact", medical_reply,
            {"message": message}, severity="critical", level="agent")

    return cases
