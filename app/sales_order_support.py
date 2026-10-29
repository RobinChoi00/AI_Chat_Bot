"""Keep order support, local visit requests and email evidence separate.

This module never looks up or changes an order, sends a message, or treats a
customer's statement as proof of notification delivery. The HTTP router supplies
the latest persisted lead state after checking both session and storefront.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_NEGATED = re.compile(r"\b(?:don't|do not|didn't|did not|never|not)\s+(?:(?:want|need|wish)\s+(?:to\s+)?)?$|\bno\s+(?:(?:quiero|necesito|deseo)\s+)?$", re.I)
_CANCEL = re.compile(r"\b(?:cancel(?:ling|led|lation)?|refund|return)\b|취소|환불|반품|\b(?:cancelar|cancele|cancelaci[oó]n|reembolso|devolver(?:la|lo)?)\b", re.I)
_ORDER = re.compile(r"\b(?:my|our|the|this|that|an|existing)\s+(?:order|purchase|(?:delivered|ordered)\s+chair)|\border\s*(?:#|number\s*)?\d+\b|\b(?:already|previously)\s+(?:ordered|purchased|bought)\b|주문|\b(?:mi|el|la|este|ese)\s+(?:pedido|silla|sill[oó]n|chair)\b", re.I)
_STATUS = re.compile(r"\b(?:where|tracking|track|status|update|progress|late|delayed|hasn't arrived|has not arrived|not arrived|not delivered|not\s+(?:yet\s+)?(?:here|shipped)|when|shipping|shipment|in transit)\b|어디|조회|배송|도착|\b(?:d[oó]nde|seguimiento|estado|tarde|retrasado|llegado|llega)\b", re.I)
_SHORT_FOLLOWUP = re.compile(r"^(?:(?:my\s+)?(?:order\s+)?(?:email|e-mail|number)\s*(?:is|:)\s*)?[^\s@]+@[^\s@]+\.[^\s@]+[.!?]?\s*$|^\d{4,20}[.!?]?\s*$|^(?:where is it|what about it|any update|what happens next)[.!?]?\s*$", re.I)
_CHECKOUT_CONTACT = re.compile(r'\b(?:checkout|purchase|order)\b|주문|결제|\b(?:pedido|compra)\b', re.I)
_SHOWROOM = re.compile(r"\b(?:showroom|visit|appointment)\b|쇼룸|방문\s*예약|\b(?:visita|cita)\b", re.I)
_SHOWROOM_CANCEL = re.compile(r"\b(?:cancel|withdraw|drop|stop|forget)\b|취소|철회|\b(?:cancelar|cancele|retirar)\b", re.I)
_RETURN_RULES = re.compile(r"\b(?:return|refund)\s+(?:rules?|polic(?:y|ies)|terms?)\b|반품\s*(?:규정|정책)|\b(?:reglas?|pol[ií]tica)\s+de\s+devoluci[oó]n\b", re.I)


def _negated_request(text: str, match) -> bool:
    return bool(_NEGATED.search(text[:match.start()]) or re.match(
        r'\s*(?:하지\s*(?:마|않)|안\s*해|하지마|(?:는|를)?\s*원하지\s*않)', text[match.end():],
    ))


def is_email_status_question(message: str, prefs: dict | None = None) -> bool:
    from sales_cta import extract_email

    text = (message or '').replace('’', "'")
    address = extract_email(text)
    if address:
        text = text.replace(address, '')
    return bool(
        re.search(r"\b(?:email|e-mail|notification|request|receipt|inbox)\b", text, re.I)
        and re.search(r"\b(?:did|has|have|was|were|sent|delivered|forwarded|saved|only|status|received|receive|confirm|reach)\b", text, re.I)
        and not re.search(r"\b(?:order|tracking|shipment|delivery date)\b", text, re.I)
        or re.search(r"(?:이메일|메일|알림).{0,35}(?:보냈|보낸|발송(?:했|됐|되었|여부|대기)|전송(?:했|됐|되었|여부)|저장(?:했|됐|만)|도착(?:했|여부)|받았)|(?:envi[oó]|enviado|enviaste|enviaron|mandaste|guardado|recibido).{0,35}(?:correo|solicitud)|(?:correo|solicitud).{0,35}(?:enviado|guardado|recibido)", text, re.I)
        or (prefs or {}).get('followup_request_saved') and re.search(r"\b(?:did|has|was)\b.{0,20}\b(?:it|that)\b.{0,20}\b(?:sent|send|delivered|saved)\b|\bdid\s+you\s+(?:actually\s+)?send\s+it\b", text, re.I)
    )


def latest_followup_status(session_id: str, domain: str, prefs: dict) -> str:
    """Return evidence for this attempt; an older success cannot fill a gap."""
    if prefs.get('sales_followup_save_failed'):
        return 'save_failed'
    if prefs.get('awaiting_email_for_pick') or prefs.get('awaiting_showroom_email'):
        return 'not_saved'
    from sales_models import SalesLead
    from warranty_models import warranty_db_session

    with warranty_db_session() as db:
        row = db.query(SalesLead).filter(
            SalesLead.session_id == session_id, SalesLead.domain == domain,
        ).order_by(SalesLead.id.desc()).first()
        return str(row.forwarded) if row is not None and row.forwarded in {'pending', 'sent', 'failed'} else 'unknown'


def _language(prefs: dict) -> str:
    value = prefs.get('conversation_language')
    return str(value) if value in {'ko', 'es'} else 'en'


def _email_status_reply(prefs: dict):
    from sales_agent import QuickReply, SalesReply

    language = _language(prefs)
    status = prefs.get('_sales_followup_delivery_status', 'unknown')
    copies = {
        'en': {
            'sent': "The latest saved follow-up notification was accepted by the email service for the sales inbox. That does not confirm inbox arrival, a staff reply, or delivery of a copy to you.",
            'pending': "Your latest follow-up request is saved. Its notification to sales is pending; I cannot confirm that an email has been sent or received.",
            'failed': "Your latest follow-up request is saved, but the notification to sales failed. I cannot confirm email delivery. Please call sales for direct help.",
            'save_failed': "The latest follow-up request could not be saved. I have no confirmed email delivery for that attempt. Please retry or call sales.",
            'not_saved': "The current follow-up request has not been submitted yet. Sharing contact details or choosing a pick does not by itself send a confirmed email.",
            'unknown': "I do not have confirmed email delivery for this request. A saved request or an email address alone is not proof that an email was sent or received.",
        },
        'ko': {
            'sent': "가장 최근에 저장한 후속 문의 알림을 이메일 서비스가 영업팀 수신함으로 접수했습니다. 받은편지함 도착, 직원 회신 또는 고객님께 보낸 사본의 도착까지 확인된 것은 아닙니다.",
            'pending': "최근 후속 문의는 저장됐습니다. 영업팀 알림은 전송 대기 중이며 이메일 발송이나 수신은 아직 확인되지 않았습니다.",
            'failed': "최근 후속 문의는 저장됐지만 영업팀 알림 전송에 실패했습니다. 이메일 전달은 확인되지 않았으니 직접 도움이 필요하면 영업팀에 전화해 주세요.",
            'save_failed': "최근 후속 문의를 저장하지 못했습니다. 해당 시도의 이메일 전달도 확인되지 않았습니다. 다시 시도하거나 영업팀에 전화해 주세요.",
            'not_saved': "현재 후속 문의는 아직 제출되지 않았습니다. 연락처를 입력하거나 상품을 선택했다는 사실만으로 이메일 발송이 확인되는 것은 아닙니다.",
            'unknown': "이 문의의 이메일 전달은 확인되지 않았습니다. 문의 저장이나 이메일 주소만으로 발송 또는 수신 완료를 판단할 수 없습니다.",
        },
        'es': {
            'sent': "El servicio de correo aceptó la última notificación guardada para el buzón de ventas. Esto no confirma su llegada al buzón, una respuesta del personal ni la entrega de una copia a usted.",
            'pending': "Su última solicitud está guardada. La notificación a ventas está pendiente; no puedo confirmar que el correo se haya enviado o recibido.",
            'failed': "Su última solicitud está guardada, pero falló la notificación a ventas. No puedo confirmar la entrega del correo. Llame a ventas para recibir ayuda directa.",
            'save_failed': "No se pudo guardar la última solicitud. No hay entrega de correo confirmada para ese intento. Inténtelo de nuevo o llame a ventas.",
            'not_saved': "La solicitud actual todavía no se ha presentado. Compartir datos de contacto o elegir una silla no confirma por sí solo el envío de un correo.",
            'unknown': "No tengo confirmación de entrega del correo para esta solicitud. Una solicitud guardada o una dirección de correo no demuestran que se haya enviado o recibido un mensaje.",
        },
    }
    return SalesReply(
        reply=copies[language].get(str(status), copies[language]['unknown']) + '\n\nSales: **+1-888-848-2630 ext. 2**.',
        intent='email_status', tools_used=['lead.delivery_status'],
        quick_replies=[QuickReply(label={'en': 'Talk to sales', 'ko': '영업팀 문의', 'es': 'Hablar con ventas'}[language], payload='human:confirm')],
    )


def followup_save_failure_reply(prefs: dict, reason: str):
    reply = _email_status_reply({**prefs, '_sales_followup_delivery_status': 'save_failed'})
    reply.intent = 'lead_capture'
    reply.flow_stage = 'lead'
    reply.prefs_patch = {'sales_followup_save_failed': True, 'followup_request_saved': False,
                         'awaiting_showroom_email': reason == 'showroom_visit',
                         'awaiting_email_for_pick': reason != 'showroom_visit'}
    return reply


def persist_prepared_followup(reply, *, prefs: dict, session_id: str, domain: str,
                             background_tasks, capture, schedule):
    """Both customer channels must persist a request before claiming saved."""
    details = reply.lead_capture or {}
    email = str(details.get('email') or '').strip()
    if not email:
        return reply
    interest = str(details.get('interest_summary') or '').strip() or None
    reason = str(details.get('reason') or 'save_pick').strip() or 'save_pick'
    try:
        lead_id = capture(session_id=session_id, domain=domain, email=email,
                          interest_summary=interest, reason=reason)
    except Exception:
        logger.warning('sales lead could not be saved')
        lead_id = None
    if lead_id is None:
        return followup_save_failure_reply(prefs, reason)
    try:
        schedule(background_tasks, lead_id=lead_id, domain=domain, email=email,
                 interest_summary=interest)
    except Exception:
        logger.warning('sales lead notification could not be scheduled')
    return reply


def maybe_sales_support_reply(message: str, prefs: dict | None, *, domain: str):
    from sales_agent import QuickReply, SalesReply
    from sales_intent import (INTENT_PARTS_TECHNICIAN, INTENT_WARRANTY_REDIRECT, classify,
                              _EXISTING_ORDER_CHANGE_RE, _requested_matches)
    from customer_safety import reported_safety_hazard
    from sales_cta import extract_email

    text = (message or '').strip().replace('’', "'")
    supplied_email = extract_email(text)
    route_text = text.replace(supplied_email, '') if supplied_email else text
    data = prefs or {}
    language = _language(data)
    # A new reported hazard or a concrete technical fault keeps its service
    # route even when an earlier turn concerned delivery or cancellation.
    current_intent = classify(route_text)
    technical_report = current_intent.label == INTENT_WARRANTY_REDIRECT and (not current_intent.matched_terms or any(
        not re.fullmatch(r"already\s+(?:bought|ordered|purchased|own(?:ed)?)|my\s+chair\s+(?:is|has|won'?t|does\s+not|isn'?t|doesn'?t)", term, re.I)
        for term in current_intent.matched_terms
    ))
    # Owning or ordering a chair does not itself establish a technical fault.
    # Concrete failures still interrupt order support through their own terms.
    if reported_safety_hazard(text=route_text) or current_intent.label == INTENT_PARTS_TECHNICIAN or technical_report:
        return None
    technical_review = bool(re.search(r"\b(?:is|was|has)\b.{0,65}\b(?:approved|free|covered)\b", route_text, re.I))
    technical_schedule = bool(re.search(
        r"\b(?:has|have|is|was|when)\b.{0,65}\b(?:scheduled|booked|technician|appointment|come\s+out|confirmed)\b", route_text, re.I,
    ))
    if data.get('technical_support_topic') and (technical_review or technical_schedule):
        lead = ("I cannot confirm a technician visit or appointment from this chat. " if technical_schedule else
                "I cannot confirm that a replacement or repair has been approved or is free. ")
        return SalesReply(
            reply=lead + "Our Warranty team must review the service request and applicable coverage before arranging service. Please contact **service@osakititan.com** or **+1-888-848-2630 ext. 3** with your case or order number.",
            intent='parts_technician', handoff=True, handoff_reason='parts_technician',
            tools_used=['support.technical_followup'], flow_stage='warranty',
            quick_replies=[QuickReply(label='Back to menu', payload='menu')],
            prefs_patch={'technical_support_topic': 'parts_technician', 'awaiting_recommend': False},
        )

    stop_search = re.search(r'\b(?:never mind|forget|stop|cancel|drop|end)\b.{0,55}\b(?:suggestions?|search|recommendations?|questionnaire|interview)\b|(?:추천|상품\s*검색).{0,15}(?:그만|중단)|\b(?:detener|cancelar)\b.{0,30}\b(?:b[uú]squeda|recomendaciones)\b', route_text, re.I)
    if stop_search and not _ORDER.search(route_text) and not _negated_request(route_text, stop_search):
        from sales_agent import _clear_compare_state
        copy = {
            'en': "I've stopped the chair search in this chat. Your preferences are kept if you choose to start again.",
            'ko': "이 채팅의 의자 추천을 중단했습니다. 다시 시작하실 때 사용할 수 있도록 입력하신 조건은 유지합니다.",
            'es': "He detenido la búsqueda de sillas en este chat. Sus preferencias se conservan si decide empezar de nuevo.",
        }[language]
        return SalesReply(reply=copy, intent='recommendation_cancel', tools_used=['shopping.stop_search'],
                          prefs_patch={**_clear_compare_state(), 'awaiting_recommend': False,
                                       'pending_tier_picks': [], 'sales_support_topic': ''},
                          quick_replies=[QuickReply(label='Back to menu', payload='menu')])

    showroom_cancel = _SHOWROOM_CANCEL.search(route_text)
    if showroom_cancel and not _negated_request(route_text, showroom_cancel) and not _ORDER.search(route_text) and (
        _SHOWROOM.search(route_text) or data.get('showroom_request_pending') and re.search(r'\b(?:it|that|request)\b|그거|요청', route_text, re.I)
    ):
        copy = {
            'en': "I've closed the showroom request form in this chat. I have not cancelled a booked appointment or withdrawn any request already submitted to sales. For an existing booking or submitted request, please contact sales at **+1-888-848-2630 ext. 2**.",
            'ko': "이 채팅의 쇼룸 요청 입력을 닫았습니다. 확정된 예약을 취소하거나 이미 영업팀에 제출한 요청을 철회한 것은 아닙니다. 기존 예약이나 제출한 요청은 **+1-888-848-2630 ext. 2**로 문의해 주세요.",
            'es': "He cerrado el formulario de visita en este chat. No he cancelado una cita reservada ni retirado una solicitud ya enviada a ventas. Para una reserva o solicitud existente, llame al **+1-888-848-2630 ext. 2**.",
        }[language]
        return SalesReply(reply=copy, intent='showroom_request_cancel', tools_used=['showroom.clear_pending'],
                          prefs_patch={'awaiting_showroom_window': False, 'awaiting_showroom_email': False,
                                       'showroom_request_pending': False, 'pending_showroom_window': ''},
                          quick_replies=[QuickReply(label='Back to menu', payload='menu')])

    pending_form = data.get('awaiting_email_for_pick') or data.get('awaiting_showroom_email') or data.get('showroom_request_pending')
    if pending_form and _RETURN_RULES.search(route_text) and not _ORDER.search(route_text):
        from sales_policy import TOPIC_RETURNS, policy_answer
        return SalesReply(reply=policy_answer(TOPIC_RETURNS, domain=domain) or '', intent='prepurchase_policy',
                          tools_used=['policy.returns'], prefs_patch={'awaiting_email_for_pick': False,
                          'awaiting_showroom_email': False, 'awaiting_showroom_window': False},
                          quick_replies=[QuickReply(label='Back to menu', payload='menu')])

    topic = data.get('sales_support_topic')
    contact_form = data.get('awaiting_email_for_pick') or data.get('awaiting_showroom_email')
    known_order_context = topic in {'order_status', 'cancel_refund', 'order_change'} and not contact_form
    # Keep the actual order task when the customer uses a pronoun or asks
    # what to prepare. Such turns are not a new chair recommendation.
    refund_followup = bool(topic == 'cancel_refund' and not contact_form and re.search(
        r"\b(?:how\s+(?:soon|long)|when)\b.{0,65}\b(?:money|back|refund|process|receive|credit|card)\b|"
        r"\b(?:money|refund|credit)\b.{0,45}\b(?:back|arriv\w*|process\w*|receiv\w*|take|show)\b|"
        r"\bwhat\b.{0,35}\b(?:provide|prepare|send|share|need|required|documents?|paperwork)\b|"
        r"\b(?:has|is|was)\b.{0,30}\b(?:approved|issued|processed|completed|cancell?ed)\b|"
        r"\b(?:return|refund)\s+(?:request|review|shipping|costs?)\b", route_text, re.I))
    explicit_order_change = bool(not contact_form and (known_order_context or _ORDER.search(route_text))
                                 and _requested_matches(_EXISTING_ORDER_CHANGE_RE, route_text))
    order_change_followup = bool(explicit_order_change or topic == 'order_change' and not contact_form and re.search(
        r"\b(?:invoice|receipt|colou?r|address|upgrade)\b|"
        r"\b(?:has|is|was|have|what)\b.{0,40}\b(?:added|changed|updated|confirmed|need|provide)\b", route_text, re.I))
    without_contact = route_text
    if contact_form and supplied_email and not (
        _ORDER.search(without_contact) and (_STATUS.search(without_contact) or _CANCEL.search(without_contact))
    ):
        # The latest explicit contact form owns its answer, including phrases
        # such as "the checkout email was ...". Do not read "was" as proof
        # the shopper is asking about delivery of a notification.
        return None
    checkout_contact = bool(known_order_context and _CHECKOUT_CONTACT.search(route_text) and supplied_email)
    if is_email_status_question(text, data) and not checkout_contact:
        return _email_status_reply(data)

    cancellation = _CANCEL.search(route_text)
    negated_cancel = cancellation and _negated_request(route_text, cancellation)
    contextual_cancel = bool(topic in {'order_status', 'cancel_refund', 'order_change'} and (
        re.search(r'\b(?:cancel|refund|return)\s+(?:it|that|this|the\s+order)\b|\b(?:that|the|this)\s+(?:return|refund|cancellation)\b', text, re.I)
        or re.fullmatch(r'(?:please\s+)?(?:cancel|refund|return)[.!?\s]*', text, re.I)
        or re.search(r'그\s*주문|그걸|그것|\b(?:cancelar|devolver)\s*(?:lo|ese|esto)\b', text, re.I)
    ))
    is_cancel = bool(refund_followup or cancellation and not negated_cancel and (bool(_ORDER.search(route_text)) or contextual_cancel))
    is_status = bool(_ORDER.search(route_text) and _STATUS.search(route_text) or known_order_context and _SHORT_FOLLOWUP.fullmatch(text) or checkout_contact)
    if not is_cancel and not is_status and not order_change_followup:
        return None
    invoice_request = bool(known_order_context and re.search(r'\b(?:invoice|receipt)\b', route_text, re.I))
    intent = ('cancel_refund' if is_cancel or checkout_contact and topic == 'cancel_refund'
              else 'order_status' if invoice_request
              else 'order_change' if order_change_followup or checkout_contact and topic == 'order_change'
              else 'order_status')
    copies = {
        'en': {
            'cancel_refund': "A sales agent must review this cancellation, return or refund request. I have not cancelled your order, approved a return or refund, or changed a payment. Please contact **sales at +1-888-848-2630 ext. 2** with your order number and checkout email. You do not need to repeat those details if you already shared them here.",
            'order_status': "This is an existing-order question for **sales at +1-888-848-2630 ext. 2**, rather than technical support. I have not looked up live tracking or confirmed a delivery date in this chat. Sales can verify the order with its order number and checkout email; I cannot verify it from an email address alone.",
        },
        'ko': {
            'cancel_refund': "주문 취소·반품·환불은 영업팀 검토가 필요합니다. 이 채팅에서 주문을 취소하거나 반품·환불을 승인하거나 결제를 변경하지 않았습니다. 주문번호와 주문 당시 이메일로 **+1-888-848-2630 ext. 2**에 문의해 주세요. 이미 여기 알려주신 내용을 다시 입력하실 필요는 없습니다.",
            'order_status': "기존 주문 문의는 기술 지원이 아닌 **영업팀 +1-888-848-2630 ext. 2**에서 확인해야 합니다. 이 채팅에서 실시간 배송 조회나 배송일 확정을 하지 않았습니다. 주문번호와 주문 당시 이메일을 함께 확인해야 하며 이메일 주소만으로 주문을 인증할 수 없습니다.",
        },
        'es': {
            'cancel_refund': "Ventas debe revisar la cancelación, devolución o reembolso. No he cancelado su pedido, aprobado una devolución o reembolso, ni cambiado un pago. Llame a **ventas al +1-888-848-2630 ext. 2** con el número de pedido y el correo usado al comprar. No necesita repetir datos que ya compartió aquí.",
            'order_status': "La consulta sobre un pedido existente corresponde a **ventas al +1-888-848-2630 ext. 2**, no a soporte técnico. No he consultado el seguimiento en vivo ni confirmado una fecha de entrega. Ventas debe verificar el número de pedido y el correo usado al comprar; el correo por sí solo no verifica el pedido.",
        },
    }
    copies['en']['order_change'] = ("Sales must review changes to an existing order and requests for an invoice or receipt. "
        "Please contact **sales at +1-888-848-2630 ext. 2** with your order number and checkout email. "
        "This chat has not changed the order, added a delivery upgrade, or issued an invoice.")
    copies['ko']['order_change'] = "기존 주문 변경과 청구서·영수증은 영업팀 검토가 필요합니다. 주문번호와 주문 당시 이메일로 **+1-888-848-2630 ext. 2**에 문의해 주세요. 이 채팅에서 주문 변경이나 배송 업그레이드, 청구서 발행을 완료하지 않았습니다."
    copies['es']['order_change'] = "Ventas debe revisar los cambios de un pedido y las solicitudes de factura o recibo. Llame al **+1-888-848-2630 ext. 2** con el número de pedido y el correo usado al comprar. Este chat no ha cambiado el pedido, agregado servicios ni emitido una factura."
    if invoice_request:
        copies['en']['order_status'] = "Sales can help locate an invoice or receipt for your existing order. Please contact **sales at +1-888-848-2630 ext. 2** with your order number and checkout email. This chat has not retrieved or issued an invoice."
    if refund_followup and re.search(r"\b(?:how\s+(?:soon|long)|when|processing|money|credit)\b", route_text, re.I):
        timing = {
            'en': "I cannot verify the refund processing time or payment status from this chat. Sales must first review the return and confirm any approved refund and timing with the payment provider. ",
            'ko': "이 채팅에서는 환불 처리 기간이나 결제 상태를 확인할 수 없습니다. 영업팀이 반품을 검토하고 승인 여부와 결제 수단별 환불 기간을 확인해야 합니다. ",
            'es': "No puedo verificar aquí el plazo ni el estado del reembolso. Ventas debe revisar la devolución y confirmar la aprobación y el plazo con el proveedor de pago. ",
        }
        copies[language]['cancel_refund'] = timing[language] + copies[language]['cancel_refund']
    acknowledgement = {
        'en': "Thanks for the checkout email. It does not by itself verify the order. ",
        'ko': "주문 당시 이메일을 알려주셔서 감사합니다. 이메일만으로 주문이 인증되는 것은 아닙니다. ",
        'es': "Gracias por el correo usado al comprar. Por sí solo no verifica el pedido. ",
    }[language] if checkout_contact else ''
    return SalesReply(reply=acknowledgement + copies[language][intent], intent=intent, handoff=True, handoff_reason=intent,
                      tools_used=['support.order_route'], flow_stage='handoff',
                      prefs_patch={'sales_support_topic': intent, 'awaiting_email_for_pick': False,
                                   'awaiting_showroom_email': False, 'awaiting_showroom_window': False},
                      quick_replies=[QuickReply(label={'en': 'Talk to sales', 'ko': '영업팀 문의', 'es': 'Hablar con ventas'}[language], payload='human:confirm')])
