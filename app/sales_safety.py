"""Stop shopping for reported physical hazards without claiming a service case."""
from __future__ import annotations

import re

from customer_safety import detect_customer_language, reported_safety_hazard, requested_customer_language


# Sales shoppers commonly discuss back pain or ask about treatment. The shared
# Warranty detector includes pain. Its physical-only mode keeps the same
# report/negation/reference processing but excludes ordinary shopping pain.


def physical_hazard_report(message: str) -> bool:
    return reported_safety_hazard(text=message, physical_only=True)


def interrupt_sales_for_safety(message: str, prefs: dict):
    """A safety report stays active across acknowledgements and button payloads."""
    incident = physical_hazard_report(message)
    if not incident and not prefs.get('sales_safety_active'):
        return None
    from sales_agent import SalesReply
    from warranty_scope import support_contact

    original = str(prefs.get('sales_safety_report') or '') or message[:2000]
    language = requested_customer_language(message) or str(prefs.get('sales_safety_language') or detect_customer_language(original))
    if language not in {'en', 'ko', 'es'}:
        language = 'en'
    business_text = re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}|https?://[^\s<>]+', '', message, flags=re.I)
    cancel_request = bool(re.search(
        r'\b(?:cancel|refund)\s+(?:(?:my|the|our|this|that)\s+)?(?:order|purchase)\b|'
        r'\b(?:cancelar|cancele|reembolsar)\s+(?:(?:mi|el|este|ese)\s+)?pedido\b|'
        r'주문[^.!?\n]{0,15}(?:취소|환불)', business_text, re.I,
    )) and not re.search(r"\b(?:don't|do not|not to|no quiero|no deseo)\s+(?:cancel|refund|cancelar)\b|취소하지\s*마", business_text, re.I)
    cancel_request = cancel_request or bool(prefs.get('sales_safety_order_action_requested'))
    phone = support_contact('warranty')['phone']
    text = {
        'en': ('Stop using the chair. Switch it off and unplug it only if you can do so safely. '
               'Do not run another test, resume use, or open panels. '
               'If there is an active fire or immediate danger, move away and contact local emergency services. '
               f'Please call warranty support at **{phone}** for direct help. '
               'I cannot confirm that staff have been notified or a callback has been booked.'),
        'es': ('Deje de usar la silla. Apáguela y desenchúfela solo si puede hacerlo de forma segura. '
               'No haga otra prueba, no vuelva a usarla ni abra paneles. '
               'Si hay fuego activo o peligro inmediato, aléjese y contacte con los servicios de emergencia locales. '
               f'Llame directamente a soporte de garantía al **{phone}**. '
               'No puedo confirmar que se haya notificado al personal ni reservado una llamada.'),
        'ko': ('의자 사용을 즉시 중단하세요. 안전하게 할 수 있을 때만 전원을 끄고 플러그를 뽑으세요. '
               '다시 시험하거나 사용을 재개하지 말고 패널을 열거나 직접 분해하지 마세요. '
               '불이 나고 있거나 즉각적인 위험이 있으면 안전한 곳으로 이동하고 현지 긴급 구조 기관에 연락하세요. '
               f'보증 지원팀 **{phone}**로 직접 전화해 주세요. '
               '직원에게 알림이 전달됐거나 콜백이 예약됐는지는 확인되지 않았습니다.'),
    }[language]
    if cancel_request:
        text += {'en': ' No order cancellation has been confirmed.',
                 'es': ' No se ha confirmado la cancelación de ningún pedido.',
                 'ko': ' 주문 취소도 확인되지 않았습니다.'}[language]
    patch = {
        'sales_safety_active': True, 'sales_safety_report': original,
        'sales_safety_language': language, 'conversation_language': language,
        'sales_safety_order_action_requested': cancel_request,
        'awaiting_recommend': False, 'awaiting_email_for_pick': False,
        'awaiting_showroom_window': False, 'awaiting_showroom_email': False,
        'awaiting_pre_human_question': False, 'showroom_request_pending': False,
        'pending_showroom_window': '', 'pending_tier_picks': [],
        'pending_primary': '', 'pending_product_url': '', 'pending_pick_summary': '',
        'awaiting_compare_pick': '', 'awaiting_compare_recommend': False,
        'pending_compare_left': '', 'pending_compare_right': '',
        'pending_compare_left_query': '', 'pending_compare_right_query': '',
        'pending_compare_candidates': [], 'pending_compare_pair': [],
    }
    return SalesReply(reply=text, intent='warranty_redirect', handoff=True,
                      handoff_reason='safety_hazard', products=[], quick_replies=[],
                      tools_used=['safety.reported_hazard', 'stage:warranty'],
                      prefs_patch=patch, flow_stage='warranty')
