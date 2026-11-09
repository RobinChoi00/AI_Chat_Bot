"""Reviewed language support for shopping inputs and customer-facing prompts.

This is deliberately a grammar adapter, not an automatic translation service.
Catalog facts are read from the original structured result, never translated or
filled in from memory. Unknown policy prose remains visible in its source form.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace


def _fold(text: str) -> str:
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))).replace('’', "'")


def _money_amount(raw: str) -> str:
    """Read explicit USD amounts without confusing Spanish thousands marks."""
    value = raw.rstrip('.,')
    if ',' in value and '.' in value:
        decimal, thousands = (',', '.') if value.rfind(',') > value.rfind('.') else ('.', ',')
        value = value.replace(thousands, '').replace(decimal, '.')
    elif re.fullmatch(r'\d{1,3}(?:[.,]\d{3})+', value):
        value = value.replace(',', '').replace('.', '')
    else:
        value = value.replace(',', '.')
    return value


def customer_language(message: str, previous: str = 'en') -> str:
    from customer_safety import requested_customer_language

    requested = requested_customer_language(message)
    if requested:
        return requested
    value = _fold(message)
    if re.search('[가-힣]', message):
        return 'ko'
    if re.search(r'\b(?:hola|busco|quiero|necesito|recomienda\w*|recomendar\w*|sillon|silla|masaje|mido|peso|pesa|paquete|presupuesto|tope|puerta|hablar|persona|asesora?|ciatica|marcapasos|operacion)\b|[¿¡]', value):
        return 'es'
    # A substantive new English question can change language naturally. Bare
    # model names, measurements and acknowledgments preserve the saved choice.
    if len(re.findall(r'\b[a-z]+\b', value)) >= 4 and re.search(
        r"\b(?:i\s+(?:am|would|want|need|weigh|have)|i'm|my\s+(?:chair|door|doorway|budget|order|father|mother|weight|height)|"
        r"(?:what|how|where|when|which|can|could|would|does|do|is|are)\b.*\b(?:chair|delivery|assembly|model|price|refund|order|return|massage|warranty|support|please)|"
        r"please\s+(?:recommend|explain|compare|show|connect)|tell\s+me\s+about)\b",
        value,
    ):
        return 'en'
    return previous if previous in {'en', 'ko', 'es'} else 'en'


def prepare_message(message: str, prefs: dict) -> tuple[str, str]:
    """Normalize explicit Spanish shopping grammar while retaining all facts."""
    language = customer_language(message, str(prefs.get('conversation_language') or 'en'))
    if language == 'ko':
        return _prepare_korean(message), language
    if language != 'es':
        return message, language
    text = message
    # A measurement must have a personal subject; product and room dimensions
    # cannot become a shopper's height or weight through this adapter.
    text = re.sub(r'\bmido\s+(\d(?:[.,]\d{1,2})?)\s*(?:metros?|m)\b',
                  lambda m: f'I am {float(m[1].replace(",", "."))*100:g} cm', text, flags=re.I)
    text = re.sub(r'\bmido\s+(\d{3})\s*(?:cent[ií]metros?|cm)\b', r'I am \1 cm', text, flags=re.I)
    text = re.sub(r'\bpeso\s+(\d{2,3}(?:[.,]\d+)?)\s*(kg|kilos?|libras?|lb[ss]?)\b',
                  lambda m: f'I weigh {m[1].replace(",", ".")} '+('kg' if m[2].lower().startswith(('kg', 'kilo')) else 'lb'), text, flags=re.I)
    if not re.search(r'€|\b(?:euros?|pesos|peso\s+mexicano|mxn|cad)\b', text, re.I):
        text = re.sub(r'\b(?:menos\s+de|por\s+debajo\s+de|hasta|m[aá]ximo(?:\s+de)?)\s*\$?\s*(\d[\d.,]*)\s*(?:d[oó]lares?|usd)\b',
                      lambda m: 'budget under $'+_money_amount(m[1]), text, flags=re.I)
        text = re.sub(r'\b(?:menos\s+de|hasta)\s*\$\s*(\d[\d.,]*)', lambda m: 'budget under $'+_money_amount(m[1]), text, flags=re.I)
        text = re.sub(r'\b(?:presupuesto|tope|l[ií]mite)\s*(?:m[aá]ximo\s*)?(?:(?:es|de|ahora)\s*)*\$?\s*(\d[\d.,]*)', lambda m: 'budget $'+_money_amount(m[1]), text, flags=re.I)
    text = re.sub(r'\b(?:la\s+|mi\s+)?puerta\s*(?:mide|es\s+de|tiene)?\s*(\d{2,3}(?:[.,]\d+)?)\s*(cm|cent[ií]metros?|pulgadas?)\b',
                  lambda m: 'my doorway is '+(f'{float(m[1].replace(",", "."))/2.54:.8f}' if m[2].lower().startswith(('cm', 'cent')) else m[1].replace(',', '.'))+' inches', text, flags=re.I)
    replacements = [
        (r'\b(?:sin\s+desmontar|no\s+se\s+puede\s+desmontar|solo\s+montad[oa])\b', 'assembled only'),
        (r'\b(?:puede\s+desmontarse|se\s+puede\s+desmontar)\b', 'OK to disassemble'),
        (r'\b(?:no\s+quiero|no\s+necesito)\s+hablar\s+con\s+(?:una?\s+)?(?:persona|asesor[ae]?|agente)\b', "I don't want to talk to a human"),
        (r'\b(?:puedo|quiero|necesito|prefiero)\s+hablar\s+con\s+(?:una?\s+)?(?:persona|asesor[ae]?|agente)\b', 'connect me to a human'),
        (r'\b(?:hablar\s+con\s+(?:una?\s+)?(?:persona|asesor[ae]?|agente)|(?:con[eé]ctame|comun[ií]came)\s+con\s+(?:(?:una?\s+)?(?:persona|asesor[ae]?|agente)|alguien\s+del\s+equipo))\b', 'connect me to a human'),
        (r'\b(?:busco|quiero|necesito)\s+(?:una?\s+)?(?:sill[oó]n|silla)\s*(?:de\s+masaje)?\b', 'recommend a chair'),
        (r'\brecomi[eé]nd(?:ame|a)|\brecomendar\w*', 'recommend'),
        (r'\b(?:espalda\s+baja|zona\s+lumbar)\b', 'lower back'),
        (r'\bcuello\b', 'neck'), (r'\bhombros\b', 'shoulders'),
        (r'\bpantorrillas\b', 'calves'), (r'\bpies\b', 'feet'),
        (r'\b(?:relajaci[oó]n\s+(?:de\s+)?cuerpo\s+completo|cuerpo\s+completo)\b', 'full body relaxation'),
        (r'\brelajarme\b', 'full body relaxation'),
        (r'\b(?:masaje\s+)?suave\b', 'gentle massage'),
        (r'\b(?:masaje\s+)?fuerte\b', 'strong massage'),
        (r'\b(?:sin\s+restricciones?\s+de\s+espacio|no\s+hay\s+problema\s+de\s+espacio)\b', 'no doorway constraint'),
        (r'\b(?:cu[aá]nto\s+cuesta|qu[eé]\s+precio\s+tiene)\b', 'what is the price of'),
        (r'\b(?:hay\s+existencias|est[aá]\s+disponible|tienen\s+en\s+stock)\b', 'is it in stock'),
        (r'\b(?:compara|comparar|comparaci[oó]n)\b', 'compare'),
        (r'\benv[ií]o\b', 'shipping'),
        (r'\bincluye\s+montaje\b', 'includes assembly'),
        (r'\b(?:sill[oó]n|silla)\b', 'chair'),
        (r'\bpaquete\b', 'package'),
        (r'\bpesa\b', 'weighs'),
        (r'\besposa\b', 'wife'), (r'\besposo\b', 'husband'),
        (r'\bhola\b', 'hello'),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text, language


_KOREAN_SHOPPING = (
    (r'키(?:는|가)?\s*(\d{3})\s*(?:cm|센티(?:미터)?)', r' I am \1 cm '),
    (r"키(?:는|가)?\s*(\d)\s*(?:피트|ft|')\s*(\d{1,2})", r" I am \1'\2 "),
    (r'(?:몸무게|체중)(?:는|이|가)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|킬로(?:그램)?)', r' I weigh \1 kg '),
    (r'(?:몸무게|체중)(?:는|이|가)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lbs?|파운드)', r' I weigh \1 lb '),
    (r'예산(?:은|는|이)?\s*\$?\s*(\d[\d,]*)\s*(?:달러|불|usd)', r' budget under $\1 '),
    (r'\$\s*(\d[\d,]*)\s*(?:이하|미만|까지|이내)', r' budget under $\1 '),
    (r'(?:분해해도|분해\s*가능|분해할\s*수\s*있)', ' OK to disassemble '),
    (r'조립(?:된)?\s*상태(?:로)?(?:만)?', ' assembled only '),
    (r'(?:허리|요추)', ' lower back '),
    (r'목(?=\s|과|·|,|이랑|하고|\s*어깨|$)', ' neck '), (r'어깨', ' shoulders '),
    (r'종아리', ' calves '), (r'발(?:바닥)?(?=\s|과|·|,|이랑|하고|$)', ' feet '),
    (r'전신', ' full body relaxation '),
    (r'(?:부드러운|부드럽게|약하게|약한)', ' gentle massage '),
    (r'(?:강한|강하게|세게)', ' strong massage '),
)


def _prepare_korean(message: str) -> str:
    """Add English shopping cues beside Korean facts; the original text is kept."""
    text = message
    door = re.search(r'(?:문|현관문?|출입문|입구)\s*(?:폭|너비)?(?:은|는|이|가)?\s*(\d{2,3}(?:\.\d+)?)\s*(cm|센티(?:미터)?|인치)', text)
    for pattern, replacement in _KOREAN_SHOPPING:
        text = re.sub(pattern, replacement, text)
    if door:
        width = float(door.group(1)) / (2.54 if not door.group(2).startswith('인치') else 1)
        text += f' my doorway is {width:.8f} inches'
    if re.search(r'의자|안마|마사지', message) and re.search(r'추천|찾고\s*있|찾아|사고\s*싶|구매|구입|알아보', message):
        text += ' recommend a chair'
    return re.sub(r'\s{2,}', ' ', text).strip()


def is_medical_scope_question(message: str) -> bool:
    """Recognize requested clinical clearance or treatment, not ordinary fit."""
    value = _fold(message)
    eligibility = re.search(r'\b(?:pacemaker|pregnan\w*|(?:first|second|third)\s+trimester|implanted\s+defibrillator|implantable\s+(?:cardioverter\s+)?defibrillator|marcapasos|embaraz\w*|desfibrilador)\b|심박\s*조율기|임신|수술\s*후', value)
    icd = re.search(r'\bicd\b', value) and re.search(r'\b(?:implant\w*|heart|cardiac|defibrillator|safe|use)\b', value)
    surgery = re.search(r'\b(?:surgery|operacion|cirugia|operaron|operado|operada|postoperative|post[ -]op)\b', value) and re.search(r'\b(?:safe\w*|use|try|after|recent\w*|segur\w*|usar|probar|despues|reciente|sin\s+riesgo|hace\s+una\s+semana)\b', value)
    condition = r'(?:sciatica|scoliosis|arthritis|neuropathy|herniated\s+disc|back\s+pain|chronic\s+pain|disease|condition|ciatica|artritis|neuropatia|dolor|enfermedad|hernia(?:\s+de\s+disco)?)'
    treatment = r'(?:cur\w*|treat\w*|heal\w*|revers\w*|eliminat\w*|fix|sanar|trat\w*|get\s+rid\s+of)'
    guaranteed_relief = re.search(r'\b(?:guarantee\w*|promise\w*)\b[^.!?]{0,50}\b(?:pain\s+relief|relie\w*\s+pain)\b|\bgarant\w*\b[^.!?]{0,50}\bdolor\b|통증.{0,20}보장', value)
    return bool(eligibility or icd or surgery or guaranteed_relief or re.search(rf'\b{treatment}\b[^.!?]{{0,70}}\b{condition}\b|\b{condition}\b[^.!?]{{0,70}}\b{treatment}\b', value)
                or re.search(r'(?:의자|안마|마사지).{0,40}(?:완치|치료|고칠|낫게)|(?:좌골|디스크|관절염|척추|허리\s*통증).{0,30}(?:완치|치료|고칠|낫게)', value))


def medical_scope_message(message: str, language: str = 'en') -> str | None:
    if not is_medical_scope_question(message):
        return None
    return {
        'en': "I can't promise that a massage chair will treat a medical condition, relieve pain, or be safe for your individual health situation. Please ask your clinician before using it, especially during pregnancy, after surgery, or with a pacemaker or implanted defibrillator. A sales specialist can explain product features but cannot provide medical clearance.",
        'es': 'No puedo prometer que un sillón de masaje cure o trate la ciática u otra enfermedad, ni confirmar que sea seguro para su situación médica. Consulte con su profesional de salud antes de usarlo, especialmente si tiene marcapasos, está embarazada o tuvo una cirugía reciente. Un asesor de ventas puede explicar funciones del producto, pero no autorizar su uso médico.',
        'ko': '마사지 의자가 좌골신경통 등 질환을 치료하거나 완치한다고 약속할 수 없으며, 개인의 건강 상태에서 안전한지도 판단할 수 없습니다. 심박 조율기 사용, 임신 또는 최근 수술을 포함한 의학적 사용 가능 여부는 사용 전에 담당 의료진에게 확인해 주세요. 판매 상담원은 제품 기능을 설명할 수 있지만 의학적 사용 허가를 할 수는 없습니다.',
    }.get(language, '') or medical_scope_message(message, 'en')


def currency_scope_message(message: str, language: str) -> str | None:
    if language != 'es' or not re.search(r'€|\b(?:euros?|pesos|peso\s+mexicano|mxn|cad)\b', message, re.I):
        return None
    return 'Los precios de este catálogo están en dólares estadounidenses (USD). No tengo un tipo de cambio confirmado y no convertiré su presupuesto automáticamente. ¿Cuál es su presupuesto máximo en USD?'


_SPANISH_PROMPTS = {
    'ask_height': '¿Cuál es la estatura de la persona que usará el sillón? Puede escribirla en centímetros o metros, por ejemplo: «Mido 1,80 metros».',
    'ask_weight': '¿Cuánto pesa la persona que usará el sillón? Puede indicarlo en kg o libras.',
    'ask_goal': '¿Qué zona o tipo de masaje le interesa más: cuello y hombros, espalda baja, pies y pantorrillas o relajación de cuerpo completo?',
    'ask_intensity': '¿Prefiere un masaje suave, medio o fuerte?',
    'ask_foot': '¿Son importantes los rodillos o el amasado para pies y pantorrillas?',
    'ask_space': '¿Hay límites de espacio o una puerta estrecha en el recorrido de entrega? Indique la medida real de la puerta si la conoce.',
    'ask_doorway': '¿Cuánto mide la abertura más estrecha por la que debe pasar el sillón? Puede escribir, por ejemplo: «Mi puerta mide 74 cm».',
    'ask_doorway_in': '¿Cuánto mide la abertura más estrecha por la que debe pasar el sillón? Puede escribir, por ejemplo: «Mi puerta mide 74 cm».',
    'ask_doorway_fit': '¿Debe pasar completamente montado o permite desmontarlo? Confirmaremos el ancho publicado que corresponde a esa opción.',
}

_SPANISH_LABELS = {
    'Talk to a human': 'Hablar con una persona', 'Recommend a chair': 'Recomendar un sillón',
    'Check a price': 'Consultar un precio', 'Availability / stock': 'Consultar existencias',
    'Compare two models': 'Comparar dos modelos', 'Neck & shoulders': 'Cuello y hombros',
    'Lower back': 'Espalda baja', 'Foot & calf': 'Pies y pantorrillas',
    'Full-body relax': 'Relajación de cuerpo completo', 'Gentle': 'Suave',
    'Medium': 'Medio', 'Strong': 'Fuerte', 'Assembled only': 'Solo montado',
    'OK to disassemble': 'Se permite desmontar', 'No space issue': 'Sin límite de espacio',
    'Not sure': 'No estoy seguro', 'Share contact for follow-up': 'Compartir contacto para seguimiento',
    'Continue chatting': 'Seguir conversando', 'Back to menu': 'Volver al menú',
    'That helped': 'Me resultó útil', 'Not what I needed': 'No era lo que necesitaba',
    'Email me this pick': 'Enviar esta opción por correo', 'Email me these picks': 'Enviar estas opciones por correo',
    'View this chair': 'Ver este sillón', 'Shop this chair': 'Ver este sillón',
    'Visit showroom': 'Visitar la sala de exposición', 'Prefer stronger': 'Prefiero más fuerte',
    'Prefer gentler': 'Prefiero más suave', 'Foot rollers matter': 'Importan los rodillos para pies',
    'Tight space / doorway': 'Espacio reducido o puerta estrecha',
    'Widen doorway / skip': 'Revisar la medida o saltar',
}


_KOREAN_PROMPTS = {
    'ask_height': '의자를 사용하실 분의 키가 어떻게 되나요? 센티미터나 피트로 적어 주세요. 예: 「키는 175cm입니다」',
    'ask_weight': '사용하실 분의 몸무게는 어떻게 되나요? kg 또는 파운드로 알려 주세요.',
    'ask_goal': '어느 부위의 마사지를 가장 원하시나요? 목·어깨, 허리, 발·종아리, 전신 이완 중에서 골라 주세요.',
    'ask_intensity': '부드러운 마사지, 중간 강도, 강한 마사지 중 어느 쪽을 선호하시나요?',
    'ask_foot': '발과 종아리 롤러 마사지가 중요하신가요?',
    'ask_space': '배송 경로에 공간 제약이나 좁은 문이 있나요? 알고 계시면 실제 문 폭을 알려 주세요.',
    'ask_doorway': '의자가 지나가야 하는 가장 좁은 문의 폭은 얼마인가요? 예: 「문 폭은 76cm입니다」',
    'ask_doorway_in': '의자가 지나가야 하는 가장 좁은 문의 폭은 얼마인가요? 예: 「문 폭은 76cm입니다」',
    'ask_doorway_fit': '완전히 조립된 상태로 들어가야 하나요, 아니면 분해해서 옮겨도 되나요? 그에 맞는 공개 폭으로 확인하겠습니다.',
}

_KOREAN_LABELS = {
    'Talk to a human': '상담원 연결', 'Recommend a chair': '의자 추천',
    'Check a price': '가격 확인', 'Check the price': '가격 확인', 'Availability / stock': '재고 확인',
    'Compare two models': '두 모델 비교', 'Neck & shoulders': '목·어깨',
    'Lower back': '허리', 'Foot & calf': '발·종아리',
    'Full-body relax': '전신 이완', 'Gentle': '부드럽게',
    'Medium': '중간', 'Strong': '강하게', 'Assembled only': '조립된 상태로만',
    'OK to disassemble': '분해해도 됨', 'No space issue': '공간 제약 없음',
    'Not sure': '잘 모르겠어요', 'Share contact for follow-up': '후속 연락처 남기기',
    'Continue chatting': '계속 대화하기', 'Back to menu': '메뉴로 돌아가기',
    'That helped': '도움이 됐어요', 'Not what I needed': '원하던 답이 아니에요',
    'Email me this pick': '이 제품 이메일로 받기', 'Email me these picks': '추천 목록 이메일로 받기',
    'View this chair': '제품 보기', 'Shop this chair': '제품 보기',
    'Visit showroom': '쇼룸 방문', 'Prefer stronger': '더 강하게',
    'Prefer gentler': '더 부드럽게', 'Foot rollers matter': '발 롤러가 중요해요',
    'Tight space / doorway': '좁은 공간 또는 문', 'Widen doorway / skip': '문 폭 다시 확인 또는 건너뛰기',
    'Keep shopping': '계속 둘러보기', 'See all models': '전체 모델 보기', 'Check doorway fit': '문 폭 확인',
    'Under 5\'4"': '163cm 미만 (5\'4" 미만)', '5\'4"–5\'11"': '163–180cm (5\'4"–5\'11")',
    '6\'0"–6\'2"': '183–188cm (6\'0"–6\'2")', '6\'3"+': '191cm 이상 (6\'3"+)',
}

_KOREAN_SOURCE_NOTE = '이 내용은 검토된 한국어 번역이 아직 없어 원문(영어) 그대로 안내드립니다. 제품 정보와 수치는 원문과 같습니다.'


def _localize_korean(result, prefs: dict):
    result.quick_replies = [replace(quick, label=_KOREAN_LABELS.get(quick.label, quick.label)) for quick in result.quick_replies]
    if any(tool in result.tools_used for tool in ['policy.medical_scope', 'policy.currency_scope']):
        return result
    shipping = any(tool in result.tools_used for tool in ['policy.shipping', 'policy.white_glove'])
    shipping_copy = ('기본 배송은 집 앞(커브사이드) 배송이며 조립은 포함되지 않습니다. 공개된 배송 기간은 최대 2주입니다. '
                     'White Glove(화이트 글러브) 배송은 실내 반입과 조립이 포함되며, 공개된 기본 요금은 $275입니다. '
                     '계단 추가 인력, 현장 시간, 거리 등에 따라 금액이 더 붙을 수 있습니다. 공개된 배송 기간은 최대 3주입니다. '
                     'White Glove 요금은 환불되지 않습니다. 이 기간은 특정 배송 날짜를 확정하는 것이 아닙니다.') if shipping else ''
    if shipping:
        urls = re.findall(r'https://[^\s)]+', result.reply)
        if urls:
            shipping_copy += '\n\n공개된 배송 정책: ' + urls[0]
    if result.flow_stage in _KOREAN_PROMPTS and 'cases.clarify' in result.tools_used:
        result.reply = (shipping_copy + '\n\n' if shipping_copy else '') + _KOREAN_PROMPTS[result.flow_stage]
    elif result.intent == 'human':
        result.reply = (('영업 상담원 연결 요청이 접수되었습니다. 이 대화에 남기신 내용은 다시 말씀하지 않으셔도 됩니다. '
                         if 'human.requested' in result.tools_used else
                         '질문을 명확하게 해결해 드리지 못했습니다. 영업 상담원이 도와드릴 수 있으니 상담원 연결 버튼을 이용해 주세요. ')
                        + '아직 상담원이 대화에 참여했거나 통화가 예약된 것은 아닙니다. '
                        '**+1-888-848-2630 ext. 2**로 전화하셔도 됩니다. 후속 연락을 위한 연락처 공유는 선택 사항입니다.')
        if prefs.get('medical_scope_pending'):
            result.reply += ' 상담원은 제품 기능을 설명할 수 있지만, 의학적 사용 가능 여부는 담당 의료진에게 확인하셔야 합니다.'
    elif result.flow_stage == 'recommend_nofit':
        rec = (result.prefs_patch or {}).get('recommend_prefs') or prefs.get('recommend_prefs') or {}
        limits = []
        for key, label, unit in [('budget_max_usd', '최대 예산', 'USD'), ('weight_lb', '최소 사용자 체중', 'lb'), ('doorway_in', '문 폭', '인치')]:
            if rec.get(key) and rec[key] != 'skip':
                limits.append(f'{label}: {rec[key]} {unit}')
        result.reply = ('등록하신 조건' + (' (' + '; '.join(limits) + ')' if limits else '') +
                        '을 모두 충족하는 것으로 공개된 의자를 찾지 못했습니다. 확인할 수 없는 제품은 추천하지 않겠습니다. '
                        '다른 배송 방법을 검토하거나 상담원에게 문의하실 수 있습니다. 실제 치수를 제품에 맞춰 바꾸지는 마세요.')
    elif result.intent == 'recommend' and result.products:
        lines = ['등록하신 조건에 맞는 카탈로그 옵션입니다:']
        for product in result.products:
            model, price = product.get('model'), product.get('price_usd')
            if model:
                url = product.get('product_url')
                lines.append(f'- **{model}**' + (f' — **${float(price):,.0f} USD**' if isinstance(price, (int, float)) and not isinstance(price, bool) else ' — 가격 미확인')
                             + (f'\n  {url}' if url else ''))
        lines.append('편안함과 강도는 직접 체험해 확인하시는 것이 좋습니다. 이 추천은 재고를 확보하거나 배송 날짜를 확정하지 않습니다.'
                     + _height_sentence(prefs, result, 'ko'))
        result.reply = '\n\n'.join(lines)
    elif result.intent in {'unclear', 'greeting'}:
        result.reply = '의자 추천, 가격과 카탈로그 정보 확인, 상담원 연결을 도와드릴 수 있습니다. 무엇이 필요하신가요?'
    elif shipping_copy:
        result.reply = shipping_copy
    elif not re.search('[가-힣]', result.reply or ''):
        result.reply = _KOREAN_SOURCE_NOTE + '\n\n' + (result.reply or '')
    return result


def rec_has_height(prefs: dict, result) -> bool:
    rec = (result.prefs_patch or {}).get('recommend_prefs') or prefs.get('recommend_prefs') or {}
    return bool(rec.get('height_in'))


def _height_sentence(prefs: dict, result, language: str) -> str:
    """The company height guidance (5'0"–6'2" standard, 6'3"–6'5" XL) in Korean or Spanish."""
    from sales_height import STANDARD_MAX_IN, STANDARD_MIN_IN, XL_MAX_IN

    rec = (result.prefs_patch or {}).get('recommend_prefs') or prefs.get('recommend_prefs') or {}
    try:
        height = float(rec.get('height_in') or 0)
    except (TypeError, ValueError):
        return ''
    if not height:
        return ''
    cm = round(height * 2.54)
    if language == 'ko':
        if height < STANDARD_MIN_IN or height > XL_MAX_IN:
            return (f' 키 {cm}cm는 공개된 설계 기준 범위(일반 라인 152–188cm, XL 라인 191–196cm) 밖이라 '
                    '쇼룸 체험이나 상담원 확인을 권해 드립니다.')
        if height <= STANDARD_MAX_IN:
            return f' 키 {cm}cm는 일반 라인의 설계 기준 범위(152–188cm) 안에 있습니다.'
        return f' 키 {cm}cm는 일반 라인 기준(188cm)을 넘어, 191–196cm용으로 설계된 XL 라인으로 추천해 드렸습니다.'
    if height < STANDARD_MIN_IN or height > XL_MAX_IN:
        return (f' Una estatura de {cm} cm está fuera de los rangos de diseño publicados (152–188 cm línea estándar, '
                '191–196 cm línea XL); le recomendamos probarlo en el showroom o consultar con un asesor.')
    if height <= STANDARD_MAX_IN:
        return f' Una estatura de {cm} cm está dentro del rango de diseño de nuestra línea estándar (152–188 cm).'
    return f' Una estatura de {cm} cm supera la línea estándar (188 cm), por eso le recomendamos la línea XL, diseñada para 191–196 cm.'


def localize_reply(result, language: str, prefs: dict):
    """Localize reviewed copy without rewriting provider or catalog evidence."""
    if language == 'ko':
        return _localize_korean(result, prefs)
    if language != 'es':
        return result
    # Most quick replies are shared static dataclass instances. Localizing one
    # customer's copy must never change another customer's English buttons.
    result.quick_replies = [replace(quick, label=_SPANISH_LABELS.get(quick.label, quick.label)) for quick in result.quick_replies]
    if any(tool in result.tools_used for tool in ['policy.medical_scope', 'policy.currency_scope']):
        return result
    shipping = any(tool in result.tools_used for tool in ['policy.shipping', 'policy.white_glove'])
    shipping_copy = ('El envío estándar es a pie de calle y no incluye montaje; el plazo publicado es de hasta 2 semanas. '
                     'White Glove añade entrega dentro de casa y montaje. La tarifa básica publicada es de $275; '
                     'pueden aplicarse cargos extra por escaleras, tiempo en sitio o distancia. El plazo publicado es de hasta 3 semanas. '
                     'El cargo de White Glove no es reembolsable. Estos plazos no confirman una fecha concreta.') if shipping else ''
    if shipping:
        urls = re.findall(r'https://[^\s)]+', result.reply)
        if urls:
            shipping_copy += '\n\nPolítica publicada: '+urls[0]
    if result.flow_stage in _SPANISH_PROMPTS and 'cases.clarify' in result.tools_used:
        result.reply = (shipping_copy+'\n\n' if shipping_copy else '')+_SPANISH_PROMPTS[result.flow_stage]
    elif result.intent == 'human':
        result.reply = (('Se ha guardado su solicitud de hablar con un asesor de ventas. No necesita repetir los datos de esta conversación. '
                         if 'human.requested' in result.tools_used else
                         'No he podido resolver su pregunta con claridad. Un asesor de ventas puede ayudar; use la opción para hablar con una persona. ')
                        + 'Todavía no se ha confirmado que una persona se haya unido ni se ha reservado una llamada. '
                        'También puede llamar al **+1-888-848-2630 ext. 2**. Compartir su contacto para seguimiento es opcional.')
        if prefs.get('medical_scope_pending'):
            result.reply += ' El asesor puede explicar funciones del producto; la autorización médica debe darla su profesional de salud.'
    elif result.flow_stage == 'recommend_nofit':
        rec = (result.prefs_patch or {}).get('recommend_prefs') or prefs.get('recommend_prefs') or {}
        limits = []
        for key, label, unit in [('budget_max_usd', 'presupuesto máximo', 'USD'), ('weight_lb', 'capacidad mínima', 'lb'), ('doorway_in', 'abertura de puerta', 'pulgadas')]:
            if rec.get(key) and rec[key] != 'skip':
                limits.append(f'{label}: {rec[key]} {unit}')
        result.reply = 'No encontré un sillón con datos publicados que cumpla los límites registrados'+(' ('+'; '.join(limits)+')' if limits else '')+'. No recomendaré uno que no pueda verificar. Podemos revisar otras opciones de entrega o consultar con un asesor; no cambie una medida real para hacerla coincidir con un producto.'
    elif result.intent == 'recommend' and result.products:
        lines = ['Opciones del catálogo para los requisitos registrados:']
        for product in result.products:
            model, price = product.get('model'), product.get('price_usd')
            if model:
                lines.append(f'- **{model}**'+(f' — **${float(price):,.2f} USD**' if isinstance(price, (int, float)) and not isinstance(price, bool) else ' — precio sin confirmar'))
        lines.append('La comodidad y la intensidad deben comprobarse personalmente. Esta recomendación no reserva existencias ni confirma una fecha de entrega. Puede abrir la ficha del producto o hablar con un asesor.'
                     + _height_sentence(prefs, result, 'es'))
        result.reply = '\n\n'.join(lines)
    elif result.intent in {'unclear', 'greeting'}:
        result.reply = 'Puedo ayudarle a buscar un sillón, revisar precios y datos del catálogo, o solicitar ayuda de una persona. ¿Qué necesita?'
    elif shipping_copy:
        result.reply = shipping_copy
    elif result.intent == 'prepurchase_policy' and 'shopping chat' in result.reply:
        result.reply = 'Puedo ayudar con la búsqueda básica de sillones en español. Algunas políticas o fichas técnicas siguen disponibles en inglés; conservaré los datos originales y puedo ofrecer contacto con un asesor cuando no haya una traducción revisada.'
    return result
