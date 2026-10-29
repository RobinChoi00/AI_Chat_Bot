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
    value = _fold(message)
    if re.search(r'\b(?:in|speak|reply|answer)\s+english\b|en ingles|영어로', value):
        return 'en'
    if re.search(r'\b(?:in|speak|reply|answer)\s+spanish\b|\bespanol\b|스페인어로', value):
        return 'es'
    if re.search(r'\b(?:in|speak|reply|answer)\s+korean\b|한국어로', value) or re.search('[가-힣]', message):
        return 'ko'
    if re.search(r'\b(?:hola|busco|quiero|necesito|recomienda\w*|recomendar\w*|sillon|silla|masaje|mido|peso|pesa|paquete|presupuesto|tope|puerta|hablar|persona|asesora?|ciatica|marcapasos|operacion)\b|[¿¡]', value):
        return 'es'
    return previous if previous in {'en', 'ko', 'es'} else 'en'


def prepare_message(message: str, prefs: dict) -> tuple[str, str]:
    """Normalize explicit Spanish shopping grammar while retaining all facts."""
    language = customer_language(message, str(prefs.get('conversation_language') or 'en'))
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


def is_medical_scope_question(message: str) -> bool:
    """Recognize requested clinical clearance or treatment, not ordinary fit."""
    value = _fold(message)
    eligibility = re.search(r'\b(?:pacemaker|pregnan\w*|marcapasos|embaraz\w*)\b|심박\s*조율기|임신|수술\s*후', value)
    surgery = re.search(r'\b(?:surgery|operacion|cirugia)\b', value) and re.search(r'\b(?:safe\w*|use|try|after|recent\w*|segur\w*|usar|probar|despues|reciente)\b', value)
    condition = r'(?:sciatica|scoliosis|arthritis|neuropathy|herniated\s+disc|back\s+pain|chronic\s+pain|disease|condition|ciatica|artritis|neuropatia|dolor|enfermedad|hernia(?:\s+de\s+disco)?)'
    treatment = r'(?:cur\w*|treat\w*|heal\w*|revers\w*|eliminat\w*|fix|sanar|trat\w*|get\s+rid\s+of)'
    guaranteed_relief = re.search(r'\b(?:guarantee\w*|promise\w*)\b[^.!?]{0,50}\b(?:pain\s+relief|relie\w*\s+pain)\b|\bgarant\w*\b[^.!?]{0,50}\bdolor\b|통증.{0,20}보장', value)
    return bool(eligibility or surgery or guaranteed_relief or re.search(rf'\b{treatment}\b[^.!?]{{0,70}}\b{condition}\b|\b{condition}\b[^.!?]{{0,70}}\b{treatment}\b', value)
                or re.search(r'(?:의자|안마|마사지).{0,40}(?:완치|치료|고칠|낫게)|(?:좌골|디스크|관절염|척추|허리\s*통증).{0,30}(?:완치|치료|고칠|낫게)', value))


def medical_scope_message(message: str, language: str = 'en') -> str | None:
    if not is_medical_scope_question(message):
        return None
    return {
        'en': "I can't promise a cure or treatment from a massage chair, or guarantee relief from pain or safe use for your individual health situation. Please ask your clinician before use, especially with a pacemaker, pregnancy, or recent surgery. A sales specialist can explain product features, but cannot give medical clearance.",
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


def localize_reply(result, language: str, prefs: dict):
    """Localize reviewed copy without rewriting provider or catalog evidence."""
    if language != 'es':
        return result
    # Most quick replies are shared static dataclass instances. Localizing one
    # customer's copy must never change another customer's English buttons.
    result.quick_replies = [replace(quick, label=_SPANISH_LABELS.get(quick.label, quick.label)) for quick in result.quick_replies]
    if any(tool in result.tools_used for tool in ['policy.medical_scope', 'policy.currency_scope']):
        return result
    shipping = any(tool in result.tools_used for tool in ['policy.shipping', 'policy.white_glove'])
    shipping_copy = ('El envío estándar es a pie de calle y no incluye montaje; el plazo publicado es de hasta 2 semanas. '
                     'White Glove añade entrega dentro de casa y montaje por un cargo adicional, con un plazo publicado de hasta 3 semanas. '
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
        lines.append('La comodidad y la intensidad deben comprobarse personalmente. Esta recomendación no reserva existencias ni confirma una fecha de entrega. Puede abrir la ficha del producto o hablar con un asesor.')
        result.reply = '\n\n'.join(lines)
    elif result.intent in {'unclear', 'greeting'}:
        result.reply = 'Puedo ayudarle a buscar un sillón, revisar precios y datos del catálogo, o solicitar ayuda de una persona. ¿Qué necesita?'
    elif shipping_copy:
        result.reply = shipping_copy
    elif result.intent == 'prepurchase_policy' and 'shopping chat' in result.reply:
        result.reply = 'Puedo ayudar con la búsqueda básica de sillones en español. Algunas políticas o fichas técnicas siguen disponibles en inglés; conservaré los datos originales y puedo ofrecer contacto con un asesor cuando no haya una traducción revisada.'
    return result
