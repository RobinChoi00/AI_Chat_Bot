"""
Deterministic intent → tool routing for the chat agent.

Forces the first tool call on high-risk intents (repair, tracking, price,
recommendations) so the LLM cannot stall or answer from training data.
"""

from __future__ import annotations

import re
from typing import Optional

_PRODUCT_INTENT_PATTERNS = [
    re.compile(r"\b(i'?m|i am)\s+(interested\s+in|looking\s+(at|for))\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+(more\s+)?about\b", re.IGNORECASE),
    re.compile(r"\b(specs?|specifications?|dimensions?|price|weight|features?)\b.*\b(of|for)\b", re.IGNORECASE),
    re.compile(
        r"\b(what\s+is|how\s+much\s+(is|does))\b.*\b("
        r"osaki|titan|hypnos|nova|amamedic|maestro|orion|fleetwood|champ|atai|soho|duke|epic|aether|vera|"
        r"solo|otamic|maji|vero|flagship|supreme|grand|jade|sol|flex|duo|pro|max|elite"
        r")\b",
        re.IGNORECASE,
    ),
]

_PRICE_INTENT_PATTERNS = [
    re.compile(r"\b(how\s+much|what'?s?\s+the\s+price|price\s+of|cost\s+of|what\s+does\s+.+\s+cost)\b", re.IGNORECASE),
    re.compile(r"\b(is\s+it\s+on\s+sale|any\s+discount|current\s+price)\b", re.IGNORECASE),
    re.compile(r"(가격|얼마|할인|세일)"),
    re.compile(r"\b(cu[aá]nto\s+(cuesta|vale)|precio|descuento|oferta)\b", re.IGNORECASE),
]

_RECOMMEND_INTENT_PATTERNS = [
    re.compile(r"\b(recommend|suggest|best\s+(?:massage\s+)?chair|top\s+\d+\s+chair)\b", re.IGNORECASE),
    re.compile(r"\b(which\s+chair|what\s+chair\s+(?:should|would|do)\s+(?:you|i))\b", re.IGNORECASE),
    re.compile(r"\b(under|around|about|budget\s+of|less\s+than)\s+\$?\d", re.IGNORECASE),
    re.compile(r"\bcompare\s+(?:the\s+)?(?:chairs?|models?)\b", re.IGNORECASE),
    re.compile(r"(추천|어떤\s*(의자|모델)|비교해)"),
    re.compile(r"\b(recomienda|recomendar|qu[eé]\s+silla|comparar\s+modelos)\b", re.IGNORECASE),
]

_REPAIR_INTENT_PATTERNS = [
    re.compile(r"\b(install(ation|ing)?|assembl(e|y|ing)|set\s*up|setup)\b", re.IGNORECASE),
    re.compile(r"\b(repair|troubleshoot|fix|broken|not\s+working|wo[nN]?'?t\s+(turn|power|start)|stopped\s+working)\b", re.IGNORECASE),
    re.compile(r"\b(error\s+code|err\s*\d+|e\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(not\s+inflating|leaking|noise|squeak|grind|stuck|jammed)\b", re.IGNORECASE),
    re.compile(r"\b(replace(ment)?|swap)\s+(the\s+)?(controller|remote|mech|roller|airbag|cable|cord|adapter)\b", re.IGNORECASE),
    re.compile(r"(고장|작동\s*안|켜지지|수리|조립|설치|에러\s*코드)"),
    re.compile(r"\b(no\s+(enciende|funciona)|averiad[oa]|reparar|arreglar|montar|instalar|c[oó]digo\s+de\s+error)\b", re.IGNORECASE),
]

_TRACKING_INTENT_PATTERNS = [
    re.compile(r"\b(track|tracking|where\s+is\s+my|order\s+(status|update|tracking)|delivery\s+(status|update))\b", re.IGNORECASE),
    re.compile(r"(배송\s*(조회|상태)|주문\s*(조회|상태)|어디쯤)"),
    re.compile(r"\b(rastrear|seguimiento|estado\s+de\s+(mi\s+)?pedido|d[oó]nde\s+est[aá]\s+mi\s+pedido)\b", re.IGNORECASE),
]

_ORDER_ID_PATTERN = re.compile(r"\b(OSKMC|OSKUS|TIDM|OSK|TI)\d{3,7}\b", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[\w\.\-+]+@[\w\.\-]+\.\w+")

_WARRANTY_CLAIM_PATTERNS = [
    re.compile(r"\b(warranty\s*(claim|request|ticket|issue|case|service|form)|file\s+a\s+warranty|under\s+warranty|submit\s+warranty)\b", re.IGNORECASE),
    re.compile(r"\b(defect(ive)?|malfunction(ing)?)\b", re.IGNORECASE),
    re.compile(r"\b(delivery\s+(damage[d]?|issue|problem|wrong)|damaged\s+in\s+transit|box\s+was\s+(damage[d]?|opened|crushed))\b", re.IGNORECASE),
    re.compile(r"\b(i\s+want|i\s+need|i\s+'d\s+like|please)\s+(a\s+)?(replacement|refund|exchange|RMA|repair\s+service|compensation)\b", re.IGNORECASE),
    re.compile(r"\b(my|the)\s+(chair|unit|product|massage\s+chair)\s+(is\s+)?(not\s+working|broken|defective|damaged|stopped)\b", re.IGNORECASE),
    re.compile(r"\bfile\s+(a\s+)?(claim|warranty|ticket|complaint)\b", re.IGNORECASE),
    re.compile(r"(보증\s*(신청|청구|접수|문제)|워런티|교환\s*신청)"),
    re.compile(r"\b(garant[ií]a\s+(reclamaci[oó]n|solicitud|problema|caso)|presentar\s+una\s+reclamaci[oó]n|solicitar\s+reemplazo)\b", re.IGNORECASE),
]

_SHOWROOM_INTENT_PATTERNS = [
    re.compile(r"\b(showroom|show\s*room)\b", re.IGNORECASE),
    re.compile(r"\b(where\s+(is|are)\s+(?:your|the)\s+(?:office|store|company|headquarters|hq)|company\s+location|store\s+location|office\s+location|headquarters)\b", re.IGNORECASE),
    re.compile(r"\b(can\s+i\s+visit|come\s+see\s+(?:the\s+)?chairs?|try\s+(?:the\s+)?chairs?\s+in\s+person|in[-\s]?store)\b", re.IGNORECASE),
    re.compile(r"\b(your|company)\s+(address|location)\b", re.IGNORECASE),
    re.compile(r"(쇼룸|전시장|매장\s*위치|회사\s*주소)"),
    re.compile(r"\b(sala\s+de\s+exhibici[oó]n|tienda|direcci[oó]n|ubicaci[oó]n)\b", re.IGNORECASE),
]


def infer_forced_tool(user_query: str) -> Optional[str]:
    """Return the tool name to force on the first agent turn, or None."""
    q = (user_query or "").strip()
    if not q:
        return None

    has_order_id = bool(_ORDER_ID_PATTERN.search(q))
    has_email = bool(_EMAIL_PATTERN.search(q))

    if any(p.search(q) for p in _SHOWROOM_INTENT_PATTERNS):
        return "get_showroom_info"

    if any(p.search(q) for p in _WARRANTY_CLAIM_PATTERNS):
        return "start_warranty_workflow"

    if any(p.search(q) for p in _REPAIR_INTENT_PATTERNS):
        return "get_repair_help"

    if any(p.search(q) for p in _TRACKING_INTENT_PATTERNS) and (has_order_id or has_email):
        return "lookup_order_status"
    if has_order_id and has_email:
        return "lookup_order_status"

    if any(p.search(q) for p in _RECOMMEND_INTENT_PATTERNS):
        return "recommend_chairs"

    if any(p.search(q) for p in _PRICE_INTENT_PATTERNS):
        return "search_chair_specs"

    if any(p.search(q) for p in _PRODUCT_INTENT_PATTERNS):
        return "search_chair_specs"

    return None
