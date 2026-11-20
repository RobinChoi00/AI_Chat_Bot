"""Narrow, high-confidence intents found by auditing rated sales answers."""

from __future__ import annotations

import re

_STOCK = re.compile(r"\b(?:in stock|out of stock|have\s+(?:it|this|that)\s+in\s+stock|stock availability|available\s+(?:to buy|to order))\b", re.I)
_PARTS = re.compile(
    r"\b(?:replacement|replace|order|buy|get|need)\b[^.!?]{0,30}\b(?:upholstery|remote(?:\s+control)?)\b"
    r"|\b(?:loose|unmatched|disconnected|unplugged|unidentified)\s+(?:wire|connector|connection)\b"
    r"|\b(?:connector|connection)\b[^.!?]{0,48}\b(?:no\s+mate|doesn['’]?t\s+(?:seem\s+to\s+)?have\s+a\s+mate)\b", re.I,
)
_PARTS_MODEL_RESTATEMENT = re.compile(
    r"(?:osaki\s+)?(?:os[-\s]*3d\s+(?:cyber\s+pro|pro\s+cyber)|(?:os[-\s]*)?4000xt)"
    r"(?:\s+massage\s+chair)?(?:[\s,]+(?:red|black|brown|taupe|beige))?[.!?\s]*", re.I,
)
_GUIDE = re.compile(r"\b(?:manual|video|tutorial|instructions?|guide|help\s+with\s+(?:the\s+)?settings)\b", re.I)
_OWNER = re.compile(r"\b(?:have|own|purchased|bought|my|got)\b", re.I)
_VIVANT = re.compile(r"\bvivant\s*5d\b", re.I)
_YOGA = re.compile(r"\bsimilar\s+to\b[^.!?]{0,30}\byoga\s*flex\b", re.I)
_HIGHER_CAPACITY = re.compile(r"\b(?:accommodat\w*|support\w*|hold\w*|handle\w*)\s+more\s+weight\b|\b(?:higher|greater)\s+(?:user\s+)?weight\s+(?:limit|capacity)\b", re.I)


def requests_stock(message: str) -> bool:
    return bool(_STOCK.search(message))


def replacement_parts_answer(message: str, *, continuing_parts: bool) -> str | None:
    if not (_PARTS.search(message) or continuing_parts and _PARTS_MODEL_RESTATEMENT.fullmatch(message.strip())):
        return None
    # Do not turn a hypothetical feature/coverage question into an order.
    if re.search(r"\b(?:does\s+it\s+come\s+with|included\s+in\s+the\s+box|what\s+if)\b", message, re.I):
        return None
    raw = message.lower()
    if "4000xt" in raw or re.search(r"\b(?:connector|connection|wire)\b", raw):
        lead = ("For an **OS-4000XT** replacement remote, Warranty Support needs to confirm the exact compatible part and availability. "
                if "4000xt" in raw else "Warranty Support needs to identify the connector or compatible replacement part. ")
        if re.search(r"\b(?:connector|connection|wire)\b", raw):
            lead += "Do not connect an unidentified plug or open the chair. Turn the chair off and unplug it before further use until Support identifies the connector. "
    elif "cyber" in raw or "upholstery" in raw:
        lead = ("For **OS-3D Cyber Pro** upholstery, Warranty Support can check availability for the exact panel and color. "
                if "cyber" in raw else "Warranty Support can check replacement upholstery by model, panel and color. ")
    else:
        lead = "Warranty Support can confirm the correct replacement remote and whether it is available. "
    return lead + ("Contact **1-888-848-2630 ext. 3** or **service@osakititan.com** and have the model label and a photo of the part ready. "
                   "Part price and coverage need review. [Official support](https://otasupport.com/)")


def vivant_settings_answer(message: str, *, focused_handle: str = "") -> str | None:
    exact_model = bool(_VIVANT.search(message)) or focused_handle == "osaki-pro-vivant-5d"
    settings_guide = bool(_GUIDE.search(message)) and (
        bool(_OWNER.search(message)) or bool(re.search(r"\b(?:settings?|manual|controller)\b", message, re.I))
    )
    if not (exact_model and settings_guide):
        return None
    return ("The **Osaki Pro Vivant 5D** uses the touchscreen controller for massage programs and settings. "
            "For operating instructions, search for **Pro Vivant 5D** in the [official support library](https://otasupport.com/) "
            "or look under **User Manual** on the [manufacturer product page](https://titanchair.com/products/pro-vivant-5d). "
            "I couldn't verify a dedicated settings video for this model. Which setting are you trying to change: program, intensity, or something else?")


def wants_higher_capacity_yoga_alternative(message: str) -> bool:
    return bool(_YOGA.search(message) and _HIGHER_CAPACITY.search(message))
