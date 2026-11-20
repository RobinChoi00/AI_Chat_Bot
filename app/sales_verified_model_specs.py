"""Answer narrow model-specific questions from reviewed manufacturer sources.

The broader catalog is good for discovery, but its fixed columns do not cover
these facts. Match exact catalog handles and the requested detail; do not infer
an engineering feature (or borrow measurements) from a neighboring model.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sales_constraints import _read_person_measurements, extract_constraints
from sales_height import feet

if TYPE_CHECKING:
    from sales_catalog import ProductSpecs


_HAMILTON = "osaki-os-3d-hamilton-le"
_BRAVO_FLEX = "osaki-bravo-duo-flex-chair"
_ORION = "osaki-4d-3d-orion-duo-mech"
_CURA = "titan-pro-cura-4d"
_VITO = "os-3d-ai-vito-massage-chair"

_HAMILTON_SOURCE = "https://osakimassagechair.com/products/osaki-os-3d-hamilton-le"
_BRAVO_SOURCE = "https://osakimassagechair.com/products/osaki-4d-bravo-duo-flex"
_ORION_MANUAL = "https://device.report/m/381c65f37233b5f354c032e8a54bc604dc69f4ffc04c807378d951533fa0bef2_optim.pdf"
_CURA_SOURCE = "https://titanchair.com/products/titan-pro-cura-4d"
_VITO_SOURCE = "https://osakimassagechair.com/products/os-3d-vito"

_CARTONS = re.compile(
    r"\b(?:how\s+many\b[^?.!]{0,45}\b(?:box(?:es)?|cartons?|packages?)|"
    r"(?:box|carton|package)\s+(?:count|dimensions?|size|weight)|"
    r"(?:shipping|delivery)\s+(?:box(?:es)?|cartons?|packages?|weight|dimensions?)|"
    r"(?:come|comes|arrive|arrives|ship|ships)\s+in\s+(?:how\s+many|(?:one|two|[12])\s+(?:box(?:es)?|cartons?|packages?))|"
    r"(?:boxed|packaged)\s+(?:weight|dimensions?|size))\b", re.I,
)
_WALKING = re.compile(r"\bwalk(?:ing)?[\s-]+(?:motion|tech(?:nology)?|feature|mode|massage)\b", re.I)
_PANELS = re.compile(r"\bside[\s-]+(?:(?:control|quick[\s-]+access)\s+)?panels?\b|\bquick[\s-]+access\s+(?:panel|controls?)\b", re.I)
_ROLLING_BASE = re.compile(
    r"\b(?:cast[eo]rs?|(?:base|back|rear|transport|moving)\s+wheels?|"
    r"wheels?\b.{0,24}\b(?:mov(?:e|ing)|transport|roll)|(?:have|has|on)\s+(?:any\s+)?wheels?)\b", re.I,
)
_MOTOR = re.compile(r"\bbrush[\s-]*less\b|\b(?:brushed|brush)\s+motors?\b|\b(?:kind|type)\s+of\s+motors?\b", re.I)
_CONTROLS = re.compile(r"\b(?:speed|pressure|intensity|strength)\b", re.I)
_FIT = re.compile(r"\b(?:fit(?:s|ting)?|suit(?:able|ed)?|work\s+for|support(?:s|ed)?|accommodate(?:s|d)?|comfortable|tall\s+enough)\b", re.I)
_USER_HEIGHT = re.compile(
    r"\b(?:how\s+tall\s+can\s+(?:someone|(?:a|the)\s+(?:user|person))|"
    r"(?:shortest|tallest)\s+(?:person|user)|"
    r"(?:min(?:imum)?|max(?:imum)?|recommended|supported|user|person)\s+(?:(?:user|person)\s+)?heighth?|"
    r"heighth?\s+(?:range|limit|minimum|maximum)|up\s+to\s+what\s+height)\b", re.I,
)
_USER_WEIGHT = re.compile(r"\b(?:max(?:imum)?\s+(?:user\s+)?weight|weight\s+(?:capacity|limit)|(?:user|person)\s+weight|how\s+much\s+weight)\b", re.I)
_OTHER_CONSTRAINTS = re.compile(r"\b(?:door(?:way)?|stair(?:s|case)?|box(?:es)?|carton|wall\s+clearance|clearance|budget|compared?|versus|vs\.?|alternatives?)\b", re.I)
_REPAIR = re.compile(r"\b(?:replacement|repair|disassembl\w*|broken|stuck|stopped\s+working|"
                     r"not\s+working|doesn['’]?t\s+work|how\s+to\s+(?:take|replace|remove|fix|open))\b", re.I)

_ORION_MIN_IN = 57  # 4 ft 9 in, model instruction manual, p. 32 (FAQ)
_ORION_MAX_IN = 76  # 6 ft 4 in
_ORION_MAX_LB = 280
_ORION_TROUBLESHOOTING_LB = 260  # p. 33 uses a different figure in the fault table


def _with_source(text: str, url: str, *, label: str = "Manufacturer source") -> str:
    return f"{text}\n\n{label}: {url}"


def _orion_fit(name: str, raw: str) -> tuple[str, bool] | None:
    if _OTHER_CONSTRAINTS.search(raw):
        return None
    ask_height, ask_weight = bool(_USER_HEIGHT.search(raw)), bool(_USER_WEIGHT.search(raw))
    # Preserve compact chat shorthand (e.g. 6ft2) while reusing the shared
    # measurement parser's negation and product-dimension exclusions.
    measurement_text = re.sub(r"(?<![\w.])([3-7])\s*ft(?=\d)", r"\1 ft ", raw, flags=re.I)
    measures = extract_constraints(measurement_text) if ask_height or ask_weight or _FIT.search(raw) else {}
    heights = list(_read_person_measurements(measures.get("user_heights_in")).values())
    weights = list(_read_person_measurements(measures.get("user_weights_lb")).values())
    if not (ask_height or ask_weight or heights or weights):
        return None

    lines = []
    if ask_height or heights:
        lines.append(f"**{name}** has a manufacturer-recommended user height range of **{feet(_ORION_MIN_IN)} to {feet(_ORION_MAX_IN)}**.")
    if ask_weight or weights:
        subject = "The manual" if lines else f"The **{name}** manual"
        lines.append(f"{subject}'s FAQ lists a **{_ORION_MAX_LB} lb maximum user weight**.")
    weight_discrepancy = (ask_weight and not weights) or any(
        _ORION_TROUBLESHOOTING_LB < weight <= _ORION_MAX_LB for weight in weights
    )
    if weight_discrepancy:
        lines.append(f"The troubleshooting section mentions **{_ORION_TROUBLESHOOTING_LB} lb**. "
                     f"I can't confirm fit above {_ORION_TROUBLESHOOTING_LB} lb from those conflicting figures; "
                     "ask Sales to verify the limit for the exact chair before ordering.")
    if heights or weights:
        outside = (any(not _ORION_MIN_IN <= size <= _ORION_MAX_IN for size in heights)
                   or any(weight > _ORION_MAX_LB for weight in weights))
        checks = [f"{feet(size)}" for size in heights] + [f"{weight:g} lb" for weight in weights]
        values = " and ".join(checks)
        if outside:
            lines.append(f"**{values}** is outside at least one of these limits; I would not recommend this model for those measurements.")
        elif not weight_discrepancy:
            lines.append(f"**{values}** is within the manual's stated {'height and weight limits' if heights and weights else 'limit for that measurement'}. That does not guarantee comfort or roller reach; a showroom trial is the best way to check.")
    return " ".join(lines), weight_discrepancy


def verified_model_answer(product: ProductSpecs, message: str) -> str | None:
    """Return an answer-first response for a reviewed detail of one exact model."""
    raw = message or ""
    if _REPAIR.search(raw):
        return None
    name = product.display_name
    if product.handle == _HAMILTON and _CARTONS.search(raw):
        text = f"The manufacturer lists **2 shipping boxes** for the **{name}**."
        if re.search(r"\b(?:dimensions?|size|measur\w*|weight|weigh\w*|heavy)\b", raw, re.I):
            text += ("\n- **Box 1:** 30.0 × 60.0 × 37.6 in (W × L × H), 269.0 lb."
                     "\n- **Box 2:** 55.9 × 28.0 × 15.0 in (W × L × H), 70.5 lb.")
        return _with_source(text, _HAMILTON_SOURCE)
    if product.handle == _BRAVO_FLEX:
        lines = []
        if _WALKING.search(raw):
            lines.append("**Walking Technology** moves the leg sections in a walking-like pattern while you stay seated, to stretch the lower body.")
        if _PANELS.search(raw):
            lines.append("**Side panels** have quick-access controls to adjust the chair position or start a massage program without using the main touchscreen.")
        if lines:
            return _with_source(f"On the **{name}**:\n" + "\n".join(f"- {line}" for line in lines), _BRAVO_SOURCE)
    if product.handle == _CURA and _MOTOR.search(raw):
        return _with_source(
            f"I can't verify whether the **{name}** roller motor is brushless. Titan lists a **4D massage mechanism**, "
            "but that doesn't specify the motor type. A product specialist would need to confirm the roller motor.",
            _CURA_SOURCE,
        )
    if product.handle == _VITO and _ROLLING_BASE.search(raw):
        return _with_source(
            f"I can't confirm that the **{name}** has built-in casters. The listed foot rollers are for massage, not transport. "
            "The user manual describes using a dolly with floor protection and warns against dragging the assembled chair. "
            "Ask Sales to confirm the base hardware before planning to roll it.", _VITO_SOURCE,
        )
    if product.handle == _ORION:
        lines = []
        if _CONTROLS.search(raw) and not re.search(r"\b(?:blood|arterial|tire|shipping|delivery)\s+(?:pressure|speed)\b", raw, re.I):
            lines.append(
                f"Yes. On the **{name}**, the touchscreen lets you adjust massage strength/speed during an auto program. "
                "The manual lists **6 levels of 4D roller depth** and **5 levels each for Kneading, Rapping (tapping), and Airbags**. "
                "It doesn't give a pressure measurement or state that every setting can be changed independently."
            )
        fit = _orion_fit(name, raw)
        pages = "p. 22"
        if fit:
            fit_text, weight_discrepancy = fit
            fit_pages = "32–33" if weight_discrepancy else "32"
            if lines:
                pages = f"pp. 22 and {fit_pages}"
            else:
                pages = "pp. 32–33" if weight_discrepancy else "p. 32"
            lines.append(fit_text)
        if lines:
            return _with_source("\n\n".join(lines), _ORION_MANUAL, label=f"Manufacturer instructions ({pages})")
    return None
