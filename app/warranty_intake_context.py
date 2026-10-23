"""
Persist and reuse free-text intake from smart-start / natural-start / chat tools.

The intake summary is prepended to knowledge search queries so early workflow
steps can surface Freshdesk tips before the customer has clicked many buttons.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

COL_MODEL_CONFIRMED = "model_confirmed"

_NON_MODEL_INPUTS = frozenset({
    "return", "returns", "refund", "cancel", "warranty", "sales", "help", "human", "agent",
    "none", "unknown", "not sure", "i don't know", "i dont know", "no model", "defect", "broken",
    "delivery", "installation", "setup", "yes", "no", "error code", "hello", "hi", "okay", "ok", "thanks",
})
_LOCAL_FILE_RE = re.compile(r"(?:file://|/Users/|/home/|/Downloads/|[A-Za-z]:\\|\b[^\s]+\.(?:jpe?g|png|heic|pdf|mov|mp4)\b)", re.I)
_EXTRACTION_META_RE = re.compile(
    r"\b(?:customer (?:provided|gave|entered|reports? only)|model name only|no issue (?:description|described)|"
    r"without (?:describing|an? issue)|no (?:symptom|problem) (?:description|provided))\b", re.I
)


def normalize_model_input(value: str) -> str:
    """Remove an explicit model-answer wrapper without guessing a different model."""
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    raw = re.sub(r"^(?:(?:my|the)\s+)?(?:chair\s+)?model(?:\s+name)?\s*(?:is\s+|[:=-]\s*)(?:the\s+)?", "", raw, flags=re.I)
    return raw.strip(" .:;\"'")


def is_valid_model_input(value: str) -> bool:
    """Reject intent, file references and issue text before model confirmation."""
    raw = normalize_model_input(value)
    if not raw or len(raw) > 100 or raw.lower() in _NON_MODEL_INPUTS:
        return False
    if _LOCAL_FILE_RE.search(raw) or _EXTRACTION_META_RE.search(raw):
        return False
    if re.search(r"[@\n\r]|https?://", raw, re.I):
        return False
    if re.search(r"\b(?:not working|doesn'?t work|does not work|won'?t|does not|call me|talk to|want to return|need help|no air|not inflating|hurts|pain|discomfort|i need|i want|can you|please help|(?:human|agent|refund|return) please|please (?:human|agent|refund|return))\b", raw, re.I):
        return False
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    if tokens and all(token in _MODEL_NOISE for token in tokens):
        return False
    return bool(re.search(r"[A-Za-z0-9\uac00-\ud7a3]", raw)) and len(raw.split()) <= 10


def _usable_customer_intake(value: str, current_model: str = "") -> str:
    text = str(value or "").strip()
    if not text or _LOCAL_FILE_RE.search(text) or _EXTRACTION_META_RE.search(text):
        return ""
    # A model label, return intent, or extraction summary is not a symptom.
    if text.lower() in _NON_MODEL_INPUTS:
        return ""
    if current_model and _model_identity(normalize_model_input(text)) == _model_identity(current_model):
        return ""
    return text

_MODEL_NOISE = frozenset(
    {"osaki", "titan", "os", "massage", "chair", "model", "the", "my", "is"}
)
_MODEL_DEPENDENT_COLLECTED_KEYS = (
    "error_code",
    "error_code_gate_completed",
    "pending_terminal",
    "fonz_meaning",
    "fonz_parts_internal",
    "fonz_severity",
    "fonz_lookup_failed",
    "fonz_category_aligned",
)


def _model_identity(value: str) -> str:
    """Comparable model key that ignores brands and display punctuation."""
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return "".join(token for token in tokens if token not in _MODEL_NOISE)


def _strip_previous_model(text: str, previous_model: str) -> str:
    """Remove a superseded model name while retaining any symptom description."""
    value = str(text or "").strip()
    if not value:
        return ""

    previous_identity = _model_identity(previous_model)
    if previous_identity and _model_identity(value) == previous_identity:
        return ""

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", str(previous_model or "").lower())
        if token not in _MODEL_NOISE
    ]
    if tokens:
        flexible = r"[\s_\-/]*".join(re.escape(token) for token in tokens)
        value = re.sub(
            # In particular, the model Haven must not remove "haven" from
            # "haven't", which would reverse a customer's attempted-check report.
            rf"(?<![a-z0-9'’]){flexible}(?![a-z0-9'’])",
            " ",
            value,
            flags=re.I,
        )

    value = re.sub(
        r"\b(?:my|the)?\s*(?:chair\s*)?model\s*(?:is|was|:|-)?\s*",
        " ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" ,;:-.()[]")
    meaningful = [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _MODEL_NOISE
    ]
    return value if meaningful else ""


@lru_cache(maxsize=1)
def _known_model_labels() -> tuple[str, ...]:
    """Model labels used to clean stale context from sessions created pre-fix."""
    try:
        from fonz_warranty_data import load_model_diagnostic_records  # noqa: WPS433

        labels = {
            str(row.get("model") or "").strip()
            for row in load_model_diagnostic_records()
            if str(row.get("model") or "").strip()
        }
        return tuple(
            sorted(
                (label for label in labels if len(_model_identity(label)) >= 5),
                key=len,
                reverse=True,
            )
        )
    except Exception:
        return ()


def _sanitize_intake_for_current_model(text: str, current_model: str) -> str:
    value = str(text or "").strip()
    current_identity = _model_identity(current_model)
    if not value or not current_identity:
        return value

    allowed = {current_identity}
    try:
        from model_families import resolve_family_canonical  # noqa: WPS433

        canonical = resolve_family_canonical(current_model)
        if canonical:
            allowed.add(_model_identity(canonical))
    except Exception:
        pass

    for label in _known_model_labels():
        identity = _model_identity(label)
        if identity in allowed or identity in current_identity or identity not in _model_identity(value):
            continue
        value = _strip_previous_model(value, label)
        if not value:
            break
    return value


def reconcile_model_change(ticket, previous_model: str, current_model: str) -> bool:
    """
    Remove context that belongs to a superseded chair model.

    Troubleshooting text is preserved when possible, but old model references,
    captured error codes, and Fonz lookup results are cleared so a corrected
    model can never inherit a diagnosis from the previous one.
    """
    previous = str(previous_model or "").strip()
    current = str(current_model or "").strip()
    if not previous or not current:
        return False
    if _model_identity(previous) == _model_identity(current):
        return False
    if ticket is None or not hasattr(ticket, "set_collected"):
        return False

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    for key in ("intake_summary", "intake_raw_message"):
        cleaned = _strip_previous_model(str(collected.get(key) or ""), previous)
        ticket.set_collected(key, cleaned)

    for key in _MODEL_DEPENDENT_COLLECTED_KEYS:
        ticket.set_collected(key, "")
    ticket.set_collected(COL_MODEL_CONFIRMED, "")
    return True


def mark_model_confirmed(ticket) -> None:
    if ticket is None or not hasattr(ticket, "set_collected"):
        return
    ticket.set_collected(COL_MODEL_CONFIRMED, "1")


def is_model_confirmed(ticket) -> bool:
    if ticket is None:
        return False
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    return str(collected.get(COL_MODEL_CONFIRMED) or "") == "1"


def needs_model_confirmation(ticket) -> bool:
    model = str(getattr(ticket, "model_name", "") or "").strip()
    if not model:
        return False
    return not is_model_confirmed(ticket)


def build_model_confirmation_message(model_name: str) -> str:
    display = (model_name or "your chair").strip()
    return (
        f"I have **{display}** as your chair model.\n\n"
        "Is that correct? Tap **Yes, that's my model** below, or type the correct model name."
    )


def persist_intake_summary(
    ticket,
    *,
    summary: str = "",
    raw_message: str = "",
) -> None:
    """Store a customer-safe one-line intake on the ticket for later enrichment."""
    if ticket is None or not hasattr(ticket, "set_collected"):
        return

    summary_text = str(summary or "").strip()
    raw_text = str(raw_message or "").strip()
    model = str(getattr(ticket, "model_name", "") or "")
    primary = _usable_customer_intake(summary_text, model) or _usable_customer_intake(raw_text, model)
    if not primary:
        return

    ticket.set_collected("intake_summary", primary[:500])
    if _usable_customer_intake(raw_text, model):
        ticket.set_collected("intake_raw_message", raw_text[:800])

    try:
        from warranty_error_code_gate import capture_error_code_from_intake  # noqa: WPS433

        capture_error_code_from_intake(ticket, raw_text or primary)
    except Exception:
        pass


def get_intake_summary(ticket) -> str:
    if ticket is None:
        return ""
    collected: dict = {}
    if hasattr(ticket, "get_collected"):
        collected = ticket.get_collected() or {}
    elif isinstance(ticket, dict):
        collected = ticket.get("collected_data") or {}
    current_model = str(getattr(ticket, "model_name", "") or "").strip()
    for key in ("intake_summary", "intake_raw_message"):
        raw = _usable_customer_intake(str(collected.get(key) or ""), current_model)
        value = _usable_customer_intake(_sanitize_intake_for_current_model(
            raw,
            current_model,
        ), current_model)
        if value:
            return value
    return ""


def enrich_path_text(path_text: str, ticket) -> str:
    """Prepend stored intake text when it is not already in the path."""
    intake = get_intake_summary(ticket)
    base = str(path_text or "").strip()
    if not intake:
        return base
    if intake.lower() in base.lower():
        return base
    return f"{intake} {base}".strip()


def intake_aware_step_summary(
    *,
    ticket,
    turns,
    summary: str,
    max_turns: int = 5,
) -> str:
    """
    Personalize early step enrichment with what the customer said at intake.
    """
    intake = get_intake_summary(ticket)
    body = str(summary or "").strip()
    if not intake or len(turns or ()) > max_turns:
        return body
    if intake.lower() in body.lower():
        return body
    return f"You mentioned: **{intake}**. {body}"


_CHECK_TOPICS = {
    "outlet": re.compile(r"\b(?:outlet|wall socket)\b", re.I),
    "power_cord": re.compile(r"\bpower\s*cord\b", re.I),
    "power_cycle": re.compile(r"\b(?:power[ -]?cycl\w*|restart\w*|reboot\w*|toggl\w*.{0,35}switch|turn\w*.{0,25}off.{0,25}on)\b", re.I),
    "remote_cable": re.compile(r"\b(?:remote.{0,30}(?:cable|connection)|cable.{0,30}remote)\b", re.I),
    "heat_warmup": re.compile(r"\b(?:heat\w*.{0,55}(?:10|ten)\s*minutes?|(?:10|ten)\s*minutes?.{0,55}heat\w*)\b", re.I),
    "heat_on": re.compile(r"\bheat\w*.{0,35}(?:turned? on|enabled|switched on)|\bturn\w*.{0,35}heat\w*.{0,15}on\b", re.I),
}


def completed_check_topics(ticket, turns) -> set[str]:
    """Use explicit completed attempts, never assume a suggested step was tried."""
    texts = [get_intake_summary(ticket)]
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    texts.append(str(collected.get("intake_raw_message") or ""))
    topics: set[str] = set()
    for turn in turns or ():
        key = str(getattr(turn, "answer_key", "") or "")
        if key == "still_no_heat":
            topics.update({"heat_warmup", "heat_on"})
        texts.append(str(getattr(turn, "customer_answer", "") or ""))
    for text in texts:
        text = text.replace("’", "'")
        if not re.search(r"\b(?:already|tried|checked|tested|completed|reseated|reconnected|cycled|restarted|rebooted)\b", text, re.I):
            continue
        if re.search(r"\b(?:haven'?t|have not|not yet|never|didn'?t|did not)\b", text, re.I):
            continue
        topics.update(name for name, pattern in _CHECK_TOPICS.items() if pattern.search(text))
    return topics


def filter_completed_check_steps(steps: list[str], *, ticket, turns) -> list[str]:
    completed = completed_check_topics(ticket, turns)
    if not completed:
        return list(steps)
    kept = []
    for step in steps:
        topics = {name for name, pattern in _CHECK_TOPICS.items() if pattern.search(step)}
        if topics and topics <= completed:
            continue
        kept.append(step)
    return kept


def reported_safety_hazard(ticket=None, turns=(), text: str = "") -> bool:
    """Recognize an explicit unsafe-use report before suggesting more testing."""
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    texts = [text, str(collected.get("intake_summary") or ""), str(collected.get("intake_raw_message") or "")]
    texts.extend(str(getattr(turn, "customer_answer", "") or "") for turn in turns or ())
    hazard = r"(?:too hot|overheat\w*|smok\w*|burn(?:ing|t) smell|sparks?|electric(?:al)? shock|short circuit|has a short|pain(?:ful)?|hurts?|discomfort)"
    pattern = re.compile(rf"\b{hazard}\b", re.I)
    # Negation applies only to a concrete hazard phrase/list, not the rest of
    # the sentence: "no smoke, but there are sparks" still requires a stop.
    negative = re.compile(
        rf"\b(?:no|not|without|never|isn't|aren't|hasn't|doesn't|don't)\s+"
        rf"(?:(?:any|signs? of|experiencing|feel(?:ing)?|caus(?:e|es|ed|ing)|have|had|emit(?:ting)?|produc(?:e|es|ed|ing)|show(?:ing)?)\s+)*"
        rf"{hazard}(?:\s*(?:,|or|and)\s*(?:any\s+)?{hazard})*\b", re.I,
    )
    for value in texts:
        value = negative.sub("", value.replace("’", "'"))
        if pattern.search(value):
            return True
    return False


def try_side_question_for_ticket(engine, ticket_id: str, answer: str) -> Optional[str]:
    """Shared side-question handler for REST API and chat agent tools."""
    from warranty_side_questions import try_answer_side_question  # noqa: WPS433

    node = engine.get_current_node(ticket_id)
    ticket = engine.get_ticket(ticket_id)
    if not node or ticket is None:
        return None

    return try_answer_side_question(
        node=node,
        answer=answer,
        model_name=str(getattr(ticket, "model_name", None) or ""),
        issue_type=str(getattr(ticket, "issue_type", None) or ""),
        turns=engine.get_turns(ticket_id),
        ticket=ticket,
    )
