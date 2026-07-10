"""
Pre-terminal error-code gate (engine intercept — Option B).

Before a defect terminal, supported models are asked whether an error code
is visible. Answers are stored on the ticket; Fonz lookup enriches the
terminal message that follows.
"""

from __future__ import annotations

from typing import Any, Optional

GATE_VISIBLE_ID = "defect_error_code_visible_q"
GATE_PICK_ID = "defect_error_code_pick"
GATE_ENTER_ID = "defect_error_code_enter"

COL_PENDING_TERMINAL = "pending_terminal"
COL_GATE_COMPLETED = "error_code_gate_completed"
COL_ERROR_CODE = "error_code"
COL_FONZ_MEANING = "fonz_meaning"
COL_FONZ_PARTS = "fonz_parts_internal"
COL_FONZ_SEVERITY = "fonz_severity"
COL_FONZ_LOOKUP_FAILED = "fonz_lookup_failed"
COL_FONZ_CATEGORY_ALIGNED = "fonz_category_aligned"

_SKIP_DEFECT_TYPES = frozenset({"voice", "cosmetic"})
_PICK_CODE_LIMIT = 8


def is_gate_node(node_id: str) -> bool:
    return node_id in (GATE_VISIBLE_ID, GATE_PICK_ID, GATE_ENTER_ID)


def model_supports_error_codes(model_name: str) -> bool:
    from error_code_lookup import get_model_diagnostic  # noqa: WPS433

    diag = get_model_diagnostic(model_name)
    if not diag:
        return False
    flag = str(diag.get("entry_method") or "").strip().lower()
    return flag in {"yes", "y", "true", "1"}


def _existing_error_code(ticket) -> str:
    if not hasattr(ticket, "get_collected"):
        return ""
    collected = ticket.get_collected() or {}
    return str(collected.get(COL_ERROR_CODE) or "").strip()


def should_intercept_terminal(ticket, terminal_node_id: str) -> bool:
    """Return True when the engine should show the error-code gate first."""
    if str(getattr(ticket, "issue_type", "") or "").lower() != "defect":
        return False
    if not str(terminal_node_id or "").startswith("defect_"):
        return False
    if str(getattr(ticket, "status", "") or "") != "in_progress":
        return False

    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    if collected.get(COL_GATE_COMPLETED):
        return False
    if collected.get(COL_PENDING_TERMINAL):
        return False
    if _existing_error_code(ticket):
        return False

    defect_type = str(getattr(ticket, "defect_type", "") or "").lower()
    if defect_type in _SKIP_DEFECT_TYPES:
        return False

    model_name = str(getattr(ticket, "model_name", "") or "").strip()
    if not model_name or not model_supports_error_codes(model_name):
        return False

    return True


def intercept_terminal_node_id(ticket, terminal_node_id: str) -> Optional[str]:
    if should_intercept_terminal(ticket, terminal_node_id):
        return GATE_VISIBLE_ID
    return None


def capture_error_code_from_intake(ticket, text: str) -> bool:
    """If intake mentions a code, store it and skip the gate later."""
    from error_code_lookup import extract_error_codes_from_text  # noqa: WPS433

    if not text or not hasattr(ticket, "set_collected"):
        return False
    if _existing_error_code(ticket):
        return False
    model_name = str(getattr(ticket, "model_name", "") or "").strip()
    for code in extract_error_codes_from_text(text):
        if not code:
            continue
        finalize_error_code_submission(ticket, code, model_name=model_name or None)
        ticket.set_collected(COL_GATE_COMPLETED, "intake")
        return True
    return False


def _entry_procedure_hint(model_name: str) -> str:
    from error_code_lookup import get_model_diagnostic  # noqa: WPS433

    diag = get_model_diagnostic(model_name) or {}
    return str(diag.get("entry_procedure") or "").strip()


def build_enter_prompt(model_name: str) -> str:
    base = (
        "Please type the error code exactly as shown on your remote or display "
        "(for example: C6, E5, 63)."
    )
    proc = _entry_procedure_hint(model_name)
    if proc:
        return (
            f"To read the error code on your **{model_name.strip()}**:\n"
            f"{proc}\n\n{base}"
        )
    return base


def build_pick_prompt(model_name: str, defect_type: str) -> str:
    model_display = (model_name or "your chair").strip()
    area = (defect_type or "this issue").replace("_", " ")
    return (
        f"Which error code is showing on your **{model_display}**? "
        f"These are the most common codes for **{area}** issues on this model."
    )


def build_visible_prompt(model_name: str) -> str:
    model_display = (model_name or "your chair").strip()
    return (
        f"Before we wrap up — on your **{model_display}**, do you see an error code "
        "on the remote or display right now?"
    )


def _pick_options(ticket) -> list[dict[str, str]]:
    from error_code_lookup import list_model_error_codes  # noqa: WPS433

    model_name = str(getattr(ticket, "model_name", "") or "")
    defect_type = str(getattr(ticket, "defect_type", "") or "")
    codes = list_model_error_codes(
        model_name,
        workflow_category=defect_type or None,
        limit=_PICK_CODE_LIMIT,
    )
    if not codes:
        codes = list_model_error_codes(model_name, limit=_PICK_CODE_LIMIT)

    options: list[dict[str, str]] = []
    for entry in codes:
        code = str(entry.get("error_code") or "").strip()
        if not code:
            continue
        meaning = str(entry.get("meaning") or "").strip()
        short = meaning.split(".")[0][:55] if meaning else code
        options.append(
            {
                "label": f"{code} — {short}",
                "answer_key": f"pick_{code}",
            }
        )
    options.append(
        {
            "label": "Other / type code manually",
            "answer_key": "error_code_other",
        }
    )
    return options


def resolve_gate_node(node_id: str, ticket) -> Optional[dict[str, Any]]:
    model_name = str(getattr(ticket, "model_name", "") or "")
    defect_type = str(getattr(ticket, "defect_type", "") or "")

    if node_id == GATE_VISIBLE_ID:
        return {
            "node_id": GATE_VISIBLE_ID,
            "type": "question",
            "prompt": build_visible_prompt(model_name),
            "options": [
                {
                    "label": "Yes, I see an error code",
                    "answer_key": "error_code_yes",
                },
                {
                    "label": "No error code is showing",
                    "answer_key": "error_code_no",
                },
            ],
        }
    if node_id == GATE_PICK_ID:
        return {
            "node_id": GATE_PICK_ID,
            "type": "question",
            "prompt": build_pick_prompt(model_name, defect_type),
            "options": _pick_options(ticket),
        }
    if node_id == GATE_ENTER_ID:
        return {
            "node_id": GATE_ENTER_ID,
            "type": "question_text",
            "prompt": build_enter_prompt(model_name),
            "answer_key": COL_ERROR_CODE,
        }
    return None


def build_gate_assistant_message(ticket, node: dict) -> Optional[str]:
    node_id = str(node.get("node_id") or "")
    model_name = str(getattr(ticket, "model_name", "") or "")
    prompt = str(node.get("prompt") or "").strip()
    proc = _entry_procedure_hint(model_name)

    if node_id in (GATE_VISIBLE_ID, GATE_PICK_ID, GATE_ENTER_ID) and proc:
        if proc not in prompt:
            return f"{prompt}\n\n*How to open the diagnostic screen:*\n{proc}"
    return prompt or None


def fonz_hit_from_ticket(ticket) -> Optional[dict[str, Any]]:
    from error_code_lookup import extract_error_codes_from_text, lookup_error_code  # noqa: WPS433

    code_raw = _existing_error_code(ticket)
    if not code_raw:
        return None
    model_name = str(getattr(ticket, "model_name", "") or "")
    for code in extract_error_codes_from_text(code_raw) or [code_raw]:
        hit = lookup_error_code(model_name, code)
        if hit:
            return hit
    return None


def persist_fonz_internal_fields(ticket, hit: dict[str, Any]) -> None:
    if not hasattr(ticket, "set_collected"):
        return
    meaning = str(hit.get("meaning") or "").strip()
    parts = str(hit.get("parts_required") or "").strip()
    severity = str(hit.get("severity") or "").strip()
    if meaning:
        ticket.set_collected(COL_FONZ_MEANING, meaning[:600])
    if parts:
        ticket.set_collected(COL_FONZ_PARTS, parts[:400])
    if severity:
        ticket.set_collected(COL_FONZ_SEVERITY, severity[:120])
    ticket.set_collected(COL_FONZ_LOOKUP_FAILED, "")

    defect_type = str(getattr(ticket, "defect_type", "") or "")
    from error_code_lookup import code_aligns_with_defect  # noqa: WPS433

    if code_aligns_with_defect(hit, defect_type):
        ticket.set_collected(COL_FONZ_CATEGORY_ALIGNED, "1")
    else:
        ticket.set_collected(COL_FONZ_CATEGORY_ALIGNED, "")


def finalize_error_code_submission(
    ticket,
    raw_code: str,
    *,
    model_name: Optional[str] = None,
) -> None:
    from error_code_lookup import extract_error_codes_from_text, lookup_error_code  # noqa: WPS433

    if not hasattr(ticket, "set_collected"):
        return

    text = str(raw_code or "").strip()
    codes = extract_error_codes_from_text(text) if text else []
    code = codes[0] if codes else text
    ticket.set_collected(COL_ERROR_CODE, code)

    model = model_name or str(getattr(ticket, "model_name", "") or "")
    hit = lookup_error_code(model, code) if code else None
    if hit:
        persist_fonz_internal_fields(ticket, hit)
    else:
        ticket.set_collected(COL_FONZ_LOOKUP_FAILED, "1")
        ticket.set_collected(COL_FONZ_MEANING, "")
        ticket.set_collected(COL_FONZ_PARTS, "")
        ticket.set_collected(COL_FONZ_SEVERITY, "")
        ticket.set_collected(COL_FONZ_CATEGORY_ALIGNED, "")


def _lookup_failed(ticket) -> bool:
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    return str(collected.get(COL_FONZ_LOOKUP_FAILED) or "") == "1"


def _category_aligned(ticket) -> bool:
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    return str(collected.get(COL_FONZ_CATEGORY_ALIGNED) or "") == "1"


def merge_fonz_into_diagnosis(diagnosis: dict[str, Any], ticket) -> dict[str, Any]:
    out = dict(diagnosis)
    hit = fonz_hit_from_ticket(ticket)

    if hit:
        persist_fonz_internal_fields(ticket, hit)
        code = str(hit.get("error_code") or "?")
        meaning = str(hit.get("meaning") or "").strip()
        troubleshooting = str(hit.get("troubleshooting") or "").strip()

        summary = str(out.get("summary") or "").strip()
        fonz_line = (
            f"**Error code {code}:** {meaning}"
            if meaning
            else f"**Error code {code}** was recorded for your model."
        )
        if _category_aligned(ticket):
            fonz_line = (
                f"{fonz_line} This code matches the type of issue you described in the workflow."
            )
        out["summary"] = f"{summary}\n\n{fonz_line}".strip() if summary else fonz_line

        steps = list(out.get("steps") or [])
        if troubleshooting and len(troubleshooting) >= 12:
            tip = troubleshooting[:240].strip()
            if tip and tip.lower() not in {s.lower() for s in steps}:
                steps.insert(0, tip)
        out["steps"] = steps

        sources = list(out.get("sources") or [])
        if "fonz_error_code" not in sources:
            sources.insert(0, "fonz_error_code")
        out["sources"] = sources
        out["fonz_match"] = {"model": hit.get("model"), "error_code": code}
        out["top_match"] = out.get("top_match") or f"{hit.get('model')} — error {code}"
        return out

    if _lookup_failed(ticket):
        code = _existing_error_code(ticket)
        summary = str(out.get("summary") or "").strip()
        note = (
            f"We recorded error code **{code}**, but it is not listed for your model in our "
            "reference sheet. Our team will verify it during review."
        )
        out["summary"] = f"{summary}\n\n{note}".strip() if summary else note
        return out

    return append_soft_hints_to_diagnosis(out, ticket)


def append_soft_hints_to_diagnosis(diagnosis: dict[str, Any], ticket) -> dict[str, Any]:
    from error_code_lookup import suggest_error_codes_for_ticket  # noqa: WPS433

    model_name = str(getattr(ticket, "model_name", "") or "")
    defect_type = str(getattr(ticket, "defect_type", "") or "")
    suggestions = suggest_error_codes_for_ticket(model_name, defect_type, limit=2)
    if not suggestions:
        return diagnosis

    out = dict(diagnosis)
    codes = ", ".join(f"**{row.get('error_code')}**" for row in suggestions)
    summary = str(out.get("summary") or "").strip()
    hint = (
        f"If an error code appears later, these codes sometimes relate to similar symptoms "
        f"on your model: {codes}. Confirm on your display before relying on them."
    )
    out["summary"] = f"{summary}\n\n*{hint}*".strip() if summary else f"*{hint}*"
    out["fonz_suggestions"] = [
        {"error_code": row.get("error_code"), "meaning": (row.get("meaning") or "")[:120]}
        for row in suggestions
    ]
    return out


def append_fonz_to_message(message: str, ticket) -> str:
    if "**Error code" in message or "not listed for your model" in message:
        return message

    hit = fonz_hit_from_ticket(ticket)
    if hit:
        persist_fonz_internal_fields(ticket, hit)
        code = str(hit.get("error_code") or "?")
        meaning = str(hit.get("meaning") or "").strip()
        troubleshooting = str(hit.get("troubleshooting") or "").strip()

        parts = [message.strip()]
        if meaning:
            line = f"**Error code {code}:** {meaning}"
            if _category_aligned(ticket):
                line += " This aligns with the issue path you selected."
            parts.append(f"\n\n{line}")
        if troubleshooting and len(troubleshooting) >= 12:
            parts.append(f"\n\n**From the manufacturer error list:** {troubleshooting[:260]}")
        return "".join(parts).strip()

    if _lookup_failed(ticket):
        code = _existing_error_code(ticket)
        return (
            f"{message.strip()}\n\n"
            f"We recorded error code **{code}**, but it is not listed for your model in our "
            "reference sheet. Our team will verify it during review."
        ).strip()

    from error_code_lookup import suggest_error_codes_for_ticket  # noqa: WPS433

    model_name = str(getattr(ticket, "model_name", "") or "")
    defect_type = str(getattr(ticket, "defect_type", "") or "")
    suggestions = suggest_error_codes_for_ticket(model_name, defect_type, limit=2)
    if not suggestions:
        return message

    codes = ", ".join(f"**{row.get('error_code')}**" for row in suggestions)
    return (
        f"{message.strip()}\n\n"
        f"*If an error code appears later, related codes for this symptom on your model "
        f"may include: {codes}.*"
    ).strip()


def append_fonz_to_terminal_enrichment(
    enrichment: Optional[dict[str, Any]],
    ticket,
) -> Optional[dict[str, Any]]:
    if not enrichment:
        return enrichment
    out = dict(enrichment)
    diagnosis = out.get("diagnosis")
    if isinstance(diagnosis, dict):
        out["diagnosis"] = merge_fonz_into_diagnosis(diagnosis, ticket)
    message = str(out.get("message") or "").strip()
    if message:
        out["message"] = append_fonz_to_message(message, ticket)
    elif isinstance(out.get("diagnosis"), dict):
        from warranty_self_help import format_diagnosis_message  # noqa: WPS433

        out["message"] = append_fonz_to_message(
            format_diagnosis_message(out["diagnosis"]),
            ticket,
        )
    return out


def build_admin_fonz_payload(ticket) -> dict[str, Any]:
    collected = ticket.get_collected() if hasattr(ticket, "get_collected") else {}
    hit = fonz_hit_from_ticket(ticket)
    return {
        "error_code": str(collected.get(COL_ERROR_CODE) or "").strip() or None,
        "meaning": str(collected.get(COL_FONZ_MEANING) or "").strip() or None,
        "parts_internal": str(collected.get(COL_FONZ_PARTS) or "").strip() or None,
        "severity": str(collected.get(COL_FONZ_SEVERITY) or "").strip() or None,
        "lookup_failed": str(collected.get(COL_FONZ_LOOKUP_FAILED) or "") == "1",
        "category_aligned": str(collected.get(COL_FONZ_CATEGORY_ALIGNED) or "") == "1",
        "gate_completed": str(collected.get(COL_GATE_COMPLETED) or "").strip() or None,
        "match_model": hit.get("model") if hit else None,
        "match_code": hit.get("error_code") if hit else None,
    }
