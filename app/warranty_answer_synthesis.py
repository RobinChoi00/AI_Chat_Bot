"""
Unified customer-facing answer synthesis for warranty workflow.

Merges workflow outcomes with Fonz error-code data and symptom-based hints
into one coherent message (terminal and soft-hint paths).
"""

from __future__ import annotations

from typing import Any, Optional


def format_fonz_suggestion_entries(
    suggestions: list[dict[str, Any]],
    *,
    header: str,
    max_entries: int = 2,
) -> str:
    """Prominent block listing related error codes with meanings."""
    if not suggestions:
        return ""

    lines: list[str] = [header.strip()]
    for row in suggestions[:max_entries]:
        code = str(row.get("error_code") or "").strip()
        if not code:
            continue
        meaning = str(row.get("meaning") or "").strip()
        short = meaning.split(".")[0][:100] if meaning else ""
        troubleshooting = str(row.get("troubleshooting") or "").strip()
        tip = troubleshooting[:160] if troubleshooting else ""

        if short:
            lines.append(f"• **{code}** — {short}")
        else:
            lines.append(f"• **{code}**")
        if tip and tip.lower() not in (short or "").lower():
            lines.append(f"  *Try:* {tip}")

    return "\n".join(lines) if len(lines) > 1 else ""


def build_symptom_fonz_block(ticket) -> str:
    """
    Symptom-based Fonz insight when no exact error code is on the ticket.
    Used at terminals (including models without on-screen codes).
    """
    from error_code_lookup import suggest_error_codes_for_ticket  # noqa: WPS433

    model_name = str(getattr(ticket, "model_name", "") or "")
    if not model_name:
        return ""

    defect_type = str(getattr(ticket, "defect_type", "") or "")
    suggestions = suggest_error_codes_for_ticket(model_name, defect_type, limit=2)
    if not suggestions:
        return ""

    area = (defect_type or "this issue").replace("_", " ")
    header = (
        f"**Related manufacturer error codes for {area} on your model** "
        f"(confirm on your display if one appears):"
    )
    return format_fonz_suggestion_entries(suggestions, header=header)


def append_symptom_insights_to_message(message: str, ticket) -> str:
    """Append a visible Fonz symptom block when the message lacks code detail."""
    base = (message or "").strip()
    if not base:
        return base
    if "**Error code" in base or "**Related manufacturer error codes" in base:
        return base

    block = build_symptom_fonz_block(ticket)
    if not block:
        return base
    return f"{base}\n\n{block}".strip()


def enrich_diagnosis_with_symptom_fonz(
    diagnosis: dict[str, Any],
    ticket,
) -> dict[str, Any]:
    """Merge prominent symptom-based Fonz hints into a diagnosis dict."""
    if not isinstance(diagnosis, dict):
        return diagnosis

    summary = str(diagnosis.get("summary") or "").strip()
    if "**Error code" in summary or "**Related manufacturer error codes" in summary:
        return diagnosis

    block = build_symptom_fonz_block(ticket)
    if not block:
        return diagnosis

    out = dict(diagnosis)
    out["summary"] = f"{summary}\n\n{block}".strip() if summary else block
    suggestions = []
    try:
        from error_code_lookup import suggest_error_codes_for_ticket  # noqa: WPS433

        model_name = str(getattr(ticket, "model_name", "") or "")
        defect_type = str(getattr(ticket, "defect_type", "") or "")
        suggestions = suggest_error_codes_for_ticket(model_name, defect_type, limit=2)
    except Exception:
        pass
    if suggestions:
        out["fonz_suggestions"] = [
            {
                "error_code": row.get("error_code"),
                "meaning": (row.get("meaning") or "")[:120],
            }
            for row in suggestions
        ]
    sources = list(out.get("sources") or [])
    if "fonz_error_code" not in sources:
        sources.append("fonz_symptom_hint")
    out["sources"] = sources
    return out
