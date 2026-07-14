"""
Unified customer-facing answer synthesis for warranty workflow.

Merges workflow outcomes with Fonz error-code data and symptom-based hints
into one coherent message (terminal and soft-hint paths).
"""

from __future__ import annotations

from typing import Any


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
        tip = ""
        if troubleshooting:
            from warranty_knowledge import _extract_customer_steps  # noqa: WPS433

            safe_steps = _extract_customer_steps(troubleshooting, meaning)
            tip = safe_steps[0] if safe_steps else ""

        if short:
            lines.append(f"• **{code}** — {short}")
        else:
            lines.append(f"• **{code}**")
        if tip and tip.lower() not in (short or "").lower():
            lines.append(f"  *Try:* {tip}")

    return "\n".join(lines) if len(lines) > 1 else ""


def build_symptom_fonz_block(ticket) -> str:
    """
    Do not expose model-similar error codes until the customer confirms a code.

    Those suggestions remain available in internal/admin diagnostics. Showing
    them to a customer as possibilities creates noise and can expose repair-only
    notes such as PCB replacements or manual page references.
    """
    return ""


def append_symptom_insights_to_message(message: str, ticket) -> str:
    """Return customer copy unchanged; unconfirmed code hints stay internal."""
    return (message or "").strip()


def enrich_diagnosis_with_symptom_fonz(
    diagnosis: dict[str, Any],
    ticket,
) -> dict[str, Any]:
    """Return diagnosis unchanged; unconfirmed code hints stay internal."""
    return diagnosis
