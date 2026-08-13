"""
warranty_step_paraphrase.py
===========================
Optional LLM rewrite for non-terminal workflow step messages.

After ``build_step_enrichment`` assembles Freshdesk/Q&A tips + the next
workflow question, this module asks a small model to make the prose warmer
while keeping facts and the exact branching question intact.

Design contract
---------------
- Graceful no-op when OPENAI_API_KEY is missing or paraphrase is disabled.
- The workflow ``base_prompt`` MUST appear verbatim in the output — we
  validate and fall back to the draft if the model edits the question.
- Never add replacement / dispatch / refund promises.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Off by default: customer copy must stay on the drafted facts.
_ENABLED = os.getenv("WARRANTY_STEP_PARAPHRASE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_MODEL = os.getenv("WARRANTY_STEP_PARAPHRASE_MODEL", "").strip()
_MAX_TOKENS = int(os.getenv("WARRANTY_STEP_PARAPHRASE_MAX_TOKENS", "450"))

_CUSTOMER_FORBIDDEN_REFERENCE_MARKERS = (
    "from past cases",
    "past support ticket",
    "similar support cases",
    "similar support history",
    "from the knowledge base",
    "related topic:",
    "refer to:",
    "refer to page",
    "ticket #",
    "freshdesk",
)

# If paraphrase invents a defect topic that was not in the draft, reject it.
# Keys are workflow answer_keys; values are distinctive phrases for that topic.
_TOPIC_MARKER_PHRASES: dict[str, tuple[str, ...]] = {
    "air": (
        "air compression",
        "airbag",
        "air bags",
        "inflate",
        "inflating",
        "hissing",
        "air hose",
        "no air",
        "compressor",
    ),
    "remote": (
        "remote control",
        "hand controller",
        "tablet remote",
        "controller cable",
    ),
    "power": (
        "wall outlet",
        "power cord",
        "back switch",
        "won't turn on",
        "will not turn on",
    ),
    "footrest": (
        "footrest",
        "legrest",
        "leg rest",
        "calf roller",
    ),
    "heat": (
        "heating",
        "heater",
        "won't heat",
        "not heating",
    ),
    "voice": (
        "voice control",
        "voice command",
        "microphone",
        "false trigger",
    ),
    "rolling": (
        "massage rollers",
        "roller mechanism",
        "kneading motor",
        "rolling noise",
    ),
    "recline": (
        "recline",
        "zero gravity",
        "won't recline",
    ),
}


def _paraphrase_enabled() -> bool:
    return _ENABLED and bool(os.environ.get("OPENAI_API_KEY"))


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        from config import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT
    except ImportError:
        return None
    return OpenAI(
        api_key=api_key,
        timeout=float(OPENAI_REQUEST_TIMEOUT),
        max_retries=int(OPENAI_MAX_RETRIES),
    )


def _model_name() -> str:
    if _MODEL:
        return _MODEL
    try:
        from config import ROUTER_MODEL  # type: ignore

        if ROUTER_MODEL:
            return ROUTER_MODEL
    except Exception:
        pass
    return os.environ.get("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _question_preserved(output: str, base_prompt: str) -> bool:
    """Ensure the branching question survived the rewrite."""
    if not base_prompt.strip():
        return True
    if base_prompt.strip() in output:
        return True
    # Tolerate minor whitespace differences only.
    return _normalize(base_prompt) in _normalize(output)


def _introduced_error_code_claim(output: str, draft: str) -> bool:
    """Block conversion of a model/number into an unconfirmed error code."""
    return "error code" in _normalize(output) and "error code" not in _normalize(draft)


def _contains_internal_reference(output: str) -> bool:
    """Reject customer copy that exposes internal provenance or manual row labels."""
    normalized = _normalize(output)
    return any(marker in normalized for marker in _CUSTOMER_FORBIDDEN_REFERENCE_MARKERS)


def _introduced_off_topic_claim(
    output: str,
    draft: str,
    *,
    defect_category: Optional[str] = None,
) -> bool:
    """
    True when the rewrite adds a defect-topic phrase that was not in the draft.

    Example: remote-path draft gets paraphrased into “air compression suddenly
    stopping” — reject and keep the safer draft.
    """
    out = _normalize(output)
    src = _normalize(draft)
    selected = (defect_category or "").strip().lower()
    for topic, phrases in _TOPIC_MARKER_PHRASES.items():
        if selected and topic == selected:
            continue
        # rolling/recline both map to mech tips; still block inventing them
        # on an unrelated selected topic.
        for phrase in phrases:
            if phrase in out and phrase not in src:
                return True
    return False


def _format_options_hint(options: list[dict[str, Any]]) -> str:
    labels = [
        str(opt.get("label") or "").strip()
        for opt in options
        if str(opt.get("label") or "").strip()
    ]
    if not labels:
        return ""
    if len(labels) <= 4:
        joined = "; ".join(labels)
        return f"The customer can pick from these options: {joined}."
    return "The customer will choose from the buttons shown below."


def build_paraphrase_system_prompt(*, base_prompt: str) -> str:
    return (
        "You rewrite Osaki/Titan warranty chatbot messages to sound warm, clear, "
        "and mobile-friendly.\n"
        "Rules:\n"
        "- Keep every troubleshooting tip and fact from the draft — do not invent steps.\n"
        "- Do NOT promise replacement, refund, technician dispatch, compensation, or approval.\n"
        "- Do NOT mention support tickets, ticket numbers, Freshdesk, past cases, "
        "knowledge-base sources, internal references, or manual page references.\n"
        "- Soft wording like \"symptoms like yours\" or \"often help\" is OK when "
        "already present in the draft — do not invent ticket subjects or case IDs.\n"
        "- Do NOT invent a different problem type than the draft (e.g. do not "
        "turn a remote issue into air compression / airbags / rollers).\n"
        "- Do NOT add new questions beyond the one provided.\n"
        "- The message MUST end with this exact workflow question copied verbatim "
        f"(same punctuation):\n{base_prompt.strip()}\n"
        "- You may rephrase everything above that final question.\n"
        "- Use short sentences. Bullet numbering is OK.\n"
        "- Output ONLY the customer-facing message — no preamble or meta commentary."
    )


def _normalize_step_draft(draft: str, base_prompt: str) -> str:
    """Light cleanup when LLM paraphrase is unavailable."""
    text = (draft or "").strip()
    prompt = (base_prompt or "").strip()
    if not text:
        return text
    if prompt and text.count(prompt) > 1:
        head = text[: text.rfind(prompt)].replace(prompt, "").strip()
        text = f"{head}\n\n{prompt}".strip() if head else prompt
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def paraphrase_step_message(
    draft: str,
    *,
    base_prompt: str,
    model_name: str = "",
    node_id: str = "",
    options: Optional[list[dict[str, Any]]] = None,
    defect_category: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Rewrite ``draft`` with a small LLM, or return it unchanged.

    Returns ``(message, paraphrased)``.
    """
    draft = (draft or "").strip()
    base_prompt = (base_prompt or "").strip()
    if not draft or not _paraphrase_enabled():
        return _normalize_step_draft(draft, base_prompt), False

    client = _openai_client()
    if client is None:
        return _normalize_step_draft(draft, base_prompt), False

    user_parts = [f"Draft message:\n{draft}"]
    if model_name:
        user_parts.append(f"Chair model: {model_name}")
    if node_id:
        user_parts.append(f"Workflow node: {node_id}")
    if defect_category:
        user_parts.append(
            f"Customer-selected defect topic: {defect_category}. "
            "Do not rewrite this into a different problem type."
        )
    opt_hint = _format_options_hint(list(options or []))
    if opt_hint:
        user_parts.append(opt_hint)

    try:
        completion = client.chat.completions.create(
            model=_model_name(),
            max_tokens=_MAX_TOKENS,
            temperature=0.25,
            messages=[
                {
                    "role": "system",
                    "content": build_paraphrase_system_prompt(base_prompt=base_prompt),
                },
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Step paraphrase failed (%s): %s", node_id, exc)
        return _normalize_step_draft(draft, base_prompt), False

    if (
        not text
        or not _question_preserved(text, base_prompt)
        or _introduced_error_code_claim(text, draft)
        or _contains_internal_reference(text)
        or _introduced_off_topic_claim(
            text, draft, defect_category=defect_category
        )
    ):
        logger.info(
            "Step paraphrase rejected — invariant failed (node=%s)",
            node_id,
        )
        return _normalize_step_draft(draft, base_prompt), False

    return text, True
