"""
ringcentral_voice.py
====================
Adapt warranty flowchart nodes for RingCentral phone IVR (DTMF + TTS scripts).

Phone UX uses 1-based DTMF indices — WarrantyEngine._match_option already
supports "1", "2", … as option selectors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
RC_AUDIO_CACHE_DIR = Path(__file__).resolve().parent.parent / "rc_audio_cache"
RC_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# After-hours IVR: 0 replays the current prompt (no live agent transfer).
REPEAT_DTMF = "0"
POST_DIY_FIXED_DTMF = "1"


class IvrPhase(str, Enum):
    MENU = "menu"
    POST_DIY = "post_diy"
    DONE = "done"


@dataclass
class VoiceCallContext:
    session_id: str
    party_id: str
    ticket_id: str
    caller_phone: str = ""
    phase: IvrPhase = IvrPhase.MENU
    awaiting_command: Optional[str] = None  # "Play" | "Collect"
    last_audio_key: str = ""


# In-memory call state keyed by RingCentral telephony sessionId.
_call_contexts: dict[str, VoiceCallContext] = {}


def get_call_context(session_id: str) -> Optional[VoiceCallContext]:
    return _call_contexts.get(session_id)


def set_call_context(ctx: VoiceCallContext) -> None:
    _call_contexts[ctx.session_id] = ctx


def pop_call_context(session_id: str) -> Optional[VoiceCallContext]:
    return _call_contexts.pop(session_id, None)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_menu_script(node: dict) -> str:
    """Build TTS script for a question node with numbered DTMF options."""
    prompt = _strip_markdown(str(node.get("prompt") or ""))
    options = node.get("options") or []
    lines = [prompt]
    for idx, opt in enumerate(options, start=1):
        label = _strip_markdown(str(opt.get("label") or f"Option {idx}"))
        # Keep labels short for phone — truncate very long flowchart labels.
        if len(label) > 120:
            label = label[:117] + "..."
        lines.append(f"Press {idx} for {label}.")
    lines.append(f"Press {REPEAT_DTMF} to hear these options again.")
    return " ".join(lines)


def build_terminal_script(node: dict, enrichment: Optional[dict[str, Any]] = None) -> str:
    """Build TTS for a terminal node — DIY steps capped for phone length."""
    prompt = _strip_markdown(str(node.get("prompt") or ""))
    parts = [prompt]

    steps: list[str] = []
    if enrichment:
        msg = str(enrichment.get("customer_message") or "")
        for line in msg.splitlines():
            cleaned = _strip_markdown(line.strip())
            if cleaned.startswith(("•", "-", "Step", "1.", "2.", "3.", "4.", "5.")):
                steps.append(cleaned)
            elif cleaned and len(steps) < 5 and len(cleaned) > 20:
                steps.append(cleaned)
        steps = steps[:5]

    if steps:
        parts.append("Here are the steps to try.")
        parts.extend(steps)
    else:
        parts.append("Please follow the guidance we just described.")

    parts.append(
        f"Press {POST_DIY_FIXED_DTMF} if that fixed the issue. "
        f"Press {REPEAT_DTMF} to hear these steps again."
    )
    return " ".join(parts)


def build_after_hours_closure_script() -> str:
    """Closing message when the workflow ends outside live agent hours."""
    try:
        from config import WARRANTY_BUSINESS_HOURS  # type: ignore

        hours = str(WARRANTY_BUSINESS_HOURS or "Mon-Fri, 10:00 AM - 6:00 PM CST")
    except Exception:
        hours = "Monday through Friday, ten A M to six P M Central time"
    return (
        "Thank you. We have recorded your answers for our warranty team. "
        f"Their phone line is open {hours}. "
        "You can also start a warranty chat on our website anytime. "
        f"Press {POST_DIY_FIXED_DTMF} to end this call. "
        f"Press {REPEAT_DTMF} to hear this message again."
    )


def menu_dtmf_patterns(node: dict) -> list[str]:
    """Allowed DTMF keys for a question node (+ agent escape)."""
    count = len(node.get("options") or [])
    patterns = [str(i) for i in range(1, count + 1)]
    patterns.append(REPEAT_DTMF)
    return patterns


def post_diy_dtmf_patterns() -> list[str]:
    return [POST_DIY_FIXED_DTMF, REPEAT_DTMF]


def audio_cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return digest


def audio_public_url(cache_key: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL must be set for RingCentral play URIs.")
    return f"{PUBLIC_BASE_URL}/rc/audio/{cache_key}.wav"


def ensure_audio_file(text: str) -> tuple[str, Path]:
    """
    Return (cache_key, local_path) for the TTS audio file.

    Generates via OpenAI TTS when OPENAI_API_KEY is set; otherwise writes a
    placeholder marker file so dev can stub the /rc/audio endpoint.
    """
    key = audio_cache_key(text)
    path = RC_AUDIO_CACHE_DIR / f"{key}.wav"
    if path.exists() and path.stat().st_size > 100:
        return key, path

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model=os.getenv("RC_TTS_MODEL", "tts-1"),
                voice=os.getenv("RC_TTS_VOICE", "nova"),
                input=text[:4096],
                response_format="wav",
            )
            path.write_bytes(response.content)
            return key, path
        except Exception as exc:
            logger.warning("OpenAI TTS failed, using placeholder: %s", exc)

    # Placeholder for local dev without TTS — replace before production.
    path.write_bytes(b"RIFF")
    return key, path


def resolve_play_uri(text: str) -> str:
    """Generate/cache audio and return the HTTPS URI RingCentral will fetch."""
    key, _path = ensure_audio_file(text)
    return audio_public_url(key)
