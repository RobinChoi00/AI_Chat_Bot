"""
ringcentral_voice.py
====================
Adapt warranty flowchart nodes for RingCentral phone IVR (DTMF + TTS scripts).

Phone UX uses 1-based DTMF indices — WarrantyEngine._match_option already
supports "1", "2", … as option selectors.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
_DEFAULT_RC_AUDIO_CACHE = Path(__file__).resolve().parent.parent / "rc_audio_cache"
RC_AUDIO_CACHE_DIR = Path(os.getenv("RC_AUDIO_CACHE_DIR", str(_DEFAULT_RC_AUDIO_CACHE)))
RC_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# After-hours IVR: 0 replays the current prompt (no live agent transfer).
REPEAT_DTMF = "0"
POST_DIY_FIXED_DTMF = "1"


class IvrPhase(str, Enum):
    CONNECTING = "connecting"
    MENU = "menu"
    POST_DIY = "post_diy"
    SALES_TRANSFER = "sales_transfer"
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
    cached = _call_contexts.get(session_id)
    if cached is not None:
        return cached
    if not session_id:
        return None
    try:
        from warranty_models import RingCentralCallState, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = (
                db.query(RingCentralCallState)
                .filter(RingCentralCallState.session_id == session_id)
                .first()
            )
            if row is None:
                return None
            try:
                phase = IvrPhase(str(row.phase))
            except ValueError:
                phase = IvrPhase.MENU
            restored = VoiceCallContext(
                session_id=str(row.session_id),
                party_id=str(row.party_id),
                ticket_id=str(row.ticket_id),
                caller_phone=str(row.caller_phone or ""),
                phase=phase,
                awaiting_command=str(row.awaiting_command) if row.awaiting_command else None,
                last_audio_key=str(row.last_audio_key or ""),
            )
        _call_contexts[session_id] = restored
        logger.info("RC IVR restored persisted call state session=%s", session_id)
        return restored
    except Exception:
        logger.exception("RC IVR could not restore call state session=%s", session_id)
        return None


def set_call_context(ctx: VoiceCallContext) -> None:
    _call_contexts[ctx.session_id] = ctx
    try:
        from warranty_models import RingCentralCallState, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            row = (
                db.query(RingCentralCallState)
                .filter(RingCentralCallState.session_id == ctx.session_id)
                .first()
            )
            if row is None:
                row = RingCentralCallState(session_id=ctx.session_id)
                db.add(row)
            row.party_id = ctx.party_id
            row.ticket_id = ctx.ticket_id
            row.caller_phone = ctx.caller_phone
            row.phase = ctx.phase.value
            row.awaiting_command = ctx.awaiting_command
            row.last_audio_key = ctx.last_audio_key
    except Exception:
        logger.exception("RC IVR could not persist call state session=%s", ctx.session_id)
        if os.getenv("APP_ENV", "development").strip().lower() == "production":
            raise


def pop_call_context(session_id: str) -> Optional[VoiceCallContext]:
    ctx = _call_contexts.pop(session_id, None)
    if ctx is None:
        ctx = get_call_context(session_id)
        _call_contexts.pop(session_id, None)
    try:
        from warranty_models import RingCentralCallState, warranty_db_session  # noqa: WPS433

        with warranty_db_session() as db:
            db.query(RingCentralCallState).filter(
                RingCentralCallState.session_id == session_id
            ).delete(synchronize_session=False)
    except Exception:
        logger.exception("RC IVR could not delete call state session=%s", session_id)
        if os.getenv("APP_ENV", "development").strip().lower() == "production":
            raise
    return ctx


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
    from ringcentral_hours import next_warranty_open_phrase, warranty_hours_text  # noqa: WPS433

    hours = warranty_hours_text()
    next_open = next_warranty_open_phrase()
    return (
        "Thank you. We have recorded your answers for our warranty team. "
        f"Warranty phone support is open {hours}. "
        f"Please call back {next_open}. "
        "You can also continue on our website warranty chat anytime. "
        "When you hang up, we will text you a link to pick up where you left off. "
        f"Press {POST_DIY_FIXED_DTMF} to end this call. "
        f"Press {REPEAT_DTMF} to hear this message again."
    )


def build_after_hours_welcome_script() -> str:
    """Opening message when the warranty line is closed — sets expectations."""
    from ringcentral_hours import (  # noqa: WPS433
        next_warranty_open_phrase,
        sales_hours_text,
        warranty_hours_text,
    )

    hours = warranty_hours_text()
    next_open = next_warranty_open_phrase()
    sales_note = sales_hours_text()
    parts = [
        "Thank you for calling Osaki and Titan warranty support.",
        "Our warranty service department is closed right now.",
        f"Warranty phone hours are {hours}.",
        f"Please call back {next_open}.",
        "To help us assist you faster, please have your invoice, order number, "
        "serial number photos, or any ticket reference ready before you call.",
        "After this call you will receive a text message with a link to continue online.",
    ]
    if sales_note:
        parts.append(sales_note)
    parts.append("You can still use our automated warranty assistant now.")
    return " ".join(parts)


def build_business_hours_connect_script() -> str:
    """Played before connecting to a live warranty agent during open hours."""
    from ringcentral_hours import warranty_hours_text  # noqa: WPS433

    hours = warranty_hours_text()
    return (
        "Thank you for calling Osaki and Titan warranty support. "
        f"Our warranty team is open {hours}. "
        "Please have your invoice, order number, or ticket reference ready. "
        "We are now connecting you to the next available warranty specialist. "
        "Please stay on the line."
    )


def build_sales_transfer_script() -> str:
    """Announce before transferring to sales during business hours."""
    return (
        "This option is handled by our sales team, not warranty. "
        "We are now transferring your call to sales. Please stay on the line."
    )


def build_after_hours_sales_closed_script() -> str:
    """Sales handoff is not available when warranty is closed."""
    from ringcentral_hours import next_warranty_open_phrase, warranty_hours_text  # noqa: WPS433

    hours = warranty_hours_text()
    next_open = next_warranty_open_phrase()
    return (
        "Our warranty service department is closed, so we cannot transfer you to sales for warranty help. "
        f"Warranty phone hours are {hours}. "
        f"Please call back {next_open}, or use our website warranty chat. "
        "When you hang up, we will text you a link to continue your case online. "
        f"Press {POST_DIY_FIXED_DTMF} to end this call. "
        f"Press {REPEAT_DTMF} to hear this message again."
    )


def build_question_text_handoff_script() -> str:
    """Phone cannot capture free-text — direct caller to SMS resume link."""
    return (
        "This step needs your order number, tracking details, or other written information, "
        "which is easier on our website. "
        "When you finish this call we will text you a link to continue. "
        f"Press {REPEAT_DTMF} to hear the previous options again."
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
    model = os.getenv("RC_TTS_MODEL", "tts-1")
    voice = os.getenv("RC_TTS_VOICE", "nova")
    digest = hashlib.sha256(f"{model}\0{voice}\0{text}".encode("utf-8")).hexdigest()[:24]
    return digest


def audio_public_url(cache_key: str) -> str:
    public_base_url = os.getenv("PUBLIC_BASE_URL", PUBLIC_BASE_URL).rstrip("/")
    if not public_base_url.startswith("https://"):
        raise RuntimeError("PUBLIC_BASE_URL must be set for RingCentral play URIs.")
    return f"{public_base_url}/rc/audio/{cache_key}.wav"


def _atomic_audio_write(path: Path, content: bytes) -> None:
    if len(content) <= 100 or not content.startswith(b"RIFF"):
        raise RuntimeError("TTS provider returned an invalid WAV payload.")
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(content)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _development_silence_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


def ensure_audio_file(text: str) -> tuple[str, Path]:
    """
    Return (cache_key, local_path) for the TTS audio file.

    Generates via OpenAI TTS. Production never serves a placeholder file.
    """
    key = audio_cache_key(text)
    path = RC_AUDIO_CACHE_DIR / f"{key}.wav"
    if path.exists() and path.stat().st_size > 100:
        return key, path

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI

            try:
                timeout = float(os.getenv("RC_TTS_TIMEOUT_SECONDS", "20"))
            except ValueError:
                timeout = 20.0
            client = OpenAI(api_key=api_key, timeout=max(5.0, min(timeout, 60.0)), max_retries=2)
            response = client.audio.speech.create(
                model=os.getenv("RC_TTS_MODEL", "tts-1"),
                voice=os.getenv("RC_TTS_VOICE", "nova"),
                input=text[:4096],
                response_format="wav",
            )
            _atomic_audio_write(path, response.content)
            return key, path
        except Exception as exc:
            logger.exception("OpenAI TTS failed: %s", type(exc).__name__)

    if os.getenv("APP_ENV", "development").strip().lower() == "production":
        raise RuntimeError("RingCentral TTS audio generation is unavailable.")

    # A valid short silent WAV keeps local route tests realistic without
    # pretending that a four-byte RIFF marker is playable audio.
    _atomic_audio_write(path, _development_silence_wav())
    return key, path


def resolve_play_uri(text: str) -> str:
    """Generate/cache audio and return the HTTPS URI RingCentral will fetch."""
    try:
        key, _path = ensure_audio_file(text)
        return audio_public_url(key)
    except Exception:
        fallback = os.getenv("RC_FALLBACK_AUDIO_URI", "").strip()
        if fallback.startswith("https://"):
            logger.exception("RC TTS unavailable; using configured fallback audio")
            return fallback
        raise
