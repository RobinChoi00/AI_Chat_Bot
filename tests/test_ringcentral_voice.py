"""
tests/test_ringcentral_voice.py
===============================
Unit tests for RingCentral voice adapter (no live RC API).
"""

import sys
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from ringcentral_voice import (  # noqa: E402
    IvrPhase,
    REPEAT_DTMF,
    VoiceCallContext,
    build_after_hours_closure_script,
    build_after_hours_welcome_script,
    build_business_hours_connect_script,
    build_menu_script,
    build_sales_transfer_script,
    build_terminal_script,
    menu_dtmf_patterns,
    post_diy_dtmf_patterns,
    ensure_audio_file,
    get_call_context,
    pop_call_context,
    resolve_play_uri,
    set_call_context,
)


def test_build_menu_script_includes_dtmf_options():
    node = {
        "prompt": "What type of warranty issue?",
        "options": [
            {"label": "Installation Issue", "answer_key": "installation"},
            {"label": "Delivery Issue", "answer_key": "delivery"},
            {"label": "Defect", "answer_key": "defect"},
        ],
    }
    script = build_menu_script(node)
    assert "Press 1 for Installation Issue" in script
    assert "Press 2 for Delivery Issue" in script
    assert "Press 3 for Defect" in script
    assert f"Press {REPEAT_DTMF} to hear these options again" in script


def test_menu_dtmf_patterns_includes_repeat():
    node = {"options": [{"label": "A"}, {"label": "B"}]}
    assert menu_dtmf_patterns(node) == ["1", "2", REPEAT_DTMF]


def test_build_terminal_script_includes_post_diy_prompt():
    node = {"prompt": "Try reconnecting the air hose."}
    script = build_terminal_script(node, None)
    assert "Press 1 if that fixed the issue" in script
    assert f"Press {REPEAT_DTMF} to hear these steps again" in script
    assert "specialist" not in script.lower()


def test_build_after_hours_closure_script_mentions_business_hours():
    script = build_after_hours_closure_script()
    assert "Press 1 to end this call" in script
    assert f"Press {REPEAT_DTMF} to hear this message again" in script
    assert "call back" in script.lower()
    assert "text" in script.lower()


def test_build_after_hours_welcome_mentions_closed_and_docs():
    script = build_after_hours_welcome_script()
    assert "closed" in script.lower()
    assert "warranty" in script.lower()
    assert "invoice" in script.lower() or "order number" in script.lower()
    assert "text message" in script.lower()


def test_build_business_hours_connect_script_mentions_connecting():
    script = build_business_hours_connect_script()
    assert "connecting" in script.lower()
    assert "invoice" in script.lower() or "order number" in script.lower()


def test_build_sales_transfer_script_announces_sales():
    script = build_sales_transfer_script()
    assert "sales" in script.lower()
    assert "transfer" in script.lower()


def test_post_diy_patterns_are_repeat_or_hangup_only():
    assert post_diy_dtmf_patterns() == ["1", REPEAT_DTMF]


def test_call_context_is_restored_after_in_memory_state_is_lost():
    from ringcentral_voice import _call_contexts  # noqa: WPS433

    session_id = f"persist-{uuid.uuid4().hex}"
    ctx = VoiceCallContext(
        session_id=session_id,
        party_id="party-1",
        ticket_id="ticket-1",
        caller_phone="+15551234567",
        phase=IvrPhase.POST_DIY,
        awaiting_command="Collect",
        last_audio_key="abc123",
    )
    set_call_context(ctx)
    _call_contexts.clear()

    restored = get_call_context(session_id)
    assert restored is not None
    assert restored.ticket_id == "ticket-1"
    assert restored.phase == IvrPhase.POST_DIY
    assert restored.awaiting_command == "Collect"
    pop_call_context(session_id)


def test_production_never_writes_placeholder_audio(tmp_path, monkeypatch):
    import ringcentral_voice as voice

    monkeypatch.setattr(voice, "RC_AUDIO_CACHE_DIR", tmp_path)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        ensure_audio_file(f"unique-{uuid.uuid4().hex}")
    except RuntimeError as exc:
        assert "unavailable" in str(exc).lower()
    else:
        raise AssertionError("production TTS unexpectedly created placeholder audio")
    assert list(tmp_path.iterdir()) == []


def test_configured_fallback_audio_is_used_when_tts_fails(monkeypatch):
    import ringcentral_voice as voice

    fallback = "https://cdn.example.com/temporary-message.wav"
    monkeypatch.setenv("RC_FALLBACK_AUDIO_URI", fallback)
    monkeypatch.setattr(voice, "ensure_audio_file", lambda _text: (_ for _ in ()).throw(RuntimeError("tts down")))
    assert resolve_play_uri("hello") == fallback
