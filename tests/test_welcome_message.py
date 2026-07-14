"""Tests for opening welcome message helpers."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from chat_welcome import build_chat_welcome_message, is_conversation_start, is_opening_greeting


def test_opening_greeting_detects_hello():
    assert is_opening_greeting("Hello!")
    assert is_opening_greeting("hi there")
    assert is_opening_greeting("안녕하세요")
    assert is_opening_greeting("¡Hola!")


def test_opening_greeting_rejects_specific_question():
    assert not is_opening_greeting("How much is the Solo Flex?")


def test_conversation_start_empty_history():
    assert is_conversation_start([])
    assert is_conversation_start(None)


def test_welcome_message_asks_for_model():
    msg = build_chat_welcome_message()
    assert "Which massage chair model" in msg
    assert "Hello" in msg


def test_localized_welcome_messages():
    assert "모델" in build_chat_welcome_message("ko")
    assert "modelo" in build_chat_welcome_message("es").lower()
