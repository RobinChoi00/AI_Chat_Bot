"""
sales_tidio_buttons.py
======================
Tidio Flow cannot invent Decision (quick reply) buttons from an API
response — those nodes are static in the Flow editor.

So we bridge the gap in three ways the Flow / visitor can use today:

1. Cap ``quick_replies`` to a Tidio-friendly size (default 5).
2. Flatten ``button_1_label`` … ``button_5_payload`` for Flow variable mapping
   and static Decision nodes that mirror the same labels.
3. Append a numbered menu to ``reply_plain`` and resolve the visitor's next
   message ("1", "Shop this chair", …) back to the stored payload.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

_MAX_BUTTONS = max(1, min(8, int(os.getenv("TIDIO_MAX_QUICK_REPLIES", "5"))))

# Lower = keep first when capping.
_PREFIX_PRIORITY: list[tuple[str, int]] = [
    ("open:", 10),
    ("cta:financing", 20),
    ("lead:", 25),
    ("cta:showroom", 30),
    ("recommend:height:", 50),
    ("recommend:weight:", 50),
    ("recommend:space:", 50),
    ("recommend:goal:", 50),
    ("recommend:intensity:", 55),
    ("recommend:foot:", 55),
    ("recommend:", 70),
    ("specs:", 80),
    ("stock:", 85),
    ("human", 35),
    ("menu", 95),
]


def tidio_max_buttons() -> int:
    return _MAX_BUTTONS


def _priority(payload: str) -> int:
    raw = (payload or "").strip().lower()
    for prefix, rank in _PREFIX_PRIORITY:
        if raw.startswith(prefix) or raw == prefix.rstrip(":"):
            return rank
    return 100


def _open_url(payload: str) -> Optional[str]:
    raw = (payload or "").strip()
    if raw.lower().startswith("open:"):
        url = raw.split(":", 1)[1].strip()
        if url.startswith("https://"):
            return url
    if raw.lower().startswith("cta:financing:"):
        url = raw.split(":", 2)[2].strip()
        if url.startswith("https://"):
            return url
    return None


def prioritize_quick_replies(
    quick_replies: list[Any],
    *,
    limit: Optional[int] = None,
) -> list[dict[str, str]]:
    """Dedupe + rank + cap for Tidio chat."""
    cap = limit if limit is not None else _MAX_BUTTONS
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in quick_replies or []:
        if isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            payload = str(item.get("payload") or "").strip()
        else:
            label = str(getattr(item, "label", "") or "").strip()
            payload = str(getattr(item, "payload", "") or "").strip()
        if not label or not payload:
            continue
        key = payload.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"label": label[:80], "payload": payload[:200]})

    ranked = sorted(
        enumerate(cleaned),
        key=lambda pair: (_priority(pair[1]["payload"]), pair[0]),
    )
    return [q for _, q in ranked][:cap]


def flatten_buttons_for_flow(buttons: list[dict[str, str]]) -> dict[str, Any]:
    """
    Flat fields Tidio Flow API Call → session variables can map 1:1.

    button_1_label / button_1_payload / button_1_url …
    button_count
    """
    out: dict[str, Any] = {"button_count": len(buttons)}
    for idx in range(1, _MAX_BUTTONS + 1):
        out[f"button_{idx}_label"] = ""
        out[f"button_{idx}_payload"] = ""
        out[f"button_{idx}_url"] = ""
    for i, btn in enumerate(buttons, start=1):
        out[f"button_{i}_label"] = btn["label"]
        out[f"button_{i}_payload"] = btn["payload"]
        url = _open_url(btn["payload"]) or ""
        out[f"button_{i}_url"] = url
    return out


def format_numbered_menu(buttons: list[dict[str, str]]) -> str:
    if not buttons:
        return ""
    lines = ["", "—", "Tap a choice below, or reply with the number:"]
    for i, btn in enumerate(buttons, start=1):
        lines.append(f"{i}) {btn['label']}")
    return "\n".join(lines)


def append_numbered_menu(reply_plain: str, buttons: list[dict[str, str]]) -> str:
    menu = format_numbered_menu(buttons)
    if not menu:
        return (reply_plain or "").strip()
    base = (reply_plain or "").rstrip()
    # Avoid duplicating if we already appended once.
    if "reply with the number:" in base.lower():
        return base
    return f"{base}{menu}"


# Must match ``prioritize_quick_replies(_menu_quick_replies())`` numbering so
# visitors can type ``1`` even when session ``last_quick_replies`` was lost
# (common Tidio Flow bug when ``session_id`` is not passed between turns).
_DEFAULT_MENU_BUTTONS: list[dict[str, str]] = [
    {"label": "Recommend a chair", "payload": "recommend"},
    {"label": "Availability / stock", "payload": "stock"},
    {"label": "Talk to a human", "payload": "human"},
    {"label": "Check a price", "payload": "price"},
    {"label": "Compare two models", "payload": "compare"},
]


def resolve_button_choice(
    message: str,
    last_buttons: list[dict[str, str]] | None,
) -> Optional[str]:
    """
    Map a visitor tap/type back to a payload.

    Accepts:
      - exact / case-insensitive label match
      - bare number ``1`` … ``N``
      - ``1) Shop this chair`` style echoes

    If ``last_buttons`` is empty, falls back to the default main-menu order
    so ``1`` / ``1.`` still means Recommend a chair.
    """
    text = (message or "").strip()
    if not text:
        return None

    buttons = list(last_buttons or []) or list(_DEFAULT_MENU_BUTTONS)

    # Bare index: "1", "1)", "1.", "1:"
    bare = re.fullmatch(r"([1-9])[).:\s]*", text)
    if bare:
        idx = int(bare.group(1)) - 1
        if 0 <= idx < len(buttons):
            return buttons[idx]["payload"]

    # "1) Shop this chair" / "1. Recommend a chair"
    numbered = re.match(r"^([1-9])[).:\-\s]+(.+)$", text)
    if numbered:
        idx = int(numbered.group(1)) - 1
        if 0 <= idx < len(buttons):
            return buttons[idx]["payload"]
        text = numbered.group(2).strip()

    lowered = text.lower()
    for btn in buttons:
        if btn["label"].lower() == lowered:
            return btn["payload"]

    # Soft match: visitor typed a unique keyword from a label
    hits = [
        btn
        for btn in buttons
        if lowered in btn["label"].lower() or btn["label"].lower() in lowered
    ]
    if len(hits) == 1:
        return hits[0]["payload"]
    return None


def normalize_stored_buttons(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        payload = str(item.get("payload") or "").strip()
        if label and payload:
            out.append({"label": label, "payload": payload})
    return out


def is_probably_url_host(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False
