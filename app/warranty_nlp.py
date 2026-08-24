"""
Natural-language helpers for the warranty workflow (Phase 1 hybrid).

Maps free-text customer messages to flowchart answer_keys while keeping the
deterministic WarrantyEngine as the source of truth for branching and records.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ISSUE_TYPES = frozenset({"installation", "delivery", "defect"})

_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "delivery": (
        "deliver",
        "delivery",
        "shipping",
        "shipped",
        "tracking",
        "package",
        "fedex",
        "ups",
        "usps",
        "carrier",
        "arrived damaged",
        "box is broken",
        "box was crushed",
        "box was damaged",
    ),
    "installation": (
        "install",
        "installation",
        "assembly",
        "assemble",
        "setup",
        "set up",
        "put together",
        "manual",
    ),
    "defect": (
        "defect",
        "broken",
        "malfunction",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't turn on",
        "wont turn on",
        "fault",
        "repair",
        "remote",
        "power",
        "recline",
        "inflate",
        "airbag",
        "air bag",
        "footrest",
        "foot rest",
        "not inflating",
        "won't inflate",
        "wont inflate",
        "calf",
        "legrest",
    ),
}

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|affirmative|i do|i have)\b",
    re.I,
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|negative|i don't|i dont|don't have|dont have)\b",
    re.I,
)
# Generic yes/no must be the whole message. Otherwise "no power" / "never came"
# / "I have tracking" steal the no/yes option on unrelated questions.
_BARE_YES_RE = re.compile(
    r"^(yes|yeah|yep|yup|sure|correct|affirmative)( please| it (is|was|did))?$",
    re.I,
)
_BARE_NO_RE = re.compile(
    r"^(no|nope|nah|negative|not really)$",
    re.I,
)
_SUGGEST_CONFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|exactly|yes please|"
    r"that'?s (it|the one|right|correct)|yes that'?s (it|right))\s*[.!]?\s*$",
    re.I,
)
_SUGGEST_REJECT_RE = re.compile(
    r"^\s*(no|nope|nah|not that|wrong|neither)\s*[.!]?\s*$",
    re.I,
)
_STEPS_DONE_RE = re.compile(
    r"\b(tried (all )?(the )?steps|i('ve| have) tried|watched the (guide|video)|"
    r"completed (the )?(prep|steps|setup|guide)|done with (the )?steps|"
    r"all the steps|i('ve| have) (watched|done|finished)|"
    r"i did (that|it|them)|tried it|\bdone\b)\b",
    re.I,
)
_UNABLE_RE = re.compile(
    r"\b(can'?t (safely )?(do|complete|try)|cannot (safely )?(do|complete)|"
    r"unable to|too unsafe|don'?t have (the )?(photos|paperwork|video))\b",
    re.I,
)
_NEED_TEAM_RE = re.compile(
    r"\b(need help|please help|send (a )?(tech|technician|someone)|"
    r"send someone( out)?|come out|warranty team|"
    r"call me|contact me|submit (my )?(case|claim)|file a claim)\b",
    re.I,
)
_STILL_BROKEN_RE = re.compile(
    r"\b(still (broken|there|not working|happening)|not (fixed|working)|"
    r"didn'?t work|doesn'?t work|issue is still)\b",
    re.I,
)
_WORKING_NOW_RE = re.compile(
    r"\b(working now|it'?s working|it'?s fixed|fixed now|"
    r"problem (is )?gone|resolved)\b",
    re.I,
)
_SETUP_DONE_RE = re.compile(
    r"\b(set up now|all set up|assembled|"
    r"install(ation)? (is )?(done|complete)|chair is (ready|set up))\b",
    re.I,
)
_ALL_SET_RE = re.compile(
    r"\b(all set|no (more )?help needed|i'?m good|no thanks)\b",
    re.I,
)
_COME_BACK_RE = re.compile(
    r"\b(come back|not yet|later|i'?ll wait)\b",
    re.I,
)
_DELIVERY_SUBMIT_RE = re.compile(
    r"\b(submit|send (the|this|my)? ?case|file (the |a )?(claim|case))\b",
    re.I,
)

PENDING_SUGGESTED_KEY = "pending_suggested_answer_key"
PENDING_SUGGESTED_LABEL = "pending_suggested_label"

# Words that appear in many labels and must not unique-match an option.
_WEAK_WORDS = frozenset(
    {
        "won't",
        "wont",
        "doesn't",
        "doesnt",
        "don't",
        "dont",
        "coming",
        "working",
        "issue",
        "chair",
        "problem",
        "still",
        "there",
        "from",
        "something",
        "other",
        "about",
        "please",
        "help",
        "with",
        "that",
        "this",
        "have",
        "been",
        "not",
        "the",
        "and",
        "for",
        "are",
        "was",
        "has",
        "your",
        "my",
        "any",
        "all",
        "look",
        "looks",
        "looked",
        "fine",
        "area",
        "part",
        "massage",
        "warm",
        "here",
        "sound",
        "move",
        "moves",
        "back",
        "opened",
        "down",
        "turn",
        "turns",
    }
)

# Phrases → answer_key. Applied only when that key is on the current node,
# and only when exactly one current option hits.
_OPTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "power": (
        "won't turn on",
        "wont turn on",
        "doesn't turn on",
        "not turning on",
        "no power",
        "won't power on",
        "won't power up",
        "wont power up",
        "won't start",
        "wont start",
        "doesn't start",
        "no lights",
        "no lights at all",
        "chair is dead",
        "dead chair",
    ),
    "air": (
        "airbag",
        "air bag",
        "not inflating",
        "won't inflate",
        "wont inflate",
        "inflation",
        "air not working",
        "losing air",
        "deflating",
        "won't stay inflated",
        "wont stay inflated",
        "keeps deflating",
    ),
    "cosmetic": (
        "scratch",
        "scratched",
        "crack",
        "cracked",
        "dent",
        "scuff",
        "appearance",
        "looks damaged",
        "cosmetic",
    ),
    "remote": (
        "controller",
        "remote",
        "remote is dead",
        "remote dead",
        "blank screen",
        "no display",
        "screen is off",
    ),
    "rolling": (
        "rollers not moving",
        "roller",
        "kneading",
        "massage head",
        "massage mechanism",
        "rollers",
        "heads stuck",
        "mechanism stuck",
    ),
    "recline": (
        "won't recline",
        "wont recline",
        "not reclining",
        "lay back",
        "zero gravity",
        "recline",
    ),
    "heat": (
        "heater",
        "heating",
        "not warm",
        "no heat",
        "too hot",
        "temperature",
        "doesn't heat",
        "not getting warm",
    ),
    "voice": ("alexa", "voice control", "hey osaki", "voice command", "voice"),
    "footrest": (
        "ottoman",
        "legrest",
        "leg rest",
        "foot rest",
        "footrest",
        "foot roller",
        "foot rollers",
        "calf roller",
        "calf rollers",
    ),
    "feet_calves": ("calves", "calf", "feet", "foot", "legs", "ankles"),
    "shoulders_hips": (
        "shoulders",
        "shoulder",
        "hips",
        "hip",
        "lumbar",
        "upper back",
    ),
    "arms": ("armrest", "arms", "arm"),
    "side_panel": ("side panel", "side panels"),
    "base": ("bottom of the chair", "under the chair", "the base"),
    "status_check": (
        "where's my order",
        "where is my order",
        "where is my package",
        "where's my chair",
        "where is my chair",
        "track my",
        "tracking status",
        "shipment status",
        "delivery status",
        "when will it arrive",
    ),
    "damage_issue": (
        "arrived damaged",
        "box damaged",
        "damaged on arrival",
        "crushed",
        "box was crushed",
        "missing parts",
        "never came",
        "never showed",
        "delivery problem",
        "problem with delivery",
        "issue with my delivery",
    ),
    "damaged_in_transit": (
        "box crushed",
        "arrived damaged",
        "damaged in transit",
        "chair damaged",
        "box was crushed",
        "box is broken",
        "the box is broken",
        "box broken",
        "box damaged",
    ),
    "missing_parts": (
        "missing parts",
        "incomplete",
        "missing pieces",
        "parts missing",
        "parts are missing",
    ),
    "wrong_item": (
        "wrong chair",
        "wrong model",
        "wrong item",
        "sent the wrong",
        "wrong product",
        "not the model i ordered",
        "not what i ordered",
    ),
    "never_arrived": (
        "never came",
        "never showed",
        "never arrived",
        "marked delivered",
        "stolen",
        "package never arrived",
        "says delivered but missing",
        "marked delivered but missing",
    ),
    "other_delivery_problem": (
        "something else",
        "other delivery",
        "different delivery problem",
        "another delivery issue",
    ),
    "yes_chair_inside_damage": (
        "damaged inside",
        "chair inside was damaged",
        "damage inside the box",
        "chair was damaged inside",
        "inside damage",
    ),
    "no_chair_inside_damage": (
        "no damage inside",
        "chair inside was fine",
        "nothing damaged inside",
        "inside looked fine",
    ),
    "yes_box_damaged": (
        "box was crushed",
        "box damaged",
        "box was damaged",
        "box is broken",
        "the box is broken",
    ),
    "hose_clear": (
        "hose is clear",
        "hoses look fine",
        "hoses are clear",
        "clear and connected",
        "hose looks fine",
    ),
    "no_sound": (
        "no sound",
        "silent",
        "no noise",
        "completely silent",
        "no sound at all",
    ),
    "power_but_no_move": (
        "barely move",
        "have power but",
        "power but barely",
        "weak rollers",
        "barely moving",
        "powered but won't move",
    ),
    "bad_connection": (
        "loose cable",
        "connection loose",
        "bad connection",
        "loose connection",
        "cable is loose",
    ),
    "all_checked_ok": (
        "nothing obvious",
        "all checked",
        "nothing found",
        "everything looks fine",
        "checked everything",
        "nothing wrong that i can see",
    ),
    "still_no_heat": (
        "still no warmth",
        "still no heat",
        "still not warm",
        "ran heat still cold",
        "heat on still cold",
    ),
    "will_try_warmup": (
        "i'll try warm-up",
        "i'll try warmup",
        "try 10 minutes",
        "will try heat",
        "i'll run the heat",
    ),
    "commands_not_responding": (
        "buttons don't work",
        "buttons dont work",
        "some buttons",
        "certain commands",
        "partial remote",
        "only some buttons work",
        "some commands don't work",
    ),
    "late_delivery": ("was late", "delayed", "still hasn't arrived", "running late"),
    "no_tracking": (
        "don't have tracking",
        "dont have tracking",
        "no tracking",
        "don't have a tracking",
    ),
    "has_tracking": (
        "i have tracking",
        "here's my tracking",
        "have my tracking",
        "tracking number is",
    ),
    "yes_worked": (
        "used to work",
        "worked before",
        "it worked then stopped",
        "stopped working",
        "worked then",
    ),
    "never_worked": (
        "never worked",
        "never has",
        "from day one",
        "out of the box it didn't",
        "never worked out of the box",
    ),
    "air_blowing": (
        "air is coming out",
        "air coming out",
        "i feel air",
        "air is blowing",
        "yes air",
        "air blowing",
    ),
    "no_air": ("no air coming", "nothing coming out", "no air at all", "no air"),
    "yes_hissing": ("hissing", "i hear air", "hear hissing", "hear air"),
    "no_hissing": ("no sound", "silent", "no hiss", "no sound at all"),
    "hose_issue": ("kinked", "disconnected hose", "hose disconnected", "pinched"),
    "hoses_ok": ("hoses look fine", "hose is fine", "hoses are fine"),
    "yes_white_glove": ("white glove", "white-glove"),
    "no_white_glove": ("standard delivery", "left at the door"),
    "signed_cleared": ("signed as cleared", "signed clear", "signed cleared"),
    "signed_damaged": ("signed as damaged", "signed damaged"),
    "visible_at_unboxing": ("when i opened", "unboxing", "opened the box"),
    "noticed_later": ("noticed later", "after a few days", "found it later"),
    "installation": (
        "assemble",
        "assembling",
        "assembly",
        "set up",
        "set it up",
        "setup",
        "put together",
        "install",
    ),
    "delivery": (
        "package",
        "shipping",
        "tracking",
        "delivery",
        "fedex",
        "ups",
        "box is broken",
        "box was crushed",
        "box was damaged",
    ),
    "defect": ("broken", "won't turn on", "not working", "defect", "malfunction"),
    "general_setup": (
        "put together",
        "how do i assemble",
        "set it up",
        "setup video",
        "how to install",
        "assembly",
        "assemble",
        "assembling",
        "setup",
        "install",
    ),
    "footrest_or_no_air": (
        "no air anywhere",
        "footrest air",
        "footrest",
        "foot roller",
        "no air",
    ),
    "yes_box_damage": ("box was crushed", "box damaged", "box was damaged"),
    "no_box_damage": (
        "box looked fine",
        "box was fine",
        "box was ok",
        "looked fine",
        "appeared fine",
        "it looked fine",
    ),
    "warranty": ("warranty", "claim"),
    "too_hot": ("too hot", "too warm", "uneven heat", "burns"),
    "not_heating": (
        "doesn't heat",
        "doesnt heat",
        "no warmth",
        "not heating",
        "won't heat",
        "wont heat",
        "doesn't warm",
    ),
    "voice_no_response": (
        "doesn't listen",
        "doesnt listen",
        "won't respond",
        "doesn't respond",
        "no response",
        "ignores me",
    ),
    "false_triggers": (
        "random commands",
        "false trigger",
        "picks up random",
        "picks up commands",
        "randomly turns",
        "false triggers",
    ),
    "heads_not_moving": (
        "heads don't move",
        "heads dont move",
        "heads not moving",
        "massage heads not moving",
    ),
    "legrest_not_lowering": (
        "won't go down",
        "wont go down",
        "not lowering",
        "won't lower",
        "won't retract",
        "won't raise",
        "wont raise",
        "not raising",
    ),
    "legrest_not_extend": (
        "won't extend",
        "wont extend",
        "won't come out",
        "won't go out",
        "does not extend",
        "not extending",
    ),
    "foot_rollers": ("foot roller", "foot rollers", "foot rolling"),
    "calf_roller": ("calf roller", "calf rollers"),
    "air_not_inflating": (
        "airbag",
        "air bag",
        "not inflating",
        "won't inflate",
        "wont inflate",
        "no air in the footrest",
    ),
    "zero_gravity": ("zero gravity", "zg", "zero-g", "zero g"),
    "fuse_broken": ("fuse blown", "fuse is blown", "blown fuse", "fuse is broken"),
    "fuse_blown": ("fuse blown", "fuse is blown", "blown fuse", "fuse is broken"),
    "stays_stuck": ("stuck", "stays stuck", "doesn't move back", "doesnt move back"),
    "moves_on_off": (
        "moves back",
        "returns to default",
        "goes back to default",
        "moves to default",
    ),
    "clicking_sound": ("clicking", "clicking sound", "i heard a click", "heard a click"),
    "blank_screen_commands_ok": (
        "blank screen",
        "screen is blank",
        "screen blank",
    ),
    "cable_damaged": (
        "cable damaged",
        "cable is cut",
        "cut cable",
        "damaged cable",
        "cable looks cut",
    ),
    "worked_before_stopped": (
        "used to work",
        "worked before",
        "stopped working",
        "worked then stopped",
    ),
    "pump_running": (
        "hissing",
        "hear the pump",
        "pump is running",
        "i hear the pump",
        "pump running",
    ),
    "no_movement": (
        "don't move at all",
        "dont move at all",
        "no movement",
        "heads do not move",
    ),
    "footrest_recline": ("footrest", "foot rest", "footrest recline", "legrest recline"),
    "intermittent": (
        "intermittent",
        "on and off",
        "sometimes works",
        "works sometimes",
        "works intermittently",
    ),
    "pops": ("pops", "popping", "clicks during", "clicking during massage"),
    # --- remaining flowchart keys (unique match only when on the current node) ---
    "sales": (
        "sales",
        "buy a chair",
        "looking to buy",
        "want to purchase",
        "pricing question",
        "which chair should i buy",
        "shopping for a chair",
    ),
    "other": (
        "other installation",
        "something else with setup",
        "different install issue",
        "backrest damage",
        "other cosmetic",
        "damage elsewhere",
    ),
    "remote_on": (
        "remote turns on",
        "remote powers on",
        "remote comes on",
        "controller turns on",
        "yes the remote turns on",
    ),
    "remote_off": (
        "remote does not turn on",
        "remote won't turn on",
        "remote wont turn on",
        "remote doesn't turn on",
        "controller won't turn on",
        "remote stays off",
    ),
    "has_power": (
        "remote has power",
        "screen shows something",
        "display is on",
        "screen lights up",
        "remote screen is on",
        "controller has power",
    ),
    "no_power": (
        "remote completely unresponsive",
        "remote is completely blank",
        "remote totally dead",
        "controller completely dead",
        "remote no power at all",
        "completely unresponsive remote",
    ),
    "quick_control_ok": (
        "quick control works",
        "side panel works",
        "quick panel works",
        "side controls work",
        "panel works but remote doesn't",
        "quick controls work but remote doesn't",
    ),
    "no_response": (
        "remote turns on but doesn't respond",
        "remote on but no commands",
        "won't accept any commands",
        "doesn't respond to any commands",
        "remote ignores all buttons",
    ),
    "recline_not_working": (
        "recline not working",
        "recline functions not working",
        "backrest and footrest won't recline",
        "zg not working either",
        "recline buttons don't work",
    ),
    "back_switch_sound": (
        "heard something from the chair",
        "heard a sound from the chair",
        "back switch made a sound",
        "heard noise when i flipped the switch",
        "switch on and heard something",
    ),
    "outlet_no_power": (
        "outlet has no power",
        "wall outlet dead",
        "outlet not working",
        "no power at the outlet",
        "outlet doesn't work",
    ),
    "powercord_issue": (
        "power cord",
        "powercord",
        "power cable loose",
        "cord connection",
        "plug connection issue",
        "cord looks loose",
    ),
    "no_clicking": (
        "no clicking",
        "no click",
        "didn't hear a click",
        "no clicking sound",
        "silent when switched on",
    ),
    "noise_up_down": (
        "loud noise when moving up",
        "loud noise going up or down",
        "noisy when it travels",
        "screeching up and down",
        "grinding when moving up",
    ),
    "noise_massaging": (
        "noise while massaging",
        "noisy during massage",
        "makes noise during the massage",
        "loud while massaging",
        "grinding during massage",
    ),
    "backrest": (
        "backrest",
        "back rest",
        "back recline",
        "seat back won't recline",
    ),
    "none_working": (
        "none of the recline",
        "no recline works",
        "all recline broken",
        "nothing reclines",
        "none of the recline functions work",
    ),
    "multiple_not_working": (
        "other recline also broken",
        "more than one recline",
        "several recline functions",
        "multiple recline issues",
        "other recline functions also",
    ),
    "panels_fixed": (
        "panels fixed",
        "now fixed",
        "reinstalled the panels",
        "panels weren't installed",
        "panels not installed properly but fixed",
        "side panels fixed now",
    ),
    "still_damaged": (
        "still damaged",
        "panels installed but damaged",
        "still looks damaged",
        "installed correctly but damaged",
        "damage remains",
    ),
    "voice_not_sure": (
        "not sure about voice",
        "unsure about the voice",
        "don't know which voice issue",
        "not sure which voice problem",
    ),
}


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _keyword_issue_type(text: str) -> Optional[str]:
    """Cheap keyword vote before calling the LLM."""
    norm = _normalize(text)
    if re.search(r"\bbox\b", norm) and re.search(
        r"\b(broken|crushed|damaged|dented)\b", norm
    ):
        return "delivery"
    scores = {key: 0 for key in _ISSUE_KEYWORDS}
    for issue, words in _ISSUE_KEYWORDS.items():
        for word in words:
            if _phrase_in_norm(word, norm):
                scores[issue] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] <= 0:
        return None
    tied = [k for k, v in scores.items() if v == scores[best]]
    if len(tied) > 1:
        return None
    return best


def _phrase_in_norm(phrase: str, norm: str) -> bool:
    """Match multi-word phrases as substrings; single tokens as whole words."""
    phrase = (phrase or "").strip()
    if not phrase or not norm:
        return False
    if " " in phrase or "'" in phrase or "-" in phrase:
        return phrase in norm
    return bool(re.search(rf"\b{re.escape(phrase)}\b", norm))


def _synonym_option_match(options: list[dict], norm: str) -> Optional[str]:
    """Unique synonym hit among the current node's answer_keys."""
    valid = {
        str(opt.get("answer_key") or "")
        for opt in options
        if str(opt.get("answer_key") or "")
    }
    hits: list[str] = []
    for key, phrases in _OPTION_SYNONYMS.items():
        if key not in valid:
            continue
        if any(_phrase_in_norm(phrase, norm) for phrase in phrases):
            hits.append(key)
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    # Foot/calf rollers are a footrest path, not the rolling mechanism.
    if set(uniq) == {"footrest", "rolling"} and re.search(
        r"\b(foot|calf|leg) rollers?\b", norm
    ):
        return "footrest"
    return None


def _is_yes_key(key: str) -> bool:
    return (
        key == "yes"
        or key.startswith("yes")
        or key in {
            "yes_worked",
            "air_blowing",
            "yes_hissing",
            "pump_running",
            "has_power",
            "has_tracking",
            "visible_at_unboxing",
        }
    )


def _is_no_key(key: str) -> bool:
    return (
        key == "no"
        or key.startswith("no")
        or key in {
            "never_worked",
            "noticed_later",
            "no_air",
            "no_hissing",
            "no_sound",
            "no_power",
            "no_tracking",
            "no_white_glove",
        }
    )


def _heuristic_option_match(options: list[dict], text: str) -> Optional[str]:
    """Match obvious yes/no or label fragments without an LLM call."""
    norm = _normalize(text)
    if not norm:
        return None

    for opt in options:
        label = _normalize(str(opt.get("label", "")))
        key = str(opt.get("answer_key", ""))
        key_norm = _normalize(key)
        if key_norm and norm == key_norm:
            return key
        if label and norm == label:
            return key
        # Require a substantial label phrase in the user's text — avoids
        # matching short fragments like "no" or "air" to the wrong option.
        if label and len(label) >= 12 and label in norm:
            return key

    synonym = _synonym_option_match(options, norm)
    if synonym:
        return synonym

    keys = [str(o.get("answer_key", "")) for o in options]
    keys_set = set(keys)
    if keys_set & _ISSUE_TYPES:
        if "delivery" in keys_set and re.search(r"\bbox\b", norm) and re.search(
            r"\b(broken|crushed|damaged|dented)\b", norm
        ):
            return "delivery"
        guessed = _keyword_issue_type(text)
        if guessed in keys_set:
            return guessed

    has_yes = any(_is_yes_key(k) for k in keys)
    has_no = any(_is_no_key(k) for k in keys)
    if has_yes and has_no:
        if _BARE_YES_RE.match(norm):
            for opt in options:
                key = str(opt.get("answer_key", ""))
                if _is_yes_key(key):
                    return key
        if _BARE_NO_RE.match(norm):
            for opt in options:
                key = str(opt.get("answer_key", ""))
                if _is_no_key(key):
                    return key

    if "tracking" in norm:
        for opt in options:
            key = str(opt.get("answer_key", ""))
            if "tracking" in key or "tracking" in _normalize(str(opt.get("label", ""))):
                if _NO_RE.search(norm) and key.startswith("no"):
                    return key
                if _YES_RE.search(norm) and key.startswith("has"):
                    return key

    return None


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


def _llm_json(prompt: str, *, task: str = "mapping") -> Optional[dict[str, Any]]:
    client = _openai_client()
    if client is None:
        return None

    from config import ROUTER_MODEL

    system = (
        "You are a strict classifier for a warranty workflow. "
        "You NEVER invent facts, outcomes, repair steps, or new options. "
        "You ONLY pick from the allowed values given in the user message. "
        "If the message is ambiguous, off-topic, or you are not confident, "
        'return null for the target field and confidence="low". '
        "Reply with JSON only."
    )
    if task == "issue_type":
        system += (
            " Map the message to installation, delivery, or defect ONLY when clearly indicated."
        )

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.warning("warranty_nlp LLM call failed: %s", exc)
        return None


def _accept_llm_choice(
    parsed: Optional[dict[str, Any]],
    field: str,
    valid_values: list[str],
) -> Optional[str]:
    """Accept an LLM choice only when confident and within the allowed set."""
    if not parsed:
        return None

    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence != "high":
        return None

    value = str(parsed.get(field, "")).strip()
    if value.lower() in ("null", "none", ""):
        return None
    if value in valid_values:
        return value
    return None


def _format_option_bullets(options: list[dict], *, limit: int = 8) -> str:
    lines: list[str] = []
    for opt in options[:limit]:
        label = str(opt.get("label") or opt.get("answer_key") or "").strip()
        if label:
            lines.append(f"• **{label}**")
    return "\n".join(lines)


def suggest_closest_option(options: list[dict], text: str) -> Optional[dict]:
    """Heuristic best-guess option for did-you-mean clarifying (does not auto-submit)."""
    norm = _normalize(text)
    if not norm or not options:
        return None

    best: Optional[dict] = None
    best_score = 0
    for opt in options:
        label = _normalize(str(opt.get("label") or ""))
        key = _normalize(str(opt.get("answer_key") or ""))
        score = 0
        key_words = key.replace("_", " ")
        if key_words and len(key_words) >= 3 and _phrase_in_norm(key_words, norm):
            score += 4
        if label and len(label) >= 8 and label in norm:
            score += 5
        for word in norm.split():
            if len(word) >= 4 and word not in _WEAK_WORDS and _phrase_in_norm(word, label):
                score += 2
        if score > best_score:
            best_score = score
            best = opt

    if best_score >= 2:
        return best
    return None


def option_label_for(node: dict, answer_key: str) -> str:
    key = str(answer_key or "").strip()
    for opt in node.get("options") or []:
        if str(opt.get("answer_key") or "") == key:
            return str(opt.get("label") or key).strip() or key
    return key


def public_option(opt: dict) -> dict[str, str]:
    key = str(opt.get("answer_key") or "").strip()
    label = str(opt.get("label") or key).strip() or key
    return {"answer_key": key, "label": label}


def node_has_yes_no_options(node: dict) -> bool:
    keys = [str(o.get("answer_key") or "") for o in (node.get("options") or [])]
    has_yes = any(k == "yes" or k.startswith("yes") for k in keys)
    has_no = any(k == "no" or k.startswith("no") for k in keys)
    return has_yes and has_no


def is_suggestion_confirmation(text: str) -> bool:
    return bool(_SUGGEST_CONFIRM_RE.match((text or "").strip()))


def is_suggestion_rejection(text: str) -> bool:
    return bool(_SUGGEST_REJECT_RE.match((text or "").strip()))


def build_mapped_ack(label: str) -> str:
    pretty = str(label or "").strip()
    if not pretty:
        return "Got it."
    return f"Got it — **{pretty}**."


def build_clarifying_workflow_message(
    node: dict,
    user_text: str,
    *,
    closest: Optional[dict] = None,
) -> str:
    """Customer-facing re-prompt when free text did not map to a menu option."""
    prompt = str(node.get("prompt") or "").strip()
    options = list(node.get("options") or [])
    trimmed = (user_text or "").strip()
    suggested = closest if closest is not None else suggest_closest_option(options, trimmed)

    if trimmed:
        lead = (
            f'I wasn\'t fully sure how **"{trimmed[:120]}"** maps to the choices below.'
        )
    else:
        lead = "I want to make sure I pick the right next step for you."

    parts = [lead]
    if suggested:
        label = str(suggested.get("label") or suggested.get("answer_key") or "").strip()
        if label:
            parts.append(
                f"Did you mean **{label}**? Tap **Yes — {label}** below, "
                "or type **yes** to confirm."
            )
            parts.append("If that’s not it, pick a different option or rephrase.")
            return "\n\n".join(parts)

    if options:
        parts.append("Please tap one of the options below, or rephrase to match one of them.")
    elif prompt:
        parts.append("Please answer the question below.")
    if prompt:
        parts.append(prompt)
    return "\n\n".join(p for p in parts if p)


def build_intent_confirmation_message(
    node: dict,
    mapped_key: str,
    user_text: str,
) -> str:
    """Ask the customer to tap the matched option — never auto-advance on guess."""
    options = list(node.get("options") or [])
    label = str(mapped_key or "").strip()
    for opt in options:
        if str(opt.get("answer_key") or "") == mapped_key:
            label = str(opt.get("label") or mapped_key).strip()
            break

    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'Just to confirm — for **"{trimmed[:120]}"**, did you mean **{label}**?'
        )
    else:
        parts.append(f"Just to confirm — did you mean **{label}**?")
    parts.append(
        "Please tap that option below to continue. I won't choose and move forward for you."
    )
    bullets = _format_option_bullets(options)
    if bullets:
        parts.append(bullets)
    return "\n\n".join(parts)


_ISSUE_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("installation", "Setup & installation"),
    ("delivery", "Delivery & tracking"),
    ("defect", "Warranty / defect"),
)


def build_clarifying_issue_type_message(
    user_text: str,
    *,
    model_name: str = "",
) -> str:
    """Re-prompt when issue type could not be inferred from free text."""
    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'I couldn\'t tell whether **"{trimmed[:120]}"** is installation, '
            "delivery, or a product defect."
        )
    if model_name:
        parts.append(f"For your **{model_name}**, what type of issue can we help with?")
    else:
        parts.append("What type of issue can we help with?")
    parts.append("Choose one of these, or describe your issue a bit more specifically:")
    for _key, label in _ISSUE_TYPE_LABELS:
        parts.append(f"• **{label}**")
    return "\n\n".join(parts)


def build_suggested_issue_type_message(issue_type: str, user_text: str = "") -> str:
    """Ask the customer to confirm an inferred issue type by tapping a button."""
    label = issue_type
    for key, pretty in _ISSUE_TYPE_LABELS:
        if key == issue_type:
            label = pretty
            break
    trimmed = (user_text or "").strip()
    parts: list[str] = []
    if trimmed:
        parts.append(
            f'Based on **"{trimmed[:120]}"**, this sounds like **{label}**.'
        )
    else:
        parts.append(f"This sounds like **{label}**.")
    parts.append(
        f"Please tap **{label}** below to confirm. I won't start that path until you choose."
    )
    parts.append("Or pick a different option if I misunderstood:")
    for _key, pretty in _ISSUE_TYPE_LABELS:
        parts.append(f"• **{pretty}**")
    return "\n\n".join(parts)


def interpret_issue_type(user_text: str) -> Optional[str]:
    """
    Map natural language to installation | delivery | defect.
    Returns None when the intent is unclear.
    """
    text = user_text.strip()
    if not text:
        return None

    keyword = _keyword_issue_type(text)
    if keyword:
        return keyword

    prompt = (
        "Classify this customer warranty message into exactly one issue_type.\n"
        f'Message: "{text}"\n\n'
        "Valid issue_type values:\n"
        '- "installation" — setup, assembly, how to install\n'
        '- "delivery" — shipping, tracking, box damage on arrival\n'
        '- "defect" — product malfunction, broken parts, not working\n\n'
        "Rules:\n"
        "- Pick only when the message clearly fits ONE category.\n"
        "- If unclear or mixed, return issue_type=null and confidence=low.\n"
        '- Return JSON: {"issue_type":"installation|delivery|defect|null","confidence":"high|low"}'
    )
    parsed = _llm_json(prompt, task="issue_type")
    if not parsed:
        return None

    issue = _accept_llm_choice(parsed, "issue_type", list(_ISSUE_TYPES))
    return issue


def append_unmapped_phrase(
    existing: Any,
    *,
    node_id: str,
    text: str,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Keep the last few unmatched typed answers for admin review."""
    rows: list[dict[str, str]] = []
    if isinstance(existing, list):
        rows = [row for row in existing if isinstance(row, dict)]
    elif isinstance(existing, str) and existing.strip():
        try:
            parsed = json.loads(existing)
        except (TypeError, json.JSONDecodeError):
            parsed = []
        if isinstance(parsed, list):
            rows = [row for row in parsed if isinstance(row, dict)]

    trimmed = " ".join((text or "").strip().split())[:160]
    node = str(node_id or "").strip()
    if not trimmed or not node:
        return rows[-limit:]
    if rows and rows[-1].get("node_id") == node and rows[-1].get("text") == trimmed:
        return rows[-limit:]
    rows.append({"node_id": node, "text": trimmed})
    return rows[-limit:]


def interpret_warranty_answer(node: dict, user_text: str) -> Optional[str]:
    """
    Map natural language to an answer_key for the current workflow node.

    For question_text nodes, returns the trimmed user text unchanged.
    For option nodes, returns a valid answer_key or None.
    """
    text = user_text.strip()
    if not text:
        return None

    node_type = node.get("type")
    if node_type == "question_text":
        return text

    options = node.get("options") or []
    if not options:
        return None

    heuristic = _heuristic_option_match(options, text)
    if heuristic:
        return heuristic

    valid_keys = [str(o.get("answer_key", "")) for o in options]
    option_lines = [
        f'- answer_key="{o.get("answer_key")}" label="{o.get("label", "")}"'
        for o in options
    ]
    prompt = (
        "Pick the single best matching answer_key for the customer's message.\n\n"
        f'Question: "{node.get("prompt", "")}"\n'
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\n"
        f'Customer message: "{text}"\n\n'
        f"Valid answer_keys ONLY: {valid_keys}\n"
        "Rules:\n"
        "- Choose exactly one answer_key from Valid answer_keys when clearly matched.\n"
        "- Do NOT invent keys, do NOT guess between close options.\n"
        "- If ambiguous or unrelated, return answer_key=null and confidence=low.\n"
        '- Return JSON: {"answer_key":"<one valid key or null>","confidence":"high|low"}'
    )
    parsed = _llm_json(prompt)
    if not parsed:
        return None

    return _accept_llm_choice(parsed, "answer_key", valid_keys)


def interpret_troubleshooting_outcome(
    user_text: str,
    *,
    issue_type: str = "",
    previous_outcome: str = "",
) -> Optional[str]:
    """
    Map clear free text to a terminal troubleshooting outcome.

    Bare yes/no is never mapped — those conflict across install/delivery/defect.
    """
    text = _normalize(user_text)
    if not text:
        return None
    if text in {"yes", "yeah", "yep", "yup", "no", "nope", "nah"}:
        return None

    at_outcome = (previous_outcome or "").strip().lower() == "steps_completed"
    issue = (issue_type or "").strip().lower()

    if not at_outcome:
        if _UNABLE_RE.search(text) or _NEED_TEAM_RE.search(text):
            return "unable_to_attempt"
        if _STEPS_DONE_RE.search(text):
            return "steps_completed"
        return None

    if issue == "delivery":
        if _DELIVERY_SUBMIT_RE.search(text) or _NEED_TEAM_RE.search(text):
            return "unresolved"
        if _COME_BACK_RE.search(text) or _ALL_SET_RE.search(text):
            return "resolved"
        return None

    if issue == "installation":
        if _NEED_TEAM_RE.search(text) or _STILL_BROKEN_RE.search(text):
            return "unresolved"
        if _SETUP_DONE_RE.search(text) or _ALL_SET_RE.search(text):
            return "resolved"
        return None

    if _NEED_TEAM_RE.search(text) or _STILL_BROKEN_RE.search(text):
        return "unresolved"
    if _WORKING_NOW_RE.search(text) or _ALL_SET_RE.search(text):
        return "resolved"
    return None
