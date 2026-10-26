"""Redact customer identifiers and bearer capabilities before any log handler runs.

Uvicorn uses its own access/error handlers, so a filter on the application logger
alone cannot protect those records. The record factory keeps its access tuple
shape intact while removing capabilities, query strings, and exception values.
"""

from __future__ import annotations

import logging
import re
import traceback
from urllib.parse import unquote, urlsplit, urlunsplit

_EMAIL = re.compile(r"[A-Z0-9._%+*-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_SIGNED = re.compile(r"\b[A-Za-z0-9_-]{32,}\.[A-Za-z0-9_-]{40,}\b")
_PHONE = re.compile(r"(?<![\w])(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)")
_FIELD = re.compile(
    r"\b(session(?:_id)?|ticket(?:_id)?|contact(?:_id)?|case_reference|"
    r"order(?:_id|_number)?|tracking(?:_number)?|resume_url|followup_token|token|"
    r"customer_email|email|phone|recipient)\s*([=:])\s*"
    r"(?:'[^']*'|\"[^\"]*\"|[^\s,;]+)", re.I,
)
_URL = re.compile(r"https?://[^\s'\"<>]+", re.I)
_PATH = re.compile(r"/(?:api/|admin/)[^\s'\"<>]+")
_PRIVATE_KEYS = {"session", "session_id", "ticket", "ticket_id", "contact", "contact_id", "email",
                 "customer_email", "phone", "recipient", "message", "messages", "content", "query",
                 "user_query", "answer", "token", "resume_url", "followup_token", "name", "address",
                 "case_reference", "order_number", "order_id", "tracking_number", "first_name", "last_name",
                 "full_name", "customer_name", "address1", "address2", "billing_address", "shipping_address"}
_WARRANTY_STATIC = {"session", "resume", "status", "ocr", "quality", "tickets", "knowledge-reviews",
                    "metrics", "freshdesk-status", "freshdesk-field-catalog", "freshdesk-solutions",
                    "sync-freshdesk", "rebuild-faiss", "faiss"}


def safe_log_path(value: str) -> str:
    """Keep endpoint names, never user-selected path values or query parameters."""
    try:
        path = urlsplit(value).path
    except (ValueError, TypeError):
        return "[invalid-path]"
    path = re.sub(r"[\x00-\x1f\x7f]", "?", unquote(path).split("?", 1)[0])
    parts = path.split("/")
    for index in range(1, len(parts)):
        previous = parts[index - 1]
        if previous in {"session", "resume"}:
            parts[index] = "{session_id}" if previous == "session" else "{token}"
        elif previous == "warranty" and parts[index] not in _WARRANTY_STATIC:
            parts[index] = "{ticket_id}"
    path = _UUID.sub("[id]", _SIGNED.sub("[token]", "/".join(parts)))
    return _EMAIL.sub("[email]", _PHONE.sub("[phone]", path))


def safe_log_text(value: object) -> str:
    if isinstance(value, BaseException):
        return type(value).__name__
    text = str(value)

    def safe_url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group())
            return urlunsplit((parsed.scheme, parsed.hostname or "[host]", safe_log_path(parsed.path), "", ""))
        except ValueError:
            return "[url]"

    text = _URL.sub(safe_url, text)
    text = _PATH.sub(lambda match: safe_log_path(match.group()), text)
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    text = _SIGNED.sub("[token]", text)
    text = _UUID.sub("[id]", text)
    return _FIELD.sub(lambda match: f"{match[1]}{match[2]}[redacted]", text)


def install_log_redaction() -> None:
    """Install once, before application imports can log customer data."""
    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, "_customer_redaction", False):
        return

    def safe_argument(value):
        if isinstance(value, (str, BaseException)):
            return safe_log_text(value)
        if isinstance(value, dict):
            return {key: "[redacted]" if str(key).lower() in _PRIVATE_KEYS else safe_argument(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(safe_argument(item) for item in value)
        return value

    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            _address, method, path, version, status = record.args
            record.args = ("[client]", method, safe_log_path(str(path)), version, status)
        else:
            if isinstance(record.msg, (BaseException, dict, list, tuple)):
                record.msg = safe_argument(record.msg)
            record.args = safe_argument(record.args)
            try:
                record.msg = safe_log_text(record.getMessage())
            except (TypeError, ValueError):
                record.msg = "Log message formatting failed"
            record.args = ()
        if record.exc_info:
            error_type, _value, error_traceback = record.exc_info
            kind = error_type.__name__ if error_type is not None else "Exception"
            record.exc_text = safe_log_text("".join(traceback.format_tb(error_traceback))) + f"{kind}: [exception details redacted]"
        if record.stack_info:
            record.stack_info = safe_log_text(record.stack_info)
        return record

    factory._customer_redaction = True
    logging.setLogRecordFactory(factory)
