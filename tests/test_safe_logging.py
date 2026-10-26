"""Application and Uvicorn handlers must not receive customer capabilities."""

import io
import logging
import sys
from pathlib import Path

import pytest
from uvicorn.logging import AccessFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from safe_logging import install_log_redaction, safe_log_path, safe_log_text


@pytest.fixture
def safe_factory():
    previous = logging.getLogRecordFactory()
    install_log_redaction()
    yield
    logging.setLogRecordFactory(previous)


@pytest.mark.parametrize("path,expected", [
    ("/api/v1/warranty/resume/private-token?email=person@example.test", "/api/v1/warranty/resume/{token}"),
    ("/api/v1/warranty/session/shared-session/resume-link", "/api/v1/warranty/session/{session_id}/resume-link"),
    ("/api/v1/warranty/arbitrary-ticket/answer", "/api/v1/warranty/{ticket_id}/answer"),
    ("/admin/warranty/private-ticket/evidence/12/download", "/admin/warranty/{ticket_id}/evidence/12/download"),
    ("/api/v1/warranty/ocr/serial", "/api/v1/warranty/ocr/serial"),
    ("//[", "[invalid-path]"),
    ("http://[", "[invalid-path]"),
    ("/unknown/private@example.test", "/unknown/[email]"),
    ("/unknown/+15551234567", "/unknown/[phone]"),
])
def test_paths_keep_routes_without_customer_values(path, expected):
    assert safe_log_path(path) == expected


def test_uvicorn_access_formatter_retains_its_tuple_and_redacts_query_and_token(safe_factory):
    logger = logging.getLogger("uvicorn.access")
    record = logger.makeRecord("uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("192.0.2.20:123", "GET", "/api/v1/warranty/resume/secret-capability?email=private@example.test", "1.1", 200), None)
    output = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(record)
    assert "200 OK" in output
    assert "/api/v1/warranty/resume/{token}" in output
    assert all(secret not in output for secret in ("192.0.2.20", "secret-capability", "private@example.test"))


def test_generic_logs_strip_identifiers_and_contact_information(safe_factory):
    record = logging.getLogger("customer").makeRecord("customer", logging.INFO, __file__, 1,
        "session=%s ticket=%s contact=%s email=%s phone=%s", ("browser-capability", "case-capability", "123456", "person@example.test", "+1 (555) 123-4567"), None)
    output = logging.Formatter("%(message)s").format(record)
    assert all(value not in output for value in ("browser-capability", "case-capability", "123456", "person@example.test", "555"))


def test_error_tracebacks_keep_stack_and_type_without_customer_exception_values(safe_factory):
    secret = "Private Customer Name at private@example.test"
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        record = logging.getLogger("uvicorn.error").makeRecord("uvicorn.error", logging.ERROR,
            __file__, 1, "Request failed at %s", ("/api/v1/warranty/session/private-session/answer",), sys.exc_info())
    output = logging.Formatter("%(message)s").format(record)
    assert "RuntimeError" in output
    assert "test_error_tracebacks_keep_stack" in output
    assert secret not in output
    assert "private-session" not in output


def test_signed_values_and_url_query_are_redacted_without_losing_host():
    token = "a" * 50 + "." + "b" * 43
    output = safe_log_text(f"URL=https://support.example.test/warranty?resume={token} signed={token}")
    assert "support.example.test/warranty" in output
    assert token not in output
    assert "resume=" not in output


def test_structured_log_values_redact_private_fields_without_losing_counts(safe_factory):
    record = logging.getLogger("provider").makeRecord("provider", logging.ERROR, __file__, 1,
        "Provider result: %s", ({"count": 2, "name": "Private Customer", "nested": {"session_id": "bearer-capability"}},), None)
    output = logging.Formatter("%(message)s").format(record)
    assert "Private Customer" not in output
    assert "bearer-capability" not in output
    assert "'count': 2" in output
