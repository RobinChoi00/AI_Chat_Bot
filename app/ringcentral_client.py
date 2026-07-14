"""
ringcentral_client.py
=====================
RingCentral REST API client for Automated Voice Apps (IVR).

Uses JWT auth flow with in-memory access-token cache. Call Control helpers:
  play_prompt, collect_digits, forward_call, hangup.

Environment
-----------
RC_SERVER                        https://platform.ringcentral.com
RC_CLIENT_ID
RC_CLIENT_SECRET
RC_USER_JWT                        JWT token from Developer Console → Credentials (recommended)
RC_USER_JWT_FILE                   Optional file path (avoids .env paste corruption)
RC_JWT_PRIVATE_KEY                 Optional PEM path — only if using self-signed JWT
RC_JWT_CLAIM_SUB                   Extension ID — only with RC_JWT_PRIVATE_KEY
RC_WARRANTY_TRANSFER_TO          E.164 for ext.3 queue (PSTN/direct number)
RC_WARRANTY_TRANSFER_EXTENSION     Optional internal extension, e.g. 103
RC_SMS_FROM_NUMBER               E.164 outbound SMS sender (Warranty line)
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

RC_SERVER = os.getenv("RC_SERVER", "https://platform.ringcentral.com").rstrip("/")
RC_CLIENT_ID = os.getenv("RC_CLIENT_ID", "")
RC_CLIENT_SECRET = os.getenv("RC_CLIENT_SECRET", "")
RC_JWT_CLAIM_SUB = os.getenv("RC_JWT_CLAIM_SUB", "")
RC_WARRANTY_TRANSFER_TO = os.getenv("RC_WARRANTY_TRANSFER_TO", "")
RC_WARRANTY_TRANSFER_EXTENSION = os.getenv("RC_WARRANTY_TRANSFER_EXTENSION", "")
RC_SMS_FROM_NUMBER = os.getenv("RC_SMS_FROM_NUMBER", "")

_token_lock = threading.Lock()
_cached_token: Optional[str] = None
_token_expires_at: float = 0.0


def _request_timeout() -> tuple[float, float]:
    try:
        connect = float(os.getenv("RC_CONNECT_TIMEOUT_SECONDS", "5"))
        read = float(os.getenv("RC_READ_TIMEOUT_SECONDS", "25"))
    except ValueError:
        connect, read = 5.0, 25.0
    return max(1.0, min(connect, 30.0)), max(1.0, min(read, 60.0))


def _max_retries() -> int:
    try:
        return max(0, min(int(os.getenv("RC_API_MAX_RETRIES", "3")), 5))
    except ValueError:
        return 3


def _retry_delay(resp: Optional[requests.Response], attempt: int) -> float:
    if resp is not None:
        raw = (resp.headers.get("Retry-After") or "").strip()
        try:
            if raw:
                return max(0.0, min(float(raw), 30.0))
        except ValueError:
            pass
    base = min(0.5 * (2 ** attempt), 8.0)
    return base + random.uniform(0.0, min(base * 0.25, 1.0))


def _load_private_key() -> str:
    raw = os.getenv("RC_JWT_PRIVATE_KEY", "").strip()
    if not raw:
        raise RuntimeError("RC_JWT_PRIVATE_KEY is not configured.")
    path = Path(raw)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return raw.replace("\\n", "\n")


def _read_env(name: str) -> str:
    """Read env var and strip whitespace / optional surrounding quotes."""
    return os.getenv(name, "").strip().strip('"').strip("'")


def _load_user_jwt() -> str:
    """Load RC user JWT from file (preferred) or env var."""
    file_path = _read_env("RC_USER_JWT_FILE")
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            raise RuntimeError(f"RC_USER_JWT_FILE not found: {file_path}")
        token = path.read_text(encoding="utf-8").strip().strip('"').strip("'")
        if token:
            return token

    return _read_env("RC_USER_JWT") or _read_env("RC_JWT")


def _build_jwt_assertion() -> str:
    """Return the JWT assertion for the OAuth token request."""
    user_jwt = _load_user_jwt()
    if user_jwt:
        if not user_jwt.startswith("eyJ"):
            raise RuntimeError(
                "RC_USER_JWT must be a JWT string starting with 'eyJ' — "
                "check for truncation or extra characters in .env."
            )
        return user_jwt

    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError(
            "Set RC_USER_JWT (Developer Console → Credentials → Create JWT), "
            "or install PyJWT for RC_JWT_PRIVATE_KEY auth."
        ) from exc

    if not RC_CLIENT_ID or not RC_JWT_CLAIM_SUB:
        raise RuntimeError(
            "Set RC_USER_JWT, or RC_CLIENT_ID + RC_JWT_CLAIM_SUB + RC_JWT_PRIVATE_KEY."
        )
    now = int(time.time())
    payload = {
        "iss": RC_CLIENT_ID,
        "sub": RC_JWT_CLAIM_SUB,
        "aud": f"{RC_SERVER}/restapi/oauth/token",
        "exp": now + 3600,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(payload, _load_private_key(), algorithm="RS256")
    return token if isinstance(token, str) else token.decode("utf-8")


def get_access_token(*, force_refresh: bool = False) -> str:
    """Return a cached RingCentral access token, refreshing ~1 min before expiry."""
    global _cached_token, _token_expires_at
    with _token_lock:
        if (
            not force_refresh
            and _cached_token
            and time.time() < _token_expires_at - 60
        ):
            return _cached_token

        assertion = _build_jwt_assertion()
        client_id = _read_env("RC_CLIENT_ID")
        client_secret = _read_env("RC_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("RC_CLIENT_ID and RC_CLIENT_SECRET must be set.")
        resp = requests.post(
            f"{RC_SERVER}/restapi/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
            auth=(client_id, client_secret),
            timeout=_request_timeout(),
        )
        if resp.status_code >= 400:
            logger.error(
                "RingCentral token error %s: %s",
                resp.status_code,
                resp.text[:300],
            )
            resp.raise_for_status()

        body = resp.json()
        _cached_token = str(body["access_token"])
        _token_expires_at = time.time() + int(body.get("expires_in", 3600))
        return _cached_token


def _auth_headers(*, force_refresh: bool = False) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_access_token(force_refresh=force_refresh)}",
        "Content-Type": "application/json",
    }


def _party_url(session_id: str, party_id: str, action: str = "") -> str:
    base = (
        f"{RC_SERVER}/restapi/v1.0/account/~/telephony/sessions/"
        f"{session_id}/parties/{party_id}"
    )
    return f"{base}/{action}" if action else base


def _request(
    method: str,
    url: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
) -> requests.Response:
    """Issue an authenticated RC request with bounded transient retries."""
    transient_statuses = {429, 500, 502, 503, 504}
    max_retries = _max_retries()
    refreshed = False
    force_refresh_next = False
    last_exc: Optional[requests.RequestException] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=_auth_headers(force_refresh=force_refresh_next),
                json=json_body,
                timeout=_request_timeout(),
            )
            force_refresh_next = False
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            delay = _retry_delay(None, attempt)
            logger.warning(
                "RingCentral network error; retrying attempt=%s delay=%.2fs error=%s",
                attempt + 1,
                delay,
                type(exc).__name__,
            )
            time.sleep(delay)
            continue

        if resp.status_code == 401 and not refreshed:
            logger.info("RingCentral 401 — refreshing access token")
            refreshed = True
            force_refresh_next = True
            if attempt >= max_retries:
                # Always allow the single credential refresh, even when
                # transient retries are configured to zero.
                return requests.request(
                    method,
                    url,
                    headers=_auth_headers(force_refresh=True),
                    json=json_body,
                    timeout=_request_timeout(),
                )
            continue

        if resp.status_code not in transient_statuses or attempt >= max_retries:
            return resp

        delay = _retry_delay(resp, attempt)
        logger.warning(
            "RingCentral transient response; retrying status=%s attempt=%s delay=%.2fs",
            resp.status_code,
            attempt + 1,
            delay,
        )
        time.sleep(delay)

    if last_exc is not None:  # defensive; loop normally returns or raises
        raise last_exc
    raise RuntimeError("RingCentral request retry loop ended unexpectedly.")


def play_prompt(
    *,
    session_id: str,
    party_id: str,
    audio_uri: str,
    interrupt_by_dtmf: bool = False,
) -> dict[str, Any]:
    """Play an audio file from a public HTTPS URL to the caller."""
    resp = _request(
        "POST",
        _party_url(session_id, party_id, "play"),
        json_body={
            "resources": [{"uri": audio_uri}],
            "interruptByDtmf": interrupt_by_dtmf,
            "repeatCount": 1,
        },
    )
    if resp.status_code >= 400:
        logger.error("RingCentral play error %s: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
    return resp.json() if resp.text else {}


def collect_digits(
    *,
    session_id: str,
    party_id: str,
    patterns: list[str],
    timeout_ms: int = 120_000,
    inter_digit_timeout_ms: int = 5_000,
) -> dict[str, Any]:
    """Wait for the caller to press one of the allowed DTMF patterns."""
    resp = _request(
        "POST",
        _party_url(session_id, party_id, "collect"),
        json_body={
            "patterns": patterns,
            "timeout": timeout_ms,
            "interDigitTimeout": inter_digit_timeout_ms,
        },
    )
    if resp.status_code >= 400:
        logger.error(
            "RingCentral collect error %s: %s",
            resp.status_code,
            resp.text[:300],
        )
        resp.raise_for_status()
    return resp.json() if resp.text else {}


def _forward_payload(phone_number: str) -> dict[str, Any]:
    """
    Build a forward body for RC IVR apps.

    Prefer extensionNumber for internal Call Queue (ext.3); fall back to phoneNumber.
    """
    ext = RC_WARRANTY_TRANSFER_EXTENSION.strip()
    target = (phone_number or RC_WARRANTY_TRANSFER_TO).strip()
    if ext:
        return {"extensionNumber": ext}
    if target:
        return {"phoneNumber": target}
    raise RuntimeError(
        "Set RC_WARRANTY_TRANSFER_EXTENSION or RC_WARRANTY_TRANSFER_TO."
    )


def forward_call(
    *,
    session_id: str,
    party_id: str,
    phone_number: str = "",
) -> dict[str, Any]:
    """Transfer the caller to ext.3 queue or another target."""
    resp = _request(
        "POST",
        _party_url(session_id, party_id, "forward"),
        json_body=_forward_payload(phone_number),
    )
    if resp.status_code >= 400:
        logger.error(
            "RingCentral forward error %s: %s",
            resp.status_code,
            resp.text[:300],
        )
        resp.raise_for_status()
    return resp.json() if resp.text else {}


def hangup(*, session_id: str, party_id: str) -> None:
    """Drop the caller party (Call Control DELETE party)."""
    resp = _request("DELETE", _party_url(session_id, party_id))
    if resp.status_code >= 400:
        logger.error(
            "RingCentral hangup error %s: %s",
            resp.status_code,
            resp.text[:300],
        )
        resp.raise_for_status()


def send_sms(
    *,
    to: str,
    text: str,
    from_number: str = "",
) -> dict[str, Any]:
    """Send an SMS from the configured Warranty extension."""
    to_num = (to or "").strip()
    from_num = (from_number or RC_SMS_FROM_NUMBER).strip()
    body = (text or "").strip()
    if not to_num:
        raise RuntimeError("send_sms requires a recipient phone number.")
    if not from_num:
        raise RuntimeError("Set RC_SMS_FROM_NUMBER or pass from_number.")
    if not body:
        raise RuntimeError("send_sms requires non-empty text.")

    resp = _request(
        "POST",
        f"{RC_SERVER}/restapi/v1.0/account/~/extension/~/sms",
        json_body={
            "from": {"phoneNumber": from_num},
            "to": [{"phoneNumber": to_num}],
            "text": body,
        },
    )
    if resp.status_code >= 400:
        logger.error(
            "RingCentral SMS error %s: %s",
            resp.status_code,
            resp.text[:300],
        )
        resp.raise_for_status()
    return resp.json() if resp.text else {}


def reset_token_cache() -> None:
    """Test helper — clear cached access token."""
    global _cached_token, _token_expires_at
    with _token_lock:
        _cached_token = None
        _token_expires_at = 0.0
