"""
Lightweight ops alerts for cron / monitoring scripts (Shopify sync, RC health).

Uses the same SMTP settings as warranty email. Recipients:
  OPS_ALERT_EMAIL (comma-separated) or WARRANTY_TEAM_EMAIL.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def ops_alert_recipients() -> list[str]:
    from config import WARRANTY_TEAM_EMAIL  # noqa: WPS433

    raw = os.environ.get("OPS_ALERT_EMAIL", "").strip()
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    env_fallback = os.environ.get("WARRANTY_TEAM_EMAIL", "").strip()
    fallback = env_fallback or (WARRANTY_TEAM_EMAIL or "").strip()
    return [fallback] if fallback else []


def send_ops_alert(subject: str, body: str, *, recipients: list[str] | None = None) -> bool:
    """Send a plain-text ops email. Returns False when SMTP is not configured."""
    from config import EMAIL_PASSWORD, EMAIL_SENDER, SMTP_PORT, SMTP_SERVER  # noqa: WPS433

    to_addrs = recipients if recipients is not None else ops_alert_recipients()
    if not to_addrs:
        logger.warning("ops_notify: no recipients configured")
        return False
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        logger.warning("ops_notify: SMTP not configured")
        return False

    message = MIMEText(body.strip() + "\n", "plain", "utf-8")
    message["Subject"] = subject.strip()
    message["From"] = EMAIL_SENDER
    message["To"] = ", ".join(to_addrs)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_addrs, message.as_string())
        return True
    except smtplib.SMTPException as exc:
        logger.error("ops_notify: SMTP failed: %s", exc)
        return False
