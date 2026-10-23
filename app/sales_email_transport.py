"""Bounded Sales email transport with non-sensitive delivery failure codes.

SMTP acceptance establishes submission to the mail server, not inbox delivery
or a staff response. No retry is performed here.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


class SalesEmailError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _settings() -> tuple[str, str, str, int]:
    # Read environment values at delivery time, rather than importing main
    # (which initializes the application and captures an older config value).
    sender = os.getenv('EMAIL_SENDER', '').strip()
    password = os.getenv('EMAIL_PASSWORD', '')
    host = os.getenv('SMTP_SERVER', 'smtp.gmail.com').strip()
    try:
        port = int(os.getenv('SMTP_PORT', '587'))
    except ValueError:
        raise SalesEmailError('smtp_invalid_configuration') from None
    if not sender or not password or not host:
        raise SalesEmailError('smtp_not_configured')
    if not 1 <= port <= 65535:
        raise SalesEmailError('smtp_invalid_configuration')
    return sender, password, host, port


def transport_health() -> dict:
    try:
        _sender, _password, _host, port = _settings()
    except SalesEmailError as exc:
        return {'configured': False, 'failure_code': exc.code, 'acceptance_is_staff_confirmation': False}
    return {'configured': True, 'failure_code': None,
            'tls_mode': 'implicit_tls' if port == 465 else 'starttls',
            'timeout_seconds': 8, 'acceptance_is_staff_confirmation': False}


def _send(to_address: str, subject: str, text: str) -> bool:
    sender, password, host, port = _settings()
    if not to_address or '\r' in to_address or '\n' in to_address or '@' not in to_address:
        raise SalesEmailError('smtp_invalid_recipient')
    message = EmailMessage()
    message['From'] = sender
    message['To'] = to_address
    message['Subject'] = subject
    message.set_content(text)
    try:
        context = ssl.create_default_context()
        if port == 465:
            connection = smtplib.SMTP_SSL(host, port, timeout=8, context=context)
        else:
            connection = smtplib.SMTP(host, port, timeout=8)
        with connection as server:
            server.ehlo()
            if port != 465:
                server.starttls(context=context)
                server.ehlo()
            server.login(sender, password)
            refused = server.send_message(message)
            if refused:
                raise SalesEmailError('smtp_recipient_rejected')
    except smtplib.SMTPAuthenticationError:
        raise SalesEmailError('smtp_authentication_failed') from None
    except smtplib.SMTPRecipientsRefused:
        raise SalesEmailError('smtp_recipient_rejected') from None
    except smtplib.SMTPNotSupportedError:
        raise SalesEmailError('smtp_tls_unavailable') from None
    except (TimeoutError, ConnectionError, OSError):
        raise SalesEmailError('smtp_connection_failed') from None
    except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError):
        raise SalesEmailError('smtp_connection_failed') from None
    except smtplib.SMTPException:
        raise SalesEmailError('smtp_submission_failed') from None
    return True


def send_sales_notification(contact: str, details: str, product_info: str, domain: str) -> bool:
    from config import SALES_EMAIL_BY_DOMAIN
    from store_config import get_store_key_prefix, is_supported_storefront_domain, normalize_storefront_domain

    if not is_supported_storefront_domain(domain):
        raise SalesEmailError('sales_inbox_unconfigured')
    store = normalize_storefront_domain(domain)
    inbox_key = {'TITAN': 'titanchair', 'OSAKI': 'osakiusa',
                 'OSAKIMASSAGE': 'osakimassagechair', 'OSAKITITAN': 'titanchair'}[get_store_key_prefix(store)]
    recipient = SALES_EMAIL_BY_DOMAIN.get(inbox_key)
    if not recipient:
        raise SalesEmailError('sales_inbox_unconfigured')
    # Contact details belong in the authorized follow-up body, not the subject
    # or error logs, which often travel through monitoring systems.
    return _send(recipient, '[AI Lead] Customer requested Sales follow-up',
                 f'Store: {store}\n\n{details}\n\n{product_info}'.strip())


def send_shopper_receipt(email: str, summary: str, domain: str) -> bool:
    return _send(email, 'We saved your Osaki request',
                 'We saved the follow-up request you made in the Osaki chat.\n\n'
                 f'{summary}\n\n'
                 'This confirms your request was submitted; it does not confirm an appointment or that a specialist has replied.\n'
                 'If you did not request this message, you can ignore it.')
