#!/usr/bin/env python3
"""Fail-fast production configuration and runtime artifact validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(ROOT / ".env")


def _is_https(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        "OPENAI_API_KEY",
        "SHOPIFY_WEBHOOK_SECRET",
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD",
        "ADMIN_SESSION_SECRET",
        "ADMIN_API_KEY",
        "PUBLIC_BASE_URL",
        "NEXT_PUBLIC_API_BASE_URL",
        "CORS_ALLOWED_ORIGINS",
        "TRUSTED_HOSTS",
        "EMAIL_SENDER",
        "EMAIL_PASSWORD",
        "RC_WEBHOOK_VERIFICATION_TOKEN",
        "RC_CLIENT_ID",
        "RC_CLIENT_SECRET",
        "RC_SMS_FROM_NUMBER",
    ]
    for key in required:
        if not os.getenv(key, "").strip():
            errors.append(f"missing required environment variable: {key}")

    for key in ("ADMIN_SESSION_SECRET", "ADMIN_API_KEY"):
        value = os.getenv(key, "")
        if value and len(value) < 32:
            errors.append(f"{key} must contain at least 32 characters")

    if not (
        os.getenv("RC_USER_JWT", "").strip()
        or os.getenv("RC_USER_JWT_FILE", "").strip()
        or os.getenv("RC_JWT_PRIVATE_KEY", "").strip()
    ):
        errors.append("configure RC_USER_JWT, RC_USER_JWT_FILE, or RC_JWT_PRIVATE_KEY")
    jwt_file = os.getenv("RC_USER_JWT_FILE", "").strip()
    if jwt_file and not Path(jwt_file).is_file():
        errors.append("RC_USER_JWT_FILE does not exist")
    if not (
        os.getenv("RC_WARRANTY_TRANSFER_EXTENSION", "").strip()
        or os.getenv("RC_WARRANTY_TRANSFER_TO", "").strip()
    ):
        errors.append("configure RC_WARRANTY_TRANSFER_EXTENSION or RC_WARRANTY_TRANSFER_TO")
    sms_from = os.getenv("RC_SMS_FROM_NUMBER", "").strip()
    if sms_from and not sms_from.startswith("+"):
        errors.append("RC_SMS_FROM_NUMBER must use E.164 format beginning with '+'")

    public_base = os.getenv("PUBLIC_BASE_URL", "")
    if public_base and not _is_https(public_base):
        errors.append("PUBLIC_BASE_URL must be an absolute HTTPS URL")

    browser_api_base = os.getenv("NEXT_PUBLIC_API_BASE_URL", "")
    if browser_api_base and not _is_https(browser_api_base):
        errors.append("NEXT_PUBLIC_API_BASE_URL must be an absolute HTTPS URL")

    origins = [
        item.strip()
        for item in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    ]
    if "*" in origins:
        errors.append("CORS_ALLOWED_ORIGINS must not contain '*' in production")
    for origin in origins:
        if not _is_https(origin):
            errors.append(f"CORS origin must be an absolute HTTPS URL: {origin}")

    trusted_hosts = [
        item.strip()
        for item in os.getenv("TRUSTED_HOSTS", "").split(",")
        if item.strip()
    ]
    if not trusted_hosts or "*" in trusted_hosts:
        errors.append("TRUSTED_HOSTS must contain explicit hostnames in production")

    for relative in ("db_data", "faiss_index", "uploaded_evidence", "data", "rc_audio_cache"):
        path = ROOT / relative
        if not path.is_dir():
            errors.append(f"required runtime directory is missing: {relative}")
        elif not os.access(path, os.R_OK | os.W_OK):
            errors.append(f"runtime directory is not readable/writable: {relative}")

    for name in ("osaki_products", "freshdesk_qa", "web_data"):
        for filename in ("index.faiss", "index.pkl"):
            path = ROOT / "faiss_index" / name / filename
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty search index artifact: {path.relative_to(ROOT)}")

    for relative in (
        "data/warranty_flowchart.json",
        "data/warranty_evidence_specs.json",
        "data/model_families.json",
    ):
        if not (ROOT / relative).is_file():
            errors.append(f"required application data is missing: {relative}")

    if os.getenv("WARRANTY_FRESHDESK_CREATE_CASE", "0") == "1":
        for key in ("FRESHDESK_DOMAIN", "FRESHDESK_API_KEY"):
            if not os.getenv(key, "").strip():
                errors.append(f"{key} is required when Freshdesk case creation is enabled")

    if not os.getenv("BACKUP_S3_URI", "").strip():
        warnings.append("BACKUP_S3_URI is empty; backups will remain on the same host")

    agent_model = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.6").strip()
    if not agent_model.startswith("gpt-5.6"):
        warnings.append(
            "OPENAI_AGENT_MODEL is not GPT-5.6; compare it against the current quality profile before release"
        )

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Preflight failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("Preflight passed: configuration, storage, and search indexes are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
