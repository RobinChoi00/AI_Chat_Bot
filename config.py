# config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ==========================================
# 1. Business Logic & UI Strings (비즈니스 설정)
# ==========================================
SUPPORT_BUSINESS_HOURS = "Mon-Fri, 9:30 AM - 6:30 PM / Sat, 10:00 AM - 4:00 PM CST"

WARRANTY_PHONE = "+1-888-848-2630"

SALES_PHONE_BY_DOMAIN = {
    "osakiusa":          "+1-888-501-5988",
    "titanchair":        "1-888-848-2630",
    "osakimassagechair": "+1-214-613-1630",
}

SALES_EMAIL_BY_DOMAIN = {
    "titanchair": "info@osakititan.com",
    "osakiusa": "osakiusa@osakititan.com",
    "osakimassagechair": "oskmc@osakititan.com",
}

WARRANTY_TEAM_EMAIL = os.environ.get("WARRANTY_TEAM_EMAIL", "service@osakititan.com")

# Warranty evidence uploads — notify these team inboxes (additive; customer still sees service@).
WARRANTY_EVIDENCE_NOTIFY_RECIPIENTS: list[tuple[str, str]] = [
    ("Cong Huynh-Tran", "cong.t@osakititan.com"),
    ("Jose Alfonzo", "jose.a@osakititan.com"),
    ("Alfonso Cardenas", "alfonso.c@osakititan.com"),
    ("Roman Medrano", "roman.m@osakititan.com"),
    ("Fred Dominguez", "fred.d@osakititan.com"),
    ("Ryan Harp", "ryan.h@osakititan.com"),
]

EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

SUPPORT_CONTACT_MSG = f"If you need further assistance, please contact our support team at {WARRANTY_PHONE}. Our business hours are {SUPPORT_BUSINESS_HOURS}."

# 본사/쇼룸 주소 — 고객이 showroom / location / visit / address 질문 시 사용
COMPANY_ADDRESS = "1001 W Crosby Rd, Carrollton, TX 75006"
SHOWROOM_LOCATION = COMPANY_ADDRESS


def get_contact_msg(routing: str, target_domain: str) -> str:
    """Return the correct footer based on intent (Sales vs Warranty) and domain."""
    is_sales = "PRODUCTS" in routing.upper()

    if is_sales:
        domain_lower = (target_domain or "").lower()
        phone = WARRANTY_PHONE
        for key, number in SALES_PHONE_BY_DOMAIN.items():
            if key in domain_lower:
                phone = number
                break
        return (
            f"For sales inquiries, please contact us at {phone}. "
            f"Our business hours are {SUPPORT_BUSINESS_HOURS}."
        )

    return (
        f"If you need further assistance, please contact our support team at {WARRANTY_PHONE}. "
        f"Our business hours are {SUPPORT_BUSINESS_HOURS}."
    )

# 프론트엔드에서 도메인을 넘겨주지 않았을 때 사용할 기본 폴백(Fallback) 도메인
DEFAULT_TARGET_DOMAIN = "https://titanchair.com"

# 수리 매뉴얼 링크 중앙 관리
REPAIR_MANUAL_URL = "https://www.otasupport.com"


# ==========================================
# 2. AI Engine & System Settings (시스템 튜닝 설정)
# ==========================================
MAX_RETRIES = 3

# LLM 모델 설정 (추후 gpt-5가 나오면 여기서만 수정하면 됨)
AGENT_MODEL = "gpt-4o"
ROUTER_MODEL = "gpt-4o-mini"

# 💰 [Cost Optimization] Embedding model:
# - text-embedding-3-small : 5x cheaper than ada-002 AND higher quality.
# - Switching requires rebuilding the FAISS index (`python script/master_ingester.py`)
#   because vector dimensions differ from ada-002.
# - Set to "text-embedding-ada-002" if you can't rebuild yet.
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI client behavior (applied via the OpenAI SDK).
# - timeout: per-call hard ceiling. Tool-calling rounds rarely exceed 30s.
# - max_retries: SDK handles 429/5xx with exponential backoff automatically.
OPENAI_REQUEST_TIMEOUT = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", 45))
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", 3))

# LLM 창의성 통제 (0.0 = 극강의 팩트 위주, 1.0 = 창의적 지어내기)
# (참고: main.py에서 스트리밍 응답 시 0.1을 쓰고 있으므로 일치시킴)
LLM_TEMPERATURE = 0.1 

# RAG(FAISS) 검색 설정
FAISS_SEARCH_K = 5 # 사용자 질문 시 Vector DB에서 가져올 최대 문서(Chunk) 개수


# ==========================================
# 3. Cost Observability — token pricing per 1M tokens (USD)
# ==========================================
# Used by the usage logger to estimate per-request cost. Update if OpenAI
# changes prices. Cached input is billed at 50% of the regular input rate
# (OpenAI prompt caching).
MODEL_PRICING_USD_PER_1M = {
    "gpt-4o":              {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini":         {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1":             {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini":        {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "cached_input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "cached_input": 0.10, "output": 0.0},
}


# ==========================================
# 4. Rate Limiting (slowapi)
# ==========================================
# Per-IP rate limits to protect against accidental loops / abuse / cost bombs.
# Override via env if you need to scale up.
RATE_LIMIT_PER_MINUTE = os.environ.get("RATE_LIMIT_PER_MINUTE", "30/minute")
RATE_LIMIT_PER_HOUR = os.environ.get("RATE_LIMIT_PER_HOUR", "200/hour")


# ==========================================
# 5. Response cache for repeat FAQ queries
# ==========================================
# Common questions ("what's your warranty?", "business hours?") repeat
# constantly. Caching their answers for a short TTL bypasses the full
# agent-loop cost. PII-tinted queries (email/order id) bypass the cache.
CHAT_CACHE_ENABLED = os.environ.get("CHAT_CACHE_ENABLED", "1") == "1"
CHAT_CACHE_TTL_SECONDS = int(os.environ.get("CHAT_CACHE_TTL_SECONDS", 600))  # 10 min
CHAT_CACHE_MAX_ENTRIES = int(os.environ.get("CHAT_CACHE_MAX_ENTRIES", 512))

# Pre-LLM scope gate (blocks off-topic before the agent loop).
SCOPE_CLASSIFIER_ENABLED = os.environ.get("SCOPE_CLASSIFIER_ENABLED", "1") == "1"
SCOPE_CLASSIFIER_LLM = os.environ.get("SCOPE_CLASSIFIER_LLM", "1") == "1"


# ==========================================
# 6. CORS — restrict allowed origins in production
# ==========================================
# Comma-separated list, or "*" to allow all (dev only).
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "*",  # ⚠️ Tighten in production: https://titanchair.com,https://osakiusa.com,...
).split(",")