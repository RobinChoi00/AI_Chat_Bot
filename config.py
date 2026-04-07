# config.py

# ==========================================
# 1. Business Logic & UI Strings (비즈니스 설정)
# ==========================================
SUPPORT_PHONE_NUMBER = "1-888-848-2630 (Ext. 3)"
SUPPORT_BUSINESS_HOURS = "Mon-Fri, 9:30 AM - 6:30 PM / Sat, 10:00 AM - 4:00 PM CST"

# 💡 [수술 완료] 대괄호를 제거하고 AI가 생략하지 못하도록 완벽한 자연어 문장으로 결합했습니다.
SUPPORT_CONTACT_MSG = f"If you need further assistance, please contact our support team at {SUPPORT_PHONE_NUMBER}. Our business hours are {SUPPORT_BUSINESS_HOURS}."

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

# LLM 창의성 통제 (0.0 = 극강의 팩트 위주, 1.0 = 창의적 지어내기)
# (참고: main.py에서 스트리밍 응답 시 0.1을 쓰고 있으므로 일치시킴)
LLM_TEMPERATURE = 0.1 

# RAG(FAISS) 검색 설정
FAISS_SEARCH_K = 5 # 사용자 질문 시 Vector DB에서 가져올 최대 문서(Chunk) 개수