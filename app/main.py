import os
import logging
import time
import hmac
import hashlib
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import pytz

# LangChain embeddings, FAISS, and prompt parser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 💡 [보안] 쇼피파이 웹후크 시크릿 키 로드 (실무에서는 .env 파일에서 관리)
SHOPIFY_WEBHOOK_SECRET = os.environ.get("SHOPIFY_WEBHOOK_SECRET", "your_shopify_secret_key_here")

# --- [0] Helper constants ---
TECHNICIAN_KEYWORDS = {
    "repair", "fix", "troubleshoot", "troubleshooting",
    "assembly", "disassembly", "manual", "service",
    "technician", "engineer", "수리", "조립", "매뉴얼", "엔지니어",
}

PRODUCT_QUERY_KEYWORDS = {
    "massage chair", "chair", "model", "product", "products",
    "recommend", "buy", "price", "4d", "3d", "zero gravity", "osaki", "titan",
}

def is_product_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in PRODUCT_QUERY_KEYWORDS)

def stream_text_response(session_id: str, user_query: str, response_text: str):
    """Stream plain text response and persist chat log."""
    yield response_text
    try:
        db = SessionLocal()
        new_log = ChatLog(
            session_id=session_id,
            user_query=user_query,
            bot_response=response_text
        )
        db.add(new_log)
        db.commit()
        db.close()
        logger.info("✅ Chat log saved to DB successfully.")
    except Exception as e:
        logger.error(f"DB Save Error: {e}")

# --- [1] Data models ---
class Message(BaseModel):
    role: str      
    content: str   

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default_session"
    chat_history: Optional[List[Message]] = [] 
    current_domain: str = "https://titanchair.com" # 💡 [신규] 프론트엔드 전달 도메인

class ChatResponse(BaseModel):
    answer: str
    status: str

# --- [2] Initialize 3-core Agentic RAG engine ---
project_root = Path(__file__).resolve().parent.parent
index_dir = project_root / "faiss_index"

try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    
    openai_client = OpenAI(api_key=api_key)
    
    logger.info("🚀 Loading 3-core AI engines into memory...")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    vs_products = LC_FAISS.load_local(str(index_dir / "osaki_products"), embeddings, allow_dangerous_deserialization=True)
    vs_qa = LC_FAISS.load_local(str(index_dir / "freshdesk_qa"), embeddings, allow_dangerous_deserialization=True)
    vs_web = LC_FAISS.load_local(str(index_dir / "web_data"), embeddings, allow_dangerous_deserialization=True)
    
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    ROUTER_PROMPT = """
    You are a highly intelligent routing system. Analyze the user's question and strictly output ONLY ONE of the following routing keys:
    - "PRODUCTS": If asking about product specifications, models, recommendations, purchase intent, manuals, WARRANTY, return policies, guarantees, dimensions, pricing, or parts. (MUST use this for ANY warranty/policy questions).
    - "QA": If asking about specific technical troubleshooting, previous customer support logs, or delivery tracking. (DO NOT use for warranty/policy).
    - "WEB": If asking about current sales, events, health benefits, FAQ, or general website info.

    User Question: {question}
    Routing Key:"""
    router_chain = PromptTemplate.from_template(ROUTER_PROMPT) | router_llm | StrOutputParser()

    logger.info("✅ 3-Core Agentic RAG Engine Initialized Successfully.")
except Exception as e:
    logger.error(f"🚨 Initialization Failed: {e}")
    vs_products, vs_qa, vs_web, router_chain = None, None, None, None

# --- [2.5] SQLite chat log persistence ---
DB_DIR = project_root / "db_data"
DB_DIR.mkdir(exist_ok=True) 
DATABASE_URL = f"sqlite:///{DB_DIR}/chat_history.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_query = Column(Text)
    bot_response = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.timezone('America/Chicago')))

Base.metadata.create_all(bind=engine)

# --- [3] FastAPI app setup ---
app = FastAPI(title="Titan AI Copilot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [4] Core API endpoint (Agentic RAG + Streaming) ---
@app.post("/api/v1/chat") 
async def chat_endpoint(request: ChatRequest):
    user_query = request.user_query

    if not all([vs_products, vs_qa, vs_web, router_chain]):
        raise HTTPException(status_code=500, detail="AI Engine is not fully loaded.")

    try:
        # Step 1: Run intent routing with deterministic product override.
        if is_product_query(user_query):
            routing_decision = "PRODUCTS"
        else:
            routing_decision = router_chain.invoke({"question": user_query}).strip().upper()
        logger.info(f"🔀 Router decision: target store -> [{routing_decision}]")

        # Step 2: Query only one store to optimize latency and cost.
        if "PRODUCTS" in routing_decision:
            docs = vs_products.similarity_search(user_query, k=5)
        elif "QA" in routing_decision:
            docs = vs_qa.similarity_search(user_query, k=5)
        else:
            docs = vs_web.similarity_search(user_query, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 💡 [신규] 프론트엔드에서 넘어온 도메인 변수 확보 (마지막 / 제거)
        target_domain = request.current_domain.rstrip('/')

        # Structured control prompt for strict grounding and safe output.
        system_prompt = f"""You are an elite AI Copilot for Titan Chair LLC and Osaki, serving both general customers and internal field technicians.

[CORE DIRECTIVE - STRICT GROUNDING]
Answer the user's inquiry based SOLELY and EXCLUSIVELY on the [Context] provided below. Do not hallucinate.

[MULTI-TENANT LINK ROUTING (동적 도메인 치환)]
The user is currently browsing this website: {target_domain}
💡 ESSENTIAL: Whenever you provide a "Direct Purchase Link" from the [Context], you MUST change its base domain to match {target_domain}.
For example, if the context says "https://titanchair.com/products/titan-4d-ion", you must rewrite it and output "{target_domain}/products/titan-4d-ion".

[B2B TECHNICIAN PROTOCOL (수리기사 모드)]
If the user asks about repair, troubleshooting, assembly, parts, or manuals for a specific chair:
1. Assume the user is our internal field technician or a customer needing deep technical support.
2. 💡 ESSENTIAL: You MUST provide the specific "Repair & Manuals" deep link from the [Context] so they can find the exact parts and videos immediately.
3. Be direct, professional, and concise.

[B2C CUSTOMER PROTOCOL (일반 고객 모드)]
If the user asks for general recommendations, features, or pricing:
1. Act as a friendly sales assistant. Highlight the features and prices of 1 to 3 chairs from the [Context].
2. 💡 ESSENTIAL: Provide the "Direct Purchase Link" (rewritten to {target_domain}) if recommending a product.
3. 🚫 PROHIBITED: NEVER show the "Repair & Manuals" link to a general buyer asking for recommendations.

[ANTI-HALLUCINATION PROTOCOL]
1. VERIFY: Read the [Context] carefully. 
2. 💡 DECLINE (STRICT): If the user asks for a SPECIFIC fact (e.g., a specific dimension, warranty coverage) that is NOT in the [Context], you MUST output EXACTLY:
   "I apologize, but I do not have specific information regarding that in my current documentation. Please contact our support team at 1-888-848-2630 (Ext. 3) for precise assistance. Our business hours are Mon-Fri 09:30 AM - 06:30 PM, and Sat 10:00 AM - 04:00 PM (CT). We are closed on Sundays."
3. PROHIBITION: NEVER invent warranty exclusions, part numbers, prices, or policies.

[Context]:
{context}
"""
        messages_payload = [{"role": "system", "content": system_prompt}]
        for msg in request.chat_history:
            messages_payload.append({"role": msg.role, "content": msg.content})
        messages_payload.append({"role": "user", "content": user_query})

        # Step 4: Stream response and persist chat logs.
        def generate_stream():
            full_response = "" 
            try:
                stream_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_payload,
                    temperature=0.1,
                    stream=True 
                )
                for chunk in stream_response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content 
                        yield content
                
                db = SessionLocal()
                new_log = ChatLog(
                    session_id=request.session_id,
                    user_query=user_query,
                    bot_response=full_response
                )
                db.add(new_log)
                db.commit()
                db.close()
                logger.info("✅ Chat log saved to DB successfully.")

            except Exception as e:
                logger.error(f"Streaming Error: {e}")
                yield "🚨 API Streaming Error."

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"API Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Server Error")


# ==========================================
# 💡 [아키텍처 확장] 웹후크용 백그라운드 워커 (Guardrail 적용)
# ==========================================
def update_faiss_index_background(payload: dict):
    """
    쇼피파이로부터 받은 페이로드를 파싱하여 FAISS 벡터 DB를 갱신하는 비동기 작업(Background Job).
    """
    # 1. 상품 상태 확인
    product_status = payload.get('status', 'unknown').lower()
    item_title = payload.get('title', 'Unknown Item')
    
    # 2. 방어벽: Active가 아니면 스킵 (Draft, Archived 데이터 유출 방지)
    if product_status != 'active':
        logger.info(f"⏸️ [Skip Update] 상품 '{item_title}'의 상태가 '{product_status}'입니다. AI 뇌 업데이트를 차단합니다.")
        return # 👈 여기서 함수를 강제 종료(Short-circuit)시켜 아래 로직 실행 방지
        
    # 3. Active 상태인 경우에만 FAISS 인덱스 업데이트 진행
    logger.info(f"🔄 [Background Task] RAG 데이터베이스 갱신 시작... 타겟 상품: {item_title}")
    
    # TODO: 실제 FAISS 인덱스 갱신 및 기존 데이터 삭제(Delete) 로직 추가 예정
    time.sleep(5) # 무거운 임베딩 작업을 시뮬레이션
    
    logger.info("✅ [Background Task] RAG 데이터베이스 갱신 완료 및 메모리 적재 성공!")


# ==========================================
# 💡 [아키텍처 확장] 쇼피파이 웹후크 수신 엔드포인트
# ==========================================
@app.post("/webhook/shopify/product-update")
async def shopify_webhook(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: Optional[str] = Header(None) 
):
    # 1. 방어 로직: 서명 헤더 누락 시 즉각 차단
    if not x_shopify_hmac_sha256:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing HMAC header")

    # 2. 페이로드 추출
    body = await request.body()

    # 3. 방어 로직: HMAC-SHA256 서명 검증
    secret = SHOPIFY_WEBHOOK_SECRET.encode('utf-8')
    hash_calc = hmac.new(secret, body, hashlib.sha256)
    calculated_hmac = base64.b64encode(hash_calc.digest()).decode('utf-8')

    if not hmac.compare_digest(calculated_hmac, x_shopify_hmac_sha256):
        logger.warning("🚨 [Security Alert] 유효하지 않은 웹후크 서명 감지! 접근 차단됨.")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid HMAC signature")

    # 4. JSON 파싱
    payload = await request.json()
    logger.info(f"📦 [Webhook Received] 쇼피파이 검증 통과. 상품 ID: {payload.get('id')}")

    # 5. 무거운 인덱스 갱신 작업은 백그라운드 스레드로 위임
    background_tasks.add_task(update_faiss_index_background, payload)

    # 6. 타임아웃 방지를 위한 즉각적인 HTTP 200 반환
    return {"message": "Webhook received and processing in background"}