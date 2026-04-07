import os
import logging
import time
import hmac
import hashlib
import base64
import re
import threading
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
from langchain_core.documents import Document
from config import SUPPORT_CONTACT_MSG

def handle_support_query():
    return SUPPORT_CONTACT_MSG

faiss_lock = threading.Lock()

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
app = FastAPI(title="Titan AI Agent API", version="2.0")

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
        # 💡 [신규] B2C CUSTOMER PROTOCOL에 Instant Checkout Link 제공 지침 추가
        system_prompt = f"""You are an elite AI Copilot for Titan Chair LLC and Osaki, serving both general customers and internal field technicians.

<CORE_DIRECTIVES>
1. STRICT GROUNDING: Answer the user's inquiry based SOLELY and EXCLUSIVELY on the <context> provided below.
2. MULTI-TENANT DOMAIN ROUTING: The user is currently browsing {target_domain}. Whenever you extract a "Direct Purchase Link" or "Instant Checkout Link" from the <context>, you MUST rewrite its base domain to match {target_domain}.
</CORE_DIRECTIVES>

<PROTOCOL_B2B_TECH_SUPPORT>
TRIGGER: The user asks about error codes, repair, troubleshooting, assembly, parts, or manuals.
RULES:
1. Role: Act as a direct, professional L1 Tech Support Engineer.
2. EXACT LINK & CONTACT FORMAT: You MUST ALWAYS output the manual link and support contact together explicitly. Use this exact format:
   👉 Repair & Manuals: https://titanchair.com/pages/repair-manuals
   {SUPPORT_CONTACT_MSG}
3. IF ISSUE IS FOUND IN <context>: Provide the troubleshooting steps, then append the EXACT LINK & CONTACT FORMAT above.
</PROTOCOL_B2B_TECH_SUPPORT>

<PROTOCOL_B2C_SALES>
TRIGGER: The user asks for general recommendations, features, or pricing.
RULES:
1. Role: Act as a friendly, conversion-focused sales assistant. Highlight features and prices of 1 to 3 chairs strictly from the <context>.
2. PURCHASE LINK: Always provide the "Direct Purchase Link" (rewritten to {target_domain}) for recommended products.
3. INSTANT CHECKOUT: If the user explicitly expresses intent to buy (e.g., "I want to buy", "checkout", "purchase now"), you MUST provide the "Instant Checkout Link" from the <context>.
4. PROHIBITION: NEVER show the "Repair & Manuals" link to a general buyer.
</PROTOCOL_B2C_SALES>

<ANTI_HALLUCINATION_GUARDRAILS>
1. VERIFY: Read the <context> carefully before generating a response.
2. STRICT DECLINE (FOR REPAIRS/ERRORS): If the user asks for a specific error code (e.g., Error 63) or issue that is NOT in the <context>, do NOT guess. You MUST output EXACTLY this template:
   "I apologize, but I do not have the specific diagnostic steps for that issue in my current documentation. 
   Please check our official Repair & Manuals page for detailed guides here:
   🔗 https://titanchair.com/pages/repair-manuals
   
   {SUPPORT_CONTACT_MSG}"
3. ZERO INVENTION: NEVER invent warranty exclusions, part numbers, prices, policies, or URLs.
</ANTI_HALLUCINATION_GUARDRAILS>

<context>
{{context}}
</context>
"""
        messages_payload = [{"role": "system", "content": system_prompt}]
        for MSG in request.chat_history:
            messages_payload.append({"role": MSG.role, "content": MSG.content})
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
# 💡 [아키텍처 확장] 웹후크용 백그라운드 워커 (Upsert & Guardrail 적용)
# ==========================================
def update_faiss_index_background(payload: dict):
    """
    쇼피파이 데이터를 파싱하여 구형 벡터를 도려내고 신규 벡터를 꽂아넣는 핵심 Upsert 로직.
    """
    product_status = payload.get('status', 'unknown').lower()
    item_title = payload.get('title', 'Unknown Item')
    item_id = str(payload.get('id', ''))
    
    # 1. 방어벽: Active 상태가 아니면 차단
    if product_status != 'active':
        logger.info(f"⏸️ [Skip Update] Product '{item_title}' status is '{product_status}'. Update aborted.")
        return
        
    logger.info(f"🔄 [Background Task] Starting RAG database update... Target product: {item_title}")
    
    try:
        # 2. 쇼피파이 JSON 파싱 및 데이터 정제(Parsing & Cleansing)
        body_html = payload.get('body_html', '') or ''
        clean_body = re.sub('<[^<]+>', '', body_html) # HTML 태그 제거
        
        variants = payload.get('variants', [])
        price = variants[0].get('price', 'N/A') if variants else 'N/A'
        
        # 💡 [신규] 다이렉트 결제를 위한 Variant ID 추출
        variant_id = variants[0].get('id', '') if variants else ''
        
        handle = payload.get('handle', '')
        product_url = f"https://titanchair.com/products/{handle}"
        checkout_url = f"https://titanchair.com/cart/{variant_id}:1" # 👈 다이렉트 결제 링크 (Cart Permalink)
        
        # 💡 [신규] AI가 읽을 최종 텍스트 조립 (Instant Checkout Link 추가)
        page_content = f"Product Name: {item_title}\nPrice: ${price}\nDescription: {clean_body}\nDirect Purchase Link: {product_url}\nInstant Checkout Link: {checkout_url}"
        
        # 메타데이터 (향후 추적 및 삭제를 위한 고유 식별자)
        metadata = {
            "source": product_url,
            "title": item_title,
            "shopify_id": item_id
        }
        new_doc = Document(page_content=page_content, metadata=metadata)
        
        # 3. 락(Lock) 획득 후 안전하게 FAISS 조작 (Thread-Safe Operation)
        global vs_products
        with faiss_lock:
            if vs_products is not None:
                # 3-1. 기존 구형 데이터 탐색 및 삭제 (Delete)
                ids_to_delete = []
                for doc_id, doc in vs_products.docstore._dict.items():
                    # 상품명(title)이 정확히 일치하거나 텍스트 안에 포함되어 있으면 구형 데이터로 간주
                    if doc.metadata.get('title') == item_title or item_title in doc.page_content:
                        ids_to_delete.append(doc_id)
                        
                if ids_to_delete:
                    vs_products.delete(ids_to_delete)
                    logger.info(f"🗑️ [FAISS] Successfully deleted {len(ids_to_delete)} existing records for product '{item_title}'.")
                
                # 3-2. 최신 데이터 임베딩 및 삽입 (Add)
                vs_products.add_documents([new_doc])
                logger.info(f"➕ [FAISS] Successfully embedded and added new product '{item_title}'.")
                
                # 3-3. 디스크에 영구 보존 (Save to Disk)
                vs_products.save_local(str(index_dir / "osaki_products"))
                logger.info("💾 [FAISS] Latest index permanently saved to local disk.")
            else:
                logger.error("🚨 [FAISS] Update failed: 'vs_products' engine is not loaded.")
                
    except Exception as e:
        logger.error(f"🚨 [Background Task] Fatal error during FAISS update: {e}")


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