import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
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

# 💡 [글로벌 스탠다드] LangChain 임베딩, FAISS, 프롬프트 파서
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [1] 데이터 모델 ---
class Message(BaseModel):
    role: str      
    content: str   

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default_session"
    chat_history: Optional[List[Message]] = [] 

class ChatResponse(BaseModel):
    answer: str
    status: str

# --- [2] 3-Core Agentic RAG 엔진 초기화 ---
project_root = Path(__file__).resolve().parent.parent
index_dir = project_root / "faiss_index"

try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    
    openai_client = OpenAI(api_key=api_key)
    
    logger.info("🚀 3-Core AI 뇌세포를 메모리에 적재합니다...")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # 💡 [아키텍처 통합] 3개의 뇌를 모두 동일한 규격으로 로드
    vs_products = LC_FAISS.load_local(str(index_dir / "osaki_products"), embeddings, allow_dangerous_deserialization=True)
    vs_qa = LC_FAISS.load_local(str(index_dir / "freshdesk_qa"), embeddings, allow_dangerous_deserialization=True)
    vs_web = LC_FAISS.load_local(str(index_dir / "web_data"), embeddings, allow_dangerous_deserialization=True)
    
    # 💡 [신규 이식] 의도 분류용 초고속 라우터 LLM (gpt-4o-mini)
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    ROUTER_PROMPT = """
    You are a highly intelligent routing system. Analyze the user's question and strictly output ONLY ONE of the following routing keys:
    - "PRODUCTS": If asking about product specifications, manuals, WARRANTY, return policies, guarantees, dimensions, or parts. (MUST use this for ANY warranty/policy questions).
    - "QA": If asking about specific technical troubleshooting, previous customer support logs, or delivery tracking. (DO NOT use for warranty/policy).
    - "WEB": If asking about current sales, events, health benefits, FAQ, or general website info.

    User Question: {question}
    Routing Key:"""
    router_chain = PromptTemplate.from_template(ROUTER_PROMPT) | router_llm | StrOutputParser()

    logger.info("✅ 3-Core Agentic RAG Engine Initialized Successfully.")
except Exception as e:
    logger.error(f"🚨 Initialization Failed: {e}")
    vs_products, vs_qa, vs_web, router_chain = None, None, None, None

# --- [2.5] SQLite 대화 기록 DB 유지 (완벽한 코드 보존) ---
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

# --- [3] FastAPI 앱 구성 ---
app = FastAPI(title="Titan AI Copilot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [4] 핵심 API 엔드포인트 (Agentic RAG + Streaming) ---
@app.post("/api/v1/chat") 
async def chat_endpoint(request: ChatRequest):
    if not all([vs_products, vs_qa, vs_web, router_chain]):
        raise HTTPException(status_code=500, detail="AI Engine is not fully loaded.")

    try:
        user_query = request.user_query
        
        # 💡 Step 1: 라우터 엔진 가동 (의도 파악)
        routing_decision = router_chain.invoke({"question": user_query}).strip().upper()
        logger.info(f"🔀 라우터 판단: 타겟 뇌 ➡️ [{routing_decision}]")

        # 💡 Step 2: 정밀 타격 (단 1개의 뇌만 검색하여 비용 및 속도 최적화)
        if "PRODUCTS" in routing_decision:
            docs = vs_products.similarity_search(user_query, k=5)
        elif "QA" in routing_decision:
            docs = vs_qa.similarity_search(user_query, k=5)
        else:
            docs = vs_web.similarity_search(user_query, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 💡 [프롬프트 엔지니어링] 구조화된 마크다운 기반의 강력한 통제 프롬프트
        system_prompt = f"""You are an elite, highly precise Professional Customer Support Agent for Titan Chair LLC and Osaki Massage Chairs.

[CORE DIRECTIVE - STRICT GROUNDING]
Your absolute primary directive is to answer the user's inquiry based SOLELY and EXCLUSIVELY on the [Context] provided below.
You are strictly FORBIDDEN from utilizing pre-trained knowledge, external assumptions, or general common sense.

[ANTI-HALLUCINATION PROTOCOL]
1. VERIFY: Read the [Context] carefully. 
2. ANSWER: If the exact information is present, synthesize it clearly and professionally.
3. DECLINE: If the answer is NOT explicitly stated in the [Context], or if the [Context] is irrelevant, you MUST NOT guess. You MUST output EXACTLY this phrase:
   "I apologize, but I do not have specific information regarding that in my current documentation. Please contact our support team at 1-888-848-2630 (Ext. 3) for precise assistance."
4. PROHIBITION: NEVER invent warranty exclusions (e.g., zippers, velcro), part numbers, or policies that are not explicitly written in the [Context].

[OUTPUT FORMATTING]
- Use clear, structured bullet points if explaining multiple conditions (e.g., Year 1, Year 2 warranty coverage).
- Maintain a polite, empathetic, yet strictly objective Professional Business English tone.

[TECHNICIAN COPILOT MODE - STRICT ROUTING]
If the user's prompt includes keywords like "assembly", "repair", "video", or "manual" along with a specific massage chair model, bypass standard CS responses and IMMEDIATELY provide the exact video download link from the [Video Database] below.
Format EXACTLY like this: "Here is the official [Assembly/Repair] video for [Model Name]: [Link]"

[Video Database]:
- Titan Prime 3D : https://www.otasupport.com/api/download?fileId=1qJUblQGZksbVBFzbbsh9miUy3eBxg11L
- Osaki Maestro : https://www.otasupport.com/api/download?fileId=1XReuOFDBwigbOANoNNigLN9F81gan7o7

[Context]:
{context}
"""
        messages_payload = [{"role": "system", "content": system_prompt}]
        for msg in request.chat_history:
            messages_payload.append({"role": msg.role, "content": msg.content})
        messages_payload.append({"role": "user", "content": user_query})

        # 💡 Step 4: 지웅님의 무결점 스트리밍 & DB 저장 로직 (완벽 보존)
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