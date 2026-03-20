import os
import json
import logging
import faiss
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional

# 💡 [신규 추가] LangChain 임베딩 및 FAISS 라이브러리
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS

# 환경 변수 로드 (.env)
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [1] 데이터 모델 (요청/응답 양식) ---
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

# --- [2] RAG 엔진 초기화 (서버 시작 시 1회만 로드) ---
base_dir = Path(__file__).resolve().parent
index_dir = base_dir / "faiss_index"

try:
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    
    openai_client = OpenAI(api_key=api_key)
    
    # 1. 💡 [기존 뇌] 제품 스펙 FAISS 로드 (오타 수정 완료!)
    faiss_index = faiss.read_index(str(index_dir / "osaki_products.faiss"))
    
    with open(index_dir / "osaki_metadata.jsonl", "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
        
    # MPS 가속 임베딩 로드
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # 2. 💡 [신규 뇌] CS 워런티 FAISS 로드 (Federated Search)
    embeddings_qa = OpenAIEmbeddings(api_key=api_key)
    faiss_index_qa = LC_FAISS.load_local(
        folder_path=str(index_dir / "freshdesk_qa"),
        embeddings=embeddings_qa,
        allow_dangerous_deserialization=True # 프로덕션 보안 승인
    )
    
    logger.info("✅ Dual-Core RAG Engine Initialized Successfully.")
except Exception as e:
    logger.error(f"🚨 Initialization Failed: {e}")
    faiss_index = None
    faiss_index_qa = None

# --- [3] FastAPI 앱 구성 ---
app = FastAPI(title="Osaki AI Support API", version="1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [4] 핵심 API 엔드포인트 ---
@app.post("/api/v1/chat") 
async def chat_endpoint(request: ChatRequest):
    """실시간 스트리밍(SSE) 메시지 전송 엔드포인트"""
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="Vector DB is not loaded.")

    try:
        retrieved_docs = []

        # 1. [제품 DB]에서 검색 (토큰 절약을 위해 상위 2개만)
        query_vector = embedding_model.encode([request.user_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_vector, 2) 
        
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                retrieved_docs.append(metadata[idx]["content"])

        # 2. 💡 [CS 워런티 DB]에서 검색 (상위 2개)
        if 'faiss_index_qa' in globals() and faiss_index_qa is not None:
            cs_docs = faiss_index_qa.similarity_search(request.user_query, k=2)
            for doc in cs_docs:
                retrieved_docs.append(doc.page_content)

        # 3. 두 뇌에서 찾아낸 지식을 하나로 융합
        context = "\n\n---\n\n".join(retrieved_docs)

        # 4. Generation 프롬프트 세팅
        system_prompt = f"""You are the official Professional Customer Support Agent for Osaki Massage Chairs.
Answer the user's question accurately based ONLY on the context below. Do not hallucinate.

[CRITICAL INSTRUCTION]
You MUST formulate your entire response STRICTLY IN ENGLISH, regardless of the language the user uses to ask the question. Even if the user asks in Korean, Spanish, or any other language, your final output must be 100% in Professional Business English.

[Context]:
{context}
"""
        messages_payload = [{"role": "system", "content": system_prompt}]
        for msg in request.chat_history:
            messages_payload.append({"role": msg.role, "content": msg.content})
        messages_payload.append({"role": "user", "content": request.user_query})

        # 실시간 스트리밍 제너레이터 함수
        def generate_stream():
            try:
                stream_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_payload,
                    temperature=0.1,
                    stream=True 
                )
                for chunk in stream_response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Streaming Error: {e}")
                yield "🚨 API Streaming Error."

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"API Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Server Error")