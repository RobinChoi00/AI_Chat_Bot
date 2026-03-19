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

# 환경 변수 로드 (.env)
load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [1] 데이터 모델 (요청/응답 양식) ---
class Message(BaseModel):
    role: str      # 'user' (고객) 또는 'assistant' (AI)
    content: str   # 대화 내용

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default_session"
    # [핵심] 과거 대화 기록을 담을 배열(List) 추가. 기본값은 빈 배열([])
    chat_history: Optional[List[Message]] = [] 

class ChatResponse(BaseModel):
    answer: str
    status: str

# --- [2] RAG 엔진 초기화 (서버 시작 시 1회만 로드) ---
base_dir = Path(__file__).resolve().parent
index_dir = base_dir / "faiss_index"

try:
    # 다시 OS의 환경 변수(.env)에서 읽어오도록 롤백
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    
    openai_client = OpenAI(api_key=api_key)
    
    # FAISS 로드
    faiss_index = faiss.read_index(str(index_dir / "osaki_products.faiss"))
    
    with open(index_dir / "osaki_metadata.jsonl", "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
        
    # MPS 가속 임베딩 로드
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    logger.info("✅ Backend RAG Engine Initialized Successfully.")
except Exception as e:
    logger.error(f"🚨 Initialization Failed: {e}")
    faiss_index = None

# --- [3] FastAPI 앱 구성 ---
app = FastAPI(title="Osaki AI Support API", version="1.0")

# CORS 설정 (쇼핑몰 프론트엔드 도메인에서 API를 호출할 수 있도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실무 배포 시에는 실제 쇼핑몰 도메인(예: https://osakimassage.com)으로 제한해야 합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [4] 핵심 API 엔드포인트 ---
@app.post("/api/v1/chat") # [엄격 검증] response_model=ChatResponse 삭제됨!
async def chat_endpoint(request: ChatRequest):
    """실시간 스트리밍(SSE) 메시지 전송 엔드포인트"""
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="Vector DB is not loaded.")

    try:
        # 1. Retrieval (검색 - 기존 로직 동일)
        query_vector = embedding_model.encode([request.user_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_vector, 3)
        
        retrieved_docs = []
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                retrieved_docs.append(metadata[idx]["content"])
        context = "\n\n---\n\n".join(retrieved_docs)

        # 2. Generation 프롬프트 세팅
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

        # [핵심 로직] 데이터를 쪼개서 실시간으로 방출(Yield)하는 Generator 함수
        def generate_stream():
            try:
                # stream=True 옵션으로 OpenAI API 호출
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

        # 완성된 문장이 아닌, 흐르는 물(Stream) 자체를 프론트엔드에 연결
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"API Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Server Error")