import json
import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 🔥 [핵심 아키텍처] 무적의 절대 경로 계산 (실행 위치 독립성 보장)
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path)) # AI-Chat-Project 루트
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

def build_vector_db():
    logger.info("🚀 AI 뇌세포(Vector DB) 압축 프로세스를 시작합니다.")

    # 1. 절대 경로를 통한 타겟 JSON 완벽 조준
    target_json = os.path.join(DATA_DIR, "freshdesk_tickets.json")
    
    try:
        with open(target_json, "r", encoding="utf-8") as f:
            tickets = json.load(f)
    except FileNotFoundError:
        logger.error(f"🚨 JSON 파일을 찾을 수 없습니다. 경로를 확인하세요: {target_json}")
        return

    if not tickets:
        logger.warning("⚠️ 데이터가 비어있습니다.")
        return

    # 2. Document 객체로 변환 (Context 융합)
    documents = []
    for t in tickets:
        # AI가 완벽하게 이해하도록 Q와 A를 하나의 문서로 병합
        page_content = f"Customer Question:\n{t['question']}\n\nOfficial Answer / Resolution:\n{t['answer']}"
        
        # 메타데이터를 달아두어 출처(Source) 추적
        metadata = {"ticket_id": t.get("ticket_id", "Unknown"), "subject": t.get("subject", "Unknown")}
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"✅ {len(documents)}개의 문서를 성공적으로 변환했습니다. OpenAI API를 호출하여 임베딩을 시작합니다...")

    # 3. OpenAI 임베딩 및 FAISS 인덱스 생성
    try:
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        # 4. 절대 경로를 이용하여 로컬에 새로운 FAISS 뇌 저장
        os.makedirs(FAISS_DIR, exist_ok=True) # 폴더가 없으면 강제 생성
        save_path = os.path.join(FAISS_DIR, "freshdesk_qa")
        vector_db.save_local(save_path)
        
        logger.info(f"🎉 성공! AI의 새로운 CS 뇌가 '{save_path}' 경로에 완벽하게 구워졌습니다.")

    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 치명적 에러 발생: {e}")

if __name__ == "__main__":
    build_vector_db()