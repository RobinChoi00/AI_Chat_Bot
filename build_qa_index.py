import json
import os
import logging
from dotenv import load_dotenv

# LangChain 기반 벡터 DB 생성 글로벌 스탠다드 라이브러리
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def build_vector_db():
    logger.info("🚀 AI 뇌세포(Vector DB) 압축 프로세스를 시작합니다.")

    # 1. 추출했던 황금 데이터(JSON) 로드
    try:
        with open("data/freshdesk_tickets.json", "r", encoding="utf-8") as f:
            tickets = json.load(f)
    except FileNotFoundError:
        logger.error("🚨 JSON 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    if not tickets:
        logger.warning("⚠️ 데이터가 비어있습니다.")
        return

    # 2. Document 객체로 변환 (Context 융합)
    documents = []
    for t in tickets:
        # 💡 [핵심 비즈니스 로직] AI가 완벽하게 이해하도록 Q와 A를 하나의 문서로 병합
        page_content = f"Customer Question:\n{t['question']}\n\nOfficial Answer / Resolution:\n{t['answer']}"
        
        # 메타데이터를 달아두면 나중에 "몇 번 티켓을 참고했는지" 출처(Source)를 추적할 수 있습니다.
        metadata = {"ticket_id": t["ticket_id"], "subject": t["subject"]}
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"✅ {len(documents)}개의 문서를 성공적으로 변환했습니다. OpenAI API를 호출하여 임베딩을 시작합니다...")

    # 3. OpenAI 임베딩 및 FAISS 인덱스 생성
    try:
        # 환경 변수에서 OpenAI API 키를 자동으로 물고 들어옵니다.
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        # 4. 로컬에 새로운 FAISS 뇌(.faiss 및 .pkl) 저장
        save_path = "faiss_index/freshdesk_qa"
        vector_db.save_local(save_path)
        
        logger.info(f"🎉 성공! AI의 새로운 CS 뇌가 '{save_path}' 폴더에 완벽하게 구워졌습니다.")

    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 치명적 에러 발생: {e}")

if __name__ == "__main__":
    build_vector_db()