import json
import os
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 🔥 무적의 절대 경로 조준 로직
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path)) 
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

def build_web_vector_db():
    logger.info("🚀 웹 크롤링 데이터 -> Vector DB 압축 프로세스 시작")

    target_json = os.path.join(DATA_DIR, "web_crawled_data.json")
    
    try:
        with open(target_json, "r", encoding="utf-8") as f:
            web_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"🚨 JSON 파일을 찾을 수 없습니다: {target_json}")
        return

    documents = []
    for data in web_data:
        # 웹 데이터는 Q&A 형태가 아니므로 내용 전체를 넣습니다.
        page_content = data['content']
        # 출처(URL)를 메타데이터로 반드시 남깁니다.
        metadata = {"source": data.get("source", "Unknown")}
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"✅ {len(documents)}개의 웹 문서를 변환합니다. 임베딩 가동!")

    try:
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        os.makedirs(FAISS_DIR, exist_ok=True)
        # 🚨 [주의] 저장 폴더 이름이 다릅니다! 세 번째 뇌: web_data
        save_path = os.path.join(FAISS_DIR, "web_data") 
        vector_db.save_local(save_path)
        
        logger.info(f"🎉 성공! 타이탄 체어 홈페이지의 지식이 '{save_path}'에 구워졌습니다.")

    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 치명적 에러: {e}")

if __name__ == "__main__":
    build_web_vector_db()