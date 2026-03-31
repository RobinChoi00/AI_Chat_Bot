import pandas as pd
import json
import logging
import docx 
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 🔥 [아키텍처 경로 동기화]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_DIR = os.path.join(BASE_DIR, "data")          # 정제된 데이터 창고 (Staging Layer)
RAW_DIR = os.path.join(BASE_DIR, "raw_data")       # 💡 날것의 원본 데이터 창고 (Raw Layer)
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")  # 서빙용 뇌 공간 (Serving Layer)

def extract_text_from_docx(file_path):
    """Word 문서에서 텍스트를 추출하고 의미 단위로 청킹(Chunking)합니다."""
    doc = docx.Document(file_path)
    chunks = []
    current_chunk = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        current_chunk += text + " "
        if len(current_chunk) > 250:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_products_index():
    logger.info("🚀 [통합 아키텍처] 워런티 & 정제된 상품 데이터 Vector DB 굽기 가동")
    
    os.makedirs(FAISS_DIR, exist_ok=True)
    documents = []

    # 1. 💡 [Staging Layer] 정제된 쇼피파이 데이터는 data/ 폴더에서 조준
    clean_csv = os.path.join(DATA_DIR, "cleaned_osaki_products.csv")
    if os.path.exists(clean_csv):
        logger.info(f"📊 정제된 상품 데이터 섭취 중: cleaned_osaki_products.csv")
        try:
            df = pd.read_csv(clean_csv)
            for _, row in df.iterrows():
                # Title과 Content가 이미 AI 친화적으로 압축되어 있음
                content = str(row['content'])
                metadata = {"source": str(row['source'])} # Handle을 출처로 사용
                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 CSV 처리 실패: {e}")
    else:
        # 정제된 파일이 없으면 치명적 에러로 간주
        logger.error("🚨 cleaned_osaki_products.csv 파일이 없습니다. 파이프라인을 중단합니다.")
        return 

    # 2. 💡 [Raw Layer] 공식 워런티 파일은 raw_data/ 폴더에서 실제 파일명으로 완벽 조준
    warranty_docx = os.path.join(RAW_DIR, "Warranty-and-Return-Policy.docx") 
    if os.path.exists(warranty_docx):
        logger.info(f"📄 워런티 정책 문서 섭취 중: Warranty-and-Return-Policy.docx")
        try:
            chunks = extract_text_from_docx(warranty_docx)
            for chunk in chunks:
                metadata = {"source": "Warranty-and-Return-Policy"}
                documents.append(Document(page_content=chunk, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 워런티 문서 처리 실패: {e}")
    else:
        logger.warning(f"⚠️ {warranty_docx} 파일을 찾을 수 없습니다. 워런티 정책이 누락됩니다.")

    if not documents:
        logger.error("🚨 텍스트 데이터가 없습니다. 파이프라인을 중단합니다.")
        return

    # 3. Load: 일괄 임베딩 (OpenAI 1536차원 벡터화)
    logger.info(f"🧠 총 {len(documents)}개의 검증된 데이터를 Vector Space로 변환합니다...")
    try:
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        save_path = os.path.join(FAISS_DIR, "osaki_products")
        vector_db.save_local(save_path)
        logger.info(f"🎉 성공! 무결점의 첫 번째 뇌가 '{save_path}'에 완벽하게 덮어쓰기 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 치명적 에러: {e}")

if __name__ == "__main__":
    build_products_index()