import pandas as pd
import json
import logging
import docx 
import os
from dotenv import load_dotenv

# 💡 [아키텍처 통합] OpenAI & LangChain 글로벌 스탠다드로 통일
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 🔥 [핵심 아키텍처] 무적의 절대 경로 (scripts/ 에서 실행해도 무조건 Root 조준)
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path)) 
DATA_DIR = os.path.join(BASE_DIR, "data")          
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

def extract_text_from_docx(file_path):
    """Word 문서에서 텍스트를 추출하고 의미 단위로 청킹(Chunking)합니다."""
    doc = docx.Document(file_path)
    chunks = []
    current_chunk = ""
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current_chunk += text + " "
        if len(current_chunk) > 250:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_multi_source_index():
    logger.info("🚀 [통합 아키텍처] 워런티 & 상품 데이터 Vector DB 압축 가동")
    
    os.makedirs(FAISS_DIR, exist_ok=True)
    documents = []

    # 1-A. Extract: data/ 폴더 안의 CSV 파일 (정형 데이터)
    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    for file_path in csv_files:
        logger.info(f"📊 CSV 섭취 중: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                # LangChain Document 규격으로 래핑
                metadata = {"source": os.path.basename(file_path)}
                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 {os.path.basename(file_path)} 처리 실패: {e}")

    # 1-B. Extract: data/ 폴더 안의 DOCX 파일 (비정형 문서 - 🚨 raw_data 폴더 폐기 및 data로 통합)
    docx_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.docx')]
    for file_path in docx_files:
        logger.info(f"📄 Word 문서 섭취 중: {os.path.basename(file_path)}")
        try:
            chunks = extract_text_from_docx(file_path)
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path)}
                documents.append(Document(page_content=chunk, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 {os.path.basename(file_path)} 처리 실패: {e}")

    if not documents:
        logger.error("🚨 텍스트 데이터가 없습니다. data 폴더를 확인하세요.")
        return

    # 2. Load: 일괄 임베딩 (OpenAI 1536차원 벡터화)
    logger.info(f"🧠 총 {len(documents)}개의 조각을 Vector Space로 변환합니다...")
    try:
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        # 3. 다른 뇌들과 똑같은 규격(LangChain)으로 저장
        save_path = os.path.join(FAISS_DIR, "osaki_products")
        vector_db.save_local(save_path)
        logger.info(f"✅ [System Halt] 첫 번째 뇌가 완벽한 규격으로 '{save_path}'에 덮어쓰기 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 에러: {e}")

if __name__ == "__main__":
    build_multi_source_index()