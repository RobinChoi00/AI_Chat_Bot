import pandas as pd
import json
import logging
import docx 
import os
import glob  # 💡 파일 패턴 매칭을 위해 추가
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 🔥 [아키텍처 경로 동기화]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_DIR = os.path.join(BASE_DIR, "data")          
RAW_DIR = os.path.join(BASE_DIR, "raw_data")       
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")  

def extract_text_from_docx(file_path):
    """Word 문서에서 텍스트를 추출하고 의미 단위로 청킹(Chunking)합니다."""
    doc = docx.Document(file_path)
    chunks = []
    current_chunk = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        current_chunk += text + " "
        if len(current_chunk) > 400: # 💡 청크 사이즈를 약간 키워 문맥 보존성 향상 (Trade-off 고려)
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def find_warranty_file(directory):
    """💡 [핵심 추가] 하드코딩 방지: Warranty라는 단어가 포함된 docx 파일을 유연하게 찾습니다."""
    # 후보 1: 직접적인 후보 리스트
    candidates = ["Warranty.docx", "Warranty-and-Return-Policy.docx", "warranty.docx"]
    for c in candidates:
        path = os.path.join(directory, c)
        if os.path.exists(path):
            return path
            
    # 후보 2: 패턴 매칭 (Warranty가 포함된 모든 docx 파일 탐색)
    pattern = os.path.join(directory, "*[Ww]arranty*.docx")
    found_files = glob.glob(pattern)
    if found_files:
        return found_files[0]
        
    return None

def build_products_index():
    logger.info("🚀 [통합 아키텍처] 워런티 & 정제된 상품 데이터 Vector DB 굽기 가동")
    
    # 💡 [방어적 설계] 디렉토리가 없으면 런타임 에러 방지를 위해 자동 생성
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True) 

    documents = []

    # 1. [Staging Layer] 정제된 쇼피파이 데이터 섭취
    clean_csv = os.path.join(DATA_DIR, "cleaned_osaki_products.csv")
    if os.path.exists(clean_csv):
        logger.info(f"📊 정제된 상품 데이터 섭취 중: {os.path.basename(clean_csv)}")
        try:
            df = pd.read_csv(clean_csv)
            for _, row in df.iterrows():
                content = str(row['content'])
                metadata = {"source": str(row['source']), "type": "product"} # 메타데이터 세분화
                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 CSV 처리 실패: {e}")
    else:
        logger.error(f"🚨 {clean_csv} 파일이 없습니다. 파이프라인을 중단합니다.")
        return 

    # 2. 💡 [Raw Layer] 유연한 워런티 파일 탐색 적용
    warranty_path = find_warranty_file(RAW_DIR)
    
    if warranty_path:
        logger.info(f"📄 워런티 정책 문서 탐지 성공: {os.path.basename(warranty_path)}")
        try:
            chunks = extract_text_from_docx(warranty_path)
            for chunk in chunks:
                metadata = {"source": "warranty_policy", "type": "policy"}
                documents.append(Document(page_content=chunk, metadata=metadata))
        except Exception as e:
            logger.error(f"🚨 워런티 문서 처리 실패: {e}")
    else:
        logger.warning(f"⚠️ raw_data 폴더 내에 워런티(.docx) 파일을 찾을 수 없습니다. 정책 데이터가 생략됩니다.")

    if not documents:
        logger.error("🚨 인덱싱할 데이터가 존재하지 않습니다.")
        return

    # 3. Load: 벡터화 및 저장
    logger.info(f"🧠 총 {len(documents)}개의 문서를 OpenAI 임베딩으로 변환 중...")
    try:
        embeddings = OpenAIEmbeddings() 
        vector_db = FAISS.from_documents(documents, embeddings)

        save_path = os.path.join(FAISS_DIR, "osaki_products")
        vector_db.save_local(save_path)
        logger.info(f"🎉 성공! 벡터 DB가 '{save_path}'에 구축되었습니다.")
        
    except Exception as e:
        logger.error(f"🚨 벡터 DB 생성 중 치명적 에러: {e}")

if __name__ == "__main__":
    build_products_index()