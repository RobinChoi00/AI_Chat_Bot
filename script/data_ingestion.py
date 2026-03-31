import pandas as pd
import faiss
import json
import torch
import logging
import docx 
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 💡 [신규 함수] 비정형 문서 시맨틱 청킹 (Semantic Chunking)
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
        # 💡 [Trade-off 최적화] 문단이 너무 짧으면 문맥(Context)을 상실하고, 너무 길면 벡터 밀도가 옅어짐.
        # 약 250~300자 내외의 황금 비율로 텍스트를 묶어서 토큰 효율과 검색 정확도를 극대화.
        if len(current_chunk) > 250:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            
    # 남은 찌꺼기 텍스트 처리
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def build_multi_source_index():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"          # 정형 데이터 (CSV)
    raw_dir = base_dir / "raw_data"       # 💡 [신규 경로] 비정형 날것의 데이터 (DOCX)
    index_dir = base_dir / "faiss_index"
    index_dir.mkdir(exist_ok=True)

    # 1. Mac Apple Silicon (MPS) 하드웨어 가속
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🚀 임베딩 하드웨어 가속기: {device} 모드로 가동합니다.")
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    all_texts = []
    
    # 2-A. Extract & Transform: data/ 폴더 안의 CSV 파일 순회
    csv_files = list(data_dir.glob("*.csv"))
    for file_path in csv_files:
        logger.info(f"📊 CSV 데이터 섭취 중 (Ingesting): {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                all_texts.append(content)
                metadata.append({"source": file_path.name, "content": content})
        except Exception as e:
            logger.error(f"🚨 {file_path.name} 파일 처리 실패: {e}")

    # 2-B. 💡 Extract & Transform: raw_data/ 폴더 안의 DOCX 파일 순회
    docx_files = list(raw_dir.glob("*.docx"))
    for file_path in docx_files:
        logger.info(f"📄 Word 문서 섭취 중 (Ingesting): {file_path.name}")
        try:
            chunks = extract_text_from_docx(file_path)
            for chunk in chunks:
                all_texts.append(chunk)
                # 메타데이터에 파일명(Warranty, Sales Policy)을 박아넣어 출처 추적 기능을 강화합니다.
                metadata.append({"source": file_path.name, "content": chunk})
        except Exception as e:
            logger.error(f"🚨 {file_path.name} 파일 처리 실패: {e}")

    if not all_texts:
        logger.error("🚨 텍스트 데이터가 없습니다.")
        return

    # 3. Load: 일괄 임베딩 (Batch Embedding)
    logger.info(f"🧠 총 {len(all_texts)}개의 데이터 조각(Chunks)을 Vector Space로 변환합니다...")
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    index.add(embeddings)

    # 4. 저장 (Serialization)
    faiss.write_index(index, str(index_dir / "osaki_products.faiss"))
    with open(index_dir / "osaki_metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta) + "\n")
            
    logger.info("✅ [System Halt] Multi-Source Hybrid Ingestion Pipeline Completed Successfully.")

if __name__ == "__main__":
    build_multi_source_index()