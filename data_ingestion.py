import pandas as pd
import faiss
import json
import torch
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_multi_source_index():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"  # [핵심] 이제 루트가 아닌 data/ 폴더만 바라봅니다.
    index_dir = base_dir / "faiss_index"
    index_dir.mkdir(exist_ok=True)

    # 1. Mac Apple Silicon (MPS) 하드웨어 가속
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🚀 임베딩 하드웨어 가속기: {device} 모드로 가동합니다.")
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    
    # 2. Extract & Transform: data/ 폴더 안의 모든 CSV 파일 순회
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"🚨 {data_dir} 폴더 안에 정제된 CSV 파일이 없습니다.")
        return

    all_texts = []
    for file_path in csv_files:
        logger.info(f"📄 데이터 섭취 중 (Ingesting): {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                all_texts.append(content)
                metadata.append({"source": file_path.name, "content": content})
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
            
    logger.info("✅ [System Halt] Multi-Source Ingestion Pipeline Completed Successfully.")

if __name__ == "__main__":
    build_multi_source_index()