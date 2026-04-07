# script/import_qa.py
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# LangChain components
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.documents import Document

# 💡 [보안] 환경변수 로드
load_dotenv(override=True)
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY is missing in .env")

# 💡 [아키텍처] 동적 경로 설정 (script 폴더 기준)
project_root = Path(__file__).resolve().parent.parent
index_dir = project_root / "faiss_index"
raw_data_dir = project_root / "raw_data"

# 정확한 파일명 매핑
QA_CSV_FILE = raw_data_dir / "Warranty Daily Report - Q&A.csv"

def ingest_qa_data():
    print("🚀 [Job Started] CS Q&A 데이터 파이프라인 가동...")
    
    # 1. 기존 FAISS 뇌(인덱스) 로드
    embeddings = OpenAIEmbeddings(api_key=api_key)
    try:
        vs_products = LC_FAISS.load_local(
            str(index_dir / "osaki_products"), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ [FAISS] 기존 osaki_products 인덱스 로드 완료. (이곳에 Q&A 지식을 추가합니다)")
    except Exception as e:
        print(f"🚨 [FAISS] 인덱스 로드 실패. 먼저 상품 데이터를 학습시켜주세요: {e}")
        return

    if not QA_CSV_FILE.exists():
        print(f"🚨 [Error] Q&A 원천 데이터를 찾을 수 없습니다: {QA_CSV_FILE}")
        return

    try:
        # 2. 데이터 전처리 (Data Cleansing)
        # 첫 번째 줄(타이틀)을 무시하고, 두 번째 줄(인덱스 1)을 헤더로 파싱
        df = pd.read_csv(QA_CSV_FILE, header=1, dtype=str).fillna("N/A")
        
        new_documents = []
        
        # 3. 시맨틱 청킹 (Semantic Chunking)
        for index, row in df.iterrows():
            # CSV 컬럼명이 불규칙할 것을 대비하여 인덱스(0, 1, 2)로 안전하게 추출
            issue = str(row.iloc[0]).strip()
            questions = str(row.iloc[1]).strip()
            solutions = str(row.iloc[2]).strip()
            
            # 유효하지 않은 데이터(쓰레기 값) 드랍
            if issue == "N/A" or not issue:
                continue
                
            # 💡 [핵심] AI가 문맥을 완벽히 이해할 수 있도록 명시적인(Explicit) 템플릿 적용
            content = (
                f"[Customer Issue/Symptom]: {issue}\n"
                f"[Diagnostic Questions to Ask]: {questions}\n"
                f"[Troubleshooting & Solutions]: {solutions}"
            )
            
            # 메타데이터에 'troubleshooting' 타입을 명시하여 향후 검색(Retrieval) 가중치 조정에 활용
            metadata = {
                "source": "warranty_qa_report",
                "type": "troubleshooting",
                "issue_category": issue
            }
            new_documents.append(Document(page_content=content, metadata=metadata))
            
        # 4. FAISS 저장 (Batch Upsert)
        if new_documents:
            print(f"🧠 [Embedding] 총 {len(new_documents)}개의 CS 트러블슈팅 지식 임베딩 시작...")
            vs_products.add_documents(new_documents)
            vs_products.save_local(str(index_dir / "osaki_products"))
            print(f"💾 [Success] Q&A 지식 이식 완료! 이제 챗봇이 장애 진단을 수행할 수 있습니다.")

    except Exception as e:
        print(f"🚨 파이프라인 실행 중 치명적 에러 발생: {e}")

if __name__ == "__main__":
    ingest_qa_data()