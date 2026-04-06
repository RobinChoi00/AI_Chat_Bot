# import_specs.py
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

# 💡 [아키텍처] 경로 설정
project_root = Path(__file__).resolve().parent.parent
index_dir = project_root / "faiss_index"
raw_data_dir = project_root / "raw_data"
SPEC_CSV_FILE = raw_data_dir / "Specification_Massage Chair - Massage Chair.csv"
SHOPIFY_CSV_FILE = raw_data_dir / "products_export.csv"

def normalize_text(text):
    """문자열 비교를 위한 정규화 (소문자화, 공백 및 불용어 제거)"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("massage chair", "").replace("-", "").replace(" ", "")
    return text.strip()

def ingest_intersected_data():
    print("🚀 [Job Started] Shopify & Spec Sheet Inner Join 파이프라인 가동...")
    
    # 1. FAISS 인덱스 로드
    embeddings = OpenAIEmbeddings(api_key=api_key)
    try:
        vs_products = LC_FAISS.load_local(
            str(index_dir / "osaki_products"), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ [FAISS] 기존 osaki_products 인덱스 로드 완료.")
    except Exception as e:
        print(f"🚨 [FAISS] 인덱스 로드 실패 (먼저 백엔드를 한 번 실행하여 인덱스를 생성하세요): {e}")
        return

    # 2. 파일 존재 여부 검증
    if not SPEC_CSV_FILE.exists():
        print(f"🚨 [Error] 스펙 시트 파일을 찾을 수 없습니다: {SPEC_CSV_FILE}")
        return
    if not SHOPIFY_CSV_FILE.exists():
        print(f"🚨 [Error] 쇼피파이 CSV 파일을 찾을 수 없습니다. (경로 및 파일명 확인 요망): {SHOPIFY_CSV_FILE}")
        return

    try:
        # 3. 쇼피파이 데이터 파싱 및 정규화
        shopify_df = pd.read_csv(SHOPIFY_CSV_FILE, dtype=str).fillna("N/A")
        shopify_df['join_key'] = shopify_df['Title'].apply(normalize_text)
        
        # 4. 스펙 시트 파싱 (4번째 줄이 헤더) 및 정규화
        spec_df = pd.read_csv(SPEC_CSV_FILE, header=3, dtype=str).fillna("N/A")
        spec_df['full_name'] = spec_df['Brand'] + " " + spec_df['Name']
        spec_df['join_key'] = spec_df['full_name'].apply(normalize_text)
        
        # 5. 교집합 추출 (Inner Join)
        merged_df = pd.merge(spec_df, shopify_df, on='join_key', how='inner')
        print(f"🔍 교집합 추출 완료: 총 {len(merged_df)}개의 활성(Active) 규격 데이터 확보!")
        
        if merged_df.empty:
            print("⚠️ [Warning] 조인 결과가 없습니다. 양쪽 CSV의 상품명이 너무 다르거나 파일 형식이 잘못되었습니다.")
            return

        # 6. 벡터 문서 변환
        new_documents = []
        for index, row in merged_df.iterrows():
            model_name = row['Title'] # 쇼피파이 기준 공식 명칭
            
            # 스펙 텍스트 조립 (스펙 시트의 데이터들을 추출)
            specs_text = []
            for col_name, value in row.items():
                # 조인을 위해 만든 임시 컬럼과 너무 긴 불필요한 컬럼 제외
                if col_name not in ['join_key', 'full_name', 'Body HTML', 'Handle'] and value != "N/A":
                    specs_text.append(f"- {col_name}: {value}")
            
            content = f"Specifications for Model [{model_name}]:\n" + "\n".join(specs_text)
            
            metadata = {
                "source": "specification_join",
                "title": model_name,
                "type": "specification"
            }
            new_documents.append(Document(page_content=content, metadata=metadata))
            
        # 7. FAISS 저장
        if new_documents:
            print(f"🧠 [Embedding] 총 {len(new_documents)}개의 규격 데이터 임베딩 시작... (API 비용 발생)")
            vs_products.add_documents(new_documents)
            vs_products.save_local(str(index_dir / "osaki_products"))
            print(f"💾 [Success] 데이터 교집합 임베딩 완료 및 FAISS 인덱스 갱신 성공!")

    except Exception as e:
        print(f"🚨 파이프라인 실행 중 치명적 에러 발생: {e}")

if __name__ == "__main__":
    ingest_intersected_data()