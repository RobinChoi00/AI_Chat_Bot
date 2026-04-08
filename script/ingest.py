import os
import pandas as pd
from langchain_core.documents import Document

def process_cleaned_csv(file_path: str, skip_rows: int) -> list[Document]:
    """
    정제된 CSV 파일을 초고속으로 파싱하여 FAISS용 Document 리스트로 반환합니다.
    """
    print(f"🔄 Processing CSV file: {file_path} ...")
    
    # 💡 엑셀 엔진 불필요! 순수 csv 파서를 사용하여 속도 극대화 (한글/특수문자 방어용 utf-8-sig)
    try:
        df = pd.read_csv(file_path, skiprows=skip_rows, encoding='utf-8-sig')
    except Exception as e:
        print(f"🚨 CSV 읽기 에러 (인코딩 폴백 시도): {e}")
        df = pd.read_csv(file_path, skiprows=skip_rows, encoding='utf-8')
        
    df = df.fillna("")
    
    docs = []
    for _, row in df.iterrows():
        # 컬럼명 매핑 (두 파일의 컬럼명이 살짝 다른 것 완벽 방어)
        code_no = str(row.get("Code No.", row.get("No.", ""))).strip()
        
        # "1.0"으로 파싱되는 플로트(Float) 현상 방어
        if code_no.endswith(".0"):
            code_no = code_no[:-2]
            
        if not code_no or code_no.lower() == 'nan':
            continue 
            
        phenomenon = str(row.get("Phenomenon", row.get("phenomenon", ""))).strip()
        troubleshooting = str(row.get("Troubleshooting Steps", row.get("steps of shooting the trouble ", ""))).strip()
        
        # 💡 [핵심] LLM 전용 자연어 템플릿 조립
        page_content = f"""
[Error Code]: {code_no}
[Symptom / Phenomenon]: {phenomenon}
[Diagnosis & Troubleshooting]: {troubleshooting}
"""
        metadata = {
            "source": os.path.basename(file_path),
            "type": "error_code_manual",
            "error_code": code_no
        }
        
        docs.append(Document(page_content=page_content.strip(), metadata=metadata))
        
    print(f"✅ {len(docs)}개의 에러 코드 문서 추출 완료! ({file_path})")
    return docs

# ==========================================
# 🚀 실제 실행부 (Execution Block)
# ==========================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # 💡 [핵심] 임베딩을 위해 .env 파일에서 OPENAI_API_KEY를 반드시 불러와야 합니다.
    load_dotenv(override=True)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("🚨 OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요!")

    # 1. 데이터 파싱 (RAM에 적재)
    docs_auto = process_cleaned_csv("raw_data/Auto-Check.csv", skip_rows=1)
    docs_fault = process_cleaned_csv("raw_data/fault_judgment.csv", skip_rows=5)
    final_error_docs = docs_auto + docs_fault
    
    print(f"\n🚀 총 {len(final_error_docs)}개의 완벽한 문서를 준비했습니다!")
    print("🧠 OpenAI 임베딩을 시작합니다. (API 통신 중... 잠시만 기다려주세요.)")
    
    # 2. FAISS Vector DB 임베딩 및 저장 (물리적 디스크 I/O)
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS as LC_FAISS
    
    embeddings = OpenAIEmbeddings()
    
    # 지웅님의 챗봇이 바라보는 FAISS 인덱스 타겟 폴더
    index_path = "faiss_index/freshdesk_qa"
    
    try:
        # 기존에 만들어둔 DB가 있다면 불러와서 그 위에 얹습니다 (Upsert)
        vs_qa = LC_FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vs_qa.add_documents(final_error_docs)
        print("🔄 기존 FAISS Vector DB에 성공적으로 문서를 추가(Add)했습니다.")
    except Exception as e:
        # 기존 DB가 아예 없다면 0부터 새로 창조합니다
        print("⚠️ 기존 DB를 찾을 수 없어 새로운 FAISS 인덱스를 생성합니다.")
        vs_qa = LC_FAISS.from_documents(final_error_docs, embeddings)
        
    # 3. 메모리에 있는 DB를 하드디스크에 영구 저장 (핵심!)
    vs_qa.save_local(index_path)
    print("💾 FAISS Vector DB 영구 저장 완료! 이제 챗봇이 이 데이터를 완벽하게 읽을 수 있습니다.")