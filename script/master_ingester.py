import os
import glob
import json
import pandas as pd
import docx
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.documents import Document

# 💡 [아키텍처] 마스터 파이프라인 경로 설정
load_dotenv(override=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
RAW_DIR = os.path.join(BASE_DIR, "raw_data")       
DATA_DIR = os.path.join(BASE_DIR, "data")          
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index") 

class MasterIngester:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # 💡 [리팩토링 핵심] 도메인별 3개의 바구니(딕셔너리) 생성
        self.domain_docs = {
            "freshdesk_qa": [],   
            "web_data": [],       
            "osaki_products": []  
        }

    # ==========================================
    # 🧠 1번 뇌: freshdesk_qa (CS & 에러 전용)
    # ==========================================
    def process_error_manuals(self):
        print("🔍 [1/7] 에러 매뉴얼 CSV 파싱 중... -> [freshdesk_qa] 할당")
        for file_name, skip_rows in {"Auto-Check.csv": 1, "fault_judgment.csv": 5}.items():
            file_path = os.path.join(RAW_DIR, file_name)
            if not os.path.exists(file_path): continue
            
            # 💡 [아키텍처] 판다스의 강제 형변환(Float)을 막고 원본 텍스트(63) 그대로 보존합니다.
            df = pd.read_csv(file_path, skiprows=skip_rows, encoding='utf-8-sig', dtype=str).fillna("")
            
            # 💡 [아키텍처] 대소문자 무시 & 앞뒤 공백 제거로 더러운 헤더를 깔끔하게 평탄화(Normalization)
            df.columns = df.columns.str.strip().str.lower()
            
            for _, row in df.iterrows():
                # 번호 추출 (No. 또는 Code No.)
                code_no = str(row.get("code no.", row.get("no.", ""))).strip()
                if not code_no or code_no.lower() == 'nan': continue
                
                # 💡 [핵심] 컬럼명이 달라도 모두 잡아내는 동적 매핑 로직
                symptom = str(row.get("phenomenon", row.get("problem description", ""))).strip()
                troubleshooting = str(row.get("troubleshooting steps", row.get("steps of shooting the trouble", ""))).strip()
                
                # 빈 데이터 필터링 방어 로직
                if not symptom and not troubleshooting: continue
                
                content = f"[Error Code]: {code_no}\n[Symptom]: {symptom}\n[Troubleshooting]: {troubleshooting}"
                self.domain_docs["freshdesk_qa"].append(Document(page_content=content, metadata={"type": "error_code", "error_code": code_no}))

    def process_qa_reports(self):
        print("🔍 [2/7] CS Q&A CSV 파싱 중... -> [freshdesk_qa] 할당")
        file_path = os.path.join(RAW_DIR, "Warranty Daily Report - Q&A.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, skiprows=1).fillna("")
            for _, row in df.iterrows():
                issue = str(row.iloc[0]).strip()
                if not issue or issue == "N/A": continue
                content = f"[Issue]: {issue}\n[Diagnostic]: {row.iloc[1]}\n[Solution]: {row.iloc[2]}"
                self.domain_docs["freshdesk_qa"].append(Document(page_content=content, metadata={"type": "troubleshooting"}))

    def process_freshdesk_tickets(self):
        print("🔍 [3/7] Freshdesk 고객 상담 티켓 파싱 중... -> [freshdesk_qa] 할당")
        file_path = os.path.join(DATA_DIR, "freshdesk_tickets.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                tickets = json.load(f)
            for t in tickets:
                content = f"Customer Question:\n{t['question']}\n\nOfficial Answer / Resolution:\n{t['answer']}"
                self.domain_docs["freshdesk_qa"].append(Document(page_content=content, metadata={"source": "freshdesk", "ticket_id": t.get("ticket_id")}))

    # ==========================================
    # 🧠 2번 뇌: web_data (정책 & 정보 전용)
    # ==========================================
    def process_word_policies(self):
        print("🔍 [4/7] 정책 문서(.docx) 파싱 중... -> [web_data] 할당")
        docx_files = glob.glob(os.path.join(RAW_DIR, "*.docx")) # 💡 Warranty.docx, Sales Policy.docx 2개 동시 처리
        for file_path in docx_files:
            doc = docx.Document(file_path)
            content = " ".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            self.domain_docs["web_data"].append(Document(page_content=content, metadata={"source": os.path.basename(file_path), "type": "policy"}))

    def process_web_data(self):
        print("🔍 [5/7] 웹사이트 크롤링 데이터 파싱 중... -> [web_data] 할당")
        file_path = os.path.join(DATA_DIR, "web_crawled_data.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                web_data = json.load(f)
            for data in web_data:
                self.domain_docs["web_data"].append(Document(page_content=data['content'], metadata={"source": data.get("source", "web")}))

    def process_curated_knowledge(self):
        print("🔍 [5.5/7] 큐레이션 지식 데이터 파싱 중... -> [web_data] 할당")
        file_path = os.path.join(DATA_DIR, "curated_knowledge.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                self.domain_docs["web_data"].append(Document(
                    page_content=entry['content'],
                    metadata={
                        "source": entry.get("source", "curated"),
                        "category": entry.get("category", "general"),
                        "type": "curated_knowledge"
                    }
                ))
            print(f"   ✅ 큐레이션 지식 {len(entries)}건 로드 완료")

    # ==========================================
    # 🧠 3번 뇌: osaki_products (상품 스펙 전용)
    # ==========================================
    def process_shopify_data(self):
        print("🔍 [6/7] 쇼피파이 상품 정제 데이터 파싱 중... -> [osaki_products] 할당")
        file_path = os.path.join(DATA_DIR, "cleaned_osaki_products.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path).fillna("")
            for _, row in df.iterrows():
                self.domain_docs["osaki_products"].append(Document(page_content=str(row.get('content', '')), metadata={"source": row.get('source', 'shopify')}))

    # 🌟 [누락 복원 완료] 7번째 파이프라인: 두 CSV를 결합(Inner Join)하는 극한의 로직
    def process_specifications(self):
        print("🔍 [7/7] 원본 쇼피파이 & 스펙 시트 융합(Join) 중... -> [osaki_products] 할당")
        spec_csv = os.path.join(RAW_DIR, "Specification_Massage Chair - Massage Chair.csv")
        shopify_csv = os.path.join(RAW_DIR, "products_export.csv")
        
        if os.path.exists(spec_csv) and os.path.exists(shopify_csv):
            # 1. 💡 [아키텍처] 다중 헤더 무시하고 4번째 줄(skiprows=3)을 헤더로 스펙 데이터 로드
            df_spec = pd.read_csv(spec_csv, skiprows=3, low_memory=False).fillna("N/A")
            df_spec.columns = df_spec.columns.str.strip() # 컬럼 공백 제거 방어 로직
            
            # 2. 쇼피파이 데이터 로드
            df_shopify = pd.read_csv(shopify_csv, low_memory=False).fillna("N/A")

            # 3. 조인을 위한 텍스트 정규화 함수
            def normalize_text(text):
                if pd.isna(text) or text == "N/A": return ""
                return str(text).lower().replace("massage chair", "").replace("-", "").replace(" ", "").strip()

            # 4. 🚨 [수정 완료] 양쪽 데이터에 모두 'join_key'를 완벽하게 생성합니다!
            df_shopify['join_key'] = df_shopify['Title'].apply(normalize_text)
            df_spec['join_key'] = df_spec['Name'].apply(normalize_text) 

            # 5. 두 데이터를 이름 기준으로 완벽하게 병합 (Inner Join)
            merged_df = pd.merge(df_shopify, df_spec, on='join_key', how='inner')
            
            for _, row in merged_df.iterrows():
                model_name = str(row.get('Title', 'Unknown'))
                specs_text = []
                for col_name, value in row.items():
                    # 너무 길거나 불필요한 HTML 이미지 태그 등은 과감히 제거
                    if col_name not in ['join_key', 'Body (HTML)', 'Handle', 'Image Src'] and value != "N/A" and str(value).strip():
                        specs_text.append(f"- {col_name}: {value}")
                
                content = f"Specifications for Model [{model_name}]:\n" + "\n".join(specs_text)
                self.domain_docs["osaki_products"].append(Document(page_content=content, metadata={"source": "specification_join", "title": model_name, "type": "specification"}))

    # ==========================================
    # 🚀 다중 벡터 DB 동시 빌드 (Multi-Index Generation)
    # ==========================================
    def build_vector_dbs(self):
        print("\n🚀 [임베딩 시작] 도메인별 3개의 Vector DB를 생성합니다...")
        
        for domain_name, docs in self.domain_docs.items():
            if not docs:
                print(f"⚠️ [{domain_name}] 수집된 데이터가 없어 건너뜁니다.")
                continue
                
            print(f"🧠 [{domain_name}] 총 {len(docs)}개의 문서 임베딩 중...")
            vs = LC_FAISS.from_documents(docs, self.embeddings)
            
            save_path = os.path.join(FAISS_DIR, domain_name)
            vs.save_local(save_path)
            print(f"💾 [{domain_name}] DB 구축 완료! ({save_path})")

if __name__ == "__main__":
    os.makedirs(FAISS_DIR, exist_ok=True)
    
    ingester = MasterIngester()
    
    # [CS/Error 뇌세포]
    ingester.process_error_manuals()    # Auto-Check, fault_judgment
    ingester.process_qa_reports()       # Warranty Daily Report - Q&A
    ingester.process_freshdesk_tickets()# freshdesk_tickets.json
    
    # [Policy/Web 뇌세포]
    ingester.process_word_policies()    # Warranty.docx, Sales Policy.docx (glob으로 한 번에 2개 섭취)
    ingester.process_web_data()         # web_crawled_data.json
    ingester.process_curated_knowledge()# curated_knowledge.json (추천/FAQ/기능교육)
    
    # [Products/Specs 뇌세포]
    ingester.process_shopify_data()     # cleaned_osaki_products.csv
    ingester.process_specifications()   # products_export.csv + Specification_Massage Chair
    
    ingester.build_vector_dbs()
    print("\n🎉 모든 파이프라인이 성공적으로 완료되었습니다! 7개의 원본 파일과 3개의 정제 파일이 모두 3개의 뇌로 흡수되었습니다.")