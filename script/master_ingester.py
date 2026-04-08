import os
import glob
import json
import pandas as pd
import docx
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.documents import Document

# 💡 [아키텍처] 마스터 파이프라인 경로 설정 (raw_data와 data 폴더 모두 타격)
load_dotenv(override=True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
RAW_DIR = os.path.join(BASE_DIR, "raw_data")       
DATA_DIR = os.path.join(BASE_DIR, "data")          # 💡 실버 존(data 폴더) 추가!
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index", "freshdesk_qa")  

class MasterIngester:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.documents = []

    # 1. 에러 매뉴얼 처리 (raw_data)
    def process_error_manuals(self):
        print("🔍 [1/6] 에러 매뉴얼 CSV 파싱 중...")
        for file_name, skip_rows in {"Auto-Check.csv": 1, "fault_judgment.csv": 5}.items():
            file_path = os.path.join(RAW_DIR, file_name)
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path, skiprows=skip_rows, encoding='utf-8-sig').fillna("")
            for _, row in df.iterrows():
                code_no = str(row.get("Code No.", row.get("No.", ""))).strip()
                if not code_no or code_no.lower() == 'nan': continue
                content = f"[Error Code]: {code_no}\n[Symptom]: {row.get('Phenomenon','')}\n[Troubleshooting]: {row.get('Troubleshooting Steps','')}"
                self.documents.append(Document(page_content=content.strip(), metadata={"type": "error_code", "error_code": code_no}))

    # 2. 정책 워드 문서 처리 (raw_data)
    def process_word_policies(self):
        print("🔍 [2/6] 정책 문서(.docx) 파싱 중...")
        docx_files = glob.glob(os.path.join(RAW_DIR, "*.docx"))
        for file_path in docx_files:
            doc = docx.Document(file_path)
            content = " ".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            self.documents.append(Document(page_content=content, metadata={"source": os.path.basename(file_path), "type": "policy"}))

    # 3. CS Q&A 처리 (raw_data)
    def process_qa_reports(self):
        print("🔍 [3/6] CS Q&A CSV 파싱 중...")
        file_path = os.path.join(RAW_DIR, "Warranty Daily Report - Q&A.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, skiprows=1).fillna("")
            for _, row in df.iterrows():
                issue = str(row.iloc[0]).strip()
                if not issue or issue == "N/A": continue
                content = f"[Issue]: {issue}\n[Diagnostic]: {row.iloc[1]}\n[Solution]: {row.iloc[2]}"
                self.documents.append(Document(page_content=content, metadata={"type": "troubleshooting"}))

    # 4. Freshdesk 티켓 처리 (data 폴더 - JSON)
    def process_freshdesk_tickets(self):
        print("🔍 [4/6] Freshdesk 고객 상담 티켓 파싱 중...")
        file_path = os.path.join(DATA_DIR, "freshdesk_tickets.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                tickets = json.load(f)
            for t in tickets:
                content = f"Customer Question:\n{t['question']}\n\nOfficial Answer / Resolution:\n{t['answer']}"
                self.documents.append(Document(page_content=content, metadata={"source": "freshdesk", "ticket_id": t.get("ticket_id")}))

    # 5. 웹 크롤링 데이터 처리 (data 폴더 - JSON)
    def process_web_data(self):
        print("🔍 [5/6] 웹사이트 크롤링 데이터 파싱 중...")
        file_path = os.path.join(DATA_DIR, "web_crawled_data.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                web_data = json.load(f)
            for data in web_data:
                self.documents.append(Document(page_content=data['content'], metadata={"source": data.get("source", "web")}))

    # 6. 쇼피파이 정제 데이터 처리 (data 폴더 - CSV)
    def process_shopify_data(self):
        print("🔍 [6/6] 쇼피파이 상품 정제 데이터 파싱 중...")
        file_path = os.path.join(DATA_DIR, "cleaned_osaki_products.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path).fillna("")
            for _, row in df.iterrows():
                self.documents.append(Document(page_content=str(row.get('content', '')), metadata={"source": row.get('source', 'shopify')}))

    # 최종 빌드 (모든 데이터를 FAISS에 한 번에 굽기)
    def build_vector_db(self):
        print(f"\n🚀 총 {len(self.documents)}개의 통합 문서를 수집했습니다. 임베딩 시작! (시간이 조금 걸릴 수 있습니다)")
        if self.documents:
            vs = LC_FAISS.from_documents(self.documents, self.embeddings)
            vs.save_local(INDEX_PATH)
            print("💾 완벽한 마스터 FAISS Vector DB 구축 완료!")
        else:
            print("🚨 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    ingester = MasterIngester()
    ingester.process_error_manuals()
    ingester.process_word_policies()
    ingester.process_qa_reports()
    ingester.process_freshdesk_tickets()
    ingester.process_web_data()
    ingester.process_shopify_data()
    ingester.build_vector_db()