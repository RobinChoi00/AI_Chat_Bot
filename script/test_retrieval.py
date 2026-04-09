import os

# 💡 [Mac(ARM64) 방어코드] OpenMP 다중 실행 충돌(OMP Error #15)을 강제로 무시합니다.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DB_PATH = os.path.join(BASE_DIR, "faiss_index", "freshdesk_qa")

print("🔍 FAISS Vector DB 직접 검색 테스트 시작...")
embeddings = OpenAIEmbeddings()

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

query = "How can i fix error code 63?"
docs = db.similarity_search(query, k=3)

for i, doc in enumerate(docs):
    print(f"\n[{i+1}번째로 검색된 원본 데이터]")
    print(doc.page_content)