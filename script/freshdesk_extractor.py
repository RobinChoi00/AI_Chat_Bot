import os
import requests
import logging
import time
import json
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class FreshdeskETL:
    def __init__(self):
        self.domain = os.environ.get("FRESHDESK_DOMAIN")
        self.api_key = os.environ.get("FRESHDESK_API_KEY")
        self.base_url = f"https://{self.domain}/api/v2"
        self.auth = (self.api_key, 'X')
        self.headers = {'Content-Type': 'application/json'}

    def fetch_conversations(self, ticket_id):
        url = f"{self.base_url}/tickets/{ticket_id}/conversations"
        try:
            response = requests.get(url, auth=self.auth, headers=self.headers, timeout=10)
            if response.status_code == 200:
                conversations = response.json()
                # 💡 None 에러 방지를 위해 get(..., "") 처리 강화
                agent_replies = [conv.get("body_text") or "" for conv in conversations if conv.get("incoming") == False]
                return "\n".join(filter(None, agent_replies))
            return ""
        except Exception:
            return ""

    def fetch_resolved_tickets(self, max_pages=5): # 💡 탐색 범위를 5페이지(150개)로 대폭 확대!
        tickets = []
        page = 1
        
        while page <= max_pages:
            logger.info(f"📄 Fetching Freshdesk Tickets - Page {page}...")
            url = f"{self.base_url}/tickets"
            params = {
                "updated_since": "2023-01-01T00:00:00Z", # 💡 더 먼 과거의 데이터까지 영혼까지 끌어모음
                "include": "description",
                "page": page,
                "per_page": 30
            }
            
            try:
                response = requests.get(url, auth=self.auth, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                
                results = response.json()
                if not results:
                    break
                
                logger.info(f"🔍 [추적] {page}페이지에서 총 {len(results)}개의 티켓을 발견했습니다. 필터링을 시작합니다.")
                
                resolved_count = 0
                valid_qa_count = 0
                
                for t in results:
                    # 1차 관문: 해결된 티켓인가?
                    if t.get("status") in [4, 5]:
                        resolved_count += 1
                        ticket_id = t["id"]
                        
                        # None 방어 로직 추가
                        question = t.get("description_text") or "" 
                        answer = self.fetch_conversations(ticket_id)
                        time.sleep(0.3) 
                        
                        # 2차 관문: 질문과 답변이 모두 존재하는가?
                        if question.strip() and answer.strip():
                            valid_qa_count += 1
                            tickets.append({
                                "ticket_id": ticket_id,
                                "subject": t.get("subject", ""),
                                "question": question,
                                "answer": answer
                            })
                            
                logger.info(f"📊 [결과] {page}페이지 통계 -> 해결된 티켓: {resolved_count}개 | 최종 합격된 Q&A: {valid_qa_count}개")
                page += 1
                
            except requests.exceptions.RequestException as e:
                logger.error(f"🚨 API Request Failed: {e}")
                break
                
        return tickets

    def execute_pipeline(self):
        logger.info("🚀 Freshdesk Data Extraction 파이프라인 가동을 시작합니다.")
        
        extracted_data = self.fetch_resolved_tickets(max_pages=5) 
        logger.info(f"✅ 총 {len(extracted_data)}개의 유의미한 Q&A 세트를 성공적으로 추출했습니다.")
        
        if extracted_data:
            os.makedirs("data", exist_ok=True)
            with open("data/freshdesk_tickets.json", "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)
            logger.info("✅ 데이터가 data/freshdesk_tickets.json 파일로 저장되었습니다.")
        else:
            logger.warning("⚠️ 추출된 데이터가 없습니다. Freshdesk에 답변이 달린 해결된 티켓이 존재하는지 웹에서 직접 확인해 보십시오.")

if __name__ == "__main__":
    etl = FreshdeskETL()
    etl.execute_pipeline()