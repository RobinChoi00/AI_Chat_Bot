import os
import requests
import logging
import time
import json
from dotenv import load_dotenv

# 로깅 설정 (운영 환경 필수)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class FreshdeskETL:
    def __init__(self):
        """API 크레덴셜 및 엔드포인트 초기화"""
        self.domain = os.environ.get("FRESHDESK_DOMAIN")
        self.api_key = os.environ.get("FRESHDESK_API_KEY")
        
        if not self.domain or not self.api_key:
            logger.error("🚨 System Error: .env 파일에 Freshdesk 크레덴셜이 누락되었습니다.")
            raise ValueError("Missing Freshdesk Credentials")
            
        self.base_url = f"https://{self.domain}/api/v2"
        # Freshdesk는 API Key를 아이디로, 비밀번호는 임의의 문자('X')를 사용하는 Basic Auth 규격을 따릅니다.
        self.auth = (self.api_key, 'X')
        self.headers = {'Content-Type': 'application/json'}

    def fetch_resolved_tickets(self, max_pages=3):
        """해결 완료된(Resolved/Closed) 고객 CS 티켓만 추출 (Pagination 적용)"""
        tickets = []
        page = 1
        
        while page <= max_pages:
            logger.info(f"📄 Fetching Freshdesk Tickets - Page {page}...")
            # 엔드포인트(Endpoint): 해결된 티켓(status 4, 5)만 필터링하여 요청
            url = f"{self.base_url}/search/tickets?query=\"status:4 OR status:5\"&page={page}"
            
            try:
                response = requests.get(url, auth=self.auth, headers=self.headers, timeout=10)
                response.raise_for_status() # HTTP 200 OK가 아니면 즉시 에러 발생 (Circuit Breaker)
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break # 더 이상 가져올 데이터가 없으면 루프 종료
                    
                tickets.extend(results)
                page += 1
                time.sleep(1) # API Rate Limit(호출 제한) 방어용 딜레이 (Trade-off)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"🚨 API Request Failed: {e}")
                break
                
        return tickets

    def execute_pipeline(self):
        """전체 ETL 파이프라인 실행 및 저장"""
        logger.info("🚀 Freshdesk Data Extraction 파이프라인 가동을 시작합니다.")
        
        tickets = self.fetch_resolved_tickets(max_pages=2) # 테스트용으로 2페이지만 추출
        logger.info(f"✅ 총 {len(tickets)}개의 CS 티켓 메타데이터를 성공적으로 추출했습니다.")
        
        # 1차 정제 및 JSON 파일로 직렬화(Serialization)
        extracted_data = []
        for t in tickets:
            extracted_data.append({
                "ticket_id": t["id"],
                "subject": t["subject"],
                "description_text": t["description_text"], # HTML 태그가 제거된 순수 본문
                "created_at": t["created_at"]
            })
            
        # 데이터를 data 폴더에 저장 (이후 벡터 DB 인덱싱을 위함)
        os.makedirs("data", exist_ok=True)
        with open("data/freshdesk_tickets.json", "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
            
        logger.info("✅ 데이터가 data/freshdesk_tickets.json 파일로 저장되었습니다.")

if __name__ == "__main__":
    etl = FreshdeskETL()
    etl.execute_pipeline()