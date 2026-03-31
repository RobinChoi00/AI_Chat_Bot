import os
import json
import asyncio
import logging
from typing import List
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔥 [아키텍처] 무적의 절대 경로 계산 (scripts/ 에서 실행해도 data/ 를 정확히 조준)
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path)) # AI-Chat-Project 루트
DATA_DIR = os.path.join(BASE_DIR, "data")

def crawl_and_chunk_website(urls: list):
    return crawl_and_chunk_website_optimized(urls, dynamic=False)

async def crawl_dynamic_website(urls: list):
    logger.info(f"🚀 동적 웹 크롤링 시작: 대상 URL {len(urls)}개")
    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["header", "footer", "script", "style", "nav"], # nav(메뉴) 노이즈 추가 제거
    )
    docs = await loader.aload()
    logger.info("✅ Playwright 동적 렌더링 HTML 로드 완료")
    return docs

def _crawl_static_website(urls: List[str]):
    """정적 HTML 페이지용 고속 로더."""
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    logger.info("✅ HTML 소스 로드 완료")
    return docs


async def crawl_and_chunk_website_optimized_async(
    urls: List[str],
    dynamic: bool = False,
    fallback_to_static: bool = True,
):
    """
    async 환경 전용 엔트리.
    dynamic=True일 때 Playwright 시도 후 실패 시 정적 로더로 폴백 가능.
    """
    if dynamic:
        try:
            docs = await crawl_dynamic_website(urls)
            if not docs and fallback_to_static:
                logger.warning("⚠️ 동적 크롤링 결과가 비어 있어 정적 크롤링으로 폴백합니다.")
                docs = _crawl_static_website(urls)
        except Exception as e:
            if not fallback_to_static:
                raise
            logger.warning(f"⚠️ 동적 크롤링 실패({e}) -> 정적 크롤링으로 폴백합니다.")
            docs = _crawl_static_website(urls)
    else:
        docs = _crawl_static_website(urls)

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    logger.info("✅ HTML 노이즈 제거 및 텍스트 추출 완료")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_docs = text_splitter.split_documents(docs_transformed)
    logger.info(f"✅ 총 {len(chunked_docs)}개의 청크(Chunk) 조각으로 분할 완료!")
    return chunked_docs


def crawl_and_chunk_website_optimized(
    urls: List[str],
    dynamic: bool = False,
    fallback_to_static: bool = True,
):
    """
    sync 환경 전용 엔트리.
    - 일반 스크립트 실행에서는 asyncio.run 사용
    - 이미 실행 중인 이벤트 루프 환경에서는 명시적 안내 예외 발생
    """
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "이미 실행 중인 이벤트 루프가 감지되었습니다. "
            "이 환경에서는 crawl_and_chunk_website_optimized_async(...)를 await 하세요."
        )
    except RuntimeError as e:
        if "실행 중인 이벤트 루프" in str(e):
            raise
        return asyncio.run(
            crawl_and_chunk_website_optimized_async(
                urls=urls,
                dynamic=dynamic,
                fallback_to_static=fallback_to_static,
            )
        )

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 🚨 콤마(,) 누락 완벽 수정
    target_urls = [
        "https://titanchair.com/",
        "https://titanchair.com/pages/about-us",
        "https://titanchair.com/pages/buyers-guide",
        "https://titanchair.com/collections/season-sale",
        "https://titanchair.com/pages/health-benefits",
        "https://titanchair.com/pages/faq",
        "https://titanchair.com/pages/q-a-titanchair-com"
    ]
    
    # 크롤링 파이프라인 가동 (Playwright 사용)
    results = crawl_and_chunk_website_optimized(target_urls, dynamic=True)
    
    # 🔥 [추가됨] 결과를 JSON 파일로 `data/` 폴더에 안전하게 저장
    os.makedirs(DATA_DIR, exist_ok=True) # data 폴더가 없으면 생성
    output_file = os.path.join(DATA_DIR, "web_crawled_data.json")
    
    saved_data = []
    for doc in results:
        saved_data.append({
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(saved_data, f, ensure_ascii=False, indent=4)
        
    if results:
        print("\n" + "="*50)
        print("🔥 [크롤링 성공] 첫 번째 청크 미리보기 🔥")
        print("="*50)
        print(results[0].page_content)
        print("="*50)
    else:
        logger.warning("⚠️ 추출된 결과가 없습니다. URL 또는 크롤링 설정을 확인하세요.")

    logger.info(f"💾 웹 데이터가 성공적으로 저장되었습니다: {output_file}")