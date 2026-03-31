import asyncio
import logging
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def crawl_and_chunk_website(urls: list):
    """
    웹사이트 URL 리스트를 받아 순수 텍스트로 변환하고 RAG용 청크로 쪼개는 함수
    """
    return crawl_and_chunk_website_optimized(urls, dynamic=False)


async def crawl_dynamic_website(urls: list):
    """
    자바스크립트 렌더링이 필요한 페이지를 Playwright로 크롤링.
    """
    logger.info(f"🚀 동적 웹 크롤링 시작: 대상 URL {len(urls)}개")
    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["header", "footer", "script", "style"],
    )
    docs = await loader.aload()
    logger.info("✅ Playwright 동적 렌더링 HTML 로드 완료")
    return docs


def crawl_and_chunk_website_optimized(urls: list, dynamic: bool = False):
    """
    dynamic=True면 Playwright 기반 동적 크롤링, 아니면 AsyncHtmlLoader 사용.
    이후 공통 텍스트 정제 + 청킹 파이프라인 실행.
    """
    if dynamic:
        docs = asyncio.run(crawl_dynamic_website(urls))
    else:
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        logger.info("✅ HTML 소스 로드 완료")

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

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 타겟 URL 설정 (테스트용으로 Titan Chair 메인과 About Us 페이지 등)
    target_urls = [
        "https://titanchair.com/",
        "https://titanchair.com/pages/about-us"
        "https://titanchair.com/collections/season-sale"
        "https://titanchair.com/pages/faq"
        "https://titanchair.com/pages/shipping-returns"
        "https://titanchair.com/pages/warranty"
        "https://titanchair.com/pages/privacy-policy"
        "https://titanchair.com/pages/terms-of-service"
        "https://titanchair.com/pages/cookie-policy"
        "https://titanchair.com/pages/sitemap"
        "https://titanchair.com/pages/disclaimer"
        "https://titanchair.com/pages/contact-us"
        "https://titanchair.com/pages/faq"
        "https://titanchair.com/pages/shipping-returns"
        "https://titanchair.com/pages/warranty"
        "https://titanchair.com/pages/privacy-policy"
        "https://titanchair.com/pages/terms-of-service"
    ]
    
    # 크롤링 파이프라인 가동
    # SPA 사이트면 dynamic=True 권장
    results = crawl_and_chunk_website_optimized(target_urls, dynamic=True)
    
    # 결과물(청크) 확인
    print("\n" + "="*50)
    print("🔥 [추출된 첫 번째 데이터 조각 미리보기] 🔥")
    print("="*50)
    print(results[0].page_content)
    print("="*50)
    
    # 이 결과를 나중에 FAISS (vector DB)에 넣으면 됩니다!