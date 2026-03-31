import pandas as pd
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_specification_data():
    base_dir = Path(__file__).resolve().parent
    
    # 1. [핵심] 아키텍처 변경에 따른 파일 경로 동적 맵핑
    raw_file_path = base_dir / "raw_data" / "Specification_Massage Chair - Massage Chair.csv"
    cleaned_file_path = base_dir / "data" / "cleaned_specification.csv"

    # [엄격 검증] 원본 파일 서킷 브레이커
    if not raw_file_path.exists():
        logger.error(f"🚨 raw_data 폴더 안에 원본 파일을 찾을 수 없습니다: {raw_file_path.name}")
        return

    try:
        # 2. Extract (추출)
        df = pd.read_csv(raw_file_path)
        logger.info(f"✅ 원본 데이터 로드 완료: 총 {len(df)}행, {len(df.columns)}열")

        # 3. Transform (변환): 제조사 원가 및 사내 대외비 컬럼 리스트
        columns_to_drop = [
            "Manufacturer",
            "Manufacturer Code",
            "Design Rep",
            "Video Rep",
            "Coder Rep",
            "etc"
        ]

        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        df_cleaned = df.drop(columns=existing_cols_to_drop)

        # 4. Load (적재)
        cleaned_file_path.parent.mkdir(exist_ok=True) 
        df_cleaned.to_csv(cleaned_file_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ 데이터 정제 완료: {len(existing_cols_to_drop)}개의 보안 컬럼이 영구 삭제되었습니다.")
        logger.info(f"💾 적재 완료: {cleaned_file_path.relative_to(base_dir)}")

    except Exception as e:
        logger.error(f"🚨 데이터 정제 파이프라인 에러 발생: {e}")

if __name__ == "__main__":
    clean_specification_data()