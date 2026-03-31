import pandas as pd
import re
import os
from pathlib import Path

# 🔥 무적의 절대 경로 설정 (어디서 실행하든 완벽 조준)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
CLEAN_CSV_PATH = BASE_DIR / "data" / "cleaned_osaki_products.csv"

def remove_html_tags(text):
    """정규식(Regex)을 사용하여 지저분한 HTML 태그를 완벽하게 날려버립니다."""
    if pd.isna(text): return ""
    clean = re.sub(r'<.*?>', ' ', str(text))
    return re.sub(r'\s+', ' ', clean).strip()

def clean_shopify_for_rag():
    print("🚀 Shopify 자식 옵션(Variants) 보존 및 RAG 평탄화 파이프라인 가동...")
    
    # 1. raw_data 폴더에서 CSV 파일 자동 탐색 (파일 이름이 바뀌어도 에러 방지)
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"🚨 {RAW_DIR} 폴더에 원본 CSV 파일이 없습니다!")
        return
    
    raw_csv_path = csv_files[0] # 첫 번째 발견된 csv 파일 사용
    print(f"📄 타겟 원본 데이터 로드 중: {raw_csv_path.name}")
    
    df = pd.read_csv(raw_csv_path, low_memory=False)

    # 2. 💡 [핵심 기술] 부모 데이터 아래로 채워넣기 (Forward Fill)
    cols_to_ffill = ['Handle', 'Title', 'Body (HTML)', 'Vendor', 'Type', 'Published']
    existing_cols = [c for c in cols_to_ffill if c in df.columns]
    df[existing_cols] = df[existing_cols].ffill()

    # 3. Active 상품만 필터링
    if 'Published' in df.columns:
        df_active = df[df['Published'].astype(str).str.lower() == 'true'].copy()
    else:
        df_active = df.copy()

    # 4. HTML 노이즈 제거
    print("🧹 HTML 웹 태그 제거 중...")
    if 'Body (HTML)' in df_active.columns:
        df_active['Clean_Description'] = df_active['Body (HTML)'].apply(remove_html_tags)
    else:
        df_active['Clean_Description'] = ""

    # 5. 💡 [데이터 압축] 동일한 상품(Handle)의 모든 옵션을 하나의 AI 문서로 융합
    rag_documents = []
    if 'Handle' in df_active.columns:
        grouped = df_active.groupby('Handle')
        
        for handle, group in grouped:
            title = group['Title'].iloc[0] if 'Title' in group.columns else handle
            desc = group['Clean_Description'].iloc[0]
            
            # 자식 옵션(Variant) 정보들을 끌어모음
            variants_info = []
            for _, row in group.iterrows():
                opt1 = row.get('Option1 Value', 'Standard')
                price = row.get('Variant Price', 'N/A')
                sku = row.get('Variant SKU', 'Unknown')
                if pd.notna(price):
                    variants_info.append(f"- Option: {opt1} | Price: ${price} | SKU: {sku}")
                    
            variants_text = "\n".join(variants_info)
            final_text = f"Product Name: {title}\nDescription: {desc}\n[Available Options & Prices]\n{variants_text}"
            
            rag_documents.append({
                "source": handle,
                "content": final_text
            })
    else:
        print("🚨 'Handle' 컬럼이 없어 그룹화를 진행할 수 없습니다.")
        return

    # 6. 정제된 데이터를 data/ 폴더에 저장
    df_rag = pd.DataFrame(rag_documents)
    df_rag.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8')
    
    print(f"✅ 총 {len(df_rag)}개의 고유 상품을 기준으로 옵션 정보 압축 성공!")
    print(f"💾 완벽한 AI 학습용 데이터가 저장되었습니다: {CLEAN_CSV_PATH}")

if __name__ == "__main__":
    clean_shopify_for_rag()