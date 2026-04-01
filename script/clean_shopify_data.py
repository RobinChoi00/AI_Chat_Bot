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
    
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"🚨 {RAW_DIR} 폴더에 원본 CSV 파일이 없습니다!")
        return
    
    raw_csv_path = csv_files[0]
    print(f"📄 타겟 원본 데이터 로드 중: {raw_csv_path.name}")
    
    df = pd.read_csv(raw_csv_path, low_memory=False)

    # 2. 부모 데이터 아래로 채워넣기 (Forward Fill)
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

    # 5. 💡 [데이터 압축 및 메타데이터 추출]
    rag_documents = []
    if 'Handle' in df_active.columns:
        grouped = df_active.groupby('Handle')
        
        for handle, group in grouped:
            title = group['Title'].iloc[0] if 'Title' in group.columns else handle
            desc = group['Clean_Description'].iloc[0]
            
            # 💡 [신규] 메타데이터 추출 (Type, Vendor)
            product_type = str(group['Type'].iloc[0]) if 'Type' in group.columns and pd.notna(group['Type'].iloc[0]) else "Unknown"
            vendor = str(group['Vendor'].iloc[0]) if 'Vendor' in group.columns and pd.notna(group['Vendor'].iloc[0]) else "Unknown"
            
            # 💡 [신규] 가격 메타데이터 추출 (필터링을 위해 수치화, 여러 옵션 중 최저가 기준)
            if 'Variant Price' in group.columns:
                # 문자가 섞여 있을 경우를 대비해 숫자로 강제 변환 (결측치는 제외)
                prices = pd.to_numeric(group['Variant Price'], errors='coerce').dropna()
                base_price = float(prices.min()) if not prices.empty else 0.0
            else:
                base_price = 0.0
            
            # 자식 옵션(Variant) 텍스트화
            variants_info = []
            for _, row in group.iterrows():
                price = row.get('Variant Price', 'N/A')
                
                # 💡 [신규] 재고(Inventory) 수량 추출
                # CSV 원본의 컬럼명이 'Variant Inventory Qty'가 맞는지 반드시 확인하십시오!
                inventory = row.get('Variant Inventory Qty', 'Unknown')
                
                try:
                    inv_count = int(float(inventory))
                    stock_status = f"{inv_count} in stock" if inv_count > 0 else "Out of Stock"
                except (ValueError, TypeError):
                    stock_status = "Stock info unavailable"
                
                # 💡 Option 1, 2, 3 동적 추출
                opts = []
                for i in range(1, 4):
                    opt_name = row.get(f'Option{i} Name', '')
                    opt_val = row.get(f'Option{i} Value', '')
                    
                    # 옵션 이름과 값이 모두 존재하고 비어있지 않은 경우에만 추가
                    if pd.notna(opt_name) and pd.notna(opt_val) and str(opt_val).strip() != '':
                        opts.append(f"{opt_name}: {opt_val}")
                
                # 추출한 옵션들을 슬래시(/)로 예쁘게 연결
                opt_string = " / ".join(opts) if opts else "Standard Option"
                
                # 💡 [핵심] 텍스트 끝에 재고(Availability) 상태를 강력하게 주입!
                if pd.notna(price):
                    variants_info.append(f"- [{opt_string}] => Total Price: ${price} | Availability: {stock_status}")
                    
            variants_text = "\n".join(variants_info)
            final_text = f"Product Name: {title}\nDescription: {desc}\n[Available Options & Prices]\n{variants_text}"
            
            # 💡 [핵심] CSV에 메타데이터 컬럼이 분리되어 저장되도록 딕셔너리에 추가
            rag_documents.append({
                "source": handle,
                "content": final_text,
                "price": base_price,           # 숫자형 메타데이터
                "product_type": product_type,  # 문자형 메타데이터 (카테고리)
                "vendor": vendor               # 문자형 메타데이터 (브랜드)
            })
    else:
        print("🚨 'Handle' 컬럼이 없어 그룹화를 진행할 수 없습니다.")
        return

    # 6. 정제된 데이터를 data/ 폴더에 저장
    # 여기서 저장할 때, 자동으로 'price', 'product_type', 'vendor' 컬럼이 생성됩니다.
    df_rag = pd.DataFrame(rag_documents)
    
    # 💡 저장될 컬럼의 데이터 타입 강제 고정
    df_rag['price'] = df_rag['price'].astype(float)
    
    df_rag.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8')
    
    print(f"✅ 총 {len(df_rag)}개의 고유 상품을 기준으로 옵션 정보 압축 및 메타데이터 추출 성공!")
    print(f"💾 완벽한 AI 학습용 데이터가 저장되었습니다: {CLEAN_CSV_PATH}")

if __name__ == "__main__":
    clean_shopify_for_rag()