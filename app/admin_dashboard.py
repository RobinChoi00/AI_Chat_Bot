import streamlit as st
import pandas as pd
import sqlite3
import os
import json # 💡 [추가] JSON 파싱을 위한 모듈
from datetime import datetime
import extra_streamlit_components as stx
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(
    page_title="Titan AI - Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# 💡 [보안/아키텍처] 브라우저 쿠키 매니저 초기화 (UI 컴포넌트는 캐싱하지 않음)
cookie_manager = stx.CookieManager(key="admin_cookie_manager")

# ==========================================
# 💡 [보안] 환경변수에서 관리자 자격 증명 동적 로드 (하드코딩 방어)
# ==========================================
load_dotenv(override=True)

try:
    # .env 파일에서 ADMIN_CREDENTIALS 값을 가져옵니다. (없으면 빈 딕셔너리 반환)
    creds_str = os.environ.get("ADMIN_CREDENTIALS", "{}")
    VALID_CREDENTIALS = json.loads(creds_str)
except json.JSONDecodeError:
    VALID_CREDENTIALS = {}
    st.error("🚨 [Security Configuration Error] Invalid credentials format in .env file.")

# ==========================================

# 💡 [핵심] 쿠키 기반 영구 로그인 검증 로직
def check_login():
    # 1. 브라우저 쿠키에 인증 토큰이 살아있는지 먼저 확인 (새로고침 방어)
    if cookie_manager.get(cookie="admin_auth_token") == "authenticated":
        st.session_state["logged_in"] = True
        return True

    # 2. 쿠키가 없다면 기본 세션 상태 확인
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 3. 로그인 창 렌더링
    if not st.session_state["logged_in"]:
        st.markdown("<h2 style='text-align: center;'>🔒 Titan AI Admin Login</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("---")
            input_id = st.text_input("👤 User ID")
            input_pw = st.text_input("🔑 Password", type="password")
            
            if st.button("Login", use_container_width=True):
                if input_id in VALID_CREDENTIALS and VALID_CREDENTIALS[input_id] == input_pw:
                    st.session_state["logged_in"] = True
                    # 💡 로그인 성공 시 브라우저 쿠키 발급 (max_age: 86400초 = 1일 유지)
                    cookie_manager.set("admin_auth_token", "authenticated", max_age=86400)
                    st.rerun()
                else:
                    st.error("🚨 Invalid User ID or Password.")
        return False
    return True

# 로그인되지 않으면 데이터 로딩 원천 차단
if not check_login():
    st.stop()

# 💡 [UX] 우측 상단 로그아웃 버튼 (쿠키 삭제)
col_blank, col_logout = st.columns([9, 1])
with col_logout:
    if st.button("🚪 Logout", use_container_width=True):
        cookie_manager.delete("admin_auth_token")
        st.session_state["logged_in"] = False
        st.rerun()

# 2. Database Connection & Data Loading
DB_PATH = os.path.join(os.getcwd(), "db_data", "chat_history.db")

DOMAIN_LABELS = {
    "All Sites": None,
    "Titan Chair": "titanchair",
    "Osaki USA": "osakiusa",
    "Osaki Massage Chair": "osakimassagechair",
}

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM chat_logs ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        if 'domain' not in df.columns:
            df['domain'] = 'unknown'
        df['brand'] = df['domain'].apply(_resolve_brand_label)
    return df

def _resolve_brand_label(domain: str) -> str:
    d = (domain or "").lower()
    if "titanchair" in d:
        return "Titan Chair"
    if "osakiusa" in d or "osaki-usa" in d:
        return "Osaki USA"
    if "osakimassage" in d:
        return "Osaki Massage Chair"
    return "Unknown"

# 3. Main Dashboard UI
st.title("📊 Titan AI Agent - Intelligence Dashboard")
st.markdown("Real-time AI Chatbot Monitoring System for Management and Marketing Teams.")

df = load_data()

if df.empty:
    st.warning("🚨 No chat records found in the database yet. Try starting a conversation in the chatbot first!")
    st.stop()

# Filter & Refresh row
filter_col, refresh_col = st.columns([3, 1])
with filter_col:
    selected_label = st.selectbox("🌐 Filter by Brand", list(DOMAIN_LABELS.keys()))
with refresh_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

keyword = DOMAIN_LABELS[selected_label]
if keyword:
    filtered = df[df['domain'].str.contains(keyword, case=False, na=False)]
else:
    filtered = df

st.markdown("---")

# 4. KPI Metrics
col1, col2, col3, col4 = st.columns(4)

total_chats = len(filtered)
unique_users = filtered['session_id'].nunique()
today_chats = len(filtered[filtered['date'] == datetime.today().date()])

col1.metric("💬 Total Conversations", f"{total_chats}", help="Total number of AI responses generated so far.")
col2.metric("👥 Unique Sessions", f"{unique_users}", help="Estimated number of unique customer sessions.")
col3.metric("🔥 Today's Chats", f"{today_chats}", help="Number of conversations initiated today.")

if keyword is None and 'brand' in df.columns:
    brand_counts = df['brand'].value_counts()
    summary_parts = [f"{b}: {c}" for b, c in brand_counts.items()]
    col4.metric("🏷️ Top Brand", brand_counts.index[0] if len(brand_counts) > 0 else "N/A",
                help=" | ".join(summary_parts))
else:
    col4.metric("🏷️ Current Filter", selected_label)

st.markdown("---")

# 5. Per-brand breakdown (only in All Sites view)
if keyword is None and 'brand' in df.columns:
    st.subheader("🏢 Conversations by Brand")
    brand_summary = df.groupby('brand').agg(
        conversations=('id', 'count'),
        unique_sessions=('session_id', 'nunique'),
    ).sort_values('conversations', ascending=False)
    st.dataframe(brand_summary, use_container_width=True)
    st.bar_chart(data=brand_summary, y='conversations', use_container_width=True)
    st.markdown("---")

# 6. Daily Chat Traffic
st.subheader("📈 Daily Chat Traffic")
daily_counts = filtered.groupby('date').size().reset_index(name='counts')
st.bar_chart(data=daily_counts, x='date', y='counts', use_container_width=True)

# 7. Raw Data Table
st.subheader("🕵️‍♂️ Raw Chat Logs (Newest First)")
display_cols = ['created_at', 'brand', 'session_id', 'user_query', 'bot_response']
display_df = filtered[[c for c in display_cols if c in filtered.columns]]
st.dataframe(display_df, use_container_width=True, height=400)