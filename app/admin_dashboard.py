import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import extra_streamlit_components as stx

# 1. Page Configuration
st.set_page_config(
    page_title="Titan AI - Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# 💡 [보안/아키텍처] 브라우저 쿠키 매니저 초기화
@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()

cookie_manager = get_cookie_manager()

# Authorized Credentials
VALID_CREDENTIALS = {
    "admin": "titan1212",
    "jiwoong": "osaki1234"
}

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
    return df

# 3. Main Dashboard UI
st.title("📊 Titan AI Copilot - Intelligence Dashboard")
st.markdown("Real-time AI Chatbot Monitoring System for Management and Marketing Teams.")

df = load_data()

if df.empty:
    st.warning("🚨 No chat records found in the database yet. Try starting a conversation in the chatbot first!")
    st.stop()

# Refresh Data Button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# 4. KPI Metrics
col1, col2, col3 = st.columns(3)

total_chats = len(df)
unique_users = df['session_id'].nunique()
today_chats = len(df[df['date'] == datetime.today().date()])

col1.metric("💬 Total Conversations", f"{total_chats}", help="Total number of AI responses generated so far.")
col2.metric("👥 Unique Sessions", f"{unique_users}", help="Estimated number of unique customer sessions.")
col3.metric("🔥 Today's Chats", f"{today_chats}", help="Number of conversations initiated today.")

st.markdown("---")

# 5. Visualizations
st.subheader("📈 Daily Chat Traffic")
daily_counts = df.groupby('date').size().reset_index(name='counts')
st.bar_chart(data=daily_counts, x='date', y='counts', use_container_width=True)

# 6. Raw Data Table
st.subheader("🕵️‍♂️ Raw Chat Logs (Newest First)")
display_df = df[['created_at', 'session_id', 'user_query', 'bot_response']]
st.dataframe(display_df, use_container_width=True, height=400)