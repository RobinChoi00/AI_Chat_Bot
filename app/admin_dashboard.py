import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime

# 1. 관리자용 페이지 기본 설정 (가로로 넓게 씁니다)
st.set_page_config(
    page_title="Titan AI - Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# 💡 [보안] 초간단 하드코딩 비밀번호 (실무에서는 환경변수나 OAuth를 써야 합니다)
# 테스트용 비밀번호: admin123
def check_password():
    def password_entered():
        if st.session_state["password"] == "admin123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 보안을 위해 비밀번호 삭제
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Admin Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Admin Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop() # 비밀번호가 틀리면 여기서 화면 렌더링을 멈춥니다.

# 2. DB 연결 및 데이터 로드 함수
DB_PATH = os.path.join(os.getcwd(), "db_data", "chat_history.db")

@st.cache_data(ttl=60) # 60초마다 캐시 만료 (실시간 반영)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    # 최신 대화가 위로 오도록 정렬하여 가져옵니다.
    query = "SELECT * FROM chat_logs ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        # 시간 데이터를 보기 편하게 변환
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
    return df

# 3. 메인 대시보드 UI
st.title("📊 Titan AI Copilot - Intelligence Dashboard")
st.markdown("사장님 및 내부 마케팅 팀을 위한 실시간 AI 챗봇 모니터링 시스템입니다.")

df = load_data()

if df.empty:
    st.warning("🚨 아직 데이터베이스에 대화 기록이 없습니다. 챗봇에서 먼저 대화를 시도해 보세요!")
    st.stop()

# 우측 상단에 새로고침 버튼
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# 4. 💡 [핵심 BI] KPI 지표 한눈에 보기
col1, col2, col3 = st.columns(3)

total_chats = len(df)
unique_users = df['session_id'].nunique()
today_chats = len(df[df['date'] == datetime.today().date()])

col1.metric("💬 Total Conversations", f"{total_chats} 건", help="지금까지 AI가 답변한 총 횟수")
col2.metric("👥 Unique Sessions", f"{unique_users} 명", help="챗봇을 사용해 본 고유 고객 수 추정치")
col3.metric("🔥 Today's Chats", f"{today_chats} 건", help="오늘 발생한 대화 횟수")

st.markdown("---")

# 5. 💡 [시각화] 일자별 트래픽 추이 차트
st.subheader("📈 Daily Chat Traffic")
daily_counts = df.groupby('date').size().reset_index(name='counts')
st.bar_chart(data=daily_counts, x='date', y='counts', use_container_width=True)

# 6. 💡 [로우 데이터 검열] 실제 대화 기록 테이블
st.subheader("🕵️‍♂️ Raw Chat Logs (최신순)")
# 볼 필요 없는 id 컬럼은 숨기고 출력
display_df = df[['created_at', 'session_id', 'user_query', 'bot_response']]
st.dataframe(display_df, use_container_width=True, height=400)