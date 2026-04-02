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

# 💡 [보안] 허용된 ID와 비밀번호 목록 (Dictionary 구조)
# 실무에서는 절대 이렇게 하드코딩하지 않지만, MVP 런칭을 위해 임시 구성합니다.
VALID_CREDENTIALS = {
    "admin": "titan1212",    # 💡 지웅님이 설정했던 비밀번호 유지
    "jiwoong": "osaki1234"   # 💡 지웅님 개인 계정 추가
}

# 💡 [핵심] 세션 기반 ID/PW 검증 로직
def check_login():
    # 세션에 로그인 상태가 없으면 False로 초기화
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # 로그인되지 않은 상태라면 중앙 정렬된 로그인 폼을 보여줌
    if not st.session_state["logged_in"]:
        st.markdown("<h2 style='text-align: center;'>🔒 Titan AI Admin Login</h2>", unsafe_allow_html=True)
        
        # UI 레이아웃 가운데 정렬을 위해 3등분 컬럼 사용
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("---")
            input_id = st.text_input("👤 User ID")
            input_pw = st.text_input("🔑 Password", type="password")
            
            # 로그인 버튼 클릭 시 이벤트 처리
            if st.button("Login", use_container_width=True):
                # ID가 존재하고, 비밀번호가 일치하는지 검증 (Authentication)
                if input_id in VALID_CREDENTIALS and VALID_CREDENTIALS[input_id] == input_pw:
                    st.session_state["logged_in"] = True
                    st.rerun() # 화면 새로고침하여 대시보드 본문 렌더링
                else:
                    st.error("🚨 Invalid User ID or Password.")
        return False
    return True

# 로그인 검증 실패 시 여기서 화면 렌더링을 완전히 차단(Block)
if not check_login():
    st.stop()

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