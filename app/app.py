import streamlit as st
import requests
import json
import uuid

# 1. 페이지 기본 설정 (가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="Titan Chair AI Copilot",
    page_icon="💺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. 💡 [UI/UX 핵심] Custom CSS 주입 (고급스러운 다크/라이트 테마 및 여백 최적화)
st.markdown("""
<style>
    /* 상단 기본 헤더 숨기기 */
    header {visibility: hidden;}
    /* 푸터 워터마크 숨기기 */
    footer {visibility: hidden;}
    /* 챗봇 메시지 폰트 및 줄간격 최적화 */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* 타이틀 디자인 */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #2C3E50;
        padding-bottom: 10px;
        border-bottom: 2px solid #EAECEE;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# 3. 세션(Session) 관리 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# 백엔드 API 주소 (Docker 배포 시 환경 변수로 분리 가능)
API_URL = "http://localhost:8000/api/v1/chat"

# 4. 사이드바 (Sidebar) UI 구성
with st.sidebar:
    st.image("https://cdn.shopify.com/s/files/1/0086/1297/0558/files/titan_logo.png", width=150) # 로고 (임시 URL, 필요시 변경)
    st.markdown("### 🤖 AI Copilot Status")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.markdown("---")
    st.markdown("💡 **Capabilities:**")
    st.markdown("- 💺 Product Specs & Pricing")
    st.markdown("- 🛠️ Assembly & Troubleshooting")
    st.markdown("- 📜 Warranty Policies")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# 5. 메인 화면 UI
st.markdown("<h1 class='main-title'>Titan Chair AI Copilot 💺</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Ask anything about our massage chairs, warranties, or troubleshooting.</p>", unsafe_allow_html=True)

# 기존 대화 내역 렌더링
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 6. 💡 [핵심 비즈니스 로직] 채팅 입력 및 스트리밍 처리
if prompt := st.chat_input("How can I help you with your massage chair today?"):
    
    # 유저 메시지 화면에 즉시 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # AI 응답 스트리밍 영역
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        # FastAPI 서버로 요청 전송 (스트리밍 모드)
        try:
            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": st.session_state.messages[:-1] # 현재 질문 제외한 이전 기록만
            }
            
            with requests.post(API_URL, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            # 스트리밍 청크를 디코딩하여 실시간으로 화면에 이어붙임
                            decoded_chunk = chunk.decode("utf-8")
                            full_response += decoded_chunk
                            message_placeholder.markdown(full_response + "▌")
                    # 스트리밍 완료 후 커서(▌) 제거
                    message_placeholder.markdown(full_response)
                else:
                    error_msg = f"🚨 API Error: {response.status_code}"
                    message_placeholder.markdown(error_msg)
                    full_response = error_msg
                    
        except requests.exceptions.ConnectionError:
            error_msg = "🚨 Error: Cannot connect to the backend server. Is FastAPI running?"
            message_placeholder.markdown(error_msg)
            full_response = error_msg

    # AI 응답을 세션 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})