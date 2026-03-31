import streamlit as st
import requests
import json
import uuid
import os # 💡 환경 변수 조작을 위해 추가

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Titan Chair AI Copilot",
    page_icon="💺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. 💡 [UI/UX 핵심] Custom CSS 주입
st.markdown("""
<style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
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

# 3. 💡 [네트워크 아키텍처 핵심] 도커와 로컬 환경 동시 대응
# 도커 환경이면 서비스 이름인 'backend'를 쓰고, 로컬 테스트 시에는 'localhost'를 사용합니다.
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend") 
API_URL = f"http://{BACKEND_HOST}:8000/api/v1/chat"

# 세션 관리 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 사이드바 UI 구성
with st.sidebar:
    st.image("https://cdn.shopify.com/s/files/1/0086/1297/0558/files/titan_logo.png", width=150)
    st.markdown("### 🤖 AI Copilot Status")
    st.caption(f"Backend Host: `{BACKEND_HOST}`") # 💡 현재 연결된 대상 확인용
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

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 6. 채팅 입력 및 스트리밍 처리
if prompt := st.chat_input("How can I help you with your massage chair today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": st.session_state.messages[:-1]
            }
            
            # 스트리밍 요청 가동
            with requests.post(API_URL, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            decoded_chunk = chunk.decode("utf-8")
                            full_response += decoded_chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    error_msg = f"🚨 API Error: {response.status_code}"
                    message_placeholder.markdown(error_msg)
                    full_response = error_msg
                    
        except requests.exceptions.ConnectionError:
            error_msg = f"🚨 Error: Backend server at `{API_URL}` is unreachable."
            message_placeholder.markdown(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})