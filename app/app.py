import streamlit as st
import requests
import json
import uuid
import os

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
    # 1. 사용자 메시지 화면 출력 및 세션 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2. AI 응답 스트리밍 영역
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 💡 [아키텍처 최적화] Sliding Window 기법
            # 전체 대화가 아닌 '최근 N개의 대화'만 잘라서 전송 (토큰 낭비 방지)
            MAX_HISTORY = 4
            recent_history = st.session_state.messages[-(MAX_HISTORY+1):-1] if len(st.session_state.messages) > 1 else []

            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": recent_history
            }
            
            # 💡 [UX & 안정성 핵심] 로딩 스피너 및 타임아웃(Timeout) 적용
            with st.spinner("🧠 Scanning Titan AI Knowledge Base..."):
                # stream=True 시 연결 타임아웃만 10초로 제한하여 무한 대기 방지
                response = requests.post(API_URL, json=payload, stream=True, timeout=10)
            
            # HTTP 4xx, 5xx 에러를 명시적으로 캐치
            response.raise_for_status()

            # 💡 [성능 최적화] 청크 사이즈를 지정하여 네트워크 버퍼링 방지
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    full_response += decoded_chunk
                    # 시각적 피드백: 깜빡이는 커서(▌) 효과 주입
                    message_placeholder.markdown(full_response + "▌")
            
            # 최종 완성본 출력 (커서 제거)
            message_placeholder.markdown(full_response)
                    
        except requests.exceptions.Timeout:
            error_msg = "🚨 Timeout Error: The AI server response is delayed. Please try again."
            message_placeholder.error(error_msg)
            full_response = error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"🚨 Network/API Error: Unable to connect to the server. ({e})"
            message_placeholder.error(error_msg)
            full_response = error_msg

    # 3. AI 최종 응답 세션 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})