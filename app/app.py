import streamlit as st
import requests
import json
import uuid
import os

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Osaki & Titan AI Agent",
    page_icon="💺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 💡 [아키텍처 확장] Multi-tenant 브랜드 설정 딕셔너리 (Config)
# 실제 쇼피파이 CDN에 올라가 있는 로고 이미지 주소로 변경하여 사용하십시오.
BRAND_CONFIG = {
    "titanchair": {
        "title": "Titan Chair AI Agent 💺",
        "logo": "https://cdn.shopify.com/s/files/1/0086/1297/0558/files/titan_logo.png",
        "domain": "https://titanchair.com"
    },
    "osakiusa": {
        "title": "Osaki USA AI Agent 🌸",
        "logo": "https://cdn.shopify.com/s/files/1/0579/8379/5374/files/osakiUSA.com_logo_black_a697883f-7819-4742-b773-af131474f374.png?v=1732232858", 
        "domain": "https://osakiusa.com"
    },
    "osakimassagechair": {
        "title": "Osaki Massage Chair AI 🛋️",
        "logo": "https://cdn.shopify.com/s/files/1/0727/1609/1700/files/logo.png?v=1677278904", 
        "domain": "https://osakimassagechair.com"
    },
    "osaki-titan": {
        "title": "Osaki & Titan Agent 🤝",
        "logo": "https://cdn.shopify.com/s/files/1/0716/6856/4151/files/Logo.png?v=1761063496", 
        "domain": "https://osaki-titan.com"
    }
}

# 💡 [핵심] URL에서 '?brand=브랜드명' 파라미터를 읽어옵니다. 없으면 기본값은 'titanchair'
current_brand_key = st.query_params.get("brand", "titanchair").lower()
current_brand = BRAND_CONFIG.get(current_brand_key, BRAND_CONFIG["titanchair"])

# 2. Custom CSS 주입
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

# 3. 네트워크 및 세션 설정
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend") 
API_URL = f"http://{BACKEND_HOST}:8000/api/v1/chat"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 사이드바 UI 구성 (💡 동적 로고 적용)
with st.sidebar:
    st.image(current_brand["logo"], width=150)
    st.markdown("### 🤖 AI Agent Status")
    st.markdown("---")
    st.markdown("💡 **Capabilities:**")
    st.markdown("- 💺 Product Specs & Pricing")
    st.markdown("- 🛠️ Assembly & Troubleshooting")
    st.markdown("- 📜 Warranty Policies")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# 5. 메인 화면 UI (💡 동적 타이틀 적용)
st.markdown(f"<h1 class='main-title'>{current_brand['title']}</h1>", unsafe_allow_html=True)
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
            MAX_HISTORY = 4
            recent_history = st.session_state.messages[-(MAX_HISTORY+1):-1] if len(st.session_state.messages) > 1 else []

            # 💡 [데이터 주입] 백엔드에 현재 도메인 전달
            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": recent_history,
                "current_domain": current_brand["domain"] 
            }
            
            with st.spinner("🧠 Scanning AI Knowledge Base..."):
                response = requests.post(API_URL, json=payload, stream=True, timeout=10)
            
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    full_response += decoded_chunk
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
                    
        except requests.exceptions.Timeout:
            error_msg = "🚨 Timeout Error: The AI server response is delayed. Please try again."
            message_placeholder.error(error_msg)
            full_response = error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"🚨 Network/API Error: Unable to connect to the server. ({e})"
            message_placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})