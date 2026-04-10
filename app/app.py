import streamlit as st
import requests
import json
import uuid
import os
import re

# --- [UI Component] 배송 타임라인 렌더링 함수 ---
def render_tracking_timeline(tracking_number, company, status, tracking_url):
    """
    배송 상태 JSON 데이터를 받아 동적인 HTML 타임라인을 Streamlit에 주입합니다.
    """
    step1 = "active" if status == "PROCESSING" else "completed"
    step2 = "active" if status == "IN_TRANSIT" else ("completed" if status == "FULFILLED" else "")
    step3 = "active" if status == "FULFILLED" else ""
    
    html_content = f"""
    <div style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin: 15px 0; border: 1px solid #e9ecef; font-family: sans-serif;">
        <h4 style="margin-top: 0; color: #212529;">📦 실시간 배송 조회 ({company})</h4>
        <p style="color: #6c757d; font-size: 0.9em;">운송장 번호: <strong>{tracking_number}</strong></p>
        
        <div style="border-left: 3px solid #dee2e6; padding-left: 20px; margin-left: 10px; margin-top: 20px;">
            <div style="margin-bottom: 20px; position: relative; color: {'#28a745' if step1=='completed' else '#007bff' if step1=='active' else '#6c757d'}; font-weight: {'bold' if step1 else 'normal'};">
                <div style="position: absolute; left: -28px; top: 4px; width: 14px; height: 14px; border-radius: 50%; background: {'#28a745' if step1=='completed' else '#007bff' if step1=='active' else '#dee2e6'};"></div>
                ✅ 주문 및 출고 준비
            </div>
            <div style="margin-bottom: 20px; position: relative; color: {'#28a745' if step2=='completed' else '#007bff' if step2=='active' else '#6c757d'}; font-weight: {'bold' if step2 else 'normal'};">
                <div style="position: absolute; left: -28px; top: 4px; width: 14px; height: 14px; border-radius: 50%; background: {'#28a745' if step2=='completed' else '#007bff' if step2=='active' else '#dee2e6'};"></div>
                🚚 화물 터미널 이동 및 배송 중
            </div>
            <div style="position: relative; color: {'#007bff' if step3=='active' else '#6c757d'}; font-weight: {'bold' if step3 else 'normal'};">
                <div style="position: absolute; left: -28px; top: 4px; width: 14px; height: 14px; border-radius: 50%; background: {'#007bff' if step3=='active' else '#dee2e6'};"></div>
                🏡 고객님 댁 도착 (설치 완료)
            </div>
        </div>
        
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #dee2e6; text-align: center;">
            <a href="{tracking_url}" target="_blank" style="background-color: #000; color: #fff; padding: 10px 20px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">{company} 공식 홈페이지에서 상세보기 ➡️</a>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def parse_tracking_response(content: str):
    """Parse deterministic tracking text response from backend."""
    required_markers = ["- Current Status:", "- Current Location:", "- Current Hub:", "- Estimated Delivery:"]
    if not all(marker in content for marker in required_markers):
        return None

    def extract(label: str):
        match = re.search(rf"{re.escape(label)}\s*(.+)", content)
        return match.group(1).strip() if match else ""

    events = []
    if "Recent Tracking Timeline:" in content:
        timeline_text = content.split("Recent Tracking Timeline:", 1)[1]
        for line in timeline_text.splitlines():
            if line.strip().startswith("- "):
                events.append(line.strip()[2:])

    return {
        "status": extract("- Current Status:"),
        "location": extract("- Current Location:"),
        "hub": extract("- Current Hub:"),
        "eta": extract("- Estimated Delivery:"),
        "last_event": extract("- Last Carrier Event:"),
        "carrier": extract("- Carrier:"),
        "tracking_number": extract("- Tracking Number:"),
        "tracking_url": extract("- Live Tracking URL:"),
        "events": events[:3],
    }

def render_tracking_summary_card(parsed: dict):
    """Render modern tracking summary card."""
    tracking_url = parsed.get("tracking_url", "")
    status = parsed.get("status", "UNKNOWN")

    st.markdown(
        f"""
        <div class="tracking-card">
            <div class="tracking-card-head">
                <div class="tracking-title">📦 Real-time Delivery Update</div>
                <div class="tracking-status">{status}</div>
            </div>
            <div class="tracking-grid">
                <div class="tracking-item"><span>Current Location</span><strong>{parsed.get("location", "-")}</strong></div>
                <div class="tracking-item"><span>Current Hub</span><strong>{parsed.get("hub", "-")}</strong></div>
                <div class="tracking-item"><span>Estimated Delivery</span><strong>{parsed.get("eta", "-")}</strong></div>
                <div class="tracking-item"><span>Last Event</span><strong>{parsed.get("last_event", "-")}</strong></div>
                <div class="tracking-item"><span>Carrier</span><strong>{parsed.get("carrier", "-")}</strong></div>
                <div class="tracking-item"><span>Tracking Number</span><strong>{parsed.get("tracking_number", "-")}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if parsed.get("events"):
        st.markdown("**Recent Timeline**")
        for row in parsed["events"]:
            st.markdown(f"- {row}")

    if tracking_url:
        st.markdown(f"[Open carrier tracking page]({tracking_url})")


# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Osaki & Titan AI Agent",
    page_icon="💺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Multi-tenant 브랜드 설정 딕셔너리
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

current_brand_key = st.query_params.get("brand", "titanchair").lower()
current_brand = BRAND_CONFIG.get(current_brand_key, BRAND_CONFIG["titanchair"])

# 2. Custom CSS 주입 (애니메이션 효과 추가)
st.markdown("""
<style>
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #020617 60%, #01040f 100%);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        animation: fadeInUp 0.4s ease-out forwards;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.75);
    }

    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(30, 41, 59, 0.85);
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(15, 23, 42, 0.85);
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: #e2e8f0 !important;
    }

    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #f8fafc;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
        margin-bottom: 20px;
    }

    .sub-copy {
        text-align: center;
        color: #94a3b8;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    .tracking-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(125, 211, 252, 0.3);
        border-radius: 14px;
        padding: 14px;
        margin: 8px 0 10px;
    }

    .tracking-card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }

    .tracking-title {
        color: #e0f2fe;
        font-weight: 700;
    }

    .tracking-status {
        color: #0f172a;
        background: #7dd3fc;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .tracking-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }

    .tracking-item {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 8px 10px;
    }

    .tracking-item span {
        display: block;
        font-size: 11px;
        color: #93c5fd;
        margin-bottom: 2px;
    }

    .tracking-item strong {
        color: #f8fafc;
        font-size: 13px;
    }

    @media (max-width: 768px) {
        .tracking-grid {
            grid-template-columns: 1fr;
        }
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

# 4. 사이드바 UI 구성
with st.sidebar:
    st.image(current_brand["logo"], width=150)
    st.markdown("### 🤖 AI Agent Status")
    st.markdown("---")
    st.markdown("💡 **Capabilities:**")
    st.markdown("- 💺 Product Specs & Pricing")
    st.markdown("- 🛠️ Assembly & Troubleshooting")
    st.markdown("- 📦 Real-time Order Tracking")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# 5. 메인 화면 UI 및 과거 대화 기록 렌더링 (💡 가로채기 1번)
st.markdown(f"<h1 class='main-title'>{current_brand['title']}</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-copy'>Ask anything about our massage chairs, warranties, or order tracking.</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        content = msg["content"]
        if msg["role"] == "assistant":
            parsed = parse_tracking_response(content)
            if parsed:
                render_tracking_summary_card(parsed)
                continue

        if msg["role"] == "assistant" and "```json\n{" in content and "tracking_number" in content:
            try:
                json_str = content.split("```json\n")[1].split("\n```")[0]
                data = json.loads(json_str)
                render_tracking_timeline(
                    tracking_number=data.get("tracking_number", ""),
                    company=data.get("company", ""),
                    status=data.get("status", "PROCESSING"),
                    tracking_url=data.get("tracking_url", "#")
                )
            except Exception:
                st.markdown(content)
        else:
            st.markdown(content)

# 6. 채팅 입력 및 스트리밍 처리 (💡 가로채기 2번)
if prompt := st.chat_input("What can i help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            MAX_HISTORY = 4
            recent_history = st.session_state.messages[-(MAX_HISTORY+1):-1] if len(st.session_state.messages) > 1 else []

            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": recent_history,
                "current_domain": current_brand["domain"] 
            }
            
            with st.spinner("🧠 Processing..."):
                response = requests.post(API_URL, json=payload, stream=True, timeout=10)
            
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    full_response += decoded_chunk
                    message_placeholder.markdown(full_response + "▌")
            
            parsed = parse_tracking_response(full_response)
            if parsed:
                message_placeholder.empty()
                with message_placeholder.container():
                    render_tracking_summary_card(parsed)
            elif "```json\n{" in full_response and "tracking_number" in full_response:
                try:
                    json_str = full_response.split("```json\n")[1].split("\n```")[0]
                    data = json.loads(json_str)
                    
                    message_placeholder.empty()
                    with message_placeholder.container():
                        render_tracking_timeline(
                            tracking_number=data.get("tracking_number", ""),
                            company=data.get("company", ""),
                            status=data.get("status", "PROCESSING"),
                            tracking_url=data.get("tracking_url", "#")
                        )
                except Exception:
                    message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown(full_response)
                    
        except requests.exceptions.Timeout:
            error_msg = "🚨 Timeout Error: The AI server response is delayed. Please try again."
            message_placeholder.error(error_msg)
            full_response = error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"🚨 Network/API Error: Unable to connect to the server. ({e})"
            message_placeholder.error(error_msg)
            full_response = error_msg

    # DB 및 과거 기록 렌더링을 위해 세션에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})