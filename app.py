import streamlit as st
import requests
import uuid

# --- [1] Page Config (글로벌 스탠다드 UX 설정) ---
st.set_page_config(page_title="Osaki AI Support", page_icon="🤖", layout="centered")
st.title("💆‍♂️ Osaki Massage Chair AI")

# --- [2] Session State Management (상태 유지 아키텍처) ---
# [엄격 검증] 변수 초기화가 무조건 가장 먼저 실행되어야 합니다.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- [3] Zero-State UX (초기 웰컴 가이드) ---
# 대화 기록이 배열에 단 하나도 없을 때만 이 안내 카드를 화면에 그립니다.
if not st.session_state.messages:
    st.info("""
    👋 **Welcome to the Official Osaki Massage Chair AI Assistant!**
    
    I am your personal expert on ultimate relaxation. Whether you are a first-time buyer or a long-time Osaki owner, I'm here to provide professional product support strictly in English.
    
    Feel free to ask me about different models, unique features, or comparison details based on our technical specifications.
    
    Here are a few examples to get you started:
    * `"What are the best 4D massage chairs?"`
    * `"Which model has the strongest massage pressure?"`
    * `"Compare the Maestro LE and Vista 4D."`
    * `"Tell me about zero gravity features."`
    
    Ask your question in the input bar below to begin your journey to relaxation!
    """)

# --- [4] UI Rendering (기존 대화 연속 출력) ---
# 이전 대화 기록을 UI에 연속 렌더링 (웰컴 카드가 뜬 상태에서는 빈 배열이므로 아무것도 그리지 않음)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- [5] User Input & API Request (데이터 조립 및 통신) ---
if prompt := st.chat_input("Ask about Osaki massage chairs..."):
    # 1. 고객의 현재 질문을 즉시 UI에 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # [핵심 로직] 백엔드로 보낼 과거 대화 기록(chat_history) 배열 추출
    chat_history_payload = [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.messages
    ]

    # UI 렌더링을 위해 현재 질문을 세션 스테이트에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- [6] Backend 통신 (HTTP POST - Streaming) ---
    with st.chat_message("assistant"):
        try:
            api_url = "http://127.0.0.1:8000/api/v1/chat"
            payload = {
                "user_query": prompt,
                "session_id": st.session_state.session_id,
                "chat_history": chat_history_payload
            }

            # [핵심] stream=True 파라미터를 추가하여 연결을 끊지 않고 계속 파이프를 열어둡니다.
            response = requests.post(api_url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            # 파이프에서 떨어지는 조각(Chunk)들을 받아내는 Generator 함수
            def stream_parser():
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        yield chunk
            
            # Streamlit의 내장 스트리밍 렌더링 함수 가동 (타자기 효과)
            answer = st.write_stream(stream_parser())
            
            # 스트리밍이 완벽히 끝난 최종 완성본을 세션 스테이트에 저장 (기억 유지)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            error_msg = f"🚨 API Connection Error: 서버와 통신할 수 없습니다. ({e})"
            st.error(error_msg)