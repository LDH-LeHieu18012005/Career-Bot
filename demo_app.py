import uuid
import time
import streamlit as st
import requests

# --- 1. Cấu hình trang ---
st.set_page_config(page_title="Career Bot Demo", page_icon="🤖", layout="centered")

# --- 2. CSS (giữ nguyên) ---
st.markdown("""
<style>
    .stApp { background-color: #F4F7FB; }
    h1 {
        color: #050A30 !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 25px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    /* User */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #E8F5FE;
        border: 1px solid #87D4F5;
    }
    /* Assistant */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFFFFF;
        border: 1px solid #8DA1F2;
        box-shadow: 0 4px 12px rgba(141, 161, 242, 0.08);
    }

    .timer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #5C6C85;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        padding: 5px 12px;
        margin: 10px 0 6px 0;
        width: fit-content;
    }
    .timer-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #4caf50;
    }
    .timer-dot.slow { background: #ff9800; }
    .timer-dot.veryslow { background: #f44336; }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

API_ENDPOINT = "http://127.0.0.1:5001/api/v1/chat"

st.title("🤖 Career Bot Demo")

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "elapsed" in msg:
            elapsed = msg["elapsed"]
            # NGƯỠNG MỚI - ĐÃ NÂNG LÊN
            if elapsed < 6:
                dot = "timer-dot"           # Xanh lá
            elif elapsed < 12:
                dot = "timer-dot slow"      # Cam
            else:
                dot = "timer-dot veryslow"  # Đỏ
            st.markdown(
                f'<div class="timer-badge"><span class="{dot}"></span>Thời gian phản hồi: {elapsed:.2f}s</div>',
                unsafe_allow_html=True
            )

# --- Input ---
if prompt := st.chat_input("Nhập câu hỏi tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        timer_placeholder = st.empty()

        full_response = ""
        t_start = time.perf_counter()

        try:
            with st.spinner("Đang tìm kiếm và phân tích..."):
                response = requests.post(
                    API_ENDPOINT,
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=120
                )

                ttft = time.perf_counter() - t_start

                if response.status_code == 200:
                    data = response.json()
                    full_response = data.get("content", "Tôi không hiểu câu hỏi của bạn.")
                else:
                    full_response = f"❌ Lỗi server: {response.status_code}"

        except Exception as e:
            ttft = time.perf_counter() - t_start
            full_response = f"❌ Lỗi kết nối: {str(e)}"

        # NGƯỠNG MỚI - ĐÃ NÂNG LÊN
        if ttft < 6:
            dot = "timer-dot"           # Xanh lá - nhanh
        elif ttft < 12:
            dot = "timer-dot slow"      # Cam - chấp nhận được
        else:
            dot = "timer-dot veryslow"  # Đỏ - chậm

        timer_placeholder.markdown(
            f'<div class="timer-badge"><span class="{dot}"></span>Thời gian phản hồi: {ttft:.2f}s</div>',
            unsafe_allow_html=True
        )

        message_placeholder.markdown(full_response)

    # Lưu vào lịch sử
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "elapsed": ttft
    })