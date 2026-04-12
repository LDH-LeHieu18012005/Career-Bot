# demo_app.py
import uuid
import time
import streamlit as st
import requests

# --- Cấu hình cơ bản ---
st.set_page_config(page_title="Career Bot Demo", page_icon="🤖")
st.title("🤖 Career Bot Demo")

API_ENDPOINT = "http://127.0.0.1:5001/api/v1/chat"

# --- CSS: timer badge nhỏ gọn ---
st.markdown("""
<style>
.timer-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #888;
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 20px;
    padding: 2px 8px;
    margin-top: 6px;
    width: fit-content;
}
.timer-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4caf50;
    flex-shrink: 0;
}
.timer-dot.slow { background: #ff9800; }
.timer-dot.veryslow { background: #f44336; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị lịch sử ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("elapsed"):
            elapsed = message["elapsed"]
            if elapsed < 5:
                dot_cls = "timer-dot"
            elif elapsed < 15:
                dot_cls = "timer-dot slow"
            else:
                dot_cls = "timer-dot veryslow"
            st.markdown(
                f'<div class="timer-badge">'
                f'<span class="{dot_cls}"></span>'
                f'{elapsed:.2f}s'
                f'</div>',
                unsafe_allow_html=True,
            )

# --- Input ---
if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        timer_placeholder   = st.empty()
        full_response = ""
        elapsed = 0.0

        # Hiện spinner nhỏ trong lúc chờ
        with st.spinner(""):
            t_start = time.perf_counter()
            try:
                response = requests.post(
                    API_ENDPOINT,
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                )
                elapsed = time.perf_counter() - t_start

                if response.status_code == 200:
                    full_response = response.json().get(
                        "content", "Xin lỗi, mình không hiểu yêu cầu của bạn."
                    )
                else:
                    full_response = f"❌ Lỗi từ API: {response.status_code} - {response.text}"
            except requests.exceptions.RequestException as e:
                elapsed = time.perf_counter() - t_start
                full_response = f"❌ Lỗi kết nối đến API: {e}"

        message_placeholder.markdown(full_response)

        # Badge thời gian
        if elapsed < 5:
            dot_cls = "timer-dot"
        elif elapsed < 15:
            dot_cls = "timer-dot slow"
        else:
            dot_cls = "timer-dot veryslow"

        timer_placeholder.markdown(
            f'<div class="timer-badge">'
            f'<span class="{dot_cls}"></span>'
            f'{elapsed:.2f}s'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "elapsed": elapsed,
    })