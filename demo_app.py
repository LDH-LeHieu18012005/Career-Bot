# demo_app.py
import streamlit as st
import requests
import json

# --- Cấu hình cơ bản ---
st.set_page_config(page_title="Career Bot Demo", page_icon="🤖")
st.title("🤖 Career Bot Demo")

# Endpoint của Career Bot API
API_ENDPOINT = "http://127.0.0.1:5001/api/v1/chat"

# --- Quản lý Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị lịch sử hội thoại ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- dùng ---
if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    # Thêm tin nhắn của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị placeholder cho phản hồi của AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # --- Gửi request đến API ---
        try:
            payload = {
                "query": prompt,
                "session_id": "demo_session_v6_clean"  # Đổi ID mới để clear lịch sử nháp cũ
            }
            headers = {'Content-Type': 'application/json'}

            # Gọi API
            response = requests.post(API_ENDPOINT, json=payload, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                full_response = response_data.get("content", "Xin lỗi, tôi không hiểu yêu cầu của bạn.")
            else:
                full_response = f"❌ Lỗi từ API: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            full_response = f"❌ Lỗi kết nối đến API: {e}"

        # Hiển thị phản hồi hoàn chỉnh từ bot
        message_placeholder.markdown(full_response)

    # Thêm phản hồi của bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})
