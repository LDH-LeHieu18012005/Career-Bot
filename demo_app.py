# demo_app.py
import uuid
import streamlit as st
import requests

# --- Cấu hình cơ bản ---
st.set_page_config(page_title="Career Bot Demo", page_icon="🤖")
st.title("🤖 Career Bot Demo")

# Endpoint của Career Bot API
API_ENDPOINT = "http://127.0.0.1:5001/api/v1/chat"

# --- Quản lý Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị lịch sử hội thoại ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input người dùng ---
if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            payload = {
                "query":      prompt,
                "session_id": st.session_state.session_id,
            }
            response = requests.post(API_ENDPOINT, json=payload)

            if response.status_code == 200:
                full_response = response.json().get("content", "Xin lỗi, mình không hiểu yêu cầu của bạn.")
            else:
                full_response = f"❌ Lỗi từ API: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            full_response = f"❌ Lỗi kết nối đến API: {e}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
