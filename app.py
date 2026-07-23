import streamlit as st
from services.gemini_service import ask_gemini

st.set_page_config(
    page_title="DHAR - Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ DHAR - Hyper Personalized Shopping Assistant")

user_prompt = st.text_input("Ask me anything about shopping:")

if st.button("Send"):
    if user_prompt:
        response = ask_gemini(user_prompt)
        st.success(response)