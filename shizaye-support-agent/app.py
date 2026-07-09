"""
Shizaye Multilinks Support Agent — Streamlit App
--------------------------------------------------
Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud
"""

import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Ada — Shizaye Multilinks Support", page_icon="💬")

st.title("💬 Ada — Shizaye Multilinks Support Agent")
st.caption("Ask about products, pricing, SIM registration, warranties, or your order status.")

# --- API key handling ---
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    api_key = st.text_input("Enter your Groq API key", type="password")

if not api_key:
    st.warning("Please provide a Groq API key to start chatting.")
    st.stop()

# --- Session memory ---
if "history" not in st.session_state:
    st.session_state.history = []  # for Claude API (role/content pairs)
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # for rendering in the UI

# --- Render past messages ---
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Ada is thinking..."):
            reply, updated_history = run_agent(
                user_input, st.session_state.history, api_key
            )
            st.markdown(reply)

    st.session_state.history = updated_history
    st.session_state.display_history.append({"role": "assistant", "content": reply})

# --- Sidebar ---
with st.sidebar:
    st.header("About Ada")
    st.write(
        "Ada is an AI support agent for Shizaye Multilinks, built with Llama 3.3 "
        "70B (via Groq's free API), retrieval-augmented FAQ search, and live "
        "order lookup."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.display_history = []
        st.rerun()
