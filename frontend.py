# streamlit_frontend.py
import os, json, requests, io, base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Study Assistant (Frontend)", page_icon="🧠", layout="wide")
st.title("🧠 Study Assistant — Frontend (API-powered)")

with st.sidebar:
    st.subheader("Backend")
    backend = st.text_input("Backend URL", BACKEND_URL)
    if backend != BACKEND_URL:
        BACKEND_URL = backend
    st.markdown("---")
    st.subheader("Corpus")
    corpus_name = st.text_input("Corpus name", value=st.session_state.get("corpus_name","default"))
    if st.button("List corpora"):
        try:
            r = requests.get(f"{BACKEND_URL}/corpora")
            st.write(r.json())
        except Exception as e:
            st.error(f"Request failed: {e}")
    uploads = st.file_uploader("Upload files", type=["pdf","docx","png","jpg","jpeg"], accept_multiple_files=True)
    if st.button("Process & Index"):
        if not uploads:
            st.error("Upload at least one file")
        else:
            files=[("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
            data={"corpus_name": corpus_name}
            with st.spinner("Processing on backend..."):
                try:
                    r = requests.post(f"{BACKEND_URL}/corpora/process", data=data, files=files, timeout=300)
                    st.write(r.json())
                    st.session_state["corpus_name"] = corpus_name
                except Exception as e:
                    st.error(f"Processing failed: {e}")
    st.markdown("---")
    st.subheader("Sessions")
    session_id = st.text_input("Session ID", value=st.session_state.get("session_id","default"))
    if st.button("List sessions"):
        try:
            st.write(requests.get(f"{BACKEND_URL}/sessions").json())
        except Exception as e:
            st.error(f"Failed: {e}")
    if st.button("Load chat history"):
        try:
            r = requests.get(f"{BACKEND_URL}/sessions/{session_id}/chats")
            st.write(r.json())
        except Exception as e:
            st.error(f"Failed: {e}")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat")
    if "history" not in st.session_state: st.session_state.history=[]
    prompt = st.chat_input("Ask a question about your notes...")
    if prompt:
        st.session_state.history.append({"role":"user","content":prompt})
        payload = {
            "question": prompt,
            "corpus_name": st.session_state.get("corpus_name","default"),
            "session_id": st.session_state.get("session_id","default"),
            "k": 6,
            "history": [
                {"role": h["role"], "content": h["content"]}
                for h in st.session_state.history[-8:]
                if h["role"] in ("user","assistant")
            ]
        }

        with st.spinner("Querying backend..."):
            try:
                r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)
                data = r.json()
                answer = data.get("answer","(no answer)")
                evidence = data.get("evidence",[])
            except Exception as e:
                answer = f"(backend error: {e})"
                evidence = []
        st.session_state.history.append({"role":"assistant","content":answer,"evidence":evidence})

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"]=="assistant" and msg.get("evidence"):
                with st.expander("Evidence", expanded=False):
                    for i, hit in enumerate(msg["evidence"], start=1):
                        st.markdown(f"**[{i}] {hit['meta'].get('source','?')} #{hit['meta'].get('i','?')}** — score={hit.get('score',0):.3f}")
                        st.write(hit["text"][:500]+"…")

with col2:
    st.subheader("Tools")
    text_for_summary = st.text_area("Paste text to summarize", height=160)
    mode = st.radio("Mode", ["short","medium","detailed"], horizontal=True, index=0)
    if st.button("Summarize"):
        try:
            r = requests.post(f"{BACKEND_URL}/summarize", json={"text": text_for_summary, "mode": mode})
            st.write(r.json())
        except Exception as e:
            st.error(f"Summarize failed: {e}")

    st.markdown("---")
    quiz_text = st.text_area("Text for MCQ", height=160)
    n = st.number_input("Questions", 1, 30, 5)
    if st.button("Generate Quiz"):
        try:
            r = requests.post(f"{BACKEND_URL}/quiz", json={"text": quiz_text, "n": int(n)})
            st.write(r.json())
        except Exception as e:
            st.error(f"Quiz failed: {e}")

    st.markdown("---")
    tts_text = st.text_area("Text to speak", height=120, key="tts")
    if st.button("Speak"):
        try:
            r = requests.post(f"{BACKEND_URL}/tts", json=tts_text)
            data = r.json()
            b64 = data.get("audio_b64","")
            if b64:
                st.audio(io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            st.error(f"TTS failed: {e}")
