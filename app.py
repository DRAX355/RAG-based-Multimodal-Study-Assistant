# app.py (patched)
import os
import io
import re
import time
import json
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import requests

# Local modules from your project (unchanged)
from parsing import read_pdf_text, pdf_to_images, read_docx, image_bytes_to_pil
from ocr import smart_ocr
from embeddings import embed_chunks, get_embedder
from rag import EphemeralVault
from tts import synthesize

# New: re-ranker
try:
    from rerank import rerank
except Exception:
    # fallback no-op rerank if module not present
    def rerank(q, candidates, top_k=5):
        return candidates[:top_k]

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

# ---------- Utilities ----------
MAX_CHARS_PER_CHUNK = 220
STRIDE = 40
HISTORY_TURNS_FOR_MEMORY = 8   # how many prior turns to pass to the model

@st.cache_data(show_spinner=False)
def chunk_text(text: str, max_len=MAX_CHARS_PER_CHUNK, stride=STRIDE):
    words = text.split()
    if not words: return []
    chunks = []
    i = 0
    # chunk by words (approx tokens)
    while i < len(words):
        chunk = " ".join(words[i:i+max_len])
        chunks.append(chunk)
        i += max(1, max_len - stride)
    return [c for c in chunks if c.strip()]

def strip_think(content: str) -> str:
    # DeepSeek-R1 may include <think> ... </think> sections. Strip them for UI.
    return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

def groq_chat(prompt: str, context_chunks: List[str], history: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Send chat with short memory of previous turns + retrieved context.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])
    sys = (
        "You are a study assistant. Answer using ONLY the provided context. "
        "When relevant, cite chunk indices like [1], [2]. If the answer isn't in context, say you don't know."
    )

    messages = [{"role": "system", "content": sys}]
    # Include the last N turns for conversational memory
    for h in history[-HISTORY_TURNS_FOR_MEMORY:]:
        if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"})

    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers=headers, timeout=90)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return strip_think(content)
    except Exception as e:
        # graceful fallback
        return f"(Groq model unavailable or error: {e})"

# ---------- Agents ----------
@dataclass
class AgentResult:
    name: str
    info: Dict[str, Any]

class IngestionAgent:
    name = "Ingestion"
    def run(self, uploaded_files) -> AgentResult:
        texts = []
        sources = []
        for f in uploaded_files:
            name = f.name.lower()
            b = f.getvalue()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
            tmp.write(b); tmp.flush(); tmp.close()
            try:
                if name.endswith('.pdf'):
                    t = read_pdf_text(tmp.name)
                    if not t.strip():
                        # scanned → images → OCR
                        pages = pdf_to_images(tmp.name, dpi=300)
                        for img in pages:
                            t += "\n" + smart_ocr(img)
                elif name.endswith('.docx'):
                    t = read_docx(tmp.name)
                else:
                    img = image_bytes_to_pil(b)
                    t = smart_ocr(img)
                texts.append(t)
                sources.append(name)
            finally:
                try: os.remove(tmp.name)
                except Exception: pass
        return AgentResult(self.name, {"texts": texts, "sources": sources})

class ChunkIndexAgent:
    name = "Chunk+Index"
    def __init__(self):
        # dimension will be set when first embeddings arrive
        self.vault = None

    def run(self, texts: List[str], sources: List[str]) -> AgentResult:
        total_chunks = 0
        for text, src in zip(texts, sources):
            chunks = chunk_text(text)
            if not chunks:
                continue
            embs = embed_chunks(chunks)
            dim = embs.shape[1]
            if self.vault is None:
                self.vault = EphemeralVault(dim=dim)
            meta = [{"source": src, "i": i} for i in range(len(chunks))]
            self.vault.add(embs, chunks, meta)
            total_chunks += len(chunks)
        return AgentResult(self.name, {"vault": self.vault, "chunks": total_chunks})

class RetrievalAgent:
    name = "Retrieval"
    def run(self, vault: EphemeralVault, question: str, k: int = 6) -> AgentResult:
        q_emb = embed_chunks([question])[0]
        hits = vault.search(q_emb, k=k)
        # Re-rank with cross-encoder for better precision (if available)
        try:
            reranked = rerank(question, hits, top_k=min(k, len(hits)))
        except Exception:
            reranked = hits[:k]
        contexts = [h["text"] for h in reranked]
        return AgentResult(self.name, {"hits": reranked, "contexts": contexts})

class ReasoningAgent:
    name = "Reasoning (Groq DeepSeek)"
    def run(self, question: str, contexts: List[str], history: List[Dict[str, str]]) -> AgentResult:
        answer = groq_chat(question, contexts, history)
        return AgentResult(self.name, {"answer": answer})

class SummarizationAgent:
    name = "Summarization"
    def run(self, text: str, mode: str = "short") -> AgentResult:
        if mode == "short":
            prompt = f"Summarize this text in 2-3 lines:\n\n{text}"
        elif mode == "medium":
            prompt = f"Summarize this text in a paragraph, keeping important points:\n\n{text}"
        else:
            prompt = f"Provide a detailed structured summary with headings for this text:\n\n{text}"
        ans = groq_chat(prompt, [], [])
        return AgentResult(self.name, {"summary": ans})

class FlashcardAgent:
    name = "FlashcardGen"
    def run(self, chunks: List[str], max_q_per_chunk: int = 1) -> AgentResult:
        qas = []
        # keep small preview per chunk to avoid long prompts
        for c in chunks:
            preview = c[:800] + ("…" if len(c)>800 else "")
            prompt = f"From the following text, generate {max_q_per_chunk} concise question-answer pair(s). Keep questions clear and answers short. Use the format: 'Q: ...\\nA: ...'\n\n{preview}"
            a = groq_chat(prompt, [], [])
            # parse Q/A pairs
            for qa in self._split_qa(a):
                qas.append(qa)
        return AgentResult(self.name, {"flashcards": qas})

    def _split_qa(self, raw: str):
        res=[]
        lines = raw.splitlines()
        cur = {"q":"", "a":""}
        state=None
        for l in lines:
            if l.strip().lower().startswith("q:"):
                if cur["q"] or cur["a"]:
                    res.append(cur); cur={"q":"", "a":""}
                state="q"
                cur["q"]=l.split(":",1)[1].strip()
            elif l.strip().lower().startswith("a:"):
                state="a"
                cur["a"]=l.split(":",1)[1].strip()
            else:
                if state=="q":
                    cur["q"] += " "+l.strip()
                elif state=="a":
                    cur["a"] += " "+l.strip()
        if cur["q"] or cur["a"]:
            res.append(cur)
        return res

class QuizAgent:
    name = "QuizGen"
    def run(self, text: str, n_mcq: int = 5) -> AgentResult:
        prompt = f"Generate {n_mcq} MCQs (question + 4 options + correct answer) from the following text. Return a JSON array of objects with keys: q, options (array of 4 strings), answer (index 0-3). If you cannot make enough, make fewer.\n\n{text[:3000]}"
        js = groq_chat(prompt, [], [])
        try:
            import json as _json
            data = _json.loads(js)
        except Exception:
            data = js
        return AgentResult(self.name, {"quiz": data})

class VoiceAgent:
    name = "Voice (TTS)"
    def run(self, text: str) -> AgentResult:
        b64 = synthesize(text)
        return AgentResult(self.name, {"audio_b64": b64})

# Orchestrator
class Orchestrator:
    def __init__(self):
        self.ingest = IngestionAgent()
        self.indexer = ChunkIndexAgent()
        self.retrieve = RetrievalAgent()
        self.reason = ReasoningAgent()
        self.summarize_agent = SummarizationAgent()
        self.flashcard_agent = FlashcardAgent()
        self.quiz_agent = QuizAgent()
        self.voice = VoiceAgent()

    def build_index(self, uploaded_files):
        ing = self.ingest.run(uploaded_files)
        idx = self.indexer.run(ing.info["texts"], ing.info["sources"])
        return {**ing.info, **idx.info}

    def answer(self, vault: EphemeralVault, question: str, history: List[Dict[str, str]], kctx: int = 6):
        ret = self.retrieve.run(vault, question, k=kctx)
        ans = self.reason.run(question, ret.info["contexts"], history)
        return {**ret.info, **ans.info}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Multimodal Study Assistant ", page_icon="🧠", layout="wide")

st.title("🧠 Multimodal AI for Information Processing ")
st.caption("Upload PDFs, Word, images, or handwriting → ask multi-turn questions → summaries, flashcards, quizzes, and TTS.")

if not GROQ_API_KEY:
    st.warning("Set GROQ_API_KEY in .env to enable Groq DeepSeek 70B. The app will still run for ingestion and local ops.")

# Sidebar controls / corpus
with st.sidebar:
    st.header("Upload & Settings")
    uploads = st.file_uploader(
        "Upload notes (PDF/DOCX/PNG/JPG)",
        type=["pdf","docx","png","jpg","jpeg"],
        accept_multiple_files=True
    )
    k_ctx = st.slider("Evidence depth (top-k chunks)", 2, 12, 6, 1)
    tts_on = st.checkbox("🔊 Read answers aloud", value=True)
    st.markdown("---")
    st.subheader("Persistence")
    vault_name = st.text_input("Corpus name (for save/load)", value="default")
    if st.button("Save vault"):
        if 'orch' in st.session_state and st.session_state.orch.indexer.vault:
            dp = os.path.join("vaults", vault_name)
            st.session_state.orch.indexer.vault.save(dp)
            st.success(f"Vault saved to {dp}")
        else:
            st.error("No vault to save. Process & Index first.")
    if st.button("Load vault"):
        p = os.path.join("vaults", vault_name)
        if os.path.exists(p):
            st.session_state.orch.indexer.vault = EphemeralVault.load(p)
            st.session_state.vault = st.session_state.orch.indexer.vault
            st.success(f"Loaded vault {vault_name} ({len(st.session_state.vault.texts)} chunks).")
        else:
            st.error("Vault not found.")
    st.markdown("---")
    st.markdown("Developer options")
    regen_idx = st.checkbox("Force re-index (clear prior)", value=False)

# Session state
if 'orch' not in st.session_state:
    st.session_state.orch = Orchestrator()
if 'vault' not in st.session_state:
    st.session_state.vault = None
if 'history' not in st.session_state:
    st.session_state.history = []  # list of {"role": "...", "content": "...", optional "evidence": [...]}
if 'analytics' not in st.session_state:
    st.session_state.analytics = {"questions":0, "summaries":0, "flashcards":0, "quizzes":0}

# Ingestion/index UI
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Ingestion Pipeline")
    if st.button("Process & Index", use_container_width=True, type="primary"):
        if not uploads:
            st.error("Please upload at least one file.")
        else:
            if regen_idx:
                st.session_state.orch.indexer = ChunkIndexAgent()
                st.session_state.vault = None
            with st.spinner("Running agents: Ingestion → Chunk+Index …"):
                out = st.session_state.orch.build_index(uploads)
                st.session_state.vault = st.session_state.orch.indexer.vault
                st.success(f"Indexed {out['chunks']} chunks from {len(out['sources'])} file(s).")
            with st.expander("Ingested files", expanded=False):
                st.write(out.get("sources", []))

with col2:
    st.subheader("Corpus Overview")
    if st.session_state.vault and st.session_state.orch.indexer.vault and st.session_state.orch.indexer.vault.texts:
        sample = "\n\n".join(st.session_state.orch.indexer.vault.texts[:5])
        st.text_area("Sample chunks", sample, height=220)
    else:
        st.caption("No corpus yet. Upload and process to view sample chunks.")

st.markdown("---")

# Extra tools: Summarize / Flashcards / Quiz
st.subheader("Generate Study Artifacts")
artifact_col1, artifact_col2 = st.columns([1,1])

with artifact_col1:
    st.markdown("**Summarize a file or selection**")
    manual_text = st.text_area("Paste text to summarize (or leave empty to use latest corpus):", height=120)
    sum_mode = st.radio("Mode", ["short","medium","detailed"], horizontal=True)
    if st.button("Generate Summary"):
        text_for_summary = manual_text.strip()
        if not text_for_summary and st.session_state.vault:
            # combine first N chunks to summarize
            text_for_summary = "\n\n".join(st.session_state.vault.texts[:30])
        if not text_for_summary:
            st.error("No text available to summarize.")
        else:
            with st.spinner("Summarizing..."):
                res = st.session_state.orch.summarize_agent.run(text_for_summary, mode=sum_mode)
                st.session_state.analytics["summaries"] += 1
                st.success("Summary generated.")
                st.markdown("**Summary:**")
                st.write(res.info["summary"])

with artifact_col2:
    st.markdown("**Flashcards / Quiz generation**")
    max_q = st.slider("Max Q per chunk (flashcards)", 1, 3, 1)
    if st.button("Generate Flashcards"):
        if not st.session_state.vault:
            st.error("No corpus. Upload & Process first.")
        else:
            with st.spinner("Generating flashcards (this may take a while for many chunks)…"):
                # only take top N chunks to avoid huge prompts
                chunks = st.session_state.vault.texts[:200]
                res = st.session_state.orch.flashcard_agent.run(chunks, max_q_per_chunk=max_q)
                st.session_state.analytics["flashcards"] += 1
                st.success(f"Generated {len(res.info['flashcards'])} flashcards.")
                # show first 10
                for i, qa in enumerate(res.info["flashcards"][:30], start=1):
                    st.markdown(f"**{i}. Q:** {qa.get('q','(no q)')}\n\n**A:** {qa.get('a','(no a)')}")
                # provide CSV export
                if st.button("Export flashcards CSV"):
                    csv_buf = io.StringIO()
                    writer = csv.writer(csv_buf)
                    writer.writerow(["question","answer"])
                    for qa in res.info["flashcards"]:
                        writer.writerow([qa.get("q",""), qa.get("a","")])
                    b = csv_buf.getvalue().encode("utf-8")
                    st.download_button("Download CSV", b, file_name="flashcards.csv", mime="text/csv")

    quiz_n = st.number_input("Number of MCQs", min_value=1, max_value=30, value=5)
    if st.button("Generate Quiz"):
        if not st.session_state.vault:
            st.error("No corpus. Upload & Process first.")
        else:
            with st.spinner("Generating quiz..."):
                text_corpus = "\n\n".join(st.session_state.vault.texts[:200])
                res = st.session_state.orch.quiz_agent.run(text_corpus, n_mcq=quiz_n)
                st.session_state.analytics["quizzes"] += 1
                st.success("Quiz generated.")
                st.write(res.info["quiz"])

st.markdown("---")

# ---------------- Chat area ----------------
st.subheader("Chat — ask questions about your notes")

# chat input
prompt = st.chat_input("Type your question…")

if prompt:
    if st.session_state.vault is None:
        st.session_state.history.append({
            "role": "assistant",
            "content": "⚠️ Please upload and **Process & Index** your notes first."
        })
    else:
        # Append the user message FIRST
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.analytics["questions"] += 1

        # Retrieve + Reason with memory
        with st.spinner("Retrieval → Re-rank → Reasoning (Groq DeepSeek)…"):
            res = st.session_state.orch.answer(
                st.session_state.orch.indexer.vault,
                prompt,
                history=st.session_state.history,
                kctx=k_ctx
            )
        answer = res.get("answer", "(no answer)")

        # Append assistant answer + evidence
        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "evidence": res.get("hits", [])
        })

# render conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            if "evidence" in msg and msg["evidence"]:
                with st.expander("Grounding snippets", expanded=False):
                    for i, hit in enumerate(msg["evidence"], start=1):
                        st.markdown(
                            f"**[{i}] score={hit['score']:.2f}** — "
                            f"`{hit['meta']['source']}` #{hit['meta']['i']}\n\n> {hit['text'][:400]}…"
                        )
            if tts_on and msg["content"].strip():
                try:
                    b64 = synthesize(msg["content"])
                    st.audio(io.BytesIO(__import__('base64').b64decode(b64)))
                except Exception as e:
                    st.caption(f"🔇 TTS unavailable: {e}")

st.markdown("---")

# Analytics panel
st.subheader("Study Analytics (session)")
a = st.session_state.analytics
st.write(f"Questions asked: **{a['questions']}**  •  Summaries: **{a['summaries']}**  •  Flashcards: **{a['flashcards']}**  •  Quizzes: **{a['quizzes']}**")

st.caption("Tip: Use Save/Load in the sidebar to persist large indexes to disk, so you don't reprocess every time.")

st.markdown("---")
st.caption("Powered by Groq DeepSeek-R1 Distill Llama-70B for reasoning; SentenceTransformers for retrieval; TrOCR/Tesseract for OCR; your TTS backend for speech.")

# End of file
