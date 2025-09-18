# backend.py
import os, io, json, tempfile, re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Local modules (expected in your repo)
from rag import EphemeralVault
from embeddings import embed_chunks
from parsing import read_pdf_text, pdf_to_images, read_docx, image_bytes_to_pil
from ocr import smart_ocr
from tts import synthesize
from db import init_db, save_chat, load_chat, list_sessions, save_corpus, list_corpora, get_corpus_path
from pydantic import BaseModel, Extra

# optional reranker
try:
    from rerank import rerank
except Exception:
    def rerank(q, candidates, top_k=5): return candidates[:top_k]

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
VAULTS_DIR = os.getenv("VAULTS_DIR", "vaults")

def strip_think(content: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", content or "").strip()

def groq_chat(prompt: str, context_chunks: List[str], history, temperature: float = 0.2) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])
    sys = (
        "You are a study assistant. Answer using ONLY the provided context. "
        "Cite chunk indices like [1], [2]. If the answer isn't in context, say you don't know."
    )
    messages = [{"role": "system", "content": sys}]

    # Handle both dicts and Pydantic ChatMessage objects
    for h in history[-8:]:
        role = h.role if hasattr(h, "role") else h.get("role")
        content = h.content if hasattr(h, "content") else h.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append(
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"}
    )

    body = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          json=body, headers=headers, timeout=90)
        r.raise_for_status()
        return strip_think(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return f"(LLM error: {e})"


# Global in-memory cache of loaded vaults
vaults_cache: Dict[str, EphemeralVault] = {}
init_db()

app = FastAPI(title="Multimodal Study Assistant API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class QueryBody(BaseModel):
    question: str
    corpus_name: str = "default"
    session_id: str = "default"
    k: int = 6
    history: Optional[List[Dict[str,str]]] = None

class SummaryBody(BaseModel):
    text: str
    mode: str = "short"

class QuizBody(BaseModel):
    text: str
    n: int = 5

# ---------- Helpers ----------
def ensure_vault(corpus_name: str) -> EphemeralVault:
    if corpus_name in vaults_cache:
        return vaults_cache[corpus_name]
    # try to load from disk (if registered in DB or present)
    path = get_corpus_path(corpus_name) or os.path.join(VAULTS_DIR, corpus_name)
    if os.path.exists(path):
        v = EphemeralVault.load(path)
        vaults_cache[corpus_name] = v
        return v
    # else create empty vault
    v = EphemeralVault(dim=384)
    vaults_cache[corpus_name] = v
    return v

class ChatMessage(BaseModel, extra=Extra.allow):
    role: str
    content: str

class QueryBody(BaseModel):
    question: str
    corpus_name: str = "default"
    session_id: str = "default"
    k: int = 6
    history: Optional[List[ChatMessage]] = None


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/corpora")
def corpora():
    rows = list_corpora()
    return [{"name": n, "path": p} for (n,p) in rows]

@app.post("/corpora/process")
async def process_corpus(corpus_name: str = Form("default"), files: List[UploadFile] = File(...)):
    texts, sources = [], []
    for f in files:
        name = (f.filename or "file").lower()
        b = await f.read()
        # parse/ocr
        if name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(b); tmp.flush(); tmp.close()
                t = read_pdf_text(tmp.name)
                if not t.strip():
                    pages = pdf_to_images(tmp.name, dpi=300)
                    for p in pages:
                        t += "\n" + smart_ocr(p)
                os.unlink(tmp.name)
        elif name.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(b); tmp.flush(); tmp.close()
                t = read_docx(tmp.name)
                os.unlink(tmp.name)
        else:
            img = image_bytes_to_pil(b)
            t = smart_ocr(img)
        texts.append(t); sources.append(name)

    # chunking
    def chunk_text(text: str, max_len: int = 220, stride: int = 40):
        words = text.split()
        if not words: return []
        chunks = []; i=0
        while i < len(words):
            chunk = " ".join(words[i:i+max_len])
            chunks.append(chunk); i += max(1, max_len - stride)
        return [c for c in chunks if c.strip()]

    chunks_all, metas_all = [], []
    for t, src in zip(texts, sources):
        cs = chunk_text(t)
        chunks_all.extend(cs)
        metas_all.extend([{"source": src, "i": i} for i in range(len(cs))])

    if not chunks_all:
        return {"status":"empty", "chunks_indexed": 0}

    embs = embed_chunks(chunks_all)

    v = ensure_vault(corpus_name)

    # 🔑 NEW: reset vault if embedding dimension mismatches
    if v.index is not None and embs.shape[1] != v.dim:
        v = EphemeralVault(dim=embs.shape[1])
        vaults_cache[corpus_name] = v  

    v.add(embs, chunks_all, metas_all)

    # persist
    out_dir = os.path.join(VAULTS_DIR, corpus_name)
    os.makedirs(out_dir, exist_ok=True)
    v.save(out_dir)
    save_corpus(corpus_name, out_dir)

    return {"status":"ok", "corpus": corpus_name, "files": sources, "chunks_indexed": len(chunks_all)}

@app.post("/query")
def query(q: QueryBody):
    v = ensure_vault(q.corpus_name)
    if getattr(v, "index", None) is None or v.index.ntotal == 0:
        return {"error":"No index for this corpus. Please upload/process first."}
    # embed question
    q_emb = embed_chunks([q.question])[0]
    hits = v.search(q_emb, k=q.k)
    hits = rerank(q.question, hits, top_k=min(q.k, len(hits)))
    contexts = [h["text"] for h in hits]

    history = q.history if q.history else []
    answer = groq_chat(q.question, contexts, history)

    # persist chat
    save_chat(q.session_id, "user", q.question)
    save_chat(q.session_id, "assistant", answer)

    return {"answer": answer, "evidence": hits}

@app.get("/sessions")
def sessions():
    return {"sessions": list_sessions()}

@app.get("/sessions/{session_id}/chats")
def session_chats(session_id: str):
    return {"session_id": session_id, "history": load_chat(session_id)}

@app.post("/summarize")
def summarize(body: SummaryBody):
    text = body.text.strip()
    if not text:
        return {"summary": ""}
    if body.mode == "short":
        prompt = f"Summarize in 2-3 lines:\n\n{text}"
    elif body.mode == "medium":
        prompt = f"Summarize in one paragraph with key points:\n\n{text}"
    else:
        prompt = f"Provide a detailed structured summary with headings and bullets:\n\n{text}"
    out = groq_chat(prompt, [], [])
    return {"summary": out}

@app.post("/quiz")
def quiz(body: QuizBody):
    text = body.text[:3000]
    prompt = f"Generate {body.n} MCQs (q + 4 options + correct index 0-3) from the text. Return strict JSON array with keys: q, options, answer.\n\n{text}"
    js = groq_chat(prompt, [], [])
    try:
        data = json.loads(js)
    except Exception:
        data = js
    return {"quiz": data}

@app.post("/tts")
def tts_api(text: str = Body(..., embed=True)):
    b64 = synthesize(text)
    return {"audio_b64": b64}
