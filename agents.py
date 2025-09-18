# agents.py
from dataclasses import dataclass
from typing import Dict, Any, List
from app import groq_chat, chunk_text  # careful: circular import if you move groq_chat. If circular issues, copy groq_chat here.
import csv
import base64
import tempfile
from typing import Tuple

@dataclass
class AgentResult:
    name: str
    info: Dict[str, Any]

class SummarizationAgent:
    name = "Summarization"
    def run(self, text: str, mode: str = "short") -> AgentResult:
        if mode == "short":
            prompt = f" this text in 2-3 lines:\n\n{text}"
        elif mode == "medium":
            prompt = f"Summarize this text in a paragraph, keeping important points:\n\n{text}"
        else:
            prompt = f"Provide a detailed structured summary with headings for this text:\n\n{text}"
        # reuse groq_chat with empty context
        ans = groq_chat(prompt, [], [])
        return AgentResult(self.name, {"summary": ans})
# Summarize

class FlashcardAgent:
    name = "FlashcardGen"
    def run(self, chunks: List[str], max_q_per_chunk: int = 1) -> AgentResult:
        qas = []
        for c in chunks:
            prompt = f"From the following text, generate {max_q_per_chunk} concise question-answer pair(s). Keep questions clear and answers short.\n\n{text_preview(c)}"
            a = groq_chat(prompt, [], [])
            # expect output in Q: ... A: ... format; do some simple parsing
            for block in split_qa(a):
                qas.append(block)
        return AgentResult(self.name, {"flashcards": qas})

class QuizAgent:
    name = "QuizGen"
    def run(self, text: str, n_mcq: int = 5) -> AgentResult:
        prompt = f"Generate {n_mcq} MCQs (question + 4 options + correct answer) from the following text. Return JSON like: [{{'q':'', 'options':['a','b','c','d'], 'answer': 'b'}}]\n\n{text}"
        js = groq_chat(prompt, [], [])
        # attempt to parse JSON-ish response; if fails, return raw
        try:
            import json
            data = json.loads(js)
        except Exception:
            data = js
        return AgentResult(self.name, {"quiz": data})

# small helpers
def text_preview(s: str, n: int = 600):
    return s[:n] + ("…" if len(s)>n else "")

def split_qa(raw: str):
    # naive parse Q: A: pairs
    res=[]
    lines = raw.splitlines()
    cur = {"q":"", "a":""}
    state=None
    for l in lines:
        if l.strip().lower().startswith("q:") or l.strip().startswith("Q:"):
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
