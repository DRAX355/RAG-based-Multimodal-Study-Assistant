# db.py
import sqlite3
import os
from typing import List, Dict, Tuple

DB_PATH = os.getenv("DB_PATH", "study_assistant.db")

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        path TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_chat(session_id: str, role: str, content: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def load_chat(session_id: str) -> List[Dict]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content, timestamp FROM chats WHERE session_id=? ORDER BY id", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r, "content": c, "timestamp": t} for (r, c, t) in rows]

def list_sessions(limit: int = 50) -> List[str]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id
        FROM chats
        GROUP BY session_id
        ORDER BY MAX(id) DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def save_corpus(name: str, path: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO corpus (name, path) VALUES (?, ?)", (name, path))
    conn.commit()
    conn.close()

def list_corpora() -> List[Tuple[str, str]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT name, path FROM corpus ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_corpus_path(name: str) -> str:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT path FROM corpus WHERE name=?", (name,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None
