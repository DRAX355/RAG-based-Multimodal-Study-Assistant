# rerank.py
from sentence_transformers import CrossEncoder

# Use cross-encoder/ms-marco-MiniLM-L-6-v2 (fast & good)
_cross = None
def get_cross():
    global _cross
    if _cross is None:
        _cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross

def rerank(question: str, candidates: list, top_k: int = 5):
    """
    candidates: list of dicts with 'text' key
    returns top_k sorted candidates
    """
    cross = get_cross()
    pairs = [[question, c['text']] for c in candidates]
    scores = cross.predict(pairs)
    for c, s in zip(candidates, scores):
        c['rr_score'] = float(s)
    candidates.sort(key=lambda x: x['rr_score'], reverse=True)
    return candidates[:top_k]
