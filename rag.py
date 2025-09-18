# rag.py
import faiss
import numpy as np
import os, json

class EphemeralVault:
    def __init__(self, dim: int = None):
        """
        If dim is None, the index will be initialized lazily
        when the first batch of embeddings is added.
        """
        self.dim = dim
        self.texts = []
        self.meta = []
        self.index = None if dim is None else faiss.IndexFlatIP(dim)

    def add(self, embeddings, chunks, meta):
        embeddings = np.array(embeddings).astype("float32")

        # lazy init if index is None or dim mismatch
        if self.index is None or embeddings.shape[1] != self.dim:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(embeddings)
        self.texts.extend(chunks)
        self.meta.extend(meta)

    def search(self, emb, k=5):
        emb = np.array([emb]).astype("float32")
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(emb, k)
        res=[]
        for s, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            res.append({
                "text": self.texts[idx],
                "score": float(s),
                "meta": self.meta[idx]
            })
        return res

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))
        with open(os.path.join(dirpath, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, dirpath: str):
        inst = cls()
        inst.index = faiss.read_index(os.path.join(dirpath, "faiss.index"))
        inst.texts = json.load(open(os.path.join(dirpath, "texts.json"), "r", encoding="utf-8"))
        inst.meta = json.load(open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8"))
        inst.dim = inst.index.d
        return inst
