"""BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..ingestion.chunker import Chunk
from .embeddings import HashingEmbedder, tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.inv[t].append(i)

    def scores(self, query: list[str], allowed: set[int] | None = None) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                if allowed is not None and i not in allowed:
                    continue
                f = self.tf[i][t]
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    via: str = "retrieved"   # retrieved | context (added by window expansion) | graph | tool


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: HashingEmbedder | None = None):
        self.chunks = chunks
        self.embedder = (embedder or HashingEmbedder()).fit([c.text for c in chunks])
        self.matrix = self.embedder.encode([c.text for c in chunks])
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.by_patient: dict[str, list[int]] = defaultdict(list)
        self.by_id: dict[str, int] = {}
        for i, c in enumerate(chunks):
            self.by_patient[c.patient_id].append(i)
            self.by_id[c.chunk_id] = i
        dates = [c.date for c in chunks if c.date]
        self.asof: str = max(dates) if dates else "1970-01-01"   # "today" for temporal reasoning

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[self.by_id[chunk_id]]

    def dense_search(self, vec, k: int = 6, patient_id: str | None = None) -> list[Hit]:
        """Nearest chunks to an arbitrary vector (used by HyDE)."""
        allowed = self._allowed(patient_id)
        if allowed is not None and not allowed:
            return []
        sims = self.matrix @ vec
        idx = np.arange(len(self.chunks)) if allowed is None else np.fromiter(allowed, dtype=int)
        order = idx[np.argsort(-sims[idx])][:k]
        return [Hit(self.chunks[int(i)], round(float(sims[i]), 4), None, r + 1) for r, i in enumerate(order) if sims[i] > 0]

    def _allowed(self, patient_id: str | None) -> set[int] | None:
        return set(self.by_patient.get(patient_id, [])) if patient_id else None

    def search(self, query: str, k: int = 6, patient_id: str | None = None, mode: str = "hybrid",
               rrf_k: int = 60) -> list[Hit]:
        allowed = self._allowed(patient_id)
        if allowed is not None and not allowed:
            return []

        bm = self.bm25.scores(tokenize(query, expand=True), allowed)
        bm_rank = {i: r for r, (i, _) in enumerate(sorted(bm.items(), key=lambda x: -x[1]))}

        qv = self.embedder.encode([query], is_query=True)[0]
        sims = self.matrix @ qv
        idx = np.arange(len(self.chunks)) if allowed is None else np.fromiter(allowed, dtype=int)
        order = idx[np.argsort(-sims[idx])]
        dense_rank = {int(i): r for r, i in enumerate(order[:200]) if sims[i] > 0}

        if mode == "bm25":
            fused = {i: 1.0 / (rrf_k + r) for i, r in bm_rank.items()}
        elif mode == "dense":
            fused = {i: 1.0 / (rrf_k + r) for i, r in dense_rank.items()}
        else:
            fused = defaultdict(float)
            for i, r in bm_rank.items():
                fused[i] += 1.0 / (rrf_k + r)
            for i, r in dense_rank.items():
                fused[i] += 1.0 / (rrf_k + r)

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(self.chunks[i], round(s * rrf_k, 4), bm_rank.get(i), dense_rank.get(i)) for i, s in top]
