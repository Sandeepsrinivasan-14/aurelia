"""Reranking and diversification of a candidate pool.

``HeuristicReranker`` is a transparent, feature-based stand-in for a
cross-encoder: it scores each candidate against the *question* (not just the
first-stage retrieval score) using term coverage, phrase match, evidence-type
fit, abnormality and recency. Every feature is exposed so the UI can show *why*
a chunk moved up or down. A learned cross-encoder can replace it behind the
same ``rerank`` signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..retrieval.embeddings import SYNONYMS, tokenize
from ..retrieval.hybrid import Hit, HybridIndex
from .analysis import content_tokens, intent

KIND_FOR_INTENT = {"lab": "lab", "medication": "medication", "allergy": "profile", "encounter": "encounter"}
ABNORMAL_WORDS = re.compile(r"\b(abnormal|high|elevated|raised|low|reduced|poor|uncontrolled|out of range)\b", re.I)


@dataclass
class Features:
    coverage: float
    phrase: float
    kind_fit: float
    abnormal: float
    recency: float
    base: float

    def total(self) -> float:
        return round(0.35 * self.base + 0.35 * self.coverage + 0.10 * self.phrase + 0.12 * self.kind_fit
                     + 0.04 * self.abnormal + 0.04 * self.recency, 4)

    def explain(self) -> str:
        return (f"coverage {self.coverage:.2f} · phrase {self.phrase:.2f} · type-fit {self.kind_fit:.0f} · "
                f"abnormal {self.abnormal:.0f} · recency {self.recency:.2f} · base {self.base:.2f}")


def _covered(term: str, chunk_tokens: set[str]) -> bool:
    return term in chunk_tokens or any(s in chunk_tokens for s in SYNONYMS.get(term, []))


def term_coverage(question: str, text: str) -> float:
    q = set(content_tokens(question))
    if not q:
        return 0.0
    ct = set(tokenize(text))
    return sum(_covered(t, ct) for t in q) / len(q)


class HeuristicReranker:
    def features(self, question: str, hit: Hit, base_norm: float, date_rank: float) -> Features:
        text = hit.chunk.text
        qtoks = content_tokens(question)
        qbi = {f"{a} {b}" for a, b in zip(qtoks, qtoks[1:])}
        ctoks = tokenize(text)
        cbi = {f"{a} {b}" for a, b in zip(ctoks, ctoks[1:])}
        phrase = (len(qbi & cbi) / len(qbi)) if qbi else 0.0
        want = KIND_FOR_INTENT.get(intent(question))
        kind_fit = 1.0 if want and hit.chunk.kind == want else 0.0
        abn = 1.0 if (ABNORMAL_WORDS.search(question) and hit.chunk.meta.get("flag") in ("HIGH", "LOW")) else 0.0
        return Features(term_coverage(question, text), phrase, kind_fit, abn, date_rank, base_norm)

    def rerank(self, question: str, hits: list[Hit]) -> tuple[list[Hit], list[Features]]:
        if not hits:
            return [], []
        top = max(h.score for h in hits) or 1.0
        dates = sorted({h.chunk.date for h in hits if h.chunk.date})
        rank = {d: (i + 1) / len(dates) for i, d in enumerate(dates)}
        scored = []
        for h in hits:
            f = self.features(question, h, h.score / top, rank.get(h.chunk.date, 0.0))
            scored.append((f.total(), h, f))
        scored.sort(key=lambda t: -t[0])
        out = [Hit(h.chunk, round(s * 100, 2), h.bm25_rank, h.dense_rank, h.via) for s, h, _ in scored]
        return out, [f for _, _, f in scored]


def mmr(index: HybridIndex, hits: list[Hit], k: int, lam: float = 0.7) -> list[Hit]:
    """Maximal Marginal Relevance: balance relevance against redundancy with what is already chosen."""
    if len(hits) <= 1:
        return hits[:k]
    rel = np.array([h.score for h in hits], dtype=float)
    rel = rel / (rel.max() or 1.0)
    vecs = np.stack([index.matrix[index.by_id[h.chunk.chunk_id]] for h in hits])
    sim = vecs @ vecs.T
    chosen: list[int] = [int(np.argmax(rel))]
    while len(chosen) < min(k, len(hits)):
        best, best_score = None, -1e9
        for i in range(len(hits)):
            if i in chosen:
                continue
            score = lam * rel[i] - (1 - lam) * max(sim[i, j] for j in chosen)
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
    return [hits[i] for i in chosen]
