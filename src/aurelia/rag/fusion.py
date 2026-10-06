from __future__ import annotations

from collections import defaultdict

from ..retrieval.hybrid import Hit


def rrf_fuse(rankings: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion across several ranked lists of hits (de-duplicated by chunk id)."""
    score: dict[str, float] = defaultdict(float)
    keep: dict[str, Hit] = {}
    for hits in rankings:
        for rank, h in enumerate(hits):
            cid = h.chunk.chunk_id
            score[cid] += 1.0 / (rrf_k + rank)
            keep.setdefault(cid, h)
    top = sorted(score.items(), key=lambda x: -x[1])[:k]
    return [Hit(keep[c].chunk, round(s * rrf_k, 4), keep[c].bm25_rank, keep[c].dense_rank, keep[c].via) for c, s in top]
