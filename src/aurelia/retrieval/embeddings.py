"""Dependency-free text embeddings.

``HashingEmbedder`` builds sparse-to-dense TF-IDF style vectors with the hashing
trick over unigrams and bigrams, after expanding common clinical shorthand into
related terms. It is deterministic, fast, needs no model download and works
offline. For production-grade semantics, swap in ``SentenceTransformerEmbedder``
(optional extra: ``pip install aurelia[semantic]``) – same interface.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from itertools import pairwise

import numpy as np

TOKEN = re.compile(r"[a-z0-9\.]+")

# Lightweight clinical query expansion so that "sugar" can reach "glucose / HbA1c".
SYNONYMS = {
    "sugar": ["glucose", "hba1c", "diabetes"], "diabetic": ["diabetes", "hba1c", "glucose"],
    "bp": ["blood", "pressure", "systolic", "hypertension"], "pressure": ["hypertension", "systolic"],
    "kidney": ["renal", "creatinine", "egfr", "ckd"], "renal": ["kidney", "creatinine", "egfr"],
    "thyroid": ["tsh", "hypothyroidism", "levothyroxine"], "heart": ["cardiac", "coronary", "troponin", "ldl"],
    "cholesterol": ["ldl", "lipid", "statin", "atorvastatin"], "lung": ["copd", "asthma", "spo2", "inhaler"],
    "breathing": ["asthma", "copd", "spo2", "wheeze"], "anemia": ["hemoglobin", "ferritin", "iron"],
    "joint": ["arthritis", "crp", "esr", "methotrexate"], "mood": ["depression", "phq-9", "sertraline"],
    "allergy": ["allergies", "allergic"], "allergic": ["allergies"], "drugs": ["medications", "medication"],
    "medicines": ["medications", "medication"], "meds": ["medications", "medication"],
    "tests": ["lab", "result"], "labs": ["lab", "result"], "admitted": ["admission", "discharge"],
    "hospitalised": ["admission", "discharge"], "hospitalized": ["admission", "discharge"],
}
STOP = {"the", "a", "an", "of", "and", "or", "is", "are", "was", "were", "to", "for", "in", "on", "at", "with",
        "what", "which", "who", "how", "does", "do", "did", "has", "have", "had", "this", "that", "patient", "me",
        "show", "tell", "give", "any", "all", "from", "by", "be", "been", "his", "her", "their", "latest", "recent"}


def tokenize(text: str, expand: bool = False) -> list[str]:
    toks = [t.strip(".") for t in TOKEN.findall(text.lower())]
    toks = [t for t in toks if t and t not in STOP]
    if expand:
        extra: list[str] = []
        for t in toks:
            extra += SYNONYMS.get(t, [])
        toks += extra
    return toks


def _bucket(token: str, dim: int) -> tuple[int, float]:
    h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0


class HashingEmbedder:
    name = "hashing-tfidf"

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.idf: dict[str, float] = {}
        self._default_idf = 1.0

    def fit(self, texts: list[str]) -> HashingEmbedder:
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._features(tokenize(t))))
        n = len(texts)
        self.idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
        self._default_idf = math.log(n + 1) + 1.0
        return self

    @staticmethod
    def _features(toks: list[str]) -> list[str]:
        return toks + [f"{a}_{b}" for a, b in pairwise(toks)]

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            feats = Counter(self._features(tokenize(text, expand=is_query)))
            for f, tf in feats.items():
                idx, sign = _bucket(f, self.dim)
                out[i, idx] += sign * (1 + math.log(tf)) * self.idf.get(f, self._default_idf)
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts, is_query: bool = False):
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)
