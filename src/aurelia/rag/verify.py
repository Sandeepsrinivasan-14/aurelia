"""Answer verification: is every cited claim actually supported by its evidence?

For each answer line that carries citations, two checks run against the *cited
chunks only*: (1) every number in the claim must appear in the evidence, and
(2) enough of the claim's content words must appear there. Uncited factual
lines are flagged. Lines starting with "≈" are model-derived (forecasts,
scores) and are reported separately rather than judged against the record.
"""
from __future__ import annotations

import re

from ..retrieval.embeddings import tokenize
from .analysis import EXTRA_STOP

CITE = re.compile(r"\[(\d+)\]")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    return {t for t in tokenize(CITE.sub(" ", text)) if t not in EXTRA_STOP}


def verify_answer(answer: str, evidence: list[dict], min_overlap: float = 0.5) -> dict:
    """``evidence`` is the list of evidence dicts with ``n`` and ``text``."""
    if not evidence and not CITE.search(answer):   # nothing retrieved, nothing claimed -> nothing to verify
        return {"faithfulness": 1.0, "claims": [], "derived_lines": 0, "supported": 0, "total": 0}
    by_n = {e["n"]: e["text"] for e in evidence}
    claims, derived = [], 0
    for raw in answer.splitlines():
        line = raw.strip(" -•\t")
        if not line or line.endswith(":"):
            continue
        if line.startswith("≈"):
            derived += 1
            continue
        cites = [int(n) for n in CITE.findall(line)]
        claim = CITE.sub("", line).strip()
        if not cites:
            claims.append({"text": claim, "cites": [], "supported": False, "reason": "no citation"})
            continue
        missing = [n for n in cites if n not in by_n]
        if missing:
            claims.append({"text": claim, "cites": cites, "supported": False, "reason": f"unknown citation {missing}"})
            continue
        ev_text = " ".join(by_n[n] for n in cites)
        ev_tokens = _tokens(ev_text)
        nums = NUM.findall(claim)
        bad_nums = [n for n in nums if not re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", ev_text)]
        toks = _tokens(claim)
        overlap = (len(toks & ev_tokens) / len(toks)) if toks else 1.0
        ok = not bad_nums and overlap >= min_overlap
        reason = "" if ok else (f"numbers not in evidence: {bad_nums}" if bad_nums else f"low overlap {overlap:.2f}")
        claims.append({"text": claim, "cites": cites, "supported": ok, "reason": reason})
    n = len(claims)
    sup = sum(c["supported"] for c in claims)
    return {"faithfulness": round(sup / n, 3) if n else 1.0, "claims": claims, "derived_lines": derived,
            "supported": sup, "total": n}
