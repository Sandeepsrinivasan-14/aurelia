"""Lexicon-based clinical entity extraction with negation detection.

Negation follows the NegEx idea: a mention is *negated* when a trigger such as
"no", "denies" or "without" appears shortly before it in the same sentence.
"Patient has no chest pain" must not be retrieved as evidence *of* chest pain.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .vocab import CONDITION_ALIASES, SYMPTOMS

NEG_TRIGGERS = re.compile(r"\b(no|denies|denied|without|negative for|not|absent|free of|ruled out|nil)\b", re.I)
TERMINATORS = re.compile(r"\b(but|however|although|though|except|apart from|aside from)\b", re.I)  # end a negation scope
WINDOW = 5   # tokens between a trigger and the mention

_SYMPTOM_RX = {c: re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.I) for c, pats in SYMPTOMS.items()}
_COND_RX = {c: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted([c.lower(), *al], key=len, reverse=True)) + r")\b", re.I)
            for c, al in CONDITION_ALIASES.items()}
_ALLERGY_RX = re.compile(r"allerg(?:y|ies)(?: status)?:\s*([A-Za-z ,]+?)(?:\.|$)", re.I)


@dataclass
class Entity:
    type: str        # symptom | condition | allergen
    text: str        # canonical name
    negated: bool
    sentence: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def is_negated(sentence: str, start: int) -> bool:
    """True if a negation trigger sits within WINDOW tokens before character offset ``start``."""
    before = sentence[:start]
    ends = [m.end() for m in TERMINATORS.finditer(before)]  # keep only text after the final scope terminator
    if ends:
        before = before[ends[-1]:]
    tokens = before.split()
    tail = " ".join(tokens[-(WINDOW + 1):])
    return bool(NEG_TRIGGERS.search(tail))


def extract_entities(text: str) -> list[Entity]:
    out: list[Entity] = []
    for sent in _sentences(text):
        for canon, rx in _SYMPTOM_RX.items():
            m = rx.search(sent)
            if m:
                out.append(Entity("symptom", canon, is_negated(sent, m.start()), sent))
        for canon, rx in _COND_RX.items():
            m = rx.search(sent)
            if m:
                out.append(Entity("condition", canon, is_negated(sent, m.start()), sent))
        a = _ALLERGY_RX.search(sent)
        if a:
            for item in re.split(r",| and ", a.group(1)):
                item = item.strip()
                if item and item.lower() not in ("none known", "none"):
                    out.append(Entity("allergen", item, False, sent))
    return out


def summarize_notes(notes: list[dict]) -> dict:
    """Aggregate entities across a patient's notes: present vs explicitly denied."""
    present: dict[str, dict] = {}
    denied: dict[str, dict] = {}
    for n in notes:
        for e in extract_entities(n["text"]):
            if e.type != "symptom":
                continue
            bucket = denied if e.negated else present
            row = bucket.setdefault(e.text, {"name": e.text, "mentions": 0, "last": n["date"]})
            row["mentions"] += 1
            row["last"] = max(row["last"], n["date"])
    key = lambda r: (-r["mentions"], r["name"])  # noqa: E731
    return {"present": sorted(present.values(), key=key), "denied": sorted(denied.values(), key=key)}
