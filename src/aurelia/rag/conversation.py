"""Conversational RAG: resolve follow-up questions against earlier turns.

"And creatinine?" means nothing alone. We carry over *modifiers* that shaped the
previous question (trend, latest, abnormal, a time window) when the follow-up
is short and introduces a new subject but no modifier of its own.
"""
from __future__ import annotations

import re

FOLLOWUP_OPENERS = re.compile(r"^\s*(and|also|what about|how about|then|now|ok(?:ay)?,?|what of)\b", re.I)
PRONOUNS = re.compile(r"\b(it|that|those|these|them|his|her|their|same)\b", re.I)
MODIFIERS = [r"\btrend\b", r"\b(latest|most recent|current|recent)\b", r"\b(abnormal|high|elevated|low)\b",
             r"\b(?:last|past)\s+\d*\s*(?:day|week|month|year)s?\b", r"\bsince\s+20\d{2}\b", r"\bin\s+20\d{2}\b",
             r"\bover time\b", r"\bforecast\b"]


def _modifiers(text: str) -> list[str]:
    out = []
    for rx in MODIFIERS:
        m = re.search(rx, text, re.I)
        if m:
            out.append(m.group(0).strip())
    return out


def condense(question: str, history: list[str]) -> tuple[str, str | None]:
    """Return (standalone question, explanation or None when unchanged)."""
    if not history:
        return question, None
    short = len(question.split()) <= 8
    if not (short and (FOLLOWUP_OPENERS.search(question) or PRONOUNS.search(question))):
        return question, None
    prev = next((h for h in reversed(history) if h.strip()), "")
    carry = [m for m in _modifiers(prev) if not re.search(re.escape(m), question, re.I)]
    if _modifiers(question) or not carry:
        return question, None
    cleaned = FOLLOWUP_OPENERS.sub("", question).strip(" ?.")
    standalone = f"{cleaned} {' '.join(carry)}".strip()
    return standalone, f"follow-up of “{prev[:60]}” → carried over: {', '.join(carry)}"
