"""Query understanding: intent, time expressions, decomposition, lay-language mapping."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from ..nlp.vocab import LAB_ALIASES
from ..retrieval.embeddings import SYNONYMS, tokenize

EXTRA_STOP = {"result", "results", "value", "values", "status", "current", "history", "information", "info", "details",
              "summary", "there", "while", "also", "please", "about", "record", "records", "level", "levels", "number",
              "numbers", "reading", "readings", "found", "mentioned", "documented", "using", "taking", "currently",
              "last", "past", "since", "over", "within", "months", "month", "years", "year", "weeks", "week", "days"}

TREND_WORDS = r"(trend|trajectory|over time|changing|changed|progress|rising|falling|worsen|worse|improv|forecast|predict|projection|next)"
ALERT_WORDS = r"(alert|danger|unsafe|safe|interaction|conflict|concern|warning|contraindic|red flag|risk)"
RISK_WORDS = r"(risk|complexity|severity|how sick|priority)"
MED_WORDS = r"(medicat|medicine|drug|meds|prescri|dose|tablet|inhaler)"
LAB_WORDS = r"(lab|result|test|level|value|hba1c|creatinine|egfr|tsh|glucose|ldl|hemoglobin|crp|esr|potassium|spo2)"
ENC_WORDS = r"(admit|admission|discharge|hospital|visit|encounter|emergency)"
COHORT_WORDS = r"(which patients|which people|who has|who have|who is|who are|how many|list (?:all )?patients|patients (?:with|on|who|taking|having)|all patients|cohort)"


def content_tokens(q: str) -> list[str]:
    return [t for t in tokenize(q) if t not in EXTRA_STOP]


def intent(q: str) -> str:
    ql = q.lower()
    if re.search(r"allerg", ql):
        return "allergy"
    if re.search(MED_WORDS, ql):
        return "medication"
    if re.search(LAB_WORDS, ql):
        return "lab"
    if re.search(ENC_WORDS, ql):
        return "encounter"
    return "general"


def is_cohort_question(q: str) -> bool:
    return bool(re.search(COHORT_WORDS, q.lower()))


def mentioned_tests(q: str) -> list[str]:
    """Lab tests named directly in the question (longest alias wins)."""
    ql = q.lower()
    found: list[str] = []
    pairs = sorted(((a, t) for t, al in LAB_ALIASES.items() for a in al), key=lambda x: -len(x[0]))
    for alias, test in pairs:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", ql) and test not in found:
            found.append(test)
            ql = re.sub(re.escape(alias), " ", ql)   # avoid double counting 'ldl' inside 'ldl cholesterol'
    return found


def lay_to_tests(q: str) -> list[str]:
    """Map lay words ('kidney', 'sugar', 'thyroid') to lab tests via the synonym table."""
    tests: list[str] = []
    for tok in tokenize(q):
        for syn in [tok, *SYNONYMS.get(tok, [])]:
            for test, aliases in LAB_ALIASES.items():
                if syn in aliases and test not in tests:
                    tests.append(test)
    return tests


# ---------------------------------------------------------------- time expressions
@dataclass
class TimeWindow:
    start: date | None = None
    end: date | None = None
    latest: bool = False
    label: str = ""

    @property
    def active(self) -> bool:
        return self.latest or self.start is not None or self.end is not None


_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def parse_time(q: str, asof: date) -> TimeWindow:
    ql = q.lower()
    m = re.search(r"(?:last|past|previous|within the last)\s+(\d+|a|one|two|three|six|twelve)?\s*(day|week|month|year)s?", ql)
    if m:
        words = {"a": 1, "one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12, None: 1}
        n = int(m.group(1)) if m.group(1) and m.group(1).isdigit() else words.get(m.group(1), 1)
        days = n * _UNIT_DAYS[m.group(2)]
        return TimeWindow(asof - timedelta(days=days), asof, False, f"last {n} {m.group(2)}{'s' if n != 1 else ''}")
    m = re.search(r"(?:in|during|from)\s+(20\d{2})\b", ql)
    if m:
        y = int(m.group(1))
        return TimeWindow(date(y, 1, 1), date(y, 12, 31), False, f"year {y}")
    m = re.search(r"since\s+(20\d{2})(?:-(\d{2}))?", ql)
    if m:
        y, mo = int(m.group(1)), int(m.group(2) or 1)
        return TimeWindow(date(y, mo, 1), asof, False, f"since {y}-{mo:02d}")
    if re.search(r"this year", ql):
        return TimeWindow(date(asof.year, 1, 1), asof, False, f"year {asof.year}")
    if re.search(r"last year", ql):
        return TimeWindow(date(asof.year - 1, 1, 1), date(asof.year - 1, 12, 31), False, f"year {asof.year - 1}")
    if re.search(r"\b(latest|most recent|newest|current|recent(?:ly)?|last (?:result|value|test|visit))\b", ql):
        return TimeWindow(None, None, True, "latest")
    return TimeWindow()


# ---------------------------------------------------------------- decomposition / rewriting
def decompose(q: str) -> list[str]:
    """Split compound questions into independent sub-questions (each needs >= 2 content words)."""
    parts = [p.strip(" ?.,") for p in re.split(r"\?\s+|;\s+|\s+and also\s+|\s+as well as\s+|\s+and\s+(?=(?:what|how|is|are|does|did|show|list|who|which)\b)", q, flags=re.I)]
    parts = [p for p in parts if len(content_tokens(p)) >= 1]
    return parts if len(parts) > 1 else [q.strip()]


def query_variants(q: str) -> list[str]:
    """Distinct rewrites of one question: original, synonym-expanded, keyword-only, intent-templated."""
    toks = content_tokens(q)
    variants = [q.strip()]
    expanded = " ".join(dict.fromkeys(tokenize(q, expand=True)))
    if expanded and expanded != " ".join(toks):
        variants.append(expanded)
    if toks:
        variants.append(" ".join(toks))
    direct = mentioned_tests(q)
    tests = direct + [t for t in lay_to_tests(q) if t not in direct]
    it = intent(q)
    if tests:
        variants.append("Lab result " + " ".join(tests))
    elif it == "medication":
        variants.append("Current medications " + " ".join(toks))
    elif it == "allergy":
        variants.append("Documented allergies")
    out: list[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out
