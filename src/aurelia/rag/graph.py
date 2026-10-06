"""Knowledge graph + structured query parsing (Graph / cohort RAG).

Text retrieval is the wrong tool for questions that are really *queries over
the whole cohort* ("which patients on metformin have eGFR below 45?"):
the answer lives in the intersection of several facts across many records, and
top-k chunk search cannot count or intersect. Here the records are lifted into
typed nodes (patients, conditions, drugs, lab tests, symptoms, allergens) and
a question is parsed into constraints that are evaluated exactly. Every match
is returned together with the chunk ids that prove each constraint, so graph
answers stay citable.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..ingestion.chunker import Chunk
from ..nlp.ner import extract_entities
from ..nlp.vocab import ALLERGENS, CONDITION_ALIASES, DRUG_CLASSES, INHALER_WORDS, LAB_ALIASES, SYMPTOMS


# ------------------------------------------------------------------ query model
@dataclass
class LabCond:
    test: str
    op: str                 # '>', '<', 'high', 'low', 'abnormal'
    value: float | None = None

    def describe(self) -> str:
        sym = {">": f"> {self.value}", "<": f"< {self.value}", "high": "flagged HIGH", "low": "flagged LOW",
               "abnormal": "abnormal"}[self.op]
        return f"{self.test} {sym}"


@dataclass
class GraphQuery:
    conditions: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    drug_label: str = ""
    allergies: list[str] = field(default_factory=list)
    labs: list[LabCond] = field(default_factory=list)
    age: tuple[str, int] | None = None
    sex: str | None = None
    symptoms: list[str] = field(default_factory=list)
    count_only: bool = False

    def is_empty(self) -> bool:
        return not (self.conditions or self.drugs or self.allergies or self.labs or self.age or self.sex or self.symptoms)

    def describe(self) -> str:
        bits = [*(f"condition = {c}" for c in self.conditions),
                *([f"on {self.drug_label or ', '.join(self.drugs)}"] if self.drugs else []),
                *(f"allergy = {a}" for a in self.allergies), *(l_.describe() for l_ in self.labs),
                *([f"age {self.age[0]} {self.age[1]}"] if self.age else []),
                *([f"sex = {self.sex}"] if self.sex else []), *(f"symptom = {s}" for s in self.symptoms)]
        return " AND ".join(bits) or "(no constraints)"


_CMP = {"above": ">", "over": ">", "greater than": ">", "higher than": ">", "more than": ">", "exceeding": ">", ">": ">",
        ">=": ">", "at least": ">", "below": "<", "under": "<", "less than": "<", "lower than": "<", "<": "<", "<=": "<"}
_CMP_RX = "|".join(sorted((re.escape(c) for c in _CMP), key=len, reverse=True))
_ALIAS_TO_TEST = sorted(((a, t) for t, al in LAB_ALIASES.items() for a in al), key=lambda x: -len(x[0]))
_TEST_RX = "|".join(re.escape(a) for a, _ in _ALIAS_TO_TEST)


def _test_of(alias: str) -> str:
    alias = alias.lower()
    return next(t for a, t in _ALIAS_TO_TEST if a == alias)


def parse_graph_query(q: str, known_drugs: set[str]) -> GraphQuery:
    """Turn a natural-language cohort question into explicit constraints."""
    gq = GraphQuery()
    ql = q.lower()
    gq.count_only = bool(re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", ql))

    # 1) numeric lab thresholds: "eGFR below 45", "HbA1c over 8"
    for m in re.finditer(rf"(?P<t>{_TEST_RX})\s*(?:level|levels|value|result|results|score)?\s*(?:is|was|of|being)?\s*(?P<c>{_CMP_RX})\s*(?P<n>\d+(?:\.\d+)?)", ql):
        gq.labs.append(LabCond(_test_of(m.group("t")), _CMP[m.group("c")], float(m.group("n"))))
    ql = re.sub(rf"(?:{_TEST_RX})\s*(?:level|levels|value|result|results|score)?\s*(?:is|was|of|being)?\s*(?:{_CMP_RX})\s*\d+(?:\.\d+)?", " ", ql)

    # 2) qualitative lab words: "high creatinine", "potassium elevated", "abnormal TSH"
    qual = {"high": "high", "elevated": "high", "raised": "high", "low": "low", "reduced": "low", "abnormal": "abnormal"}
    for m in re.finditer(rf"\b(?P<w>high|elevated|raised|low|reduced|abnormal)\s+(?:fasting\s+|serum\s+|blood\s+)?(?P<t>{_TEST_RX})\b", ql):
        gq.labs.append(LabCond(_test_of(m.group("t")), qual[m.group("w")]))
    for m in re.finditer(rf"\b(?P<t>{_TEST_RX})\s+(?:is|was|being|levels? (?:is|are))?\s*(?P<w>high|elevated|raised|low|abnormal)\b", ql):
        gq.labs.append(LabCond(_test_of(m.group("t")), qual[m.group("w")]))

    # 3) age / sex
    m = re.search(r"\b(over|older than|under|younger than)\s+(\d{2,3})\b", ql)
    if m and 18 <= int(m.group(2)) <= 110:
        gq.age = (">" if m.group(1) in ("over", "older than") else "<", int(m.group(2)))
    if re.search(r"\b(women|female|females|woman)\b", ql):
        gq.sex = "F"
    elif re.search(r"\b(men|male|males|man)\b", ql):
        gq.sex = "M"

    # 4) allergies
    if "allerg" in ql:
        for word, canon in ALLERGENS.items():
            if re.search(rf"\b{word}\b", ql) and canon not in gq.allergies:
                gq.allergies.append(canon)

    # 5) conditions
    for cond, aliases in CONDITION_ALIASES.items():
        names = sorted([cond.lower(), *aliases], key=len, reverse=True)
        if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", ql) for n in names):
            gq.conditions.append(cond)
    if re.search(r"poorly controlled|uncontrolled|poor control|not controlled", ql) and "Type 2 Diabetes Mellitus" in gq.conditions \
            and not any(l_.test == "HbA1c" for l_ in gq.labs):
        gq.labs.append(LabCond("HbA1c", ">", 8.0))   # same threshold as the glycaemic-control alert

    # 6) drugs & drug classes (only drugs actually present in the data)
    if "allerg" not in ql or re.search(r"\b(take|taking|on|prescribed|using)\b", ql):
        for d in sorted(known_drugs, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(d)}(?![a-z])", ql) and d not in gq.drugs:
                gq.drugs.append(d)
        for cls, members in DRUG_CLASSES.items():
            if re.search(rf"(?<![a-z]){re.escape(cls)}(?![a-z])", ql):
                for d in members:
                    if d in known_drugs and d not in gq.drugs:
                        gq.drugs.append(d)
                gq.drug_label = gq.drug_label or cls
        if any(re.search(rf"\b{w}\b", ql) for w in INHALER_WORDS):
            gq.drugs += [d for d in ("salbutamol", "budesonide", "tiotropium", "formoterol") if d in known_drugs and d not in gq.drugs]
            gq.drug_label = gq.drug_label or "inhaler"

    # 7) symptoms from notes (negated symptoms in the *question* are not supported)
    for canon, pats in SYMPTOMS.items():
        if re.search(r"\b(?:" + "|".join(pats) + r")\b", ql) and not re.search(rf"\b(?:no|without)\s+(?:{'|'.join(pats)})", ql):
            gq.symptoms.append(canon)
    return gq


# ------------------------------------------------------------------ graph
@dataclass
class Match:
    patient_id: str
    why: list[str]
    chunk_ids: list[str]


class KnowledgeGraph:
    def __init__(self, records: list[dict], chunks: list[Chunk]):
        self.records = {p["patient_id"]: p for p in records}
        self.cond: dict[str, dict[str, str]] = defaultdict(dict)         # pid -> condition -> chunk id
        self.drugs: dict[str, dict[str, str]] = defaultdict(dict)        # pid -> base drug -> chunk id (current only)
        self.profile: dict[str, str] = {}                                # pid -> profile chunk id
        self.lab: dict[str, dict[str, tuple]] = defaultdict(dict)        # pid -> test -> (date, value, flag, unit, chunk id) newest
        self.symptoms: dict[str, dict[str, str]] = defaultdict(dict)     # pid -> symptom (present) -> note chunk id
        self.known_drugs: set[str] = set()
        for c in chunks:
            pid = c.patient_id
            if c.kind == "profile":
                self.profile[pid] = c.chunk_id
            elif c.kind == "diagnosis":
                self.cond[pid][c.meta["condition"]] = c.chunk_id
            elif c.kind == "medication" and c.meta.get("current"):
                for d in c.meta["drugs"]:
                    self.drugs[pid][d] = c.chunk_id
                    self.known_drugs.add(d)
            elif c.kind == "medication":
                self.known_drugs.update(c.meta.get("drugs", []))
            elif c.kind == "lab":
                prev = self.lab[pid].get(c.meta["test"])
                if not prev or c.date > prev[0]:
                    self.lab[pid][c.meta["test"]] = (c.date, c.meta["value"], c.meta["flag"], c.meta["unit"], c.chunk_id)
            elif c.kind == "note":
                for e in extract_entities(c.text):
                    if e.type == "symptom" and not e.negated:
                        self.symptoms[pid].setdefault(e.text, c.chunk_id)

    def stats(self) -> dict:
        tests = {t for d in self.lab.values() for t in d}
        syms = {s for d in self.symptoms.values() for s in d}
        conds = {c for d in self.cond.values() for c in d}
        nodes = {"patients": len(self.records), "conditions": len(conds), "drugs": len(self.known_drugs),
                 "lab tests": len(tests), "symptoms": len(syms)}
        edges = {"has_condition": sum(len(d) for d in self.cond.values()),
                 "takes_drug": sum(len(d) for d in self.drugs.values()),
                 "has_lab": sum(len(d) for d in self.lab.values()),
                 "reports_symptom": sum(len(d) for d in self.symptoms.values())}
        return {"nodes": nodes, "edges": edges, "total_nodes": sum(nodes.values()), "total_edges": sum(edges.values())}

    def parse(self, q: str) -> GraphQuery:
        return parse_graph_query(q, self.known_drugs)

    # --- constraint evaluation: each returns (ok, why, chunk_id)
    def _lab_ok(self, pid: str, lc: LabCond):
        got = self.lab[pid].get(lc.test)
        if not got:
            return False, "", ""
        date, value, flag, unit, cid = got
        ok = {">": value > (lc.value or 0), "<": value < (lc.value or 0), "high": flag == "HIGH", "low": flag == "LOW",
              "abnormal": flag != "NORMAL"}[lc.op]
        return ok, f"{lc.test} {value} {unit} ({flag}, {date})", cid

    def match(self, gq: GraphQuery) -> list[Match]:
        out: list[Match] = []
        for pid, p in self.records.items():
            why: list[str] = []
            cids: list[str] = []
            ok = True
            for c in gq.conditions:
                if c in self.cond[pid]:
                    why.append(c); cids.append(self.cond[pid][c])
                else:
                    ok = False; break
            if ok and gq.drugs:
                hit = [d for d in gq.drugs if d in self.drugs[pid]]
                # several drugs in one class/phrase = OR within the group
                if hit:
                    why.append("on " + ", ".join(hit)); cids.append(self.drugs[pid][hit[0]])
                else:
                    ok = False
            if ok:
                for a in gq.allergies:
                    if a in p["allergies"].lower():
                        why.append(f"allergy: {p['allergies']}"); cids.append(self.profile[pid])
                    else:
                        ok = False; break
            if ok:
                for lc in gq.labs:
                    good, w, cid = self._lab_ok(pid, lc)
                    if good:
                        why.append(w); cids.append(cid)
                    else:
                        ok = False; break
            if ok and gq.age:
                good = p["age"] > gq.age[1] if gq.age[0] == ">" else p["age"] < gq.age[1]
                if good:
                    why.append(f"age {p['age']}"); cids.append(self.profile[pid])
                else:
                    ok = False
            if ok and gq.sex:
                if p["sex"] == gq.sex:
                    cids.append(self.profile[pid])
                else:
                    ok = False
            if ok:
                for s in gq.symptoms:
                    if s in self.symptoms[pid]:
                        why.append(f"reports {s}"); cids.append(self.symptoms[pid][s])
                    else:
                        ok = False; break
            if ok:
                out.append(Match(pid, why, list(dict.fromkeys(cids))))
        out.sort(key=lambda m: (-len(m.why), self.records[m.patient_id]["name"]))
        return out
