"""Agentic RAG: a tool-using planner in the ReAct style (Thought -> Action -> Observation).

The planner inspects the question and chooses tools: text retrieval, lab-trend
analysis, safety alerts, current medications, a cohort (graph) query, or note
entities. It executes up to ``MAX_STEPS`` actions, *reflects* on what came back
(if nothing produced evidence it retries with multi-query retrieval) and
composes one cited answer. Planning is rule-based so it is deterministic and
testable; the tool interface is what an LLM planner would drive.

Lines beginning with "≈" are model-derived (scores, projections) and carry no
citation; the verifier treats them separately from record-backed claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..analytics import clinical
from ..analytics.forecast import anomaly, forecast
from ..nlp.ner import summarize_notes
from ..retrieval.hybrid import Hit
from .analysis import ALERT_WORDS, MED_WORDS, RISK_WORDS, TREND_WORDS, is_cohort_question, lay_to_tests, mentioned_tests
from .base import Context, RAGResult, Strategy, Tracer
from .fusion import rrf_fuse

MAX_STEPS = 5


@dataclass
class ToolOut:
    lines: list[tuple[str, list[str]]] = field(default_factory=list)   # (text, chunk ids)
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.lines


def _chunk_ids(ctx: Context, pid: str, kind: str, **match) -> list[str]:
    out = []
    for i in ctx.index.by_patient.get(pid, []):
        c = ctx.index.chunks[i]
        if c.kind == kind and all(c.meta.get(k) == v for k, v in match.items()):
            out.append(c.chunk_id)
    return out


def _latest_lab_chunk(ctx: Context, pid: str, test: str) -> list[str]:
    cs = [ctx.index.chunks[i] for i in ctx.index.by_patient.get(pid, [])
          if ctx.index.chunks[i].kind == "lab" and ctx.index.chunks[i].meta["test"] == test]
    cs.sort(key=lambda c: c.date)
    return [cs[-1].chunk_id] if cs else []


ALERT_EVIDENCE = {
    "Allergy conflict": lambda ctx, pid: _chunk_ids(ctx, pid, "profile") + _chunk_ids(ctx, pid, "medication", current=True),
    "Hyperkalaemia risk": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "Serum Potassium") + _chunk_ids(ctx, pid, "medication", current=True),
    "Renal dose review": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "eGFR") + _chunk_ids(ctx, pid, "medication", current=True),
    "Methotrexate with reduced eGFR": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "eGFR") + _chunk_ids(ctx, pid, "medication", current=True),
    "Poor glycaemic control": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "HbA1c"),
    "Low oxygen saturation": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "SpO2"),
    "Significant anaemia": lambda ctx, pid: _latest_lab_chunk(ctx, pid, "Hemoglobin"),
    "Condition marked worsening": lambda ctx, pid: _chunk_ids(ctx, pid, "diagnosis", status="worsening"),
}


# ------------------------------------------------------------------ tools
def tool_lab_trend(ctx: Context, pid: str, test: str) -> ToolOut:
    p = ctx.records[pid]
    series = clinical.lab_series(p).get(test)
    if not series:
        return ToolOut(note=f"no {test} results on record")
    first, last = series[0], series[-1]
    tr = clinical.trend(series)
    cids = [c for c in dict.fromkeys(_latest_lab_chunk(ctx, pid, test)
            + [c.chunk_id for c in (ctx.index.chunks[i] for i in ctx.index.by_patient[pid])
               if c.kind == "lab" and c.meta["test"] == test and c.date == first["date"]])]
    out = ToolOut()
    out.lines.append((f"{test} is {tr}: {first['value']} on {first['date']} → {last['value']} {last['unit']} on {last['date']} "
                      f"(latest flagged {last['flag']}).", cids))
    f = forecast(series)
    if f:
        out.lines.append((f"≈ Projection (linear fit, n={f['n']}, R²={f['r2']}): {f['predicted']} {last['unit']} by {f['target_date']} "
                          f"(95% interval {f['low']}–{f['high']}). Statistical extrapolation, not a clinical prediction.", []))
    a = anomaly(series)
    if a and a["flag"]:
        out.lines.append((f"≈ Latest {test} is unusual against this patient's own history (robust z = {a['z']}).", []))
    return out


def tool_alerts(ctx: Context, pid: str) -> ToolOut:
    out = ToolOut()
    for a in clinical.alerts(ctx.records[pid]):
        cids = list(dict.fromkeys(ALERT_EVIDENCE.get(a["title"], lambda *_: [])(ctx, pid)))
        out.lines.append((f"[{a['severity'].upper()}] {a['title']}: {a['detail']}", cids))
    if out.empty:
        out.note = "no rule-based safety alerts fired"
    return out


def tool_risk(ctx: Context, pid: str) -> ToolOut:
    r = clinical.risk_score(ctx.records[pid])
    top = ", ".join(f"{f['factor']} (+{f['points']})" for f in sorted(r["factors"], key=lambda f: -f["points"])[:3])
    return ToolOut(lines=[(f"≈ Complexity score {r['score']}/100 ({r['band']}); biggest contributors: {top}. Heuristic, not a validated risk model.", [])])


def tool_medications(ctx: Context, pid: str) -> ToolOut:
    cids = _chunk_ids(ctx, pid, "medication", current=True)
    if not cids:
        return ToolOut(note="no current medications on record")
    c = ctx.index.get(cids[0])
    return ToolOut(lines=[(c.text, [cids[0]])])


def tool_entities(ctx: Context, pid: str) -> ToolOut:
    s = summarize_notes(ctx.records[pid]["notes"])
    out = ToolOut()
    notes = _chunk_ids(ctx, pid, "note")
    if s["present"]:
        out.lines.append(("Symptoms documented in notes: " + ", ".join(f"{r['name']} (×{r['mentions']})" for r in s["present"]) + ".", notes[:2]))
    if s["denied"]:
        out.lines.append(("Symptoms explicitly denied: " + ", ".join(r["name"] for r in s["denied"]) + ".", notes[:2]))
    if out.empty:
        out.note = "no symptoms recognised in notes"
    return out


# ------------------------------------------------------------------ planner
def plan(question: str, patient_id: str | None) -> list[tuple[str, str]]:
    ql = question.lower()
    steps: list[tuple[str, str]] = []
    tests = mentioned_tests(question) or lay_to_tests(question)
    if patient_id:
        if re.search(ALERT_WORDS, ql) and not re.search(r"\brisk factor", ql):
            steps.append(("alerts", ""))
        if re.search(RISK_WORDS, ql):
            steps.append(("risk", ""))
        if re.search(TREND_WORDS, ql):
            if tests:
                steps += [("lab_trend", t) for t in tests[:3]]
            else:
                steps.append(("lab_trend", "*"))
        if re.search(MED_WORDS, ql):
            steps.append(("medications", ""))
        if re.search(r"symptom|complain|present|note|report|denies", ql):
            steps.append(("entities", ""))
    elif is_cohort_question(question):
        steps.append(("cohort", ""))
    return list(dict.fromkeys(steps)) or [("retrieve", "")]


class AgenticRAG(Strategy):
    id = "agentic"
    name = "Agentic RAG"
    family = "Agentic"
    summary = "Plans which tools to call (retrieval, lab trends, alerts, medications, cohort query, note entities), runs them, reflects, then answers."
    when = "Multi-part or analytical questions: trends, safety checks, 'is this patient at risk?', anything needing more than one lookup."

    def run(self, question: str, ctx: Context, patient_id: str | None, k: int) -> RAGResult:
        tr = Tracer()
        steps = plan(question, patient_id)
        tr.add("decision", "Thought", f"Question needs: {', '.join(a + (f'({b})' if b else '') for a, b in steps)}")
        hits: list[Hit] = []
        numbering: dict[str, int] = {}
        sections: list[str] = []

        def cite(cids: list[str]) -> str:
            marks = []
            for cid in cids:
                if cid not in numbering:
                    numbering[cid] = len(numbering) + 1
                    hits.append(Hit(ctx.index.get(cid), 1.0, None, None, via="tool"))
                marks.append(f"[{numbering[cid]}]")
            return "".join(marks)

        executed = 0
        structured = None
        for name, arg in steps[:MAX_STEPS]:
            executed += 1
            if name == "cohort":
                sub = ctx.graph and self._cohort(question, ctx, k)
                if sub and sub.hits:
                    tr.add("tool", "Action: cohort_query", f"graph matched {len(sub.structured['patients'])} patient(s)")
                    structured = sub.structured
                    return RAGResult(sub.hits, tr.steps + sub.trace, sub.answer, structured, sub.confidence, False, self.id)
                tr.add("tool", "Action: cohort_query", "no graph pattern recognised")
                continue
            if name == "retrieve":
                res = ctx.index.search(question, k=k, patient_id=patient_id)
                tr.add("tool", "Action: retrieve", f"{len(res)} chunk(s) from hybrid search")
                if res:
                    sections.append("Most relevant evidence in the record:")
                    sections += [f"- {h.chunk.text} {cite([h.chunk.chunk_id])}" for h in res[:4]]
                continue
            assert patient_id
            if name == "lab_trend" and arg == "*":
                cands = [t for t, pts in clinical.lab_series(ctx.records[patient_id]).items() if len(pts) >= 2]
                cands.sort(key=lambda t: clinical.trend(clinical.lab_series(ctx.records[patient_id])[t]) == "stable")
                outs = [(t, tool_lab_trend(ctx, patient_id, t)) for t in cands[:3]]
            elif name == "lab_trend":
                outs = [(arg, tool_lab_trend(ctx, patient_id, arg))]
            else:
                fn = {"alerts": tool_alerts, "risk": tool_risk, "medications": tool_medications, "entities": tool_entities}[name]
                outs = [(name, fn(ctx, patient_id))]
            for label, out in outs:
                tr.add("tool", f"Action: {name}" + (f"({label})" if name == "lab_trend" else ""),
                       out.note or f"{len(out.lines)} finding(s)")
                if out.empty:
                    continue
                head = {"alerts": "Safety alerts:", "risk": None, "medications": "Medications:",
                        "entities": "Notes analysis:", "lab_trend": f"Lab trend – {label}:"}[name]
                if head:
                    sections.append(head)
                for text, cids in out.lines:
                    sections.append(f"- {text} {cite(cids)}".rstrip())

        if not hits and not any(s.startswith("- ≈") for s in sections):
            tr.add("decision", "Reflect", "no tool produced evidence → retry with multi-query retrieval")
            from .analysis import query_variants
            fused = rrf_fuse([ctx.index.search(v, k=k, patient_id=patient_id) for v in query_variants(question)], k)
            tr.add("retrieve", "Multi-query retrieval", f"{len(fused)} chunk(s)")
            if fused:
                sections.append("Most relevant evidence in the record:")
                sections += [f"- {h.chunk.text} {cite([h.chunk.chunk_id])}" for h in fused[:4]]
        tr.add("decision", "Answer", f"composed from {executed} action(s), {len(hits)} cited chunk(s)")
        answer = "\n".join(s.replace("- ≈", "≈") if s.startswith("- ≈") else s for s in sections) or None
        conf = 0.85 if hits else (0.5 if answer else 0.0)
        return RAGResult(hits, tr.steps, answer, structured, conf, answer is None, self.id)

    @staticmethod
    def _cohort(question: str, ctx: Context, k: int) -> RAGResult | None:
        from .strategies import GraphRAG
        res = GraphRAG().run(question, ctx, None, k)
        return res if res.structured else None
