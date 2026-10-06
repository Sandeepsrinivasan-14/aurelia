"""The strategy catalogue: thirteen ways of doing retrieval-augmented generation.

Baselines:   bm25, naive (dense), hybrid
Retrieval:   multiquery, hyde, rerank, mmr, window
Reasoning:   temporal, corrective (self-grading, abstains)
Structured:  graph (cohort queries over a knowledge graph)
Agentic:     agentic (tool-using planner)
Router:      adaptive (picks the right one per question)
"""
from __future__ import annotations

import re
from collections import OrderedDict
from datetime import date

from ..nlp.vocab import CONDITION_ALIASES
from ..retrieval.embeddings import SYNONYMS, tokenize
from ..retrieval.hybrid import Hit
from .agent import AgenticRAG
from .analysis import (
    ALERT_WORDS,
    RISK_WORDS,
    TREND_WORDS,
    content_tokens,
    decompose,
    intent,
    is_cohort_question,
    lay_to_tests,
    mentioned_tests,
    parse_time,
    query_variants,
)
from .base import Context, RAGResult, Step, Strategy, Tracer
from .fusion import rrf_fuse
from .rerank import HeuristicReranker, mmr

# ------------------------------------------------------------------ helpers
HIGH_CONF, MED_CONF = 0.6, 0.34


def grade(question: str, hits: list[Hit], top_n: int = 3) -> dict:
    """Self-assessment of retrieval: do the top chunks actually cover what the question asks about?"""
    q = set(content_tokens(question))
    if not hits:
        return {"coverage": 0.0, "label": "low", "uncovered": sorted(q)}
    ev = set()
    for h in hits[:top_n]:
        ev |= set(tokenize(h.chunk.text))
    covered = {t for t in q if t in ev or any(s in ev for s in SYNONYMS.get(t, []))}
    cov = len(covered) / len(q) if q else 1.0
    label = "high" if cov >= HIGH_CONF else "medium" if cov >= MED_CONF else "low"
    return {"coverage": round(cov, 3), "label": label, "uncovered": sorted(q - covered)}


def _asof(ctx: Context) -> date:
    return date.fromisoformat(ctx.index.asof)


def _hybrid(ctx: Context, q: str, pid: str | None, n: int) -> list[Hit]:
    return ctx.index.search(q, k=n, patient_id=pid, mode="hybrid")


def _one(tr: Tracer, kind: str, name: str, hits: list[Hit], extra: str = "") -> None:
    tr.add(kind, name, f"{len(hits)} chunk(s){(' · ' + extra) if extra else ''}")


# ------------------------------------------------------------------ baselines
class BM25RAG(Strategy):
    id, name, family = "bm25", "Keyword (BM25)", "Baseline"
    summary = "Classic lexical search over chunk text. Fast and exact on terms, blind to paraphrase."
    when = "The question uses the same words as the record (drug names, lab names, dates)."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        hits = ctx.index.search(q, k=k, patient_id=pid, mode="bm25")
        _one(tr, "retrieve", "BM25 search", hits)
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


class NaiveRAG(Strategy):
    id, name, family = "naive", "Dense vectors", "Baseline"
    summary = "Embed the question and fetch nearest chunks by cosine similarity - the textbook 'naive RAG'."
    when = "Paraphrased or lay-language questions where exact words differ from the record."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        hits = ctx.index.search(q, k=k, patient_id=pid, mode="dense")
        _one(tr, "retrieve", "Dense search", hits)
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


class HybridRAG(Strategy):
    id, name, family = "hybrid", "Hybrid (BM25 + dense)", "Baseline"
    summary = "Runs lexical and dense search, merges the rankings with Reciprocal Rank Fusion."
    when = "A strong general default when you don't know how the question is phrased."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        hits = ctx.index.search(q, k=k, patient_id=pid, mode="hybrid")
        _one(tr, "retrieve", "BM25 + dense → RRF", hits)
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


# ------------------------------------------------------------------ retrieval-side improvements
class MultiQueryRAG(Strategy):
    id, name, family = "multiquery", "Multi-query + decomposition", "Retrieval"
    summary = "Splits compound questions and rewrites each (synonyms, keywords, templates); retrieves per variant and fuses."
    when = "Compound questions ('X? and Y?') or terse questions whose wording may not match the record."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        parts = decompose(q)
        variants = list(dict.fromkeys(v for p in parts for v in query_variants(p)))
        tr.add("rewrite", "Decompose & rewrite", f"{len(parts)} sub-question(s) → {len(variants)} variant(s): " + " | ".join(variants[:5]))
        rankings = []
        for v in variants:
            r = _hybrid(ctx, v, pid, k * 2)
            rankings.append(r)
        tr.add("retrieve", "Retrieve per variant", f"{len(variants)} hybrid searches")
        hits = rrf_fuse(rankings, k)
        _one(tr, "retrieve", "Reciprocal Rank Fusion", hits)
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


def hyde_passages(question: str, ctx: Context) -> tuple[list[str], str]:
    """Hypothetical evidence the record *would* contain if it answered the question (HyDE)."""
    if ctx.llm_generate:
        out = ctx.llm_generate("Write one short hypothetical clinical record excerpt (1-2 sentences) that would directly answer "
                               f"this question. Output only the excerpt.\nQuestion: {question}")
        if out:
            return [out.strip()], "LLM-written"
    ql = question.lower()
    it = intent(question)
    direct = mentioned_tests(question)
    tests = (direct + [t for t in lay_to_tests(question) if t not in direct])[:3]
    flag = "HIGH" if re.search(r"\b(high|elevated|raised|poor|uncontrolled)\b", ql) else "LOW" if re.search(r"\b(low|reduced)\b", ql) else "NORMAL"
    conds = [c for c, al in CONDITION_ALIASES.items() if any(a in ql for a in [c.lower(), *al])]
    out: list[str] = []
    if tests:
        out += [f"Lab result {t}: value recorded on 2026-01-10, flagged {flag}." for t in tests]
    if it == "allergy":
        out.append("Documented allergies: patient has a documented allergy.")
    if it == "medication":
        out.append("Current medications: " + "; ".join(f"drug for {c}" for c in conds) if conds else "Current medications: tablets and inhalers.")
    if it == "encounter":
        out.append("Admission at the hospital for " + (conds[0] if conds else "a chronic condition") + ". Discharge summary: admitted and discharged stable.")
    for c in conds[:2]:
        out.append(f"Diagnosis: {c}, onset recorded, status active.")
    return out or [question], "template"


class HyDERAG(Strategy):
    id, name, family = "hyde", "HyDE (hypothetical document)", "Retrieval"
    summary = "Writes the evidence the record *would* contain if it answered the question, embeds that, and searches with it."
    when = "Vague or lay-language questions ('how is the kidney doing?') where the question looks nothing like the record text."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        passages, how = hyde_passages(q, ctx)
        tr.add("rewrite", f"Hypothetical passage ({how})", " | ".join(passages)[:240])
        rankings = []
        for p in passages:
            vec = ctx.index.embedder.encode([p])[0]
            rankings.append(ctx.index.dense_search(vec, k * 2, pid))
        tr.add("retrieve", "Dense search with hypothetical passage(s)", f"{len(passages)} embedding(s)")
        base = _hybrid(ctx, q, pid, k * 2)
        hits = rrf_fuse([*rankings, base], k)
        _one(tr, "retrieve", "Fuse with hybrid(question)", hits)
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


class RerankRAG(Strategy):
    id, name, family = "rerank", "Hybrid + rerank", "Retrieval"
    summary = "Retrieves a wide pool, then re-scores each chunk against the question (coverage, phrase, type fit, abnormality, recency)."
    when = "Precision matters: the right chunk is usually in the pool but not at the top."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        pool = _hybrid(ctx, q, pid, max(k * 5, 25))
        _one(tr, "retrieve", "Candidate pool (hybrid)", pool)
        ranked, feats = HeuristicReranker().rerank(q, pool)
        if feats:
            tr.add("rerank", "Feature rerank", "top: " + feats[0].explain())
        hits = ranked[:k]
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


class MMRRAG(Strategy):
    id, name, family = "mmr", "Diverse (MMR)", "Retrieval"
    summary = "Maximal Marginal Relevance: picks evidence that is relevant *and* different from what is already chosen."
    when = "Broad questions where top-k would otherwise be five near-duplicate lab rows."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        pool = _hybrid(ctx, q, pid, max(k * 4, 20))
        _one(tr, "retrieve", "Candidate pool (hybrid)", pool)
        hits = mmr(ctx.index, pool, k)
        tr.add("rerank", "MMR selection (λ = 0.7)", f"kept {len(hits)} of {len(pool)} for relevance + diversity")
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"], strategy=self.id)


class WindowRAG(Strategy):
    id, name, family = "window", "Small-to-big (context window)", "Retrieval"
    summary = "Matches on small precise chunks, then pulls in their neighbours (same test history, same-day note/encounter, related meds)."
    when = "The matched fact needs surrounding context to be interpretable (a lab value without its history)."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        seeds = _hybrid(ctx, q, pid, max(2, k // 2))
        _one(tr, "retrieve", "Seed chunks (small)", seeds)
        seen = {h.chunk.chunk_id for h in seeds}
        extra: list[Hit] = []
        for h in seeds:
            for c in self._neighbours(ctx, h):
                if c.chunk_id not in seen and len(extra) < k + 2:
                    seen.add(c.chunk_id)
                    extra.append(Hit(c, 0.0, None, None, via="context"))
        tr.add("retrieve", "Expand to parent context", f"+{len(extra)} neighbouring chunk(s) (same test history, same-day records, related meds)")
        return RAGResult(seeds + extra, tr.steps, confidence=grade(q, seeds)["coverage"], strategy=self.id)

    @staticmethod
    def _neighbours(ctx: Context, h: Hit):
        c = h.chunk
        mine = [ctx.index.chunks[i] for i in ctx.index.by_patient[c.patient_id]]
        if c.kind == "lab":
            same = sorted((x for x in mine if x.kind == "lab" and x.meta["test"] == c.meta["test"] and x.chunk_id != c.chunk_id),
                          key=lambda x: x.date, reverse=True)
            return same[:2]
        if c.kind in ("note", "encounter"):
            other = "encounter" if c.kind == "note" else "note"
            return [x for x in mine if x.kind == other and x.date == c.date][:2]
        if c.kind == "diagnosis":
            return [x for x in mine if x.kind == "medication" and x.meta.get("current")][:1]
        if c.kind == "medication":
            return [x for x in mine if x.kind == "diagnosis"][:2]
        return []


# ------------------------------------------------------------------ reasoning
def _group_key(c) -> tuple:
    sub = c.meta.get("test") or c.meta.get("condition") or c.meta.get("note_type") or c.meta.get("current")
    return (c.patient_id, c.kind, sub)


class TemporalRAG(Strategy):
    id, name, family = "temporal", "Temporal RAG", "Reasoning"
    summary = "Understands 'latest', 'last 6 months', 'since 2025', 'in 2026': filters by date window or keeps only the newest of each fact."
    when = "Any question about recency or a time period - plain similarity search ignores dates entirely."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        win = parse_time(q, _asof(ctx))
        tr.add("decision", "Parse time expression",
               (f"{win.label}" + (f" ({win.start} → {win.end})" if win.start else "")) if win.active
               else "none found → recency-weighted ranking")
        pool = _hybrid(ctx, q, pid, max(k * 8, 40))
        _one(tr, "retrieve", "Candidate pool (hybrid)", pool)
        if win.start or win.end:
            lo, hi = win.start or date.min, win.end or date.max
            pool = [h for h in pool if h.chunk.date and lo <= date.fromisoformat(h.chunk.date) <= hi]
            tr.add("decision", "Filter by window", f"{len(pool)} chunk(s) fall inside {win.label}")
            hits = pool[:k]
        elif win.latest:
            order: list[tuple] = []
            newest: dict[tuple, Hit] = {}
            for h in pool:
                key = _group_key(h.chunk)
                if key not in newest:
                    order.append(key)
                    newest[key] = h
                elif (h.chunk.date or "") > (newest[key].chunk.date or ""):
                    newest[key] = h
            hits = [newest[key] for key in order][:k]
            tr.add("decision", "Keep newest per fact", f"{len(pool)} → {len(hits)} chunk(s)")
        else:
            dates = sorted({h.chunk.date for h in pool if h.chunk.date})
            rank = {d: (i + 1) / len(dates) for i, d in enumerate(dates)}
            pool.sort(key=lambda h: -(h.score * (1 + 0.5 * rank.get(h.chunk.date, 0.0))))
            hits = pool[:k]
        return RAGResult(hits, tr.steps, confidence=grade(q, hits)["coverage"] if hits else 0.0, strategy=self.id)


class CorrectiveRAG(Strategy):
    id, name, family = "corrective", "Corrective / self-grading RAG", "Reasoning"
    summary = "Retrieves, grades its own evidence, retries with rewrites if coverage is poor, and abstains rather than guess."
    when = "Safety-critical settings: it is better to say 'not in the record' than to return the nearest-looking chunk."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        rr = HeuristicReranker()

        def attempt(label: str, hits: list[Hit]):
            ranked, _ = rr.rerank(q, hits)
            top = ranked[:k]
            g = grade(q, top)
            tr.add("grade", f"Grade after {label}", f"coverage {g['coverage']:.2f} → {g['label'].upper()}"
                   + (f" · not found in evidence: {', '.join(g['uncovered'])}" if g["uncovered"] else ""))
            return top, g

        pool = _hybrid(ctx, q, pid, max(k * 4, 20))
        _one(tr, "retrieve", "Hybrid retrieval", pool)
        best, g = attempt("hybrid + rerank", pool)
        if g["label"] == "low":
            tr.add("decision", "Retry", "evidence judged weak → rewriting the question (synonyms, keywords, lab templates)")
            fused = rrf_fuse([_hybrid(ctx, v, pid, k * 3) for v in query_variants(q)], max(k * 4, 20))
            cand, g2 = attempt("rewrite", fused)
            if g2["coverage"] > g["coverage"]:
                best, g = cand, g2
        if g["label"] == "low":
            tr.add("decision", "Retry", "still weak → HyDE hypothetical-passage retrieval")
            passages, how = hyde_passages(q, ctx)
            lists = [ctx.index.dense_search(ctx.index.embedder.encode([p])[0], k * 3, pid) for p in passages]
            cand, g3 = attempt("HyDE", rrf_fuse(lists + [pool], max(k * 4, 20)))
            if g3["coverage"] > g["coverage"]:
                best, g = cand, g3
        if g["label"] == "low":
            tr.add("decision", "Abstain", "evidence still does not cover the question → refusing to guess")
            return RAGResult(best[:2], tr.steps, None, {"uncovered": g["uncovered"]}, g["coverage"], True, self.id)
        tr.add("decision", "Accept", f"{g['label']} support ({g['coverage']:.2f})")
        return RAGResult(best, tr.steps, confidence=g["coverage"], strategy=self.id)


# ------------------------------------------------------------------ structured
class GraphRAG(Strategy):
    id, name, family = "graph", "Graph / cohort RAG", "Structured"
    summary = "Parses the question into constraints (condition, drug, lab threshold, allergy, age, symptom) and evaluates them exactly on a knowledge graph."
    when = "Cohort questions - 'which patients…', 'how many…' - where the answer is an intersection across many records."

    def run(self, q, ctx, pid, k):
        tr = Tracer()
        g = ctx.graph
        gq = g.parse(q)
        if gq.is_empty():
            tr.add("graph", "Parse constraints", "no structured pattern recognised → nothing for the graph to evaluate")
            return RAGResult([], tr.steps, confidence=0.0, strategy=self.id)
        tr.add("graph", "Parse constraints", gq.describe())
        matches = g.match(gq)
        if pid:
            matches = [m for m in matches if m.patient_id == pid]
        st = g.stats()
        tr.add("graph", "Evaluate on knowledge graph", f"{len(matches)} match(es) over {st['total_nodes']} nodes / {st['total_edges']} edges")

        hits: list[Hit] = []
        number: dict[str, int] = {}

        def cite(cids):
            marks = []
            for cid in cids:
                if cid not in number:
                    number[cid] = len(number) + 1
                    hits.append(Hit(ctx.index.get(cid), 1.0, None, None, via="graph"))
                marks.append(f"[{number[cid]}]")
            return "".join(marks)

        lines = [f"≈ {len(matches)} patient(s) match: {gq.describe()}."]
        patients = []
        for m in matches[:max(k, 8)]:
            p = ctx.records[m.patient_id]
            cids = list(dict.fromkeys([g.profile[m.patient_id], *m.chunk_ids]))
            lines.append(f"- {p['name']} ({p['age']}{p['sex']}): {'; '.join(m.why) or 'matches'} {cite(cids)}")
        for m in matches:
            p = ctx.records[m.patient_id]
            patients.append({"patient_id": m.patient_id, "name": p["name"], "age": p["age"], "sex": p["sex"], "why": m.why})
        if len(matches) > max(k, 8):
            lines.append(f"≈ …and {len(matches) - max(k, 8)} more (see structured result).")
        structured = {"type": "cohort", "query": gq.describe(), "count": len(matches), "patients": patients}
        return RAGResult(hits, tr.steps, "\n".join(lines), structured, 0.9 if matches else 0.7, False, self.id)


# ------------------------------------------------------------------ router
class AdaptiveRAG(Strategy):
    id, name, family = "adaptive", "Adaptive router", "Router"
    summary = "Classifies the question and routes it to the best strategy (graph, agentic, temporal, multi-query, HyDE, or corrective)."
    when = "The default: you don't have to know which technique fits - the router explains its choice in the trace."

    def route(self, q: str, ctx: Context, pid: str | None) -> tuple[str, str]:
        ql = q.lower()
        gq = ctx.graph.parse(q) if ctx.graph else None
        n_constraints = 0
        if gq:
            n_constraints = len(gq.conditions) + len(gq.drugs) + len(gq.allergies) + len(gq.labs) + len(gq.symptoms) + (1 if gq.age else 0) + (1 if gq.sex else 0)
        if not pid and gq and not gq.is_empty() and (is_cohort_question(q) or n_constraints >= 2):
            return "graph", f"cohort-style question with {n_constraints} structured constraint(s)"
        if pid and (re.search(ALERT_WORDS, ql) or re.search(RISK_WORDS, ql) or re.search(TREND_WORDS, ql)):
            return "agentic", "analytical question (trend / alert / risk) needs tools, not just retrieval"
        if parse_time(q, _asof(ctx)).active:
            return "temporal", "question contains a time expression or asks for the latest value"
        if len(decompose(q)) > 1:
            return "multiquery", "compound question → split into sub-questions"
        known = bool(mentioned_tests(q)) or bool(gq and (gq.conditions or gq.drugs))
        if not known and lay_to_tests(q):
            return "hyde", "lay-language wording with no exact clinical term → hypothetical-passage search"
        return "corrective", "default: retrieve, grade, retry, abstain if unsupported"

    def run(self, q, ctx, pid, k):
        target, why = self.route(q, ctx, pid)
        res = STRATEGIES[target].run(q, ctx, pid, k)
        if not res.hits and not res.answer and target not in ("corrective",):
            res2 = STRATEGIES["corrective"].run(q, ctx, pid, k)
            res2.trace = [*res.trace, Step("decision", "Fallback", f"{target} found nothing → corrective"), *res2.trace]
            res2.strategy = "corrective"
            target, res = "corrective", res2
        step = Step("route", f"Route → {STRATEGIES[target].name}", why)
        res.trace = [step, *res.trace]
        res.strategy = target
        return res


STRATEGIES: "OrderedDict[str, Strategy]" = OrderedDict((s.id, s) for s in [
    AdaptiveRAG(), BM25RAG(), NaiveRAG(), HybridRAG(), MultiQueryRAG(), HyDERAG(), RerankRAG(), MMRRAG(), WindowRAG(),
    TemporalRAG(), CorrectiveRAG(), GraphRAG(), AgenticRAG(),
])


def catalog() -> list[dict]:
    return [s.meta() for s in STRATEGIES.values()]
