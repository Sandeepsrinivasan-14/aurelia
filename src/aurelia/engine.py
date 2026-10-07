from __future__ import annotations

import time
from collections import OrderedDict

import httpx

from .analytics import clinical
from .config import Settings, get_settings
from .ingestion.chunker import chunk_patient
from .ingestion.connector import load_records
from .llm.answerers import build_answerer
from .rag.base import Context
from .rag.conversation import condense
from .rag.graph import KnowledgeGraph
from .rag.strategies import STRATEGIES, catalog
from .rag.verify import verify_answer
from .retrieval.hybrid import HybridIndex
from .security.audit import AuditLog
from .security.redaction import Redactor

QUIET_ACTORS = {"benchmark", "compare"}
LEGACY_MODES = {"hybrid": "hybrid", "bm25": "bm25", "dense": "naive"}


class Engine:
    """Owns the loaded records, search index, knowledge graph, strategies, redaction and audit."""

    def __init__(self, settings: Settings | None = None, records: list[dict] | None = None):
        self.settings = settings or get_settings()
        self.records = records if records is not None else load_records(self.settings)
        self.patients = {p["patient_id"]: p for p in self.records}
        chunks = [c for p in self.records for c in chunk_patient(p)]
        self.index = HybridIndex(chunks)
        self.graph = KnowledgeGraph(self.records, chunks)
        self.redactor = Redactor([p["name"] for p in self.records])
        self.answerer = build_answerer(self.settings.llm_backend, self.settings.ollama_url, self.settings.ollama_model)
        self.audit = AuditLog(self.settings.audit_path)
        self.ctx = Context(self.index, self.patients, self.graph, self._llm if self.settings.llm_backend == "ollama" else None)
        self.booted = time.time()
        self._space = None
        self._benchmark: dict | None = None
        # Records never change after boot, so per-patient analytics are computed once and reused.
        self._risk_rows: dict[str, dict] | None = None
        self._views: dict[str, dict] = {}
        self._llm_memo: OrderedDict[str, str] = OrderedDict()

    # optional LLM used for HyDE rewriting; failures degrade to templates (and are not memoised, so a
    # recovered Ollama is picked up on the next call)
    def _llm(self, prompt: str) -> str | None:
        if prompt in self._llm_memo:
            self._llm_memo.move_to_end(prompt)
            return self._llm_memo[prompt]
        try:
            r = httpx.post(f"{self.settings.ollama_url.rstrip('/')}/api/generate", timeout=30,
                           json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False})
            r.raise_for_status()
            out = r.json()["response"].strip()
        except Exception:
            return None
        self._llm_memo[prompt] = out
        if len(self._llm_memo) > 256:
            self._llm_memo.popitem(last=False)
        return out

    def stats(self) -> dict:
        return {"patients": len(self.patients), "chunks": len(self.index.chunks),
                "embedder": self.index.embedder.name, "answerer": self.answerer.name,
                "source": self.settings.ehr_url or "bundled synthetic cohort", "strategies": len(STRATEGIES),
                "graph": self.graph.stats()}

    def strategies(self) -> list[dict]:
        return catalog()

    # ------------------------------------------------------------------ query
    def query(self, question: str, patient_id: str | None = None, k: int | None = None, mode: str | None = None,
              redact: bool = False, actor: str = "operator", strategy: str | None = None,
              history: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        sid = strategy or LEGACY_MODES.get(mode or "", "adaptive")
        if sid not in STRATEGIES:
            raise KeyError(f"unknown strategy '{sid}'")
        k = k or self.settings.top_k
        standalone, why = condense(question, history or [])
        res = STRATEGIES[sid].run(standalone, self.ctx, patient_id, k)

        evidence = [{"n": i, "chunk_id": h.chunk.chunk_id, "patient_id": h.chunk.patient_id, "kind": h.chunk.kind,
                     "date": h.chunk.date, "text": h.chunk.text, "score": h.score, "bm25_rank": h.bm25_rank,
                     "dense_rank": h.dense_rank, "via": h.via} for i, h in enumerate(res.hits, 1)]
        if res.abstain:
            miss = (res.structured or {}).get("uncovered", [])
            answer = ("I can't answer that from the record: the retrieved evidence does not mention "
                      + (", ".join(f"“{t}”" for t in miss) if miss else "what was asked") + ". Absence of evidence is not evidence of absence - "
                      "it may simply not be documented.")
            evidence = evidence[:2]
        elif res.answer is not None:
            answer = res.answer
        else:
            answer = self.answerer.answer(standalone, res.hits)
        verification = verify_answer(answer, evidence) if not res.abstain else {"faithfulness": 1.0, "claims": [], "derived_lines": 0, "supported": 0, "total": 0}

        structured = res.structured if (res.structured and res.structured.get("type") == "cohort") else None
        trace = [s.to_dict() for s in res.trace]
        if why:
            trace.insert(0, {"kind": "rewrite", "name": "Conversation memory", "detail": f"“{standalone}” ← {why}", "ms": 0})
        if redact:
            answer = self.redactor.redact(answer)
            for e in evidence:
                e["text"] = self.redactor.redact(e["text"])
                e["patient_id"] = self.redactor.redact(e["patient_id"])
            if structured:
                structured = {**structured, "query": self.redactor.redact(structured["query"]),
                              "patients": [{**p, "name": self.redactor.redact(p["name"]), "patient_id": self.redactor.redact(p["patient_id"]),
                                            "why": [self.redactor.redact(w) for w in p["why"]]} for p in structured["patients"]]}
            for t in trace:
                t["detail"] = self.redactor.redact(t["detail"])
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if actor in QUIET_ACTORS:   # bulk internal runs (benchmark, compare) must not flood the audit trail
            return self._payload(question, standalone, answer, evidence, ms, res, sid, redact, trace, verification, structured)
        self.audit.record("query", actor=actor, patient_id=patient_id, strategy=res.strategy or sid, requested=sid,
                          redacted=redact, question_len=len(question), hits=len(evidence), abstained=res.abstain)
        return self._payload(question, standalone, answer, evidence, ms, res, sid, redact, trace, verification, structured)

    def _payload(self, question, standalone, answer, evidence, ms, res, sid, redact, trace, verification, structured) -> dict:
        return {"question": question, "standalone": standalone, "answer": answer, "evidence": evidence, "latency_ms": ms,
                "strategy": res.strategy or sid, "requested_strategy": sid, "mode": res.strategy or sid, "redacted": redact,
                "answerer": self.answerer.name if res.answer is None and not res.abstain else "strategy-composed",
                "trace": trace, "confidence": round(res.confidence, 3), "abstained": res.abstain,
                "verification": verification, "structured": structured}

    def compare(self, question: str, patient_id: str | None = None, strategies: list[str] | None = None,
                k: int | None = None) -> list[dict]:
        rows = []
        self.audit.record("compare", patient_id=patient_id, question_len=len(question))
        for sid in strategies or [s for s in STRATEGIES if s != "adaptive"]:
            r = self.query(question, patient_id, k, strategy=sid, actor="compare")
            rows.append({"strategy": sid, "name": STRATEGIES[sid].name, "family": STRATEGIES[sid].family,
                         "latency_ms": r["latency_ms"], "confidence": r["confidence"], "abstained": r["abstained"],
                         "faithfulness": r["verification"]["faithfulness"], "evidence": len(r["evidence"]),
                         "top": [e["text"][:110] for e in r["evidence"][:2]], "chunk_ids": [e["chunk_id"] for e in r["evidence"]],
                         "structured_count": (r["structured"] or {}).get("count")})
        return rows

    # ------------------------------------------------------------------ patients
    def patient_view(self, pid: str) -> dict | None:
        p = self.patients.get(pid)
        if not p:
            return None
        self.audit.record("view_patient", patient_id=pid)
        if pid not in self._views:
            self._views[pid] = self._build_view(p)
        return self._views[pid]

    @staticmethod
    def _build_view(p: dict) -> dict:
        from .analytics.forecast import anomaly, forecast
        from .nlp.ner import summarize_notes
        series = clinical.lab_series(p)
        labs = {}
        for t, pts in series.items():
            points = [{"date": x["date"], "value": x["value"], "flag": x["flag"]} for x in pts]
            labs[t] = {"unit": pts[0]["unit"], "ref_low": pts[0]["ref_low"], "ref_high": pts[0]["ref_high"],
                       "trend": clinical.trend(pts), "points": points, "forecast": forecast(points), "anomaly": anomaly(points)}
        return {
            "patient": {k: p[k] for k in ("patient_id", "name", "age", "sex", "allergies", "phone")},
            "diagnoses": p["diagnoses"], "medications": p["medications"], "labs": labs,
            "alerts": clinical.alerts(p), "risk": clinical.risk_score(p), "timeline": clinical.timeline(p),
            "entities": summarize_notes(p["notes"]),
        }

    def _patient_rows(self) -> dict[str, dict]:
        if self._risk_rows is None:
            rows = {}
            for p in self.records:
                r = clinical.risk_score(p)
                rows[p["patient_id"]] = {"patient_id": p["patient_id"], "name": p["name"], "age": p["age"], "sex": p["sex"],
                                         "conditions": [d["condition"] for d in p["diagnoses"]], "risk": r["score"],
                                         "band": r["band"], "alerts": len(clinical.alerts(p))}
            self._risk_rows = rows
        return self._risk_rows

    def patient_list(self, q: str = "", sort: str = "risk") -> list[dict]:
        rows = []
        for p in self.records:
            if q and q.lower() not in (p["name"] + p["patient_id"] + " ".join(d["condition"] for d in p["diagnoses"])).lower():
                continue
            rows.append(dict(self._patient_rows()[p["patient_id"]]))
        rows.sort(key=lambda r: (-r["risk"], r["name"]) if sort == "risk" else r["name"])
        return rows

    def cohort(self) -> dict:
        return clinical.cohort_stats(self.records)

    @property
    def space(self):
        if self._space is None:
            from .analytics.similarity import PatientSpace
            self._space = PatientSpace(self.records)
        return self._space

    def similar(self, pid: str, k: int = 5) -> list[dict] | None:
        if pid not in self.patients:
            return None
        self.audit.record("similar_patients", patient_id=pid)
        return self.space.similar(pid, k)

    def cohort_map(self) -> dict:
        return self.space.map()

    def _cached_benchmark(self) -> dict | None:
        """Pre-computed results from `python -m eval.benchmark`, used only if they match this cohort."""
        import json
        from pathlib import Path
        for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
            f = base / "docs" / "benchmark.json"
            if f.exists():
                try:
                    data = json.loads(f.read_text())
                except ValueError:
                    return None
                return data if data.get("patients") == len(self.patients) else None
        return None

    def benchmark(self, refresh: bool = False) -> dict:
        if self._benchmark is None and not refresh:
            cached = self._cached_benchmark()
            if cached:
                self._benchmark = cached
        if self._benchmark is None or refresh:
            from eval.benchmark import run_all
            self._benchmark = run_all(self)
        return self._benchmark
