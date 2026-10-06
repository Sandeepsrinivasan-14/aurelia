"""Shared types for every RAG strategy.

A strategy takes a question and returns a ``RAGResult``: the evidence it chose,
a step-by-step ``trace`` explaining how, and a confidence estimate. The trace is
what the UI shows as "reasoning" - every decision is inspectable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..retrieval.hybrid import Hit, HybridIndex


@dataclass
class Step:
    kind: str      # route | rewrite | retrieve | rerank | grade | tool | graph | decision | note
    name: str
    detail: str = ""
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "detail": self.detail, "ms": round(self.ms, 2)}


@dataclass
class RAGResult:
    hits: list[Hit]
    trace: list[Step] = field(default_factory=list)
    answer: str | None = None          # strategies that compute their own answer (graph, agentic)
    structured: dict | None = None     # e.g. {"patients": [...]} for cohort questions
    confidence: float = 0.0            # 0-1 retrieval-support estimate
    abstain: bool = False              # evidence judged insufficient
    strategy: str = ""


@dataclass
class Context:
    """Everything a strategy may use. Built once by the Engine."""
    index: HybridIndex
    records: dict[str, dict]
    graph: object = None                                    # KnowledgeGraph
    llm_generate: Callable[[str], str | None] | None = None  # optional LLM (HyDE rewriting)


class Tracer:
    """Tiny helper so strategies record steps with timings in one line."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._t = time.perf_counter()

    def add(self, kind: str, name: str, detail: str = "") -> None:
        now = time.perf_counter()
        self.steps.append(Step(kind, name, detail, (now - self._t) * 1000))
        self._t = now


class Strategy:
    id = ""
    name = ""
    family = ""      # Baseline | Retrieval | Reasoning | Structured | Agentic | Router
    summary = ""
    when = ""

    def run(self, question: str, ctx: Context, patient_id: str | None, k: int) -> RAGResult:  # pragma: no cover
        raise NotImplementedError

    def meta(self) -> dict:
        return {"id": self.id, "name": self.name, "family": self.family, "summary": self.summary, "when": self.when}
