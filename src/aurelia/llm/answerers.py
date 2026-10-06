"""Answer generation.

``ExtractiveAnswerer`` composes an answer purely from retrieved evidence, with
numbered citations, so it cannot hallucinate and needs no model. ``OllamaAnswerer``
sends the same evidence to a local LLM with a grounding prompt and falls back to
the extractive answer if the model is unreachable.
"""
from __future__ import annotations

import re
from typing import Protocol

import httpx

from ..retrieval.hybrid import Hit

KIND_LABEL = {"lab": "Lab", "diagnosis": "Diagnosis", "medication": "Medication", "note": "Clinical note",
              "encounter": "Encounter", "profile": "Profile"}


class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: list[Hit]) -> str: ...


def _intent(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(allerg|allergic)", ql):
        return "allergy"
    if re.search(r"\b(medicat|drug|meds|prescri|taking|dose)", ql):
        return "medication"
    if re.search(r"\b(lab|result|hba1c|creatinine|tsh|glucose|egfr|ldl|hemoglobin|crp|test)", ql):
        return "lab"
    if re.search(r"\b(admit|admission|discharge|hospital|visit|encounter)", ql):
        return "encounter"
    return "general"


class ExtractiveAnswerer:
    name = "extractive-grounded"

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "I could not find supporting evidence in the record for that question."
        intent = _intent(question)
        pref = {"allergy": "profile", "medication": "medication", "lab": "lab", "encounter": "encounter"}.get(intent)
        ordered = sorted(enumerate(hits, 1), key=lambda t: (t[1].chunk.kind != pref, t[0]))
        if intent in ("lab", "encounter"):  # newest first within the preferred kind
            ordered = sorted(ordered, key=lambda t: (t[1].chunk.kind != pref, -int((t[1].chunk.date or "0").replace("-", ""))))
        lines = []
        for n, h in ordered[:4]:
            text = re.sub(r"\s+", " ", h.chunk.text)
            lines.append(f"- {text} [{n}]")
        head = {"allergy": "Allergy information found in the record:", "medication": "Medication information:",
                "lab": "Relevant laboratory evidence:", "encounter": "Relevant encounters:"}.get(
            intent, "Most relevant evidence in the record:")
        return head + "\n" + "\n".join(lines)


class OllamaAnswerer:
    name = "ollama"

    PROMPT = ("You are a clinical documentation assistant. Answer ONLY using the numbered evidence. "
              "Cite sources like [1]. If the evidence is insufficient, say so. Do not give treatment orders.\n\n"
              "Evidence:\n{evidence}\n\nQuestion: {q}\nAnswer:")

    def __init__(self, url: str, model: str, fallback: Answerer | None = None):
        self.url, self.model, self.fallback = url.rstrip("/"), model, fallback or ExtractiveAnswerer()

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return self.fallback.answer(question, hits)
        evidence = "\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
        try:
            r = httpx.post(f"{self.url}/api/generate", timeout=60,
                           json={"model": self.model, "prompt": self.PROMPT.format(evidence=evidence, q=question),
                                 "stream": False})
            r.raise_for_status()
            return r.json()["response"].strip()
        except Exception:
            return self.fallback.answer(question, hits) + "\n\n(LLM unavailable - showing extractive answer.)"


def build_answerer(backend: str, url: str, model: str) -> Answerer:
    return OllamaAnswerer(url, model) if backend == "ollama" else ExtractiveAnswerer()
