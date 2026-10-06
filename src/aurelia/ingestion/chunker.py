"""Turn a structured patient record into retrievable, citable evidence chunks.

Every chunk carries typed ``meta`` (test name, condition, drugs, ...) so that
higher-level strategies - reranking, temporal filtering, graph matching - can
use structure without re-parsing text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    patient_id: str
    kind: str        # profile | diagnosis | medication | lab | note | encounter
    date: str | None
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def drug_base(name: str) -> str:
    """'Metformin 500 mg' -> 'metformin'; 'Salbutamol inhaler' -> 'salbutamol'."""
    import re
    return re.split(r"\d| inhaler| weekly", name, maxsplit=1)[0].strip().lower()


def chunk_patient(p: dict) -> list[Chunk]:
    pid = p["patient_id"]
    out: list[Chunk] = []

    def add(kind: str, date: str | None, text: str, **meta) -> None:
        out.append(Chunk(f"{pid}:{kind}:{len(out)}", pid, kind, date, text, meta))

    add("profile", None, f"{p['name']} is a {p['age']} year old {'female' if p['sex'] == 'F' else 'male'} patient. "
                         f"Documented allergies: {p['allergies']}.",
        name=p["name"], age=p["age"], sex=p["sex"], allergy=p["allergies"])
    for d in p["diagnoses"]:
        add("diagnosis", d["onset"], f"Diagnosis: {d['condition']} (code {d['code']}), onset {d['onset']}, "
                                     f"status {d['status']}.", condition=d["condition"], status=d["status"])
    meds_active = [m for m in p["medications"] if m["active"]]
    if meds_active:
        add("medication", None, "Current medications: " + "; ".join(f"{m['name']} for {m['for']}" for m in meds_active) + ".",
            drugs=[drug_base(m["name"]) for m in meds_active], current=True)
    for m in p["medications"]:
        if not m["active"]:
            add("medication", m["started"], f"Discontinued medication: {m['name']} (was for {m['for']}).",
                drugs=[drug_base(m["name"])], current=False)
    for lab in p["labs"]:
        add("lab", lab["date"], f"Lab result {lab['test']}: {lab['value']} {lab['unit']} on {lab['date']} "
                                f"(reference {lab['ref_low']}-{lab['ref_high']}), flagged {lab['flag']}.",
            test=lab["test"], value=lab["value"], unit=lab["unit"], flag=lab["flag"])
    for e in p["encounters"]:
        add("encounter", e["date"], f"{e['type']} at {e['department']} on {e['date']} for {e['reason']}.",
            department=e["department"], reason=e["reason"], etype=e["type"])
    for n in p["notes"]:
        add("note", n["date"], f"{n['type']} by {n['author']} on {n['date']}: {n['text']}", note_type=n["type"])
    return out
