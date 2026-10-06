"""Benchmark of every RAG strategy across six suites.

Ground truth is computed by *independent brute-force code over the raw records*
(never by the strategy under test), so a strategy cannot grade its own homework.
Questions are template-generated from the data, which makes this a regression
and comparison harness - **not** evidence of real-world clinical accuracy.

Suites
  exact      patient-scoped, question names the test and date          recall@5
  lay        patient-scoped, lay wording ("kidney function")           recall@5
  cohort     unscoped "which patients ..." intersections               set F1
  temporal   "latest X" (hit@1) and "X in the last 6 months" (window precision@5)
  abstain    unanswerable questions: correct-abstain rate / false-abstain rate on answerable ones
  faithful   share of answer lines supported by their own citations

Run:  python -m eval.benchmark
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from aurelia.config import Settings
from aurelia.engine import Engine
from aurelia.ingestion.chunker import chunk_patient
from aurelia.rag.strategies import STRATEGIES

K = 5


# ------------------------------------------------------------------ brute-force truth (independent of the system)
def _latest(p: dict, test: str):
    rows = [x for x in p["labs"] if x["test"] == test]
    return max(rows, key=lambda x: x["date"]) if rows else None


def _on(p: dict, base: str) -> bool:
    return any(m["active"] and m["name"].lower().startswith(base) for m in p["medications"])


def _has(p: dict, cond: str) -> bool:
    return any(d["condition"] == cond for d in p["diagnoses"])


def _lab(p: dict, test: str, op: str, thr: float) -> bool:
    x = _latest(p, test)
    return bool(x) and (x["value"] > thr if op == ">" else x["value"] < thr)


def _says(p: dict, word: str) -> bool:
    return any(word in n["text"].lower() for n in p["notes"])


# ------------------------------------------------------------------ suite builders
def build_exact(records, n=200, seed=11):
    rng = random.Random(seed)
    cases = []
    for p in records:
        chunks = chunk_patient(p)
        for lab in p["labs"]:
            gold = next(c.chunk_id for c in chunks if c.kind == "lab" and f"Lab result {lab['test']}: {lab['value']} " in c.text and lab["date"] in c.text)
            tpl = rng.choice(["What was the {t} result on {d}?", "Show the {t} value recorded on {d}"])
            cases.append({"q": tpl.format(t=lab["test"], d=lab["date"]), "pid": p["patient_id"], "gold": {gold}})
        for d in p["diagnoses"]:
            gold = next(c.chunk_id for c in chunks if c.kind == "diagnosis" and d["condition"] in c.text)
            cases.append({"q": f"What is the status of the {d['condition']} diagnosis?", "pid": p["patient_id"], "gold": {gold}})
        cases.append({"q": "Does the patient have any allergies?", "pid": p["patient_id"],
                      "gold": {next(c.chunk_id for c in chunks if c.kind == "profile")}})
    rng.shuffle(cases)
    return cases[:n]


LAY = [("How is the blood sugar control?", {"HbA1c", "Fasting Glucose"}), ("How are the kidney function numbers?", {"Creatinine", "eGFR"}),
       ("Is the thyroid under control?", {"TSH"}), ("What do the cholesterol tests show?", {"LDL Cholesterol", "Total Cholesterol"}),
       ("Is there any sign of anemia?", {"Hemoglobin", "Ferritin"}), ("How is the breathing and oxygen level?", {"SpO2", "Peak Flow", "FEV1/FVC"})]


def build_lay(records):
    cases = []
    for p in records:
        have = {x["test"] for x in p["labs"]}
        chunks = chunk_patient(p)
        for q, tests in LAY:
            if have & tests:
                gold = {c.chunk_id for c in chunks if c.kind == "lab" and c.meta["test"] in tests}
                cases.append({"q": q, "pid": p["patient_id"], "gold": gold})
    return cases


def build_cohort(records):
    cases = []

    def add(q, truth, lo=1, hi=30):
        if lo <= len(truth) <= hi:
            cases.append({"q": q, "truth": truth})

    drugs = sorted({m["name"].lower().split()[0] for p in records for m in p["medications"]
                    if m["name"].lower().split()[0] not in ("sodium", "ferrous", "folic")})
    for d in drugs:
        add(f"Which patients are on {d}?", {p["patient_id"] for p in records if _on(p, d)})
    for d in ("metformin", "atorvastatin", "amlodipine"):
        add(f"Who is taking {d}?", {p["patient_id"] for p in records if _on(p, d)})
    labcombos = [("metformin", "eGFR", "<", 60, "eGFR below 60"), ("metformin", "eGFR", "<", 45, "eGFR below 45"),
                 ("telmisartan", "Serum Potassium", ">", 4.5, "potassium above 4.5"), ("losartan", "Creatinine", ">", 1.5, "creatinine above 1.5"),
                 ("atorvastatin", "LDL Cholesterol", ">", 100, "LDL above 100"), ("aspirin", "Hemoglobin", "<", 12, "hemoglobin below 12")]
    for d, t, op, thr, txt in labcombos:
        add(f"Which patients are on {d} and have {txt}?", {p["patient_id"] for p in records if _on(p, d) and _lab(p, t, op, thr)})
    condlab = [("diabetes", "Type 2 Diabetes Mellitus", "HbA1c", ">", 8, "HbA1c above 8"), ("hypertension", "Essential Hypertension", "Systolic BP", ">", 140, "systolic above 140"),
               ("chronic kidney disease", "Chronic Kidney Disease Stage 3", "Creatinine", ">", 1.5, "creatinine above 1.5"),
               ("COPD", "Chronic Obstructive Pulmonary Disease", "SpO2", "<", 92, "SpO2 below 92"), ("anemia", "Iron Deficiency Anemia", "Hemoglobin", "<", 10, "hemoglobin below 10"),
               ("hypothyroidism", "Hypothyroidism", "TSH", ">", 5, "TSH above 5")]
    for alias, cond, t, op, thr, txt in condlab:
        add(f"Which patients have {alias} and {txt}?", {p["patient_id"] for p in records if _has(p, cond) and _lab(p, t, op, thr)})
    for allergen in ("nsaids", "penicillin", "latex"):
        add(f"Which patients are allergic to {allergen}?", {p["patient_id"] for p in records if allergen in p["allergies"].lower()})
    add("Which patients are allergic to NSAIDs and take aspirin?", {p["patient_id"] for p in records if "nsaids" in p["allergies"].lower() and _on(p, "aspirin")}, hi=30)
    for alias, cond, word in [("diabetes", "Type 2 Diabetes Mellitus", "fatigue"), ("asthma", "Asthma", "wheeze"), ("anemia", "Iron Deficiency Anemia", "palpitations")]:
        add(f"Which patients have {alias} and fatigue?" if word == "fatigue" else f"Which patients have {alias} and {word}?",
            {p["patient_id"] for p in records if _has(p, cond) and _says(p, word)})
    for alias, cond in [("hypertension", "Essential Hypertension"), ("diabetes", "Type 2 Diabetes Mellitus")]:
        add(f"How many women over 60 have {alias}?", {p["patient_id"] for p in records if p["sex"] == "F" and p["age"] > 60 and _has(p, cond)}, lo=1)
        add(f"List men under 50 with {alias}", {p["patient_id"] for p in records if p["sex"] == "M" and p["age"] < 50 and _has(p, cond)}, lo=1)
    return cases


def build_cohort_paraphrase(records):
    """Looser, conversational wording the parser was not tuned on - some of it is deliberately unsupported."""
    cases = []

    def add(q, truth, hi=40):
        if 1 <= len(truth) <= hi:
            cases.append({"q": q, "truth": truth})

    ids = lambda f: {p["patient_id"] for p in records if f(p)}  # noqa: E731
    for d in ("metformin", "atorvastatin", "levothyroxine", "sertraline"):
        add(f"Anyone prescribed {d}?", ids(lambda p, d=d: _on(p, d)))
    add("Patients receiving losartan whose creatinine is high", ids(lambda p: _on(p, "losartan") and (_latest(p, "Creatinine") or {}).get("flag") == "HIGH"))
    add("Patients receiving metformin whose eGFR is low", ids(lambda p: _on(p, "metformin") and (_latest(p, "eGFR") or {}).get("flag") == "LOW"))
    add("Anemic patients with low hemoglobin", ids(lambda p: _has(p, "Iron Deficiency Anemia") and (_latest(p, "Hemoglobin") or {}).get("flag") == "LOW"))
    add("Diabetic patients whose HbA1c is high", ids(lambda p: _has(p, "Type 2 Diabetes Mellitus") and (_latest(p, "HbA1c") or {}).get("flag") == "HIGH"))
    add("Show me everyone with hypertension", ids(lambda p: _has(p, "Essential Hypertension")))
    add("How many are asthmatic?", ids(lambda p: _has(p, "Asthma")))
    add("How many patients have depression?", ids(lambda p: _has(p, "Major Depressive Disorder")))
    add("Who has COPD and is on an inhaler?", ids(lambda p: _has(p, "Chronic Obstructive Pulmonary Disease") and any(m["active"] and "inhaler" in m["name"].lower() for m in p["medications"])))
    # known-hard / unsupported wording - included on purpose
    add("Which patients have kidney problems?", ids(lambda p: _has(p, "Chronic Kidney Disease Stage 3")))
    add("Heart patients on statins", ids(lambda p: _has(p, "Coronary Artery Disease") and _on(p, "atorvastatin")))
    add("Which patients have sugar problems and high BP?", ids(lambda p: _has(p, "Type 2 Diabetes Mellitus") and _has(p, "Essential Hypertension")))
    add("Which older patients have diabetes?", ids(lambda p: p["age"] >= 65 and _has(p, "Type 2 Diabetes Mellitus")))
    return cases


def build_temporal(records, asof: date):
    latest, window = [], []
    start = asof - timedelta(days=180)
    for p in records:
        chunks = chunk_patient(p)
        tests = sorted({x["test"] for x in p["labs"]})
        for t in tests:
            rows = sorted((x for x in p["labs"] if x["test"] == t), key=lambda x: x["date"])
            if len(rows) >= 2:
                gold = next(c.chunk_id for c in chunks if c.kind == "lab" and c.meta["test"] == t and c.date == rows[-1]["date"]
                            and c.meta["value"] == rows[-1]["value"])
                latest.append({"q": f"What is the latest {t} result?", "pid": p["patient_id"], "gold": {gold}})
            inside = [x for x in rows if start <= date.fromisoformat(x["date"]) <= asof]
            if inside and len(inside) < len(rows):
                window.append({"q": f"Show {t} results from the last 6 months", "pid": p["patient_id"], "test": t, "start": start, "end": asof})
    return latest[:80], window[:60]


UNANSWERABLE = ["Does the patient have any history of cancer?", "What was the MRI brain result?", "Is the patient on warfarin?",
                "Any record of chemotherapy?", "What is the patient's blood group?", "Does the patient smoke cigarettes?",
                "What was the colonoscopy finding?", "Is the patient pregnant?"]


# ------------------------------------------------------------------ metrics
def _hits(engine: Engine, sid: str, q: str, pid: str | None, k: int = K):
    return STRATEGIES[sid].run(q, engine.ctx, pid, k)


def _rank_metrics(engine, sid, cases):
    hit = rr = 0.0
    for c in cases:
        ids = [h.chunk.chunk_id for h in _hits(engine, sid, c["q"], c["pid"]).hits][:K]
        ranks = [i for i, cid in enumerate(ids, 1) if cid in c["gold"]]
        if ranks:
            hit += 1
            rr += 1 / ranks[0]
    n = len(cases)
    return {"value": round(hit / n, 3), "secondary": round(rr / n, 3)}


def _f1(pred: set, truth: set) -> float:
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(pred), tp / len(truth)
    return 2 * pr * rc / (pr + rc)


def _cohort_metrics(engine, sid, cases):
    scores, precs, recs = [], [], []
    for c in cases:
        r = _hits(engine, sid, c["q"], None, k=20)
        pred = {p["patient_id"] for p in r.structured["patients"]} if r.structured and r.structured.get("type") == "cohort" \
            else {h.chunk.patient_id for h in r.hits}
        scores.append(_f1(pred, c["truth"]))
        tp = len(pred & c["truth"])
        precs.append(tp / len(pred) if pred else 0.0)
        recs.append(tp / len(c["truth"]))
    n = len(cases)
    return {"value": round(sum(scores) / n, 3), "secondary": round(sum(recs) / n, 3)}


def _latest_metrics(engine, sid, cases):
    ok = 0
    for c in cases:
        hits = _hits(engine, sid, c["q"], c["pid"]).hits
        ok += bool(hits) and hits[0].chunk.chunk_id in c["gold"]
    return ok / len(cases)


def _window_metrics(engine, sid, cases):
    vals = []
    for c in cases:
        hits = _hits(engine, sid, c["q"], c["pid"]).hits[:K]
        if not hits:
            vals.append(0.0)
            continue
        good = sum(1 for h in hits if h.chunk.meta.get("test") == c["test"] and h.chunk.date
                   and c["start"] <= date.fromisoformat(h.chunk.date) <= c["end"])
        vals.append(good / len(hits))
    return sum(vals) / len(vals)


def _abstain_metrics(engine, sid, unanswerable, answerable):
    ab = sum(bool(_hits(engine, sid, c["q"], c["pid"]).abstain) for c in unanswerable)
    fa = sum(bool(_hits(engine, sid, c["q"], c["pid"]).abstain) for c in answerable)
    return {"value": round(ab / len(unanswerable), 3), "secondary": round(fa / len(answerable), 3)}


def _faith_metrics(engine, sid, cases):
    vals = []
    for c in cases:
        r = engine.query(c["q"], c["pid"], strategy=sid, actor="benchmark")
        vals.append(r["verification"]["faithfulness"])
    return {"value": round(sum(vals) / len(vals), 3)}


# ------------------------------------------------------------------ orchestration
def run_all(engine: Engine, quick: bool = False, verbose: bool = False) -> dict:
    recs = engine.records
    exact = build_exact(recs, 60 if quick else 200)
    lay = build_lay(recs)
    cohort = build_cohort(recs)
    cohort2 = build_cohort_paraphrase(recs)
    latest, window = build_temporal(recs, date.fromisoformat(engine.index.asof))
    rng = random.Random(5)
    pids = rng.sample([p["patient_id"] for p in recs], 12 if quick else 25)
    unans = [{"q": q, "pid": pid} for pid in pids for q in UNANSWERABLE]
    answ = exact[:len(unans)] if len(exact) >= len(unans) else exact
    faith = (exact[:25] + lay[:25]) if quick else (exact[:60] + lay[:40])
    if quick:
        lay, latest, window = lay[:40], latest[:30], window[:25]

    suites = [
        ("exact", "Exact-term questions", "recall@5", len(exact), lambda s: _rank_metrics(engine, s, exact), "MRR"),
        ("lay", "Lay-language questions", "recall@5", len(lay), lambda s: _rank_metrics(engine, s, lay), "MRR"),
        ("cohort", "Cohort questions (unscoped)", "set F1", len(cohort), lambda s: _cohort_metrics(engine, s, cohort), "recall"),
        ("cohort2", "Cohort questions, paraphrased", "set F1", len(cohort2), lambda s: _cohort_metrics(engine, s, cohort2), "recall"),
        ("latest", "“Latest value” questions", "hit@1", len(latest), lambda s: {"value": round(_latest_metrics(engine, s, latest), 3)}, ""),
        ("window", "“Last 6 months” questions", "window precision@5", len(window), lambda s: {"value": round(_window_metrics(engine, s, window), 3)}, ""),
        ("abstain", "Unanswerable questions", "correct abstain", len(unans), lambda s: _abstain_metrics(engine, s, unans, answ), "false abstain ↓"),
        ("faithful", "Answer faithfulness", "supported lines", len(faith), lambda s: _faith_metrics(engine, s, faith), ""),
    ]
    out = {"suites": [], "strategies": [{"id": s.id, "name": s.name, "family": s.family} for s in STRATEGIES.values()],
           "quick": quick, "patients": len(recs), "note": "Template-generated synthetic questions; a comparison harness, not clinical validation."}
    for sid, title, metric, n, fn, secondary in suites:
        rows = []
        for s in STRATEGIES:
            r = fn(s)
            rows.append({"strategy": s, **r})
        out["suites"].append({"id": sid, "title": title, "metric": metric, "secondary": secondary, "n": n, "rows": rows})
        if verbose:
            print(f"\n{title} - n={n} ({metric}{', ' + secondary if secondary else ''})")
            for r in rows:
                print(f"  {r['strategy']:<11}{r['value']:>7}" + (f"{r['secondary']:>8}" if "secondary" in r else ""))
    return out


def run(quick: bool = False, save: bool = True) -> dict:
    import json
    from pathlib import Path
    engine = Engine(Settings(audit_path="data/benchmark_audit.jsonl"))
    out = run_all(engine, quick=quick, verbose=True)
    if save and not quick:   # cached so the UI can show results instantly
        Path("docs").mkdir(exist_ok=True)
        Path("docs/benchmark.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    run(quick="--quick" in sys.argv)
