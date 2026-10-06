"""Transparent, rule-based clinical analytics.

These are *demonstration heuristics* over synthetic data, written to be readable
and auditable. They are not validated clinical decision support.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

NSAID_LIKE = ("aspirin", "ibuprofen", "diclofenac", "naproxen")
RAAS = ("losartan", "telmisartan", "enalapril", "ramipril")


def lab_series(p: dict) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = defaultdict(list)
    for lab in sorted(p["labs"], key=lambda x: x["date"]):
        series[lab["test"]].append(lab)
    return dict(series)


def _latest(p: dict, test: str) -> dict | None:
    s = lab_series(p).get(test)
    return s[-1] if s else None


def trend(points: list[dict]) -> str:
    if len(points) < 2:
        return "insufficient data"
    first, last = points[0]["value"], points[-1]["value"]
    if first == 0:
        return "stable"
    change = (last - first) / abs(first)
    return "rising" if change > 0.08 else "falling" if change < -0.08 else "stable"


def alerts(p: dict) -> list[dict]:
    out: list[dict] = []
    active_meds = [m["name"].lower() for m in p["medications"] if m["active"]]
    allergy = p["allergies"].lower()

    if "nsaids" in allergy and any(any(n in m for n in NSAID_LIKE) for m in active_meds):
        out.append({"severity": "critical", "title": "Allergy conflict",
                    "detail": "Documented NSAID allergy while on an NSAID-class medication (e.g. aspirin)."})

    k = _latest(p, "Serum Potassium")
    if k and k["value"] > 5.0 and any(any(r in m for r in RAAS) for m in active_meds):
        out.append({"severity": "high", "title": "Hyperkalaemia risk",
                    "detail": f"Potassium {k['value']} mmol/L while on an ACE-inhibitor/ARB."})

    egfr = _latest(p, "eGFR")
    if egfr and egfr["value"] < 45:
        if any("metformin" in m for m in active_meds):
            out.append({"severity": "high", "title": "Renal dose review",
                        "detail": f"eGFR {egfr['value']} with metformin: review dose or continuation."})
        if any("methotrexate" in m for m in active_meds):
            out.append({"severity": "high", "title": "Methotrexate with reduced eGFR",
                        "detail": f"eGFR {egfr['value']}: methotrexate clearance may be impaired."})

    a1c = _latest(p, "HbA1c")
    if a1c and a1c["value"] >= 8.0:
        out.append({"severity": "medium", "title": "Poor glycaemic control",
                    "detail": f"Latest HbA1c {a1c['value']}% (target below 7-8% for most adults)."})

    spo2 = _latest(p, "SpO2")
    if spo2 and spo2["value"] < 90:
        out.append({"severity": "high", "title": "Low oxygen saturation", "detail": f"SpO2 {spo2['value']}%."})

    hb = _latest(p, "Hemoglobin")
    if hb and hb["value"] < 9:
        out.append({"severity": "medium", "title": "Significant anaemia", "detail": f"Haemoglobin {hb['value']} g/dL."})

    if any(d["status"] == "worsening" for d in p["diagnoses"]):
        names = ", ".join(d["condition"] for d in p["diagnoses"] if d["status"] == "worsening")
        out.append({"severity": "medium", "title": "Condition marked worsening", "detail": names})

    order = {"critical": 0, "high": 1, "medium": 2}
    return sorted(out, key=lambda a: order[a["severity"]])


def risk_score(p: dict) -> dict:
    """Additive, explainable 0-100 complexity score (not a validated risk model)."""
    parts: list[tuple[str, int]] = []
    if p["age"] >= 65:
        parts.append(("Age 65+", 12))
    elif p["age"] >= 50:
        parts.append(("Age 50-64", 6))
    parts.append((f"{len(p['diagnoses'])} chronic condition(s)", 8 * len(p["diagnoses"])))
    n_meds = sum(1 for m in p["medications"] if m["active"])
    if n_meds >= 4:
        parts.append(("Polypharmacy (4+ drugs)", 10))
    abnormal = [x for x in p["labs"] if x["flag"] != "NORMAL"]
    parts.append((f"{len(abnormal)} abnormal lab result(s)", min(25, 2 * len(abnormal))))
    adm = sum(1 for e in p["encounters"] if e["type"] in ("Admission", "Emergency visit"))
    if adm:
        parts.append((f"{adm} acute encounter(s)", 6 * adm))
    if any(d["status"] == "worsening" for d in p["diagnoses"]):
        parts.append(("Worsening condition", 12))
    parts += [(f"Alert: {a['title']}", {"critical": 12, "high": 8, "medium": 4}[a["severity"]]) for a in alerts(p)]
    total = min(100, sum(v for _, v in parts))
    band = "high" if total >= 60 else "moderate" if total >= 35 else "low"
    return {"score": total, "band": band, "factors": [{"factor": k, "points": v} for k, v in parts]}


def timeline(p: dict) -> list[dict]:
    ev = [{"date": e["date"], "kind": "encounter", "title": f"{e['type']} · {e['department']}", "detail": e["reason"]}
          for e in p["encounters"]]
    ev += [{"date": n["date"], "kind": "note", "title": n["type"], "detail": n["text"][:140] + "…"} for n in p["notes"]]
    ev += [{"date": x["date"], "kind": "lab", "title": f"{x['test']} {x['value']} {x['unit']}", "detail": x["flag"]}
           for x in p["labs"] if x["flag"] != "NORMAL"]
    return sorted(ev, key=lambda e: e["date"], reverse=True)


def cohort_stats(patients: list[dict]) -> dict:
    cond = Counter(d["condition"] for p in patients for d in p["diagnoses"])
    ages = [p["age"] for p in patients]
    bands = Counter(risk_score(p)["band"] for p in patients)
    sev = Counter(a["severity"] for p in patients for a in alerts(p))
    age_bins = Counter(("<40" if a < 40 else "40-59" if a < 60 else "60-74" if a < 75 else "75+") for a in ages)
    return {
        "patients": len(patients), "mean_age": round(mean(ages), 1),
        "female_pct": round(100 * sum(p["sex"] == "F" for p in patients) / len(patients), 1),
        "conditions": [{"name": k, "count": v} for k, v in cond.most_common()],
        "age_bins": [{"bin": k, "count": age_bins.get(k, 0)} for k in ("<40", "40-59", "60-74", "75+")],
        "risk_bands": {b: bands.get(b, 0) for b in ("low", "moderate", "high")},
        "alerts": {s: sev.get(s, 0) for s in ("critical", "high", "medium")},
    }
