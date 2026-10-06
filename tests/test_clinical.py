from aurelia.analytics import clinical


def _p(**kw):
    base = {"age": 70, "allergies": "None known", "diagnoses": [], "medications": [], "labs": [], "encounters": [], "notes": []}
    base.update(kw)
    return base


def test_allergy_conflict_is_critical():
    p = _p(allergies="NSAIDs", medications=[{"name": "Aspirin 75 mg", "for": "CAD", "started": "2024-01-01", "active": True}])
    a = clinical.alerts(p)
    assert a[0]["severity"] == "critical" and "Allergy" in a[0]["title"]


def test_inactive_drug_does_not_trigger():
    p = _p(allergies="NSAIDs", medications=[{"name": "Aspirin 75 mg", "for": "CAD", "started": "2024-01-01", "active": False}])
    assert clinical.alerts(p) == []


def test_hyperkalaemia_with_arb():
    p = _p(medications=[{"name": "Telmisartan 40 mg", "for": "HTN", "started": "2024-01-01", "active": True}],
           labs=[{"test": "Serum Potassium", "value": 5.6, "unit": "mmol/L", "ref_low": 3.5, "ref_high": 5.0, "date": "2026-01-01", "flag": "HIGH"}])
    assert any(a["title"] == "Hyperkalaemia risk" for a in clinical.alerts(p))


def test_trend_detection():
    def pts(*v):
        return [{"value": x} for x in v]

    assert clinical.trend(pts(5, 6)) == "rising"
    assert clinical.trend(pts(6, 5)) == "falling"
    assert clinical.trend(pts(5, 5.1)) == "stable"
    assert clinical.trend(pts(5)) == "insufficient data"


def test_risk_score_bounded_and_explainable(records):
    for p in records:
        r = clinical.risk_score(p)
        assert 0 <= r["score"] <= 100 and r["band"] in ("low", "moderate", "high") and r["factors"]


def test_cohort_totals(records):
    c = clinical.cohort_stats(records)
    assert c["patients"] == len(records) and sum(c["risk_bands"].values()) == len(records)
