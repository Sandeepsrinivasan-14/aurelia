"""Tests for the RAG strategies, graph, NER, verifier, conversation memory and AI analytics."""
import pytest

from aurelia.analytics.forecast import anomaly, forecast
from aurelia.nlp.ner import extract_entities
from aurelia.rag.analysis import parse_time
from aurelia.rag.conversation import condense
from aurelia.rag.strategies import STRATEGIES
from aurelia.rag.verify import verify_answer


def _brute_metformin_low_egfr(records):
    out = set()
    for p in records:
        on = any(m["name"].lower().startswith("metformin") and m["active"] for m in p["medications"])
        low = [r for r in p["labs"] if r["test"].lower() == "egfr"]
        if on and low and sorted(low, key=lambda r: r["date"])[-1]["flag"] == "LOW":
            out.add(p["patient_id"])
    return out


def test_all_strategies_return_evidence(engine):
    for sid in STRATEGIES:
        r = engine.query("What is the latest creatinine?", strategy=sid)
        assert r["strategy"] in STRATEGIES and 0 <= r["confidence"] <= 1
        assert "verification" in r and "trace" in r


def test_graph_matches_brute_force(engine, records):
    r = engine.query("which patients on metformin have low eGFR?", strategy="graph")
    got = {p["patient_id"] for p in (r["structured"] or {}).get("patients", [])}
    assert got == _brute_metformin_low_egfr(records)


def test_router_sends_cohort_questions_to_graph(engine):
    r = engine.query("which patients on metformin have low eGFR?")
    assert r["requested_strategy"] == "adaptive" and r["strategy"] == "graph"


def test_abstains_on_unsupported_topic(engine):
    for sid in ("corrective", "adaptive"):
        r = engine.query("Does anyone have a history of cancer?", strategy=sid)
        assert r["abstained"] and "can't answer" in r["answer"]


def test_legacy_mode_still_works(engine):
    assert engine.query("metformin", mode="bm25")["strategy"] == "bm25"
    with pytest.raises(KeyError):
        engine.query("metformin", strategy="nope")


def test_audit_is_not_flooded_by_compare(engine):
    before = len(engine.audit.entries(10_000))
    rows = engine.compare("kidney function")
    assert len(rows) == len(STRATEGIES) - 1
    assert len(engine.audit.entries(10_000)) == before + 1


def test_temporal_window_parsing():
    from datetime import date
    w = parse_time("labs in the last 6 months", date(2026, 6, 30))
    assert w.start == date(2026, 1, 1) and w.end == date(2026, 6, 30)
    assert parse_time("latest HbA1c", date(2026, 1, 1)).latest
    assert not parse_time("HbA1c", date(2026, 1, 1)).active


def test_conversation_condense_carries_modifiers():
    q, why = condense("and creatinine?", ["What is the latest HbA1c?"])
    assert "latest" in q and why
    assert condense("What is the HbA1c trend?", [])[1] is None


def test_ner_negation_scope():
    ents = {e.text: e for e in extract_entities("Denies palpitations but reports fatigue and wheeze.")}
    assert ents["palpitations"].negated is True
    assert ents["fatigue"].negated is False and ents["wheeze"].negated is False


def test_verifier_flags_unsupported_number():
    ev = [{"n": 1, "text": "Lab result HbA1c: 8.1 % on 2025-01-01"}]
    assert verify_answer("- HbA1c was 8.1 % [1]", ev)["faithfulness"] == 1.0
    assert verify_answer("- HbA1c was 9.9 % [1]", ev)["faithfulness"] == 0.0


def test_forecast_needs_enough_points_and_flags_anomaly():
    pts = [{"date": f"2025-0{i}-01", "value": 7 + i * .5} for i in range(1, 6)]
    f = forecast(pts)
    assert f and f["predicted"] > pts[-1]["value"] and f["low"] < f["predicted"] < f["high"]
    assert forecast(pts[:3]) is None
    spike = pts[:-1] + [{"date": "2025-06-01", "value": 40}]
    assert anomaly(spike)["flag"] is True


def test_similarity_and_map(engine):
    pid = engine.records[0]["patient_id"]
    sim = engine.similar(pid, 3)
    assert len(sim) == 3 and all(s["patient_id"] != pid for s in sim)
    assert sim[0]["similarity"] >= sim[-1]["similarity"]
    m = engine.cohort_map()
    assert len(m["points"]) == len(engine.records) and m["clusters"]


def test_new_endpoints(client):
    assert len(client.get("/api/strategies").json()) == len(STRATEGIES)
    r = client.post("/api/query", json={"question": "latest eGFR", "strategy": "temporal", "history": []})
    assert r.status_code == 200 and r.json()["strategy"] == "temporal"
    assert client.post("/api/compare", json={"question": "kidney"}).status_code == 200
    assert client.get("/api/cohort/map").status_code == 200
    assert client.get("/api/graph").json()["total_nodes"] > 0
    pid = client.get("/api/patients").json()[0]["patient_id"]
    assert client.get(f"/api/patients/{pid}/similar").status_code == 200
    assert client.get("/api/patients/NOPE/similar").status_code == 404
