from fastapi.testclient import TestClient

from aurelia.api.app import create_app
from aurelia.config import Settings
from aurelia.engine import Engine
from mock_ehr.server import app as ehr_app


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_patient_list_and_detail(client):
    rows = client.get("/api/patients").json()
    assert rows == sorted(rows, key=lambda r: (-r["risk"], r["name"]))
    d = client.get(f"/api/patients/{rows[0]['patient_id']}").json()
    assert {"risk", "alerts", "labs", "timeline"} <= d.keys()
    assert client.get("/api/patients/AUR-NOPE").status_code == 404


def test_query_returns_cited_evidence(client):
    r = client.post("/api/query", json={"question": "latest HbA1c results"}).json()
    import re
    assert r["evidence"] and re.search(r"\[\d+\]", r["answer"])


def test_query_scoped_and_redacted(client):
    pid = client.get("/api/patients").json()[0]["patient_id"]
    r = client.post("/api/query", json={"question": "who is this patient", "patient_id": pid, "redact": True}).json()
    assert all(e["patient_id"] != pid for e in r["evidence"])


def test_validation(client):
    assert client.post("/api/query", json={"question": "x"}).status_code == 422
    assert client.post("/api/query", json={"question": "valid question", "mode": "nope"}).status_code == 422


def test_audit_records_queries(client):
    client.post("/api/query", json={"question": "kidney function"})
    a = client.get("/api/audit").json()
    assert a["verify"]["valid"] and a["entries"][0]["action"] == "query"
    assert "kidney" not in str(a)  # question text is never stored, only its length


def test_api_key_enforced(records, tmp_path):
    s = Settings(audit_path=tmp_path / "a.jsonl", api_key="secret")
    c = TestClient(create_app(s, Engine(s, records=records)))
    assert c.get("/api/patients").status_code == 401
    assert c.get("/api/patients", headers={"x-api-key": "secret"}).status_code == 200


def test_mock_ehr_requires_token():
    c = TestClient(ehr_app)
    assert c.get("/v1/patients").status_code == 401
    ok = c.get("/v1/patients", headers={"Authorization": "Bearer dev-token-change-me"})
    assert ok.status_code == 200 and ok.json()["total"] > 0


def test_ingestion_over_http_from_mock_ehr():
    from aurelia.ingestion.connector import EHRClient

    http = TestClient(ehr_app, base_url="http://ehr")
    client = EHRClient("http://ehr", "dev-token-change-me", http=http)
    ids = client.list_ids(page_size=25)  # exercises pagination
    assert len(ids) == http.get("/health").json()["patients"]
    assert client.get(ids[0])["patient_id"] == ids[0]


def test_ingestion_rejects_bad_token():
    import pytest

    from aurelia.ingestion.connector import EHRClient

    bad = EHRClient("http://ehr", "wrong", http=TestClient(ehr_app, base_url="http://ehr"))
    with pytest.raises(Exception) as err:  # exception class differs between httpx builds; status is what matters
        bad.list_ids()
    assert err.value.response.status_code == 401
