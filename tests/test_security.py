import json

from aurelia.security.audit import AuditLog
from aurelia.security.redaction import Redactor


def test_redaction_masks_identifiers():
    r = Redactor(["Meera Iyer"])
    out = r.redact("Meera Iyer (AUR-100001) call +91-9876543210 or a@b.com, seen by Dr. Raman")
    for leaked in ("Meera", "AUR-100001", "9876543210", "a@b.com", "Raman"):
        assert leaked not in out


def test_audit_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(4):
        log.record("query", n=i)
    assert log.verify() == {"valid": True, "entries": 4}
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    row = json.loads(lines[1]); row["details"]["n"] = 99
    lines[1] = json.dumps(row)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    v = AuditLog(tmp_path / "a.jsonl").verify()
    assert v["valid"] is False and v["broken_at"] == 2


def test_audit_survives_restart(tmp_path):
    AuditLog(tmp_path / "a.jsonl").record("x")
    again = AuditLog(tmp_path / "a.jsonl"); again.record("y")
    assert again.verify()["entries"] == 2
