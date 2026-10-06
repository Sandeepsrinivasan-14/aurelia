"""Tamper-evident audit trail.

Each entry stores the SHA-256 of the previous entry, forming a hash chain.
Editing or deleting any line breaks verification from that point onward.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._tail_hash()

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    @staticmethod
    def _digest(prev: str, body: dict) -> str:
        return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()

    def record(self, action: str, actor: str = "operator", **details) -> dict:
        with self._lock:
            body = {"ts": round(time.time(), 3), "action": action, "actor": actor, "details": details, "prev": self._last}
            entry = {**body, "hash": self._digest(self._last, {k: v for k, v in body.items() if k != "prev"})}
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self._last = entry["hash"]
            return entry

    def entries(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[-limit:][::-1]

    def verify(self) -> dict:
        prev, n = GENESIS, 0
        if not self.path.exists():
            return {"valid": True, "entries": 0}
        with self.path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                body = {k: e[k] for k in ("ts", "action", "actor", "details")}
                if e["prev"] != prev or e["hash"] != self._digest(prev, body):
                    return {"valid": False, "entries": n, "broken_at": n + 1}
                prev, n = e["hash"], n + 1
        return {"valid": True, "entries": n}
