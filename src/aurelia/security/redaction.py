"""PHI redaction for outputs shown in shared or demo contexts."""
from __future__ import annotations

import re

PHONE = re.compile(r"\+?\d[\d\-\s]{8,}\d")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PID = re.compile(r"\bAUR-\d{6}\b")
DOCTOR = re.compile(r"\bDr\.\s+[A-Z][a-z]+")


class Redactor:
    def __init__(self, names: list[str] | None = None):
        names = sorted({n for n in (names or []) if n}, key=len, reverse=True)
        self._names = re.compile("|".join(re.escape(n) for n in names)) if names else None

    def redact(self, text: str) -> str:
        if self._names:
            text = self._names.sub("[PATIENT]", text)
        text = PHONE.sub("[PHONE]", text)
        text = EMAIL.sub("[EMAIL]", text)
        text = PID.sub("[MRN]", text)
        return DOCTOR.sub("Dr. [CLINICIAN]", text)
