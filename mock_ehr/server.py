"""Mock EHR service.

A tiny FastAPI app that imitates the shape of a typical hospital integration
(token auth, patient list, patient record) over *synthetic* data only. It lets
the ingestion layer be developed and demoed without any real system.

Run:  uvicorn mock_ehr.server:app --port 8100
"""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from .generator import generate_patients

app = FastAPI(title="Aurelia Mock EHR", version="1.0.0",
              description="Synthetic electronic health record service for development and demos.")

_PATIENTS = {p["patient_id"]: p for p in generate_patients(int(os.getenv("MOCK_EHR_PATIENTS", "60")))}
_TOKEN = os.getenv("MOCK_EHR_TOKEN", "dev-token-change-me")


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "patients": len(_PATIENTS)}


@app.get("/v1/patients", dependencies=[Depends(require_token)])
def list_patients(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict:
    ids = sorted(_PATIENTS)
    page = ids[offset: offset + limit]
    return {"total": len(ids), "items": [{"patient_id": i, "name": _PATIENTS[i]["name"]} for i in page]}


@app.get("/v1/patients/{patient_id}", dependencies=[Depends(require_token)])
def get_patient(patient_id: str) -> dict:
    p = _PATIENTS.get(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p
