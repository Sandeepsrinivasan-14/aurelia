from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..engine import Engine


def _find_ui() -> Path:
    """Locate the static UI for editable installs, wheel installs (Docker) and custom layouts."""
    candidates = [os.getenv("AURELIA_UI_DIR"), Path(__file__).resolve().parents[3] / "ui" / "static", Path.cwd() / "ui" / "static"]
    for c in candidates:
        if c and (Path(c) / "index.html").exists():
            return Path(c)
    return Path(candidates[1])


UI_DIR = _find_ui()


class QueryIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    patient_id: str | None = None
    k: int = Field(default=6, ge=1, le=20)
    strategy: str | None = Field(default=None, max_length=32)          # see GET /api/strategies; default "adaptive"
    mode: str | None = Field(default=None, pattern="^(hybrid|bm25|dense)$")   # legacy alias for strategy
    redact: bool = False
    history: list[str] = Field(default_factory=list, max_length=6)    # previous questions, for follow-ups


class CompareIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    patient_id: str | None = None
    strategies: list[str] | None = None
    k: int = Field(default=6, ge=1, le=20)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    eng = engine or Engine(settings)
    app = FastAPI(title="Aurelia", version="1.0.0",
                  description="Clinical intelligence workspace: hybrid retrieval, grounded answers, alerts, audit.")

    def guard(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", **eng.stats()}

    @app.get("/api/patients", dependencies=[Depends(guard)])
    def patients(q: str = "", sort: str = Query("risk", pattern="^(risk|name)$")) -> list[dict]:
        return eng.patient_list(q, sort)

    @app.get("/api/patients/{pid}", dependencies=[Depends(guard)])
    def patient(pid: str) -> dict:
        v = eng.patient_view(pid)
        if not v:
            raise HTTPException(404, "Patient not found")
        return v

    @app.get("/api/strategies", dependencies=[Depends(guard)])
    def strategies() -> list[dict]:
        return eng.strategies()

    @app.post("/api/query", dependencies=[Depends(guard)])
    def query(body: QueryIn) -> dict:
        if body.patient_id and body.patient_id not in eng.patients:
            raise HTTPException(404, "Patient not found")
        try:
            return eng.query(body.question, body.patient_id, body.k, body.mode, body.redact,
                             strategy=body.strategy, history=body.history)
        except KeyError as e:
            raise HTTPException(422, str(e).strip("'\"")) from e

    @app.post("/api/compare", dependencies=[Depends(guard)])
    def compare(body: CompareIn) -> list[dict]:
        if body.patient_id and body.patient_id not in eng.patients:
            raise HTTPException(404, "Patient not found")
        known = {s["id"] for s in eng.strategies()}
        if body.strategies and not set(body.strategies) <= known:
            raise HTTPException(422, "unknown strategy in list")
        return eng.compare(body.question, body.patient_id, body.strategies, body.k)

    @app.get("/api/patients/{pid}/similar", dependencies=[Depends(guard)])
    def similar(pid: str, k: int = Query(5, ge=1, le=15)) -> list[dict]:
        rows = eng.similar(pid, k)
        if rows is None:
            raise HTTPException(404, "Patient not found")
        return rows

    @app.get("/api/cohort/map", dependencies=[Depends(guard)])
    def cohort_map() -> dict:
        return eng.cohort_map()

    @app.get("/api/graph", dependencies=[Depends(guard)])
    def graph() -> dict:
        return eng.graph.stats()

    @app.get("/api/benchmark", dependencies=[Depends(guard)])
    def benchmark(refresh: bool = False) -> dict:
        return eng.benchmark(refresh)

    @app.get("/api/cohort", dependencies=[Depends(guard)])
    def cohort() -> dict:
        return eng.cohort()

    @app.get("/api/audit", dependencies=[Depends(guard)])
    def audit(limit: int = Query(40, ge=1, le=200)) -> dict:
        return {"verify": eng.audit.verify(), "entries": eng.audit.entries(limit)}

    if UI_DIR.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app


def app_factory() -> FastAPI:  # for `uvicorn aurelia.api.app:app_factory --factory`
    return create_app()
