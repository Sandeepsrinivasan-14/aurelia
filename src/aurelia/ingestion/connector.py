"""Record sources.

By default Aurelia loads the bundled synthetic cohort in-process. If
``AURELIA_EHR_URL`` is set it pulls from any service exposing the same
``/v1/patients`` contract (the bundled ``mock_ehr`` app implements it).
"""
from __future__ import annotations

import httpx

from ..config import Settings


class EHRClient:
    def __init__(self, base_url: str, token: str, timeout: float = 15.0, http: httpx.Client | None = None):
        # `http` can be injected (e.g. a Starlette TestClient) so tests need no network.
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._http.headers["Authorization"] = f"Bearer {token}"

    def list_ids(self, page_size: int = 100) -> list[str]:
        ids: list[str] = []
        offset = 0
        while True:
            r = self._http.get("/v1/patients", params={"limit": page_size, "offset": offset})
            r.raise_for_status()
            body = r.json()
            ids += [i["patient_id"] for i in body["items"]]
            offset += page_size
            if offset >= body["total"]:
                return ids

    def get(self, patient_id: str) -> dict:
        r = self._http.get(f"/v1/patients/{patient_id}")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._http.close()


def load_records(settings: Settings) -> list[dict]:
    if settings.ehr_url:
        client = EHRClient(settings.ehr_url, settings.ehr_token)
        try:
            return [client.get(pid) for pid in client.list_ids()]
        finally:
            client.close()
    from mock_ehr.generator import generate_patients
    return generate_patients(settings.synthetic_patients)
