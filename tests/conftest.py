import pytest
from fastapi.testclient import TestClient

from aurelia.api.app import create_app
from aurelia.config import Settings
from aurelia.engine import Engine
from mock_ehr.generator import generate_patients


@pytest.fixture(scope="session")
def records():
    return generate_patients(40, seed=3)


@pytest.fixture()
def engine(records, tmp_path):
    return Engine(Settings(audit_path=tmp_path / "audit.jsonl"), records=records)


@pytest.fixture()
def client(engine):
    return TestClient(create_app(engine.settings, engine))
