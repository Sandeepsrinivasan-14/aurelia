from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from environment variables (see .env.example)."""

    ehr_url: str | None = field(default_factory=lambda: os.getenv("AURELIA_EHR_URL") or None)
    ehr_token: str = field(default_factory=lambda: os.getenv("AURELIA_EHR_TOKEN", "dev-token-change-me"))
    synthetic_patients: int = field(default_factory=lambda: int(os.getenv("AURELIA_SYNTHETIC_PATIENTS", "60")))
    llm_backend: str = field(default_factory=lambda: os.getenv("AURELIA_LLM", "extractive"))  # extractive | ollama
    ollama_url: str = field(default_factory=lambda: os.getenv("AURELIA_OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("AURELIA_OLLAMA_MODEL", "llama3.1"))
    api_key: str | None = field(default_factory=lambda: os.getenv("AURELIA_API_KEY") or None)
    audit_path: Path = field(default_factory=lambda: Path(os.getenv("AURELIA_AUDIT_PATH", "data/audit.jsonl")))
    top_k: int = field(default_factory=lambda: int(os.getenv("AURELIA_TOP_K", "6")))


def get_settings() -> Settings:
    return Settings()
