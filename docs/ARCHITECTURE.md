# Architecture notes

## Data flow

1. **Connector** loads patient records – from the bundled synthetic generator, or over HTTP from any service that implements `GET /v1/patients` and `GET /v1/patients/{id}` with a bearer token (the included `mock_ehr` does).
2. **Chunker** converts each record into small, typed evidence units (`profile`, `diagnosis`, `medication`, `lab`, `encounter`, `note`). One lab result = one chunk, so a citation points at a single verifiable fact.
3. **Hybrid index** keeps a BM25 inverted index and a dense matrix. A query runs both, and the two rankings are merged with Reciprocal Rank Fusion.
4. **Engine** applies patient scoping, calls the answerer, optionally redacts, and writes an audit entry.
5. **API/UI** expose the engine and the clinical analytics.

## Key decisions

| Decision | Why | Trade‑off |
|---|---|---|
| Reciprocal Rank Fusion instead of weighted score blending | BM25 and cosine scores live on different scales; ranks need no calibration | Ignores score magnitude |
| Pre‑filter by patient *before* ranking | Guarantees a patient‑scoped query cannot surface another patient's chunk (covered by a test) | Slightly less efficient than a shared index with post‑filtering |
| Fine‑grained chunks | Precise citations and better recall on numeric questions | More chunks, less surrounding context per hit |
| Extractive answerer as default | Cannot hallucinate, needs no GPU or download, deterministic for tests | Less fluent than an LLM; the Ollama adapter covers that |
| Hashing embedder | Zero dependencies, deterministic, instant start | Weaker semantics than a trained encoder (see benchmark and README limitations) |
| Rule‑based alerts | Auditable, unit‑testable, explainable to a clinician | Coverage limited to hand‑written rules |
| Hash‑chained audit log | Tamper evidence with no external service | Detects but does not prevent tampering |
| Never log question text | Questions can contain PHI; the log stores length only | Less forensic detail |

## Extending

- **New embedder:** implement `fit(texts)` and `encode(texts, is_query)` returning L2‑normalised `float32` rows; pass it to `HybridIndex(chunks, embedder=...)`.
- **New LLM backend:** implement `Answerer.answer(question, hits) -> str` and register it in `llm/answerers.py:build_answerer`.
- **New alert rule:** add a block to `analytics/clinical.py:alerts` and a unit test in `tests/test_clinical.py`.
- **Real EHR:** write a connector returning the same record shape (see `mock_ehr/generator.py`) – ideally a FHIR R4 mapper.
