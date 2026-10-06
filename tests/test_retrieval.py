from aurelia.ingestion.chunker import chunk_patient
from aurelia.retrieval.embeddings import HashingEmbedder, tokenize
from aurelia.retrieval.hybrid import BM25, HybridIndex


def test_chunker_covers_every_record_part(records):
    p = records[0]
    kinds = {c.kind for c in chunk_patient(p)}
    assert {"profile", "diagnosis", "lab", "note", "encounter"} <= kinds
    assert len({c.chunk_id for c in chunk_patient(p)}) == len(chunk_patient(p))


def test_synonym_expansion():
    assert "glucose" in tokenize("blood sugar", expand=True)
    assert "glucose" not in tokenize("blood sugar", expand=False)


def test_embeddings_are_unit_norm_and_deterministic():
    e = HashingEmbedder(256).fit(["alpha beta", "beta gamma"])
    a, b = e.encode(["alpha beta"]), e.encode(["alpha beta"])
    assert abs((a[0] ** 2).sum() - 1) < 1e-5 and (a == b).all()


def test_bm25_ranks_matching_doc_first():
    bm = BM25([["kidney", "failure"], ["thyroid", "hormone"], ["kidney", "stone", "pain"]])
    s = bm.scores(["thyroid"])
    assert max(s, key=s.get) == 1


def test_patient_scoping_never_leaks(records):
    idx = HybridIndex([c for p in records for c in chunk_patient(p)])
    pid = records[5]["patient_id"]
    hits = idx.search("diagnosis", k=10, patient_id=pid)
    assert hits and all(h.chunk.patient_id == pid for h in hits)
    assert idx.search("diagnosis", patient_id="AUR-NOPE") == []


def test_all_modes_return_results(engine):
    for mode in ("bm25", "dense", "hybrid"):
        assert engine.index.search("HbA1c result", k=3, mode=mode)
