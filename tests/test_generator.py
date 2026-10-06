from mock_ehr.generator import generate_patients


def test_deterministic():
    assert generate_patients(10, seed=1) == generate_patients(10, seed=1)


def test_shape_and_synthetic_ids(records):
    assert len(records) == 40
    for p in records:
        assert p["patient_id"].startswith("AUR-")
        assert p["diagnoses"] and p["labs"] and p["notes"]


def test_flags_match_reference_ranges(records):
    for p in records:
        for lab in p["labs"]:
            expected = "HIGH" if lab["value"] > lab["ref_high"] else "LOW" if lab["value"] < lab["ref_low"] else "NORMAL"
            assert lab["flag"] == expected
