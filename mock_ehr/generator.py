"""Deterministic synthetic patient generator.

Everything produced here is fabricated. Names, identifiers and clinical
histories are randomly composed and do not describe any real person.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

FIRST_F = ["Meera", "Ananya", "Diya", "Isha", "Naina", "Priya", "Sneha", "Lakshmi", "Pooja", "Divya"]
FIRST_M = ["Aarav", "Rohan", "Vikram", "Kabir", "Arjun", "Rahul", "Karthik", "Aditya", "Nikhil", "Sanjay"]
LAST = ["Raman", "Iyer", "Menon", "Nair", "Sharma", "Verma", "Pillai", "Reddy", "Kapoor", "Das",
        "Bose", "Rao", "Gupta", "Joshi", "Natarajan", "Krishnan", "Singh", "Patel", "Chandra", "Mehta"]

# condition -> (ICD-10 style code, typical meds, typical labs (name, unit, normal range, abnormal high?), note template)
CONDITIONS = {
    "Type 2 Diabetes Mellitus": ("E11.9", ["Metformin 500 mg", "Glimepiride 1 mg", "Sitagliptin 100 mg"],
                                 [("HbA1c", "%", (4.0, 5.6), 8.4), ("Fasting Glucose", "mg/dL", (70, 99), 168)]),
    "Essential Hypertension": ("I10", ["Amlodipine 5 mg", "Telmisartan 40 mg", "Hydrochlorothiazide 12.5 mg"],
                               [("Systolic BP", "mmHg", (90, 120), 156), ("Serum Potassium", "mmol/L", (3.5, 5.0), 5.4)]),
    "Chronic Kidney Disease Stage 3": ("N18.3", ["Losartan 50 mg", "Sodium Bicarbonate 500 mg"],
                                       [("Creatinine", "mg/dL", (0.6, 1.2), 2.1), ("eGFR", "mL/min", (90, 120), 44)]),
    "Hypothyroidism": ("E03.9", ["Levothyroxine 75 mcg"], [("TSH", "mIU/L", (0.4, 4.0), 9.8)]),
    "Asthma": ("J45.909", ["Salbutamol inhaler", "Budesonide inhaler"], [("Peak Flow", "L/min", (400, 600), 280)]),
    "Coronary Artery Disease": ("I25.10", ["Aspirin 75 mg", "Atorvastatin 40 mg", "Metoprolol 25 mg"],
                                [("LDL Cholesterol", "mg/dL", (0, 100), 162), ("Troponin I", "ng/mL", (0, 0.04), 0.02)]),
    "Iron Deficiency Anemia": ("D50.9", ["Ferrous Sulfate 325 mg", "Folic Acid 5 mg"],
                               [("Hemoglobin", "g/dL", (12.0, 16.0), 8.9), ("Ferritin", "ng/mL", (20, 200), 7)]),
    "Chronic Obstructive Pulmonary Disease": ("J44.9", ["Tiotropium inhaler", "Formoterol inhaler"],
                                              [("SpO2", "%", (95, 100), 89), ("FEV1/FVC", "ratio", (0.7, 1.0), 0.58)]),
    "Rheumatoid Arthritis": ("M06.9", ["Methotrexate 15 mg weekly", "Folic Acid 5 mg"],
                             [("CRP", "mg/L", (0, 5), 31), ("ESR", "mm/hr", (0, 20), 58)]),
    "Major Depressive Disorder": ("F32.9", ["Sertraline 50 mg"], [("PHQ-9 Score", "points", (0, 4), 17)]),
    "Hyperlipidemia": ("E78.5", ["Atorvastatin 20 mg"], [("Total Cholesterol", "mg/dL", (0, 200), 248)]),
}

ALLERGIES = ["Penicillin", "Sulfa drugs", "Latex", "Shellfish", "NSAIDs", "None known"]
DEPARTMENTS = ["Cardiology", "Endocrinology", "Nephrology", "Pulmonology", "General Medicine", "Rheumatology",
               "Psychiatry", "Hematology"]
SYMPTOM_PHRASES = {
    "Type 2 Diabetes Mellitus": ["increased thirst", "frequent urination", "blurred vision", "fatigue"],
    "Essential Hypertension": ["occasional headaches", "dizziness on standing", "no chest pain"],
    "Chronic Kidney Disease Stage 3": ["ankle swelling", "reduced urine output", "nocturia", "fatigue"],
    "Hypothyroidism": ["cold intolerance", "weight gain", "constipation", "dry skin"],
    "Asthma": ["nocturnal cough", "wheeze on exertion", "chest tightness"],
    "Coronary Artery Disease": ["exertional chest discomfort", "breathlessness on stairs"],
    "Iron Deficiency Anemia": ["pallor", "easy fatigability", "palpitations on exertion"],
    "Chronic Obstructive Pulmonary Disease": ["productive cough", "breathlessness at rest", "wheeze"],
    "Rheumatoid Arthritis": ["morning stiffness over 60 minutes", "symmetric small joint swelling"],
    "Major Depressive Disorder": ["low mood", "poor sleep", "reduced appetite", "anhedonia"],
    "Hyperlipidemia": ["asymptomatic", "family history of early heart disease"],
}


def _lab_value(rng: random.Random, lo_hi, abnormal: float, force_abnormal: bool) -> float:
    lo, hi = lo_hi
    if force_abnormal:
        v = abnormal * rng.uniform(0.93, 1.08)
    else:
        v = rng.uniform(lo, hi)
    return round(v, 2)


def generate_patients(n: int = 60, seed: int = 10) -> list[dict]:
    rng = random.Random(seed)
    today = date(2026, 9, 1)
    patients: list[dict] = []
    for i in range(n):
        pid = f"AUR-{100000 + i}"
        age = rng.randint(24, 86)
        sex = rng.choice(["F", "M"])
        name = f"{rng.choice(FIRST_F if sex == 'F' else FIRST_M)} {rng.choice(LAST)}"
        n_cond = rng.choices([1, 2, 3], weights=[5, 4, 2])[0]
        conds = rng.sample(list(CONDITIONS), n_cond)
        # clinically common comorbidity pairs (diabetic nephropathy, hypertension + dyslipidaemia)
        if "Type 2 Diabetes Mellitus" in conds and "Chronic Kidney Disease Stage 3" not in conds and rng.random() < 0.45:
            conds.append("Chronic Kidney Disease Stage 3")
        if "Essential Hypertension" in conds and "Hyperlipidemia" not in conds and rng.random() < 0.35:
            conds.append("Hyperlipidemia")
        allergy = rng.choice(ALLERGIES)
        if "Coronary Artery Disease" in conds and rng.random() < 0.45:
            allergy = "NSAIDs"  # seeds realistic allergy-conflict alert cases for demos

        diagnoses, medications, labs, encounters, notes = [], [], [], [], []
        for c in conds:
            code, meds, lab_defs = CONDITIONS[c]
            onset = today - timedelta(days=rng.randint(200, 3200))
            diagnoses.append({"condition": c, "code": code, "onset": onset.isoformat(),
                              "status": rng.choice(["active", "active", "controlled", "worsening"])})
            for m in rng.sample(meds, k=min(len(meds), rng.randint(1, 2))):
                medications.append({"name": m, "for": c,
                                    "started": (onset + timedelta(days=rng.randint(0, 60))).isoformat(),
                                    "active": rng.random() > 0.1})
            for (lab, unit, rng_n, abn) in lab_defs:
                for _ in range(rng.randint(2, 4)):
                    when = today - timedelta(days=rng.randint(10, 600))
                    force = rng.random() < 0.45
                    val = _lab_value(rng, rng_n, abn, force)
                    labs.append({"test": lab, "value": val, "unit": unit, "ref_low": rng_n[0], "ref_high": rng_n[1],
                                 "date": when.isoformat(), "flag": "HIGH" if val > rng_n[1] else ("LOW" if val < rng_n[0] else "NORMAL")})

        for _ in range(rng.randint(2, 5)):
            when = today - timedelta(days=rng.randint(5, 700))
            c = rng.choice(conds)
            dept = rng.choice(DEPARTMENTS)
            kind = rng.choice(["Outpatient visit", "Outpatient visit", "Emergency visit", "Follow-up", "Admission"])
            sym = ", ".join(rng.sample(SYMPTOM_PHRASES[c], k=min(2, len(SYMPTOM_PHRASES[c]))))
            encounters.append({"date": when.isoformat(), "type": kind, "department": dept, "reason": c})
            notes.append({
                "date": when.isoformat(), "type": "Progress note", "author": f"Dr. {rng.choice(LAST)}",
                "text": (f"{kind} in {dept}. Patient with {c} presents with {sym}. "
                         f"Allergies: {allergy}. Plan: continue current therapy, review labs, follow up in "
                         f"{rng.choice([2, 4, 6, 12])} weeks. Counselled on diet, adherence and warning signs."),
            })
        if rng.random() < 0.5:
            c = rng.choice(conds)
            when = today - timedelta(days=rng.randint(30, 500))
            notes.append({
                "date": when.isoformat(), "type": "Discharge summary", "author": f"Dr. {rng.choice(LAST)}",
                "text": (f"Admitted with decompensation of {c}. Treated with supportive care and adjustment of "
                         f"medication. Discharged stable. Advised repeat investigations and outpatient review. "
                         f"Allergy status: {allergy}."),
            })

        patients.append({
            "patient_id": pid, "name": name, "age": age, "sex": sex, "allergies": allergy,
            "phone": f"+91-9{rng.randint(100000000, 999999999)}",
            "diagnoses": diagnoses, "medications": medications, "labs": labs,
            "encounters": sorted(encounters, key=lambda e: e["date"]), "notes": sorted(notes, key=lambda e: e["date"]),
        })
    return patients
