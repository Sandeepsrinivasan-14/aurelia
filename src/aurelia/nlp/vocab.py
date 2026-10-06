"""Clinical vocabulary used by entity extraction and structured (graph) retrieval.

A hand-built lexicon keeps behaviour transparent and testable. In a real
deployment this layer would be backed by a terminology service (SNOMED CT,
RxNorm, LOINC); the interface (alias -> canonical name) stays the same.
"""
from __future__ import annotations

CONDITION_ALIASES: dict[str, list[str]] = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes", "diabetes", "diabetic", "t2dm"],
    "Essential Hypertension": ["hypertension", "hypertensive", "high blood pressure", "htn"],
    "Chronic Kidney Disease Stage 3": ["chronic kidney disease", "kidney disease", "ckd", "renal disease"],
    "Hypothyroidism": ["hypothyroidism", "hypothyroid", "underactive thyroid"],
    "Asthma": ["asthma", "asthmatic"],
    "Coronary Artery Disease": ["coronary artery disease", "coronary", "cad", "heart disease", "ischemic heart"],
    "Iron Deficiency Anemia": ["iron deficiency", "anemia", "anaemia", "anemic"],
    "Chronic Obstructive Pulmonary Disease": ["copd", "chronic obstructive", "emphysema"],
    "Rheumatoid Arthritis": ["rheumatoid", "arthritis"],
    "Major Depressive Disorder": ["depression", "depressive", "depressed", "mdd"],
    "Hyperlipidemia": ["hyperlipidemia", "hyperlipidaemia", "high cholesterol", "dyslipidemia"],
}

LAB_ALIASES: dict[str, list[str]] = {
    "HbA1c": ["hba1c", "a1c", "glycated hemoglobin", "glycosylated"],
    "Fasting Glucose": ["fasting glucose", "blood glucose", "glucose", "blood sugar", "fasting sugar"],
    "Systolic BP": ["systolic bp", "systolic"],
    "Serum Potassium": ["potassium"],
    "Creatinine": ["creatinine"],
    "eGFR": ["egfr", "gfr"],
    "TSH": ["tsh", "thyroid stimulating"],
    "Peak Flow": ["peak flow"],
    "LDL Cholesterol": ["ldl cholesterol", "ldl"],
    "Total Cholesterol": ["total cholesterol", "cholesterol"],
    "Troponin I": ["troponin"],
    "Hemoglobin": ["hemoglobin", "haemoglobin"],
    "Ferritin": ["ferritin"],
    "SpO2": ["spo2", "oxygen saturation", "oxygen level", "saturation"],
    "FEV1/FVC": ["fev1/fvc", "fev1", "spirometry"],
    "CRP": ["crp", "c-reactive"],
    "ESR": ["esr"],
    "PHQ-9 Score": ["phq-9", "phq9"],
}

# drug class word -> base drug names (matched against what is actually in the record)
DRUG_CLASSES: dict[str, list[str]] = {
    "statin": ["atorvastatin"], "statins": ["atorvastatin"],
    "arb": ["losartan", "telmisartan"], "arbs": ["losartan", "telmisartan"],
    "beta blocker": ["metoprolol"], "beta blockers": ["metoprolol"], "beta-blocker": ["metoprolol"],
    "sulfonylurea": ["glimepiride"], "sulfonylureas": ["glimepiride"],
    "diuretic": ["hydrochlorothiazide"], "diuretics": ["hydrochlorothiazide"],
    "antidepressant": ["sertraline"], "antidepressants": ["sertraline"],
}
INHALER_WORDS = ("inhaler", "inhalers")

ALLERGENS = {"penicillin": "penicillin", "sulfa": "sulfa", "latex": "latex", "shellfish": "shellfish",
             "nsaid": "nsaids", "nsaids": "nsaids"}

# canonical symptom -> surface patterns (regex fragments)
SYMPTOMS: dict[str, list[str]] = {
    "increased thirst": [r"increased thirst", r"polydipsia"],
    "frequent urination": [r"frequent urination", r"polyuria"],
    "blurred vision": [r"blurred vision"],
    "fatigue": [r"fatigue", r"fatigability", r"tiredness"],
    "headache": [r"headaches?"],
    "dizziness": [r"dizziness"],
    "chest pain": [r"chest pain", r"chest discomfort"],
    "breathlessness": [r"breathlessness", r"shortness of breath"],
    "ankle swelling": [r"ankle swelling"],
    "reduced urine output": [r"reduced urine output", r"oliguria"],
    "nocturia": [r"nocturia"],
    "cold intolerance": [r"cold intolerance"],
    "weight gain": [r"weight gain"],
    "constipation": [r"constipation"],
    "dry skin": [r"dry skin"],
    "cough": [r"cough"],
    "wheeze": [r"wheeze", r"wheezing"],
    "chest tightness": [r"chest tightness"],
    "pallor": [r"pallor"],
    "palpitations": [r"palpitations"],
    "morning stiffness": [r"morning stiffness"],
    "joint swelling": [r"joint swelling"],
    "low mood": [r"low mood"],
    "poor sleep": [r"poor sleep"],
    "reduced appetite": [r"reduced appetite"],
    "anhedonia": [r"anhedonia"],
}
