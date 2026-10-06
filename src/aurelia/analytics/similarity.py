"""Unsupervised patient modelling: similarity search, k-means cohorts, 3-D PCA map.

Each patient becomes a feature vector (age, sex, condition multi-hot, lab
deviations from reference range, medication count, complexity score). Everything
is NumPy; nothing is learned from labels, so there is no label leakage and no
claim of predictive accuracy - the value is *exploration*: "who looks like this
patient?" and "what natural groups exist in this cohort?".
"""
from __future__ import annotations

import numpy as np

from ..ingestion.chunker import drug_base
from . import clinical

SHORT = {"Type 2 Diabetes Mellitus": "T2DM", "Essential Hypertension": "HTN", "Chronic Kidney Disease Stage 3": "CKD",
         "Chronic Obstructive Pulmonary Disease": "COPD", "Coronary Artery Disease": "CAD", "Iron Deficiency Anemia": "Anaemia",
         "Rheumatoid Arthritis": "RA", "Major Depressive Disorder": "Depression", "Hyperlipidemia": "Dyslipidaemia"}


class PatientSpace:
    def __init__(self, records: list[dict], k_clusters: int = 5, seed: int = 0):
        self.records = records
        self.ids = [p["patient_id"] for p in records]
        self.conditions = sorted({d["condition"] for p in records for d in p["diagnoses"]})
        self.tests = sorted({l_["test"] for p in records for l_ in p["labs"]})
        self.X = np.stack([self._vector(p) for p in records]).astype(float)
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.U = self.X / np.where(norms == 0, 1, norms)
        self.k = min(k_clusters, len(records))
        self.labels = self._kmeans(self.X, self.k, seed)
        self.coords = self._pca3(self.X)
        self.cluster_names = self._name_clusters()

    # ---- features
    def _vector(self, p: dict) -> np.ndarray:
        latest = {t: pts[-1] for t, pts in clinical.lab_series(p).items()}
        conds = {d["condition"] for d in p["diagnoses"]}
        labs = []
        for t in self.tests:
            l_ = latest.get(t)
            if not l_:
                labs.append(0.0)
                continue
            mid, width = (l_["ref_low"] + l_["ref_high"]) / 2, max(l_["ref_high"] - l_["ref_low"], 1e-6)
            labs.append(float(np.clip((l_["value"] - mid) / width, -3, 3)) / 3)
        n_meds = sum(1 for m in p["medications"] if m["active"])
        return np.array([p["age"] / 100, 1.0 if p["sex"] == "F" else 0.0, n_meds / 6, clinical.risk_score(p)["score"] / 100,
                         *[1.5 if c in conds else 0.0 for c in self.conditions], *labs])

    # ---- similarity
    def similar(self, pid: str, k: int = 5) -> list[dict]:
        i = self.ids.index(pid)
        sims = self.U @ self.U[i]
        order = [j for j in np.argsort(-sims) if j != i][:k]
        base = self.records[i]
        out = []
        for j in order:
            other = self.records[int(j)]
            shared_c = sorted({d["condition"] for d in base["diagnoses"]} & {d["condition"] for d in other["diagnoses"]})
            shared_m = sorted({drug_base(m["name"]) for m in base["medications"] if m["active"]}
                              & {drug_base(m["name"]) for m in other["medications"] if m["active"]})
            out.append({"patient_id": other["patient_id"], "name": other["name"], "age": other["age"], "sex": other["sex"],
                        "similarity": round(float(sims[j]), 3), "shared_conditions": shared_c, "shared_medications": shared_m})
        return out

    # ---- clustering (k-means++ init, Lloyd iterations)
    @staticmethod
    def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 60) -> np.ndarray:
        rng = np.random.default_rng(seed)
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() else np.full(len(X), 1 / len(X))
            centers.append(X[rng.choice(len(X), p=probs)])
        C = np.stack(centers)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
            new = d.argmin(1)
            if (new == labels).all() and _ > 0:
                break
            labels = new
            for c in range(k):
                if (labels == c).any():
                    C[c] = X[labels == c].mean(0)
        # stable cluster ids: order by size
        order = np.argsort([-(labels == c).sum() for c in range(k)])
        remap = {int(old): new for new, old in enumerate(order)}
        return np.array([remap[int(l_)] for l_ in labels])

    @staticmethod
    def _pca3(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Xc @ vt[:3].T
        if comps.shape[1] < 3:
            comps = np.hstack([comps, np.zeros((len(comps), 3 - comps.shape[1]))])
        span = np.abs(comps).max(0)
        return comps / np.where(span == 0, 1, span)

    def _name_clusters(self) -> list[dict]:
        overall = {c: np.mean([c in {d["condition"] for d in p["diagnoses"]} for p in self.records]) for c in self.conditions}
        out = []
        for c in range(self.k):
            members = [p for p, l_ in zip(self.records, self.labels) if l_ == c]
            if not members:
                out.append({"id": c, "name": f"Cluster {c + 1}", "size": 0, "top_conditions": [], "mean_age": 0, "mean_risk": 0})
                continue
            prev = {cond: np.mean([cond in {d["condition"] for d in p["diagnoses"]} for p in members]) for cond in self.conditions}
            lift = sorted(self.conditions, key=lambda cond: -(prev[cond] - overall[cond]))
            top = [cond for cond in lift if prev[cond] > 0][:2]
            short = " + ".join(SHORT.get(t, t) for t in top) or "mixed"
            out.append({"id": c, "name": f"{short}", "size": len(members), "top_conditions": top,
                        "mean_age": round(float(np.mean([p["age"] for p in members])), 1),
                        "mean_risk": round(float(np.mean([clinical.risk_score(p)["score"] for p in members])), 1)})
        return out

    def map(self) -> dict:
        pts = [{"patient_id": pid, "name": p["name"], "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                "cluster": int(l_), "risk": clinical.risk_score(p)["score"], "age": p["age"]}
               for pid, p, c, l_ in zip(self.ids, self.records, self.coords, self.labels)]
        return {"points": pts, "clusters": self.cluster_names}
