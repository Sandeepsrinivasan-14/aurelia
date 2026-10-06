"""Lab-value forecasting and anomaly detection (pure NumPy, fully explainable).

* ``forecast``: ordinary least squares on (date, value) with a 95% *prediction*
  interval that widens with few points and with distance from the data. With
  n=3 the interval is deliberately huge - that is the honest answer.
* ``anomaly``: robust z-score (median/MAD) of the newest value against the
  patient's own history; flags jumps without being fooled by one old outlier.

These are statistical extrapolations, not clinical predictions.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def _series(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted(points, key=lambda p: p["date"])
    x = np.array([date.fromisoformat(p["date"]).toordinal() for p in pts], dtype=float)
    y = np.array([p["value"] for p in pts], dtype=float)
    return x, y


def forecast(points: list[dict], horizon_days: int = 90) -> dict | None:
    if len(points) < 4:   # with 3 points the 95% interval is meaningless (1 degree of freedom)
        return None
    x, y = _series(points)
    n = len(x)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0:
        return None
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    dof = n - 2
    sigma = (ss_res / dof) ** 0.5
    x0 = x[-1] + horizon_days
    pred = intercept + slope * x0
    se = sigma * (1 + 1 / n + (x0 - x.mean()) ** 2 / sxx) ** 0.5
    t = _T95.get(dof, 1.96)
    target = date.fromordinal(int(x[-1])) + timedelta(days=horizon_days)
    return {"predicted": round(float(pred), 2), "low": round(float(pred - t * se), 2), "high": round(float(pred + t * se), 2),
            "slope_per_30d": round(slope * 30, 3), "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else 0.0, "n": n,
            "horizon_days": horizon_days, "target_date": target.isoformat()}


def anomaly(points: list[dict], threshold: float = 3.5) -> dict | None:
    if len(points) < 4:
        return None
    _, y = _series(points)
    hist, last = y[:-1], float(y[-1])
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med))) * 1.4826
    if mad == 0:
        mad = float(hist.std()) or 1e-9
    z = (last - med) / mad
    return {"z": round(z, 2), "median": round(med, 2), "flag": abs(z) > threshold, "last": last}
