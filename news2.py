"""NEWS2 (National Early Warning Score 2), vectorized.

Scoring follows the Royal College of Physicians NEWS2 chart (2017).
Inputs may be pandas Series or numpy arrays. Missing values score 0.
"""
import numpy as np
import pandas as pd


def _bins(x, edges, points):
    """Score x by upper-inclusive edges: points[i] applies when x <= edges[i]; last point otherwise."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, points[-1], dtype=float)
    assigned = np.zeros(x.shape, dtype=bool)
    for e, p in zip(edges, points[:-1]):
        m = (~assigned) & (x <= e)
        out[m] = p
        assigned |= m
    out[np.isnan(x)] = 0
    return out


def score_rr(rr):
    return _bins(np.round(rr), [8, 11, 20, 24], [3, 1, 0, 2, 3])


def score_spo2(spo2, scale=1, on_oxygen=0):
    s = np.round(np.asarray(spo2, dtype=float))
    scale = np.broadcast_to(np.asarray(scale), s.shape)
    o2 = np.broadcast_to(np.asarray(on_oxygen).astype(bool), s.shape)
    s1 = _bins(s, [91, 93, 95], [3, 2, 1, 0])
    # Scale 2 (hypercapnic respiratory failure, target 88-92%)
    s2 = _bins(s, [83, 85, 87, 92], [3, 2, 1, 0, 0])
    hi = s >= 93
    s2 = np.where(hi & ~o2, 0, s2)
    s2 = np.where(hi & o2 & (s <= 94), 1, s2)
    s2 = np.where(hi & o2 & (s >= 95) & (s <= 96), 2, s2)
    s2 = np.where(hi & o2 & (s >= 97), 3, s2)
    out = np.where(scale == 2, s2, s1)
    out[np.isnan(s)] = 0
    return out


def score_air(on_oxygen):
    return np.where(np.asarray(on_oxygen).astype(bool), 2, 0).astype(float)


def score_sbp(sbp):
    return _bins(np.round(sbp), [90, 100, 110, 219], [3, 2, 1, 0, 3])


def score_hr(hr):
    return _bins(np.round(hr), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])


def score_consciousness(c):
    c = pd.Series(c).fillna("A").astype(str).str.upper().str[0].to_numpy()
    return np.where(c == "A", 0, 3).astype(float)


def score_temp(t):
    return _bins(np.round(np.asarray(t, dtype=float), 1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])


def news2_components(rr, spo2, scale, on_oxygen, sbp, hr, consciousness, temp):
    return pd.DataFrame({
        "rr": score_rr(rr),
        "spo2": score_spo2(spo2, scale, on_oxygen),
        "air": score_air(on_oxygen),
        "sbp": score_sbp(sbp),
        "hr": score_hr(hr),
        "consciousness": score_consciousness(consciousness),
        "temp": score_temp(temp),
    })


def news2_total(components):
    return components.sum(axis=1).to_numpy()


def news2_band(components):
    """Map to the four labels the hackathon dataset uses.

    0 = Normal, 1-4 = Low, 5-6 or any single parameter scoring 3 = Medium, 7+ = High.
    """
    total = components.sum(axis=1).to_numpy()
    red = (components.to_numpy() >= 3).any(axis=1)
    band = np.where(total == 0, "Normal", "Low")
    band = np.where((total >= 5) | red, "Medium", band)
    band = np.where(total >= 7, "High", band)
    return band


# Clinical response per NEWS2 guidance, used by the app and slides
RESPONSE = {
    "Normal": "Routine monitoring (at least every 12 hours).",
    "Low": "Nurse assessment; monitor every 4-6 hours.",
    "Medium": "Urgent review by a clinician; monitor at least hourly.",
    "High": "Emergency response team; continuous monitoring.",
}
