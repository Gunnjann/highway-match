"""Three ways to decide which road each GNSS fix belongs to.

1. Nearest road  : snap every fix to the closest road. No memory.
2. HMM           : Hidden Markov Model + Viterbi (Newson & Krumm, 2009). Vehicles move
                   smoothly along a road and change road only at junctions.
3. HMM + ML      : the same HMM, with a logistic-regression model that reads driving
                   behaviour (speed, stops, which road the fixes lean toward) added to
                   the evidence for each fix.
"""
from __future__ import annotations

import numpy as np

from roads import Road

PARAMS = {
    "candidate_radius": 60.0,    # metres: roads further than this are not candidates
    "sigma": 8.0,                # metres: expected GNSS error, for the emission probability
    "beta": 12.0,                # metres: tolerance between route distance and fix-to-fix distance
    "junction_radius": 160.0,    # metres: road switches are plausible only this close to a junction
    "switch_logp": float(np.log(0.15)),
    "no_junction_switch_logp": -40.0,
    "ml_weight": 1.5,
    "window": 5,                 # fixes on each side used for behaviour features
}

FEATURES = ["speed", "speed_std", "stop_frac", "lean", "lean_std", "dist_hw"]


def project_all(roads: list[Road], gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance and arc position of every fix on every road: arrays of shape (n_fixes, n_roads)."""
    n = len(gps)
    dist = np.zeros((n, len(roads)))
    arc = np.zeros((n, len(roads)))
    for j, r in enumerate(roads):
        for i in range(n):
            dist[i, j], arc[i, j] = r.project(gps[i:i + 1])
    return dist, arc


def candidates(dist: np.ndarray) -> list[list[int]]:
    out = []
    for row in dist:
        c = [j for j, d in enumerate(row) if d <= PARAMS["candidate_radius"]]
        out.append(c or [int(np.argmin(row))])
    return out


# ---------------------------------------------------------------- 1. nearest road
def match_nearest(dist: np.ndarray) -> np.ndarray:
    return np.argmin(dist, axis=1)


# ---------------------------------------------------------------- behaviour features
def behaviour_features(gps: np.ndarray, t: np.ndarray, dist: np.ndarray, roads: list[Road]) -> np.ndarray:
    """Per-fix features from a window of neighbouring fixes."""
    n = len(gps)
    w = PARAMS["window"]
    step = np.linalg.norm(np.diff(gps, axis=0), axis=1)
    dt = np.maximum(np.diff(t), 1e-6)
    sp = np.concatenate([[0.0], step / dt])
    sp[0] = sp[1] if n > 1 else 0.0

    hw = [j for j, r in enumerate(roads) if r.kind == "highway"][0]
    sv = [j for j, r in enumerate(roads) if r.kind == "service"]
    d_hw = dist[:, hw]
    d_sv = dist[:, sv].min(axis=1) if sv else np.full(n, 99.0)
    lean = np.clip(d_hw - d_sv, -40, 40)          # > 0: fixes sit closer to the service road

    feats = np.zeros((n, len(FEATURES)))
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        s = np.median(sp[lo:hi])
        feats[i] = [
            s,
            np.std(sp[lo:hi]),
            np.mean(sp[lo:hi] < 1.5),
            np.mean(lean[lo:hi]),
            np.std(lean[lo:hi]),
            np.mean(d_hw[lo:hi]),
        ]
    return feats


def service_probability(feats: np.ndarray, model: dict) -> np.ndarray:
    z = (feats - np.array(model["mean"])) / np.array(model["scale"])
    logit = z @ np.array(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------- 2 & 3. HMM + Viterbi
def match_hmm(
    gps: np.ndarray,
    dist: np.ndarray,
    arc: np.ndarray,
    roads: list[Road],
    junctions: list[np.ndarray],
    p_service: np.ndarray | None = None,
) -> np.ndarray:
    n = len(gps)
    cands = candidates(dist)
    J = np.array(junctions)
    near_junction = np.min(np.linalg.norm(gps[:, None, :] - J[None, :, :], axis=2), axis=1) <= PARAMS["junction_radius"]
    sigma, beta = PARAMS["sigma"], PARAMS["beta"]

    def emission(i: int, j: int) -> float:
        lp = -0.5 * (dist[i, j] / sigma) ** 2
        if p_service is not None and len(cands[i]) > 1:
            p = np.clip(p_service[i], 1e-4, 1 - 1e-4)
            lp += PARAMS["ml_weight"] * np.log(p if roads[j].kind == "service" else 1 - p)
        return lp

    score = [np.array([emission(0, j) for j in cands[0]])]
    back = [np.zeros(len(cands[0]), int)]
    for i in range(1, n):
        gap = np.linalg.norm(gps[i] - gps[i - 1])
        prev_c, cur_c = cands[i - 1], cands[i]
        sc = np.full(len(cur_c), -np.inf)
        bp = np.zeros(len(cur_c), int)
        for b, j in enumerate(cur_c):
            e = emission(i, j)
            for a, k in enumerate(prev_c):
                if k == j:
                    trans = -abs(abs(arc[i, j] - arc[i - 1, k]) - gap) / beta
                else:
                    ok = near_junction[i] or near_junction[i - 1]
                    trans = PARAMS["switch_logp"] if ok else PARAMS["no_junction_switch_logp"]
                v = score[-1][a] + trans + e
                if v > sc[b]:
                    sc[b], bp[b] = v, a
        score.append(sc)
        back.append(bp)

    path = np.zeros(n, int)
    k = int(np.argmax(score[-1]))
    for i in range(n - 1, -1, -1):
        path[i] = cands[i][k]
        k = back[i][k]
    return path
