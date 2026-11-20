"""Full pipeline: simulate trips, train the behaviour model, evaluate all three matchers,
and export everything the web demo needs.

    python train.py            # ~1-2 minutes
"""
from __future__ import annotations

import json
import zlib
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from matching import FEATURES, PARAMS, behaviour_features, match_hmm, match_nearest, project_all, service_probability
from roads import SERVICE_OFFSET, SERVICE_SEGMENTS, build_network
from simulate import NOISE_LEVELS, random_plan, run_trip, scenario_plans

OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "data"
TOLL_RATE = 2.5   # ₹ per km, example rate for the demo

roads, junctions = build_network()
for _r in roads:                      # use exactly the coordinates the web demo receives
    _r.points = np.round(_r.points, 2)
    _r.__post_init__()
road_ids = [r.id for r in roads]
HW = road_ids.index("highway")


def prepare(trip: dict) -> dict:
    dist, arc = project_all(roads, trip["gps"])
    trip["dist"], trip["arc"] = dist, arc
    trip["y"] = (trip["label"] == "service").astype(int)
    trip["feats"] = behaviour_features(trip["gps"], trip["t"], dist, roads)
    trip["ambiguous"] = np.zeros(len(trip["t"]), bool)
    for a, b in SERVICE_SEGMENTS.values():
        trip["ambiguous"] |= (trip["s"] >= a) & (trip["s"] <= b)
    return trip


def highway_km(is_highway: np.ndarray, pos: np.ndarray) -> float:
    """Tolled distance: for each continuous run on the highway, distance from where it started to where it ended.
    Using run end points (not fix-to-fix steps) stops GNSS jitter from adding phantom kilometres."""
    km, i, n = 0.0, 0, len(is_highway)
    while i < n:
        if not is_highway[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_highway[j + 1]:
            j += 1
        km += abs(pos[j] - pos[i])
        i = j + 1
    return float(km / 1000)


def evaluate(trip: dict, model: dict) -> dict:
    preds = {
        "nearest": match_nearest(trip["dist"]),
        "hmm": match_hmm(trip["gps"], trip["dist"], trip["arc"], roads, junctions),
        "hmm_ml": match_hmm(trip["gps"], trip["dist"], trip["arc"], roads, junctions,
                            service_probability(trip["feats"], model)),
    }
    true_service = trip["y"] == 1
    true_km = highway_km(~true_service, trip["s"])
    out = {"true_km": true_km, "preds": preds, "methods": {}}
    for m, p in preds.items():
        pred_service = np.array([roads[j].kind == "service" for j in p])
        correct = pred_service == true_service
        amb = trip["ambiguous"]
        out["methods"][m] = {
            "acc": float(correct.mean()),
            "acc_amb": float(correct[amb].mean()) if amb.any() else None,
            "km": highway_km(~pred_service, trip["arc"][:, HW]),
            "switches": int(np.sum(np.diff(pred_service.astype(int)) != 0)),
        }
    out["true_switches"] = int(np.sum(np.diff(true_service.astype(int)) != 0))
    return out


def main() -> None:
    t0 = time.time()
    levels = list(NOISE_LEVELS)

    # ---------------- training data
    rng = np.random.default_rng(7)
    train = [prepare(run_trip(random_plan(rng, i), levels[i % 3], rng)) for i in range(150)]
    X = np.vstack([tr["feats"][tr["ambiguous"]] for tr in train])
    y = np.concatenate([tr["y"][tr["ambiguous"]] for tr in train])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    model = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    print(f"Trained on {len(train)} trips, {len(y)} fixes in town zones ({y.mean():.0%} service road)  [{time.time()-t0:.0f}s]")

    # ---------------- evaluation on unseen trips
    rng = np.random.default_rng(2024)
    metrics = {"levels": {}, "test_trips_per_level": 60}
    all_p, all_y = [], []
    for level in levels:
        rows = []
        for i in range(60):
            trip = prepare(run_trip(random_plan(rng, 1000 + i), level, rng))
            rows.append(evaluate(trip, model))
            amb = trip["ambiguous"]
            all_p.append(service_probability(trip["feats"][amb], model))
            all_y.append(trip["y"][amb])
        lvl = {}
        for m in ["nearest", "hmm", "hmm_ml"]:
            acc = np.mean([r["methods"][m]["acc"] for r in rows])
            acc_amb = np.mean([r["methods"][m]["acc_amb"] for r in rows if r["methods"][m]["acc_amb"] is not None])
            km_err = np.mean([abs(r["methods"][m]["km"] - r["true_km"]) for r in rows])
            km_err_pct = 100 * np.sum([abs(r["methods"][m]["km"] - r["true_km"]) for r in rows]) / np.sum([r["true_km"] for r in rows])
            extra_sw = np.mean([max(0, r["methods"][m]["switches"] - r["true_switches"]) for r in rows])
            lvl[m] = {
                "accuracy": round(float(acc) * 100, 1),
                "accuracy_town": round(float(acc_amb) * 100, 1),
                "toll_km_error": round(float(km_err), 3),
                "toll_error_pct": round(float(km_err_pct), 1),
                "false_switches": round(float(extra_sw), 1),
            }
        metrics["levels"][level] = lvl
        print(f"\n[{level} noise]")
        for m, v in lvl.items():
            print(f"  {m:8s} acc {v['accuracy']:5.1f}%  town {v['accuracy_town']:5.1f}%  toll err {v['toll_error_pct']:4.1f}%  extra switches {v['false_switches']}")
    metrics["model_auc"] = round(float(roc_auc_score(np.concatenate(all_y), np.concatenate(all_p))), 3)
    metrics["coefficients"] = dict(zip(FEATURES, [round(c, 3) for c in model["coef"]]))
    print(f"\nBehaviour model AUC on unseen town fixes: {metrics['model_auc']}")

    # ---------------- demo scenarios
    trips_out = []
    for plan in scenario_plans():
        for level in levels:
            raw = run_trip(plan, level, np.random.default_rng(zlib.crc32(f"{plan.name}-{level}".encode())))
            raw["gps"] = np.round(raw["gps"], 1)          # same precision as the exported JSON
            trip = prepare(raw)
            ev = evaluate(trip, model)
            trips_out.append({
                "id": f"{plan.name}-{level}",
                "scenario": plan.name,
                "description": plan.description,
                "vehicle": plan.vehicle,
                "noise": level,
                "t": np.round(trip["t"], 1).tolist(),
                "gps": np.round(trip["gps"], 1).tolist(),
                "truth": np.round(trip["true_xy"], 1).tolist(),
                "label": trip["label"].tolist(),
                "speed": np.round(trip["speed"] * 3.6, 1).tolist(),
                "true_km": round(ev["true_km"], 3),
                "python": {m: [road_ids[j] for j in p] for m, p in ev["preds"].items()},
                "python_stats": ev["methods"],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    network = {
        "roads": [{"id": r.id, "kind": r.kind, "points": np.round(r.points, 2).tolist()} for r in roads],
        "junctions": np.round(np.array(junctions), 2).tolist(),
        "service_offset": SERVICE_OFFSET,
        "towns": {k: v for k, v in SERVICE_SEGMENTS.items()},
        "toll_rate": TOLL_RATE,
    }
    (OUT / "network.json").write_text(json.dumps(network))
    (OUT / "model.json").write_text(json.dumps({"params": PARAMS, "model": model}))
    (OUT / "trips.json").write_text(json.dumps(trips_out))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nExported to {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
