"""Simulated GNSS (GPS) traces for vehicles on the highway and service roads.

Real GNSS error is not just random scatter. It has:
  - white noise (a few metres, every fix),
  - a slowly drifting bias (multipath, atmosphere), so errors are correlated in time,
  - occasional outliers (reflections off buildings and trucks),
  - gaps (dropouts, overpasses),
  - larger error inside towns (urban canyon).
The simulator models all five, because a matcher that only survives white noise is not useful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roads import HIGHWAY_LENGTH, SERVICE_OFFSET, SERVICE_SEGMENTS, position

DT = 2.0                 # seconds between GNSS fixes
RAMP_LENGTH = 120.0      # metres of travel to move between highway and service road
OVERPASS = (7_400.0, 7_650.0)   # GNSS gap: vehicle passes under a flyover

NOISE_LEVELS = {
    #          white σ, bias σ, outlier rate, town multiplier
    "low":    (3.5, 2.0, 0.01, 1.2),
    "normal": (6.0, 4.0, 0.02, 1.4),
    "high":   (9.0, 6.0, 0.04, 1.6),
}

VEHICLES = {
    #                highway km/h, service km/h
    "car":          ((80, 100), (30, 50)),
    "truck":        ((55, 80), (25, 40)),
    "slow truck":   ((38, 52), (22, 32)),
}


@dataclass
class Plan:
    name: str
    s_start: float
    s_end: float
    service_use: dict            # service road id -> (enter_s, exit_s); vehicle is on it between these
    vehicle: str
    stops: list                  # list of (s, dwell seconds) on service roads
    jam: tuple | None = None     # (s_from, s_to, km/h) slow traffic on the highway
    description: str = ""


def offset_at(s: float, plan: Plan) -> float:
    """Lateral offset of the true path at highway parameter s (0 highway, SERVICE_OFFSET service)."""
    # Vehicles join and leave a service road only at its junctions (its two ends).
    # If the trip starts or ends inside a town, the vehicle is already on / stays on the service road.
    for rid, (enter, exit_) in plan.service_use.items():
        if not (enter <= s <= exit_):
            continue
        ramps_in = plan.s_start < enter       # came from the highway
        ramps_out = plan.s_end > exit_        # goes back to the highway
        frac = 1.0
        if ramps_in and s < enter + RAMP_LENGTH:
            frac = min(frac, (s - enter) / RAMP_LENGTH)
        if ramps_out and s > exit_ - RAMP_LENGTH:
            frac = min(frac, (exit_ - s) / RAMP_LENGTH)
        return SERVICE_OFFSET * max(0.0, frac)
    return 0.0


def simulate_truth(plan: Plan, rng: np.random.Generator):
    """Second-by-second true motion. Returns arrays t, s, offset, speed."""
    v_hw = rng.uniform(*VEHICLES[plan.vehicle][0]) / 3.6
    v_sv = rng.uniform(*VEHICLES[plan.vehicle][1]) / 3.6
    stops = sorted(plan.stops)
    s, v, t = plan.s_start, 0.0 if plan.service_use else v_hw * 0.8, 0.0
    wait = 0.0
    jitter = 0.0
    T, S, O, V = [], [], [], []
    h = 1.0
    while s < plan.s_end and t < 4 * 3600:
        off = offset_at(s, plan)
        on_service = off > SERVICE_OFFSET / 2
        target = v_sv if on_service else v_hw
        if 0 < off < SERVICE_OFFSET:              # on a ramp
            target = min(target, 35 / 3.6)
        if plan.jam and plan.jam[0] <= s <= plan.jam[1]:
            target = plan.jam[2] / 3.6
        jitter = 0.97 * jitter + rng.normal(0, 0.25)
        target = max(0.0, target * (1 + 0.06 * jitter))
        if stops:
            gap = stops[0][0] - s
            if gap < v * v / (2 * 1.2) + 3:
                target = 0.0
                if v < 0.3 and gap < 8:
                    wait += h
                    if wait >= stops[0][1]:
                        stops.pop(0)
                        wait = 0.0
        acc = np.clip(target - v, -1.8, 1.2)
        v = max(0.0, v + acc * h)
        s += v * h
        t += h
        T.append(t); S.append(s); O.append(offset_at(s, plan)); V.append(v)
    return np.array(T), np.array(S), np.array(O), np.array(V)


def add_gnss_noise(xy: np.ndarray, s: np.ndarray, level: str, rng: np.random.Generator):
    """Apply realistic GNSS error. Returns noisy xy and a mask of kept fixes."""
    white, bias_sigma, outlier_rate, town_mult = NOISE_LEVELS[level]
    n = len(xy)
    in_town = np.zeros(n, bool)
    for a, b in SERVICE_SEGMENTS.values():
        in_town |= (s >= a) & (s <= b)
    mult = np.where(in_town, town_mult, 1.0)

    rho = 0.96
    bias = np.zeros((n, 2))
    b = rng.normal(0, bias_sigma, 2)
    for i in range(n):
        b = rho * b + rng.normal(0, bias_sigma * np.sqrt(1 - rho**2), 2)
        bias[i] = b
    noise = rng.normal(0, white, (n, 2)) * mult[:, None] + bias * mult[:, None]
    outliers = rng.random(n) < outlier_rate
    ang = rng.uniform(0, 2 * np.pi, n)
    mag = rng.uniform(25, 60, n)
    noise[outliers] += np.stack([np.cos(ang), np.sin(ang)], 1)[outliers] * mag[outliers, None]

    keep = rng.random(n) > 0.03
    keep &= ~((s >= OVERPASS[0]) & (s <= OVERPASS[1]))
    return xy + noise, keep


def run_trip(plan: Plan, level: str, rng: np.random.Generator) -> dict:
    t, s, off, v = simulate_truth(plan, rng)
    idx = np.arange(0, len(t), int(DT))          # one fix every DT seconds
    t, s, off, v = t[idx], s[idx], off[idx], v[idx]
    true_xy = position(s, off)
    gps, keep = add_gnss_noise(true_xy, s, level, rng)
    on_service = off > SERVICE_OFFSET / 2
    labels = np.where(on_service, "service", "highway")
    return {
        "name": plan.name,
        "description": plan.description,
        "vehicle": plan.vehicle,
        "noise": level,
        "t": t[keep],
        "s": s[keep],
        "gps": gps[keep],
        "true_xy": true_xy[keep],
        "speed": v[keep],
        "label": labels[keep],
    }


def random_plan(rng: np.random.Generator, i: int) -> Plan:
    """A random trip: varied start/end, vehicle type, service-road use, stops and jams."""
    vehicle = rng.choice(list(VEHICLES), p=[0.35, 0.45, 0.20])
    s_start = rng.uniform(0, HIGHWAY_LENGTH - 4_000)
    s_end = rng.uniform(s_start + 3_500, HIGHWAY_LENGTH)
    service_use, stops = {}, []
    for rid, (a, b) in SERVICE_SEGMENTS.items():
        if s_end < a or s_start > b:
            continue
        if rng.random() < 0.55:
            enter = max(a, s_start)
            exit_ = min(b, s_end)
            if exit_ - enter < 400:
                continue
            service_use[rid] = (enter, exit_)
            for _ in range(rng.integers(0, 4)):
                stops.append((rng.uniform(enter + 150, exit_ - 150), rng.uniform(15, 120)))
    jam = None
    if rng.random() < 0.2:
        j = rng.uniform(s_start, s_end - 600)
        jam = (j, j + rng.uniform(300, 900), rng.uniform(8, 25))
    return Plan(f"random-{i}", s_start, s_end, service_use, str(vehicle), stops, jam)


def scenario_plans() -> list[Plan]:
    """Hand-made trips shown in the web demo, each testing one hard case."""
    A, B = SERVICE_SEGMENTS["service_a"], SERVICE_SEGMENTS["service_b"]
    return [
        Plan("through-truck", 0, 15_000, {}, "truck", [],
             description="Truck stays on the highway end to end. Every km should be tolled."),
        Plan("town-delivery", 0, 15_000, {"service_a": A}, "truck", [(3_100, 70), (4_600, 45)],
             description="Exits to Town A's service road for two deliveries, then rejoins the highway."),
        Plan("local-van", A[0] + 100, A[1] - 100, {"service_a": (A[0] + 100, A[1] - 100)}, "car",
             [(2_900, 40), (3_900, 60), (5_000, 30)],
             description="Local van that never leaves the service road. It should pay no toll."),
        Plan("slow-highway-truck", 1_000, 14_000, {}, "slow truck", [], jam=(9_800, 10_600, 12),
             description="Slow, loaded truck that also hits a traffic jam near Town B: it looks like service-road traffic."),
        Plan("both-towns", 0, 15_000, {"service_a": A, "service_b": B}, "car", [(10_400, 50)],
             description="Car uses both service roads, with one stop in Town B."),
        Plan("fast-service-car", 8_000, 15_000, {"service_b": B}, "car", [],
             description="Car drives fast on Town B's service road with no stops: it looks like highway traffic."),
    ]
