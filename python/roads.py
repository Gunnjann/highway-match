"""Road network: a highway with parallel service roads near two towns.

Coordinates are in metres on a local plane (x east, y north).
Each road is a polyline sampled every few metres, with cumulative distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HIGHWAY_LENGTH = 15_000.0     # metres
SERVICE_OFFSET = 22.0         # metres between highway centre line and service road centre line
SAMPLE_STEP = 5.0             # metres between polyline points

# Service roads exist only near towns (start, end along the highway, in metres)
SERVICE_SEGMENTS = {
    "service_a": (2_000.0, 5_600.0),   # Town A
    "service_b": (9_200.0, 12_600.0),  # Town B
}


def highway_xy(s: np.ndarray) -> np.ndarray:
    """Point on the highway centre line at parameter s (approximately arc length)."""
    x = s
    y = 380.0 * np.sin(s / 2_400.0) + 120.0 * np.sin(s / 900.0)
    return np.stack([x, y], axis=-1)


def highway_normal(s: np.ndarray) -> np.ndarray:
    """Unit normal pointing to the service-road side (left of travel direction)."""
    eps = 0.5
    d = highway_xy(s + eps) - highway_xy(s - eps)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.stack([-d[..., 1], d[..., 0]], axis=-1)


def position(s: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """True position given highway parameter s and lateral offset (0 = highway, SERVICE_OFFSET = service road)."""
    return highway_xy(s) + highway_normal(s) * np.asarray(offset)[..., None]


@dataclass
class Road:
    id: str
    kind: str                      # "highway" | "service"
    points: np.ndarray             # (n, 2)
    cum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def project(self, p: np.ndarray) -> tuple[float, float]:
        """Distance from point p to this road and the arc position of the projection."""
        a, b = self.points[:-1], self.points[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        seg_len = np.linalg.norm(ab[i])
        return float(d[i]), float(self.cum[i] + t[i] * seg_len)


def build_network() -> tuple[list[Road], list[np.ndarray]]:
    """Return the roads and the junction points (where vehicles can switch roads)."""
    s = np.arange(0.0, HIGHWAY_LENGTH + SAMPLE_STEP, SAMPLE_STEP)
    roads = [Road("highway", "highway", highway_xy(s))]
    junctions = []
    for rid, (a, b) in SERVICE_SEGMENTS.items():
        ss = np.arange(a, b + SAMPLE_STEP, SAMPLE_STEP)
        roads.append(Road(rid, "service", position(ss, np.full_like(ss, SERVICE_OFFSET))))
        junctions.append(highway_xy(np.array(a)))
        junctions.append(highway_xy(np.array(b)))
    return roads, junctions
