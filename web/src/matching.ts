// Browser port of python/matching.py. Same maths, so the demo reproduces the Python results.
export type Pt = [number, number];
export type Method = 'nearest' | 'hmm' | 'hmm_ml';

export interface Road {
  id: string;
  kind: 'highway' | 'service';
  points: Pt[];
}
export interface Network {
  roads: Road[];
  junctions: Pt[];
  service_offset: number;
  towns: Record<string, [number, number]>;
  toll_rate: number;
}
export interface Params {
  candidate_radius: number;
  sigma: number;
  beta: number;
  junction_radius: number;
  switch_logp: number;
  no_junction_switch_logp: number;
  ml_weight: number;
  window: number;
}
export interface Model {
  features: string[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
}

interface PreparedRoad extends Road {
  cum: number[];
}

export function prepareRoads(roads: Road[]): PreparedRoad[] {
  return roads.map((r) => {
    const cum = [0];
    for (let i = 1; i < r.points.length; i++) {
      const [ax, ay] = r.points[i - 1];
      const [bx, by] = r.points[i];
      cum.push(cum[i - 1] + Math.hypot(bx - ax, by - ay));
    }
    return { ...r, cum };
  });
}

/** Distance from p to the road, and arc position of the projection (first minimum, like numpy argmin). */
function project(road: PreparedRoad, p: Pt): [number, number] {
  let best = Infinity;
  let bestArc = 0;
  const pts = road.points;
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i];
    const abx = pts[i + 1][0] - ax;
    const aby = pts[i + 1][1] - ay;
    const len2 = abx * abx + aby * aby;
    let t = ((p[0] - ax) * abx + (p[1] - ay) * aby) / len2;
    t = Math.min(1, Math.max(0, t));
    const d = Math.hypot(ax + abx * t - p[0], ay + aby * t - p[1]);
    if (d < best) {
      best = d;
      bestArc = road.cum[i] + t * Math.sqrt(len2);
    }
  }
  return [best, bestArc];
}

export function projectAll(roads: PreparedRoad[], gps: Pt[]) {
  const dist = gps.map((p) => roads.map((r) => project(r, p)[0]));
  const arc = gps.map((p) => roads.map((r) => project(r, p)[1]));
  return { dist, arc };
}

function candidates(dist: number[][], radius: number): number[][] {
  return dist.map((row) => {
    const c = row.map((d, j) => (d <= radius ? j : -1)).filter((j) => j >= 0);
    if (c.length) return c;
    let k = 0;
    row.forEach((d, j) => { if (d < row[k]) k = j; });
    return [k];
  });
}

export function matchNearest(dist: number[][]): number[] {
  return dist.map((row) => {
    let k = 0;
    row.forEach((d, j) => { if (d < row[k]) k = j; });
    return k;
  });
}

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};

export function behaviourFeatures(gps: Pt[], t: number[], dist: number[][], roads: Road[], window: number): number[][] {
  const n = gps.length;
  const sp = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    sp[i] = Math.hypot(gps[i][0] - gps[i - 1][0], gps[i][1] - gps[i - 1][1]) / Math.max(t[i] - t[i - 1], 1e-6);
  }
  if (n > 1) sp[0] = sp[1];
  const hw = roads.findIndex((r) => r.kind === 'highway');
  const sv = roads.map((r, j) => (r.kind === 'service' ? j : -1)).filter((j) => j >= 0);
  const dHw = dist.map((row) => row[hw]);
  const dSv = dist.map((row) => (sv.length ? Math.min(...sv.map((j) => row[j])) : 99));
  const lean = dHw.map((d, i) => Math.min(40, Math.max(-40, d - dSv[i])));
  const feats: number[][] = [];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - window);
    const hi = Math.min(n, i + window + 1);
    const s = sp.slice(lo, hi);
    const l = lean.slice(lo, hi);
    feats.push([median(s), std(s), mean(s.map((x) => (x < 1.5 ? 1 : 0))), mean(l), std(l), mean(dHw.slice(lo, hi))]);
  }
  return feats;
}

export function serviceProbability(feats: number[][], m: Model): number[] {
  return feats.map((f) => {
    let z = m.intercept;
    f.forEach((x, k) => { z += ((x - m.mean[k]) / m.scale[k]) * m.coef[k]; });
    return 1 / (1 + Math.exp(-z));
  });
}

export function matchHmm(
  gps: Pt[], dist: number[][], arc: number[][], roads: Road[], junctions: Pt[], P: Params, pService?: number[],
): number[] {
  const n = gps.length;
  const cands = candidates(dist, P.candidate_radius);
  const nearJ = gps.map((p) => Math.min(...junctions.map((j) => Math.hypot(p[0] - j[0], p[1] - j[1]))) <= P.junction_radius);

  const emission = (i: number, j: number) => {
    let lp = -0.5 * (dist[i][j] / P.sigma) ** 2;
    if (pService && cands[i].length > 1) {
      const p = Math.min(1 - 1e-4, Math.max(1e-4, pService[i]));
      lp += P.ml_weight * Math.log(roads[j].kind === 'service' ? p : 1 - p);
    }
    return lp;
  };

  let score = cands[0].map((j) => emission(0, j));
  const back: number[][] = [cands[0].map(() => 0)];
  for (let i = 1; i < n; i++) {
    const gap = Math.hypot(gps[i][0] - gps[i - 1][0], gps[i][1] - gps[i - 1][1]);
    const prev = cands[i - 1];
    const cur = cands[i];
    const sc = cur.map(() => -Infinity);
    const bp = cur.map(() => 0);
    cur.forEach((j, b) => {
      const e = emission(i, j);
      prev.forEach((k, a) => {
        const trans = k === j
          ? -Math.abs(Math.abs(arc[i][j] - arc[i - 1][k]) - gap) / P.beta
          : (nearJ[i] || nearJ[i - 1] ? P.switch_logp : P.no_junction_switch_logp);
        const v = score[a] + trans + e;
        if (v > sc[b]) { sc[b] = v; bp[b] = a; }
      });
    });
    score = sc;
    back.push(bp);
  }
  const path = new Array(n).fill(0);
  let k = score.indexOf(Math.max(...score));
  for (let i = n - 1; i >= 0; i--) {
    path[i] = cands[i][k];
    k = back[i][k];
  }
  return path;
}

/** Tolled km: for each continuous highway run, distance between its first and last fix. */
export function highwayKm(isHighway: boolean[], pos: number[]): number {
  let km = 0;
  let i = 0;
  while (i < isHighway.length) {
    if (!isHighway[i]) { i++; continue; }
    let j = i;
    while (j + 1 < isHighway.length && isHighway[j + 1]) j++;
    km += Math.abs(pos[j] - pos[i]);
    i = j + 1;
  }
  return km / 1000;
}

export interface MatchResult {
  roadIds: Record<Method, string[]>;
  pService: number[];
  highwayArc: number[];
}

export function runAll(net: Network, prepared: PreparedRoad[], P: Params, model: Model, gps: Pt[], t: number[]): MatchResult {
  const { dist, arc } = projectAll(prepared, gps);
  const feats = behaviourFeatures(gps, t, dist, net.roads, P.window);
  const pService = serviceProbability(feats, model);
  const ids = (path: number[]) => path.map((j) => net.roads[j].id);
  const hw = net.roads.findIndex((r) => r.kind === 'highway');
  return {
    roadIds: {
      nearest: ids(matchNearest(dist)),
      hmm: ids(matchHmm(gps, dist, arc, net.roads, net.junctions, P)),
      hmm_ml: ids(matchHmm(gps, dist, arc, net.roads, net.junctions, P, pService)),
    },
    pService,
    highwayArc: arc.map((row) => row[hw]),
  };
}
