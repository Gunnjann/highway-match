import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { Network, Pt } from './matching';

// World coordinates are metres with y pointing north. The SVG works in (x, -y) so text renders upright.
type Box = [number, number, number, number]; // minX, minY, maxX, maxY in world metres
interface View { x: number; y: number; w: number; h: number }

interface Props {
  network: Network;
  gps: Pt[];
  truth: Pt[];
  predicted: string[];
  labels: string[];
  showTruth: boolean;
  showErrors: boolean;
  cursor: number;
  fitBox: Box;
  fitKey: string;
  follow: boolean;
}

const flip = (p: Pt): Pt => [p[0], -p[1]];

export default function MapView({ network, gps, truth, predicted, labels, showTruth, showErrors, cursor, fitBox, fitKey, follow }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 480 });
  const [view, setView] = useState<View>({ x: 0, y: -600, w: 15000, h: 900 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit to a box whenever fitKey changes (new trip, "Town A" button…)
  useEffect(() => {
    const [x0, y0, x1, y1] = fitBox;
    const pad = 0.08;
    let w = (x1 - x0) * (1 + pad * 2);
    let h = (y1 - y0) * (1 + pad * 2);
    const aspect = size.w / size.h;
    if (w / h > aspect) h = w / aspect; else w = h * aspect;
    const cx = (x0 + x1) / 2;
    const cy = -(y0 + y1) / 2;
    setView({ x: cx - w / 2, y: cy - h / 2, w, h });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, size.w, size.h]);

  // Follow the vehicle during playback
  useEffect(() => {
    if (!follow || !gps[cursor]) return;
    const [px, py] = flip(gps[cursor]);
    setView((v) => {
      const w = v.w > 2500 ? 700 : v.w;
      const h = w * (size.h / size.w);
      return { x: px - w / 2, y: py - h / 2, w, h };
    });
  }, [cursor, follow, gps, size.w, size.h]);

  const mpp = view.w / size.w; // metres per pixel

  const roadPaths = useMemo(
    () => network.roads.map((r) => ({ ...r, d: 'M' + r.points.map((p) => `${p[0].toFixed(1)},${(-p[1]).toFixed(1)}`).join('L') })),
    [network]
  );
  const truthPath = useMemo(() => 'M' + truth.map((p) => `${p[0]},${-p[1]}`).join('L'), [truth]);

  function onWheel(e: React.WheelEvent) {
    const rect = wrap.current!.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const k = Math.exp(e.deltaY * 0.0015);
    setView((v) => {
      const w = Math.min(22000, Math.max(50, v.w * k));
      const h = w * (v.h / v.w);
      return { x: v.x + (v.w - w) * fx, y: v.y + (v.h - h) * fy, w, h };
    });
  }
  function zoom(k: number) {
    setView((v) => {
      const w = Math.min(22000, Math.max(50, v.w * k));
      const h = w * (v.h / v.w);
      return { x: v.x + (v.w - w) / 2, y: v.y + (v.h - h) / 2, w, h };
    });
  }

  const scale = useMemo(() => {
    const target = mpp * 110;
    const nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const m = nice.find((n) => n >= target) ?? 5000;
    return { m, px: m / mpp };
  }, [mpp]);

  const r = 3.6 * mpp;
  const current = gps[cursor];
  const currentTruth = truth[cursor];

  return (
    <div
      ref={wrap}
      className={`map ${dragging ? 'is-dragging' : ''}`}
      onWheel={onWheel}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
        setDragging(true);
      }}
      onPointerMove={(e) => {
        if (!drag.current) return;
        const d = drag.current;
        setView((v) => ({ ...v, x: d.vx - (e.clientX - d.x) * mpp, y: d.vy - (e.clientY - d.y) * mpp }));
      }}
      onPointerUp={() => { drag.current = null; setDragging(false); }}
    >
      <svg width={size.w} height={size.h} viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} role="img" aria-label="Map of the highway, service roads and GPS fixes">
        {/* roads */}
        {roadPaths.map((rd) => (
          <g key={rd.id}>
            <path d={rd.d} className={`road-edge ${rd.kind}`} strokeWidth={Math.max(rd.kind === 'highway' ? 16 : 9, (rd.kind === 'highway' ? 7 : 5) * mpp)} />
            <path d={rd.d} className={`road ${rd.kind}`} strokeWidth={Math.max(rd.kind === 'highway' ? 13 : 7, (rd.kind === 'highway' ? 5 : 3.5) * mpp)} />
            {rd.kind === 'highway' && <path d={rd.d} className="lane" strokeWidth={Math.max(0.6, 0.8 * mpp)} strokeDasharray={`${6 * Math.max(1, mpp)} ${8 * Math.max(1, mpp)}`} />}
          </g>
        ))}
        {network.junctions.map((j, i) => {
          const [x, y] = flip(j);
          const s = 6 * mpp;
          return <rect key={i} className="junction" x={x - s / 2} y={y - s / 2} width={s} height={s} transform={`rotate(45 ${x} ${y})`} />;
        })}

        {showTruth && <path d={truthPath} className="truth" strokeWidth={1.6 * mpp} strokeDasharray={`${5 * mpp} ${4 * mpp}`} />}

        {/* GPS fixes coloured by predicted road */}
        {gps.map((p, i) => {
          const svc = predicted[i]?.startsWith('service');
          const wrong = showErrors && (svc ? 'service' : 'highway') !== labels[i];
          return (
            <g key={i} className={i > cursor ? 'future' : ''}>
              {wrong && <circle cx={p[0]} cy={-p[1]} r={r * 2.1} className="err-ring" strokeWidth={1.6 * mpp} />}
              <circle cx={p[0]} cy={-p[1]} r={r} className={svc ? 'fix svc' : 'fix hw'} strokeWidth={0.9 * mpp} />
            </g>
          );
        })}

        {current && (
          <g>
            {currentTruth && <circle cx={currentTruth[0]} cy={-currentTruth[1]} r={r * 1.1} className="truth-dot" />}
            <circle cx={current[0]} cy={-current[1]} r={r * 3.4} className="pulse" />
            <circle cx={current[0]} cy={-current[1]} r={r * 1.7} className="vehicle" strokeWidth={2 * mpp} />
          </g>
        )}

        {Object.entries(network.towns).map(([id, [a, b]]) => {
          const road = network.roads.find((x) => x.id === id);
          if (!road) return null;
          const mid = road.points[Math.floor(road.points.length / 2)];
          return (
            <text key={id} x={mid[0]} y={-mid[1] - 22 * mpp} className="town-label" fontSize={12 * mpp} textAnchor="middle">
              {id === 'service_a' ? 'Town A' : 'Town B'} service road · {((b - a) / 1000).toFixed(1)} km
            </text>
          );
        })}
      </svg>

      <div className="map-controls" onPointerDown={(e) => e.stopPropagation()}>
        <button onClick={() => zoom(0.6)} aria-label="Zoom in">+</button>
        <button onClick={() => zoom(1.6)} aria-label="Zoom out">−</button>
      </div>
      <div className="scalebar" style={{ width: scale.px }}>
        <span>{scale.m >= 1000 ? `${scale.m / 1000} km` : `${scale.m} m`}</span>
      </div>
      <div className="map-hint">Scroll to zoom · drag to pan</div>
    </div>
  );
}
