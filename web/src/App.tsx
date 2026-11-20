import { useEffect, useMemo, useRef, useState } from 'react';
import MapView from './MapView';
import { highwayKm, prepareRoads, runAll, type Method, type Model, type Network, type Params } from './matching';
import { METHODS, SCENARIO_TITLES, type Metrics, type Noise, type Trip } from './types';

interface Data { network: Network; params: Params; model: Model; trips: Trip[]; metrics: Metrics }

async function loadData(): Promise<Data> {
  const get = (f: string) => fetch(`/data/${f}`).then((r) => r.json());
  const [network, m, trips, metrics] = await Promise.all([get('network.json'), get('model.json'), get('trips.json'), get('metrics.json')]);
  return { network, params: m.params, model: m.model, trips, metrics };
}

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const NOISES: { id: Noise; label: string }[] = [
  { id: 'low', label: 'Low' },
  { id: 'normal', label: 'Normal' },
  { id: 'high', label: 'High' },
];

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState('');
  useEffect(() => { loadData().then(setData).catch(() => setError('Could not load the demo data.')); }, []);

  if (error) return <div className="loading">{error}</div>;
  if (!data) return <div className="loading"><span className="spinner" /> Loading road network and trips…</div>;
  return <Demo data={data} />;
}

function Demo({ data }: { data: Data }) {
  const { network, params, model, trips, metrics } = data;
  const prepared = useMemo(() => prepareRoads(network.roads), [network]);
  const scenarios = useMemo(() => [...new Set(trips.map((t) => t.scenario))], [trips]);

  const [scenario, setScenario] = useState(scenarios[1] ?? scenarios[0]);
  const [noise, setNoise] = useState<Noise>('normal');
  const [method, setMethod] = useState<Method>('hmm_ml');
  const [showTruth, setShowTruth] = useState(false);
  const [showErrors, setShowErrors] = useState(true);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [follow, setFollow] = useState(false);
  const [fit, setFit] = useState<{ key: string; box: [number, number, number, number] }>({ key: 'init', box: [0, -600, 15000, 600] });

  const trip = trips.find((t) => t.scenario === scenario && t.noise === noise)!;

  // Run all three matchers in the browser for the selected trip
  const result = useMemo(() => runAll(network, prepared, params, model, trip.gps, trip.t), [trip, network, prepared, params, model]);

  const stats = useMemo(() => {
    const out = {} as Record<Method, { acc: number; km: number; switches: number }>;
    for (const m of METHODS) {
      const pred = result.roadIds[m.id].map((r) => r.startsWith('service'));
      const correct = pred.filter((p, i) => (p ? 'service' : 'highway') === trip.label[i]).length;
      out[m.id] = {
        acc: (100 * correct) / pred.length,
        km: highwayKm(pred.map((p) => !p), result.highwayArc),
        switches: pred.slice(1).filter((p, i) => p !== pred[i]).length,
      };
    }
    return out;
  }, [result, trip]);
  const trueSwitches = trip.label.slice(1).filter((l, i) => l !== trip.label[i]).length;

  const tripBox = useMemo<[number, number, number, number]>(() => {
    const xs = trip.gps.map((p) => p[0]);
    const ys = trip.gps.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }, [trip]);

  useEffect(() => {
    setCursor(trip.t.length - 1);
    setPlaying(false);
    setFollow(false);
    setFit({ key: trip.id, box: tripBox });
  }, [trip.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Playback: one GNSS fix every 70 ms (2 s of real time)
  const timer = useRef<number>();
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setCursor((c) => {
        if (c >= trip.t.length - 1) { setPlaying(false); return c; }
        return c + 1;
      });
    }, 70);
    return () => window.clearInterval(timer.current);
  }, [playing, trip.t.length]);

  function play() {
    if (cursor >= trip.t.length - 1) setCursor(0);
    setFollow(true);
    setPlaying((p) => !p);
  }

  function focusTown(id: string) {
    const road = network.roads.find((r) => r.id === id)!;
    const xs = road.points.map((p) => p[0]);
    const ys = road.points.map((p) => p[1]);
    setFollow(false);
    setPlaying(false);
    setFit({ key: `${id}-${Date.now()}`, box: [Math.min(...xs), Math.min(...ys) - 40, Math.max(...xs), Math.max(...ys) + 40] });
  }

  const s = stats[method];
  const kmErr = s.km - trip.true_km;
  const rate = network.toll_rate;
  const predHere = result.roadIds[method][cursor] ?? '';
  const truthHere = trip.label[cursor];
  const lvl = metrics.levels.normal;

  return (
    <div className="app">
      {/* ---------------- HERO ---------------- */}
      <header className="hero">
        <nav className="topbar">
          <div className="brand">
            <span className="brand-mark"><i /><i /></span>
            <strong>Highway Match</strong>
            <span className="pill">SIH 2024 problem · working prototype</span>
          </div>
          <div className="links">
            <a href="#demo">Demo</a>
            <a href="#results">Results</a>
            <a href="#how">How it works</a>
            <a className="gh" href="https://github.com/Gunnjann/highway-match" target="_blank" rel="noreferrer">GitHub ↗</a>
          </div>
        </nav>

        <div className="hero-body">
          <div className="hero-copy">
            <p className="eyebrow">GNSS map-matching for fair highway tolling</p>
            <h1>Is this vehicle on the highway, or on the service road <em>20 metres away</em>?</h1>
            <p className="lede">
              GPS error is about the same size as the gap between the two roads, so snapping each fix to the nearest road gets
              it wrong often enough to bill local traffic. Highway Match uses a Hidden Markov Model and a behaviour model to decide
              per fix, then works out the tolled kilometres.
            </p>
            <div className="hero-cta">
              <a href="#demo" className="btn primary">Try the demo</a>
              <a href="#results" className="btn ghost">See the results</a>
            </div>
          </div>

          <div className="hero-stats">
            <p className="stats-caption">{metrics.test_trips_per_level} unseen test trips, normal GPS noise</p>
            <Stat label="Accuracy inside towns" from={`${lvl.nearest.accuracy_town}%`} to={`${lvl.hmm_ml.accuracy_town}%`} />
            <Stat label="Toll distance error" from={`${lvl.nearest.toll_error_pct}%`} to={`${lvl.hmm_ml.toll_error_pct}%`} good />
            <Stat label="False road switches per trip" from={`${lvl.nearest.false_switches}`} to={`${lvl.hmm_ml.false_switches}`} good />
            <p className="stats-foot">Nearest road → HMM + ML</p>
          </div>
        </div>
      </header>

      {/* ---------------- DEMO ---------------- */}
      <section id="demo" className="section">
        <div className="section-head">
          <div>
            <p className="eyebrow">Interactive demo</p>
            <h2>Pick a trip and watch each method decide</h2>
          </div>
          <div className="seg-wrap">
            <span className="seg-label">GPS noise</span>
            <div className="segmented">
              {NOISES.map((n) => (
                <button key={n.id} className={noise === n.id ? 'on' : ''} onClick={() => setNoise(n.id)}>{n.label}</button>
              ))}
            </div>
          </div>
        </div>

        <div className="demo">
          <aside className="trips">
            {scenarios.map((sc, i) => {
              const t = trips.find((x) => x.scenario === sc && x.noise === noise)!;
              return (
                <button key={sc} className={`trip ${sc === scenario ? 'active' : ''}`} onClick={() => setScenario(sc)} style={{ ['--i' as string]: i }}>
                  <span className="trip-title">{SCENARIO_TITLES[sc] ?? sc}</span>
                  <span className="trip-desc">{t.description}</span>
                  <span className="trip-meta">{t.vehicle[0].toUpperCase() + t.vehicle.slice(1)} · {fmtTime(t.t[t.t.length - 1] - t.t[0])} · {t.true_km.toFixed(1)} km tolled</span>
                </button>
              );
            })}
          </aside>

          <div className="stage">
            <div className="toolbar">
              <div className="segmented methods">
                {METHODS.map((m) => (
                  <button key={m.id} className={method === m.id ? 'on' : ''} onClick={() => setMethod(m.id)} title={m.blurb}>
                    {m.short}
                    <span className="acc">{stats[m.id].acc.toFixed(1)}%</span>
                  </button>
                ))}
              </div>
              <div className="toggles">
                <Toggle on={showErrors} set={setShowErrors} label="Mark errors" />
                <Toggle on={showTruth} set={setShowTruth} label="True path" />
              </div>
            </div>

            <div className="map-card">
              <MapView
                network={network}
                gps={trip.gps}
                truth={trip.truth}
                predicted={result.roadIds[method]}
                labels={trip.label}
                showTruth={showTruth}
                showErrors={showErrors}
                cursor={cursor}
                fitBox={fit.box}
                fitKey={fit.key}
                follow={follow}
              />
              <div className="map-legend">
                <span><i className="dot hw" /> Matched to highway</span>
                <span><i className="dot svc" /> Matched to service road</span>
                {showErrors && <span><i className="dot err" /> Wrong</span>}
                <span><i className="diamond" /> Junction</span>
              </div>
              <div className="map-jump">
                <button onClick={() => { setFollow(false); setFit({ key: `trip-${Date.now()}`, box: tripBox }); }}>Whole trip</button>
                <button onClick={() => focusTown('service_a')}>Town A</button>
                <button onClick={() => focusTown('service_b')}>Town B</button>
              </div>
            </div>

            <div className="player">
              <button className="play" onClick={play} aria-label={playing ? 'Pause' : 'Play'}>
                {playing ? <svg viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                  : <svg viewBox="0 0 24 24"><path d="M8 5.5v13a1 1 0 0 0 1.5.9l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5z" /></svg>}
              </button>
              <div className="track">
                <div className="lanes" aria-hidden="true">
                  {result.roadIds[method].map((r, i) => (
                    <i key={i} className={`${r.startsWith('service') ? 'svc' : 'hw'} ${(r.startsWith('service') ? 'service' : 'highway') !== trip.label[i] ? 'bad' : ''}`} />
                  ))}
                </div>
                <input type="range" min={0} max={trip.t.length - 1} value={cursor} onChange={(e) => { setPlaying(false); setFollow(false); setCursor(+e.target.value); }} aria-label="Timeline" />
              </div>
              <span className="time mono">{fmtTime(trip.t[cursor] - trip.t[0])}</span>
              <Toggle on={follow} set={setFollow} label="Follow" />
            </div>
          </div>

          <aside className="side">
            <div className="card toll">
              <p className="card-label">Toll for this trip</p>
              <div className="toll-row">
                <div><span>Truth</span><strong>{trip.true_km.toFixed(2)} km</strong><small>₹{(trip.true_km * rate).toFixed(2)}</small></div>
                <div className="arrow">→</div>
                <div><span>{METHODS.find((m) => m.id === method)!.short}</span><strong>{s.km.toFixed(2)} km</strong><small>₹{(s.km * rate).toFixed(2)}</small></div>
              </div>
              <div className={`delta ${Math.abs(kmErr) < 0.15 ? 'ok' : kmErr > 0 ? 'over' : 'under'}`}>
                {Math.abs(kmErr) < 0.15 ? 'Billed correctly (within 150 m)' : kmErr > 0 ? `Overcharged by ${kmErr.toFixed(2)} km` : `Undercharged by ${Math.abs(kmErr).toFixed(2)} km`}
              </div>
              <p className="fine">Example rate ₹{rate}/km</p>
            </div>

            <div className="card">
              <p className="card-label">This trip, each method</p>
              {METHODS.map((m) => (
                <div key={m.id} className={`bar-row ${method === m.id ? 'sel' : ''}`} onClick={() => setMethod(m.id)}>
                  <span className="bar-name">{m.short}</span>
                  <div className="bar"><i style={{ width: `${Math.max(0, (stats[m.id].acc - 70) / 30) * 100}%` }} /></div>
                  <span className="bar-val mono">{stats[m.id].acc.toFixed(1)}%</span>
                </div>
              ))}
              <p className="fine">Road switches: true {trueSwitches} · {METHODS.map((m) => `${m.short} ${stats[m.id].switches}`).join(' · ')}</p>
            </div>

            <div className="card now">
              <p className="card-label">At {fmtTime(trip.t[cursor] - trip.t[0])}</p>
              <dl>
                <dt>Speed</dt><dd className="mono">{trip.speed[cursor]?.toFixed(0)} km/h</dd>
                <dt>True road</dt><dd><Road kind={truthHere} /></dd>
                <dt>Matched</dt><dd><Road kind={predHere.startsWith('service') ? 'service' : 'highway'} wrong={(predHere.startsWith('service') ? 'service' : 'highway') !== truthHere} /></dd>
                <dt>ML: P(service)</dt><dd className="mono">{result.pService[cursor] !== undefined ? `${(result.pService[cursor] * 100).toFixed(0)}%` : '–'}</dd>
              </dl>
            </div>
          </aside>
        </div>
      </section>

      {/* ---------------- RESULTS ---------------- */}
      <Results metrics={metrics} />

      {/* ---------------- HOW ---------------- */}
      <section id="how" className="section">
        <div className="section-head"><div><p className="eyebrow">How it works</p><h2>Three ideas, each fixing the last one's weakness</h2></div></div>
        <div className="how">
          <Step n="1" title="Nearest road" text="Project each fix onto every road and pick the closest. Fast and simple, but GPS error of 6 to 15 m on a 22 m gap makes it flip between roads dozens of times per trip." />
          <Step n="2" title="Hidden Markov Model" text="Treat the true road as a hidden state. Emission: how far the fix is from the road. Transition: route distance should match fix-to-fix distance, and road changes are only plausible near junctions. Viterbi picks the most likely sequence." />
          <Step n="3" title="Behaviour model (ML)" text="A logistic regression reads a 20-second window of driving: median speed, speed variation, share of time stopped, and which road the fixes lean toward. Its probability is added to the HMM's evidence." />
          <div className="step limits">
            <span className="step-n">!</span>
            <h3>Honest limits</h3>
            <ul>
              <li>Trained and tested on <b>simulated</b> GPS. Real traces are the next step.</li>
              <li>At high noise the ML adds little: slow trucks and traffic jams look like service-road driving.</li>
              <li>One highway with two service roads; real networks have many more junctions.</li>
            </ul>
          </div>
        </div>
      </section>

      <footer className="footer">
        <p>Built by <a href="https://www.linkedin.com/in/gunjan-ukande-432538254/" target="_blank" rel="noreferrer">Gunjan Ukande</a> · Smart India Hackathon 2024 problem statement, prototype completed after the hackathon · Python, scikit-learn, React, TypeScript</p>
        <p className="fine">All trips and GPS traces are simulated. Matching runs in your browser and reproduces the Python results exactly.</p>
      </footer>
    </div>
  );
}

function Results({ metrics }: { metrics: Metrics }) {
  const [metric, setMetric] = useState<'accuracy_town' | 'toll_error_pct'>('accuracy_town');
  const levels: Noise[] = ['low', 'normal', 'high'];
  const max = metric === 'accuracy_town' ? 100 : Math.max(...levels.flatMap((l) => METHODS.map((m) => metrics.levels[l][m.id].toll_error_pct)));
  const min = metric === 'accuracy_town' ? 60 : 0;
  const better = metric === 'accuracy_town' ? 'Higher is better' : 'Lower is better';
  return (
    <section id="results" className="section">
      <div className="section-head">
        <div>
          <p className="eyebrow">Results</p>
          <h2>{metrics.test_trips_per_level * 3} unseen test trips across three noise levels</h2>
        </div>
        <div className="segmented">
          <button className={metric === 'accuracy_town' ? 'on' : ''} onClick={() => setMetric('accuracy_town')}>Accuracy in towns</button>
          <button className={metric === 'toll_error_pct' ? 'on' : ''} onClick={() => setMetric('toll_error_pct')}>Toll error</button>
        </div>
      </div>

      <div className="results">
        <div className="card chart-card">
          <div className="chart">
            {levels.map((l) => (
              <div key={l} className="group">
                <div className="bars">
                  {METHODS.map((m, i) => {
                    const v = metrics.levels[l][m.id][metric];
                    return (
                      <div key={m.id} className={`col m-${m.id}`} style={{ ['--h' as string]: `${((v - min) / (max - min)) * 100}%`, ['--i' as string]: i }}>
                        <span className="col-val mono">{v}%</span>
                      </div>
                    );
                  })}
                </div>
                <span className="group-label">{l[0].toUpperCase() + l.slice(1)} noise</span>
              </div>
            ))}
          </div>
          <div className="chart-legend">
            {METHODS.map((m) => <span key={m.id}><i className={`sw m-${m.id}`} />{m.label}</span>)}
            <span className="fine">{better}{metric === 'accuracy_town' ? ' · axis starts at 60%' : ''}</span>
          </div>
        </div>

        <div className="card table-card">
          <table>
            <thead>
              <tr><th>Normal noise</th>{METHODS.map((m) => <th key={m.id}>{m.short}</th>)}</tr>
            </thead>
            <tbody>
              {([
                ['Accuracy, all fixes', 'accuracy', '%'],
                ['Accuracy, town zones', 'accuracy_town', '%'],
                ['Toll distance error', 'toll_error_pct', '%'],
                ['False switches / trip', 'false_switches', ''],
              ] as const).map(([label, key, unit]) => (
                <tr key={key}>
                  <td>{label}</td>
                  {METHODS.map((m) => <td key={m.id} className="mono">{metrics.levels.normal[m.id][key]}{unit}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="insights">
            <li><b>The HMM does most of the work.</b> Remembering where the vehicle was cuts false road switches from {metrics.levels.normal.nearest.false_switches} to {metrics.levels.normal.hmm.false_switches} per trip.</li>
            <li><b>ML helps at normal noise</b>, taking town accuracy from {metrics.levels.normal.hmm.accuracy_town}% to {metrics.levels.normal.hmm_ml.accuracy_town}% (behaviour model AUC {metrics.model_auc}).</li>
            <li><b>At high noise it barely helps</b>: {metrics.levels.high.hmm.accuracy_town}% → {metrics.levels.high.hmm_ml.accuracy_town}%. Real traces would show whether richer features close the gap.</li>
          </ul>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, from, to, good }: { label: string; from: string; to: string; good?: boolean }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <div className="stat-vals">
        <span className="from">{from}</span>
        <span className="to-arrow">→</span>
        <strong className={good ? 'down' : 'up'}>{to}</strong>
      </div>
    </div>
  );
}

function Toggle({ on, set, label }: { on: boolean; set: (v: boolean) => void; label: string }) {
  return (
    <label className={`toggle ${on ? 'on' : ''}`}>
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />
      <span className="knob" />
      {label}
    </label>
  );
}

function Road({ kind, wrong }: { kind: string; wrong?: boolean }) {
  return <span className={`road-chip ${kind === 'service' ? 'svc' : 'hw'} ${wrong ? 'wrong' : ''}`}>{kind === 'service' ? 'Service road' : 'Highway'}{wrong ? ' ✕' : ''}</span>;
}

function Step({ n, title, text }: { n: string; title: string; text: string }) {
  return (
    <div className="step">
      <span className="step-n">{n}</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}
