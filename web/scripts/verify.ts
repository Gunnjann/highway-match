// Checks that the browser matcher gives the same answer as the Python pipeline on every demo trip.
import { readFileSync } from 'node:fs';
import { prepareRoads, runAll, type Method, type Network, type Params, type Model, type Pt } from '../src/matching';

const dir = new URL('../public/data/', import.meta.url);
const net: Network = JSON.parse(readFileSync(new URL('network.json', dir), 'utf8'));
const { params, model }: { params: Params; model: Model } = JSON.parse(readFileSync(new URL('model.json', dir), 'utf8'));
const trips = JSON.parse(readFileSync(new URL('trips.json', dir), 'utf8'));
const prepared = prepareRoads(net.roads);

let total = 0;
let diff = 0;
for (const trip of trips) {
  const res = runAll(net, prepared, params, model, trip.gps as Pt[], trip.t);
  for (const m of ['nearest', 'hmm', 'hmm_ml'] as Method[]) {
    const py: string[] = trip.python[m];
    const d = py.filter((r, i) => r !== res.roadIds[m][i]).length;
    total += py.length;
    diff += d;
    if (d) console.log(`  ${trip.id} ${m}: ${d} fixes differ`);
  }
}
console.log(`Checked ${trips.length} trips × 3 methods: ${diff} of ${total} fixes differ from Python.`);
process.exit(diff === 0 ? 0 : 1);
