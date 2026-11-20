import type { Method, Pt } from './matching';

export type Noise = 'low' | 'normal' | 'high';

export interface Trip {
  id: string;
  scenario: string;
  description: string;
  vehicle: string;
  noise: Noise;
  t: number[];
  gps: Pt[];
  truth: Pt[];
  label: ('highway' | 'service')[];
  speed: number[];
  true_km: number;
}

export interface MethodMetrics {
  accuracy: number;
  accuracy_town: number;
  toll_km_error: number;
  toll_error_pct: number;
  false_switches: number;
}

export interface Metrics {
  levels: Record<Noise, Record<Method, MethodMetrics>>;
  test_trips_per_level: number;
  model_auc: number;
  coefficients: Record<string, number>;
}

export const METHODS: { id: Method; label: string; short: string; blurb: string }[] = [
  { id: 'nearest', label: 'Nearest road', short: 'Nearest', blurb: 'Snaps each fix to the closest road. No memory.' },
  { id: 'hmm', label: 'HMM', short: 'HMM', blurb: 'Vehicles move smoothly and change road only at junctions.' },
  { id: 'hmm_ml', label: 'HMM + ML', short: 'HMM + ML', blurb: 'Adds a model that reads speed, stops and lean.' },
];

export const SCENARIO_TITLES: Record<string, string> = {
  'through-truck': 'Through truck',
  'town-delivery': 'Town delivery',
  'local-van': 'Local van',
  'slow-highway-truck': 'Slow truck in a jam',
  'both-towns': 'Car through both towns',
  'fast-service-car': 'Fast car on service road',
};
