// gta6-trophy-rarity-forecast-01 — forecast each GTA VI PS5 trophy's Exophase
// earn rate at the case's target (Node 22, standard library only).
//
// Method (two-count model with a refresh-aware spread), documented in full in
// METHOD.md. In short, for each trophy the program reconstructs two counts
// from the case history — the players tracked N(t) and the players who earned
// the trophy E_i(t) = rate_i(t) * N(t) — fits ln N and ln E_i linear in
// ln(1 + hours since launch), extrapolates both to the target, and forecasts
// the rate as E_target / N_target. Separating the two counts matters because
// a trophy's rate moves for two opposite reasons: new players joining dilute
// it (N grows) while players progressing raise it (E grows). The spread is
// the random-walk term (0.5 logit per sqrt-day, as the house baseline) plus
// the residual of the trend fit plus a floor, so an honest quantile band
// widens where the trend is uncertain. When the history is a single
// unchanged page (no refresh observed yet — the launch-night practice
// cases), the counts carry no trend, so the median is diluted by a factor
// calibrated on the house's own documented first refresh (638 -> 3,868
// players) and on a live Exophase read (2026-11-19T17:51Z, see
// data/exophase-snapshot-20261119T1751Z.json), and the spread is widened:
// a stale page means a refresh is due, and a refresh is a level shift the
// history cannot show.
//
// The program reads every <id>.in in the input directory and writes one
// <id>.out per case to the output directory:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// It is deterministic (no randomness, no clock, no network) and runs in well
// under the 8-second cap.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Launch origin: the house curve baseline's zero, 2026-11-19T00:00:00Z. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk volatility, logit units per square-root day (house baseline). */
const RW_SIGMA = 0.5;
/** Spread floor for a rich history: refresh risk the residuals may miss. */
const RICH_FLOOR = 0.2;
/** Spread floor when the history is a single unchanged page (refresh due). */
const FLAT_SHOCK = 1.0;
/** Trend window for the rich history: the last 48 hours of reads. */
const WINDOW_MS = 48 * 3600_000;
/** Dilution prior for a flat history, calibrated on the documented first
 *  refresh (638 -> 3,868 players over the first ~4.5 h) and the live read at
 *  2026-11-19T17:51Z (see METHOD.md): D = 6 * (H / 1.5) ^ 0.226, capped. */
const DILUTE_BASE = 6;
const DILUTE_HOURS = 1.5;
const DILUTE_EXP = 0.226;
const DILUTE_CAP = 20;

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/**
 * Inverse standard normal CDF (Acklam's rational approximation, accurate to
 * ~1e-9). Lets the program honour whatever levels a case asks for.
 */
function normQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Least-squares line y = a + b*x; returns [a, b, residualStd]. */
function linfit(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx === 0) return [my, 0, 0];
  const b = sxy / sxx;
  const a = my - b * mx;
  let resid = 0;
  if (n > 2) {
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (ys[i] - (a + b * xs[i])) ** 2;
    resid = Math.sqrt(ss / (n - 2));
  }
  return [a, b, resid];
}

/** Hours since launch for an ISO timestamp. */
const hoursSinceLaunch = (iso: string): number =>
  (Date.parse(iso) - LAUNCH_MS) / 3600_000;

/** Forecast one case: {trophies: [{id, q: [q05, q25, q50, q75, q95]}]}. */
function forecast(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const zs = c.levels.map((t) => normQuantile(t));
  const days = c.horizon_minutes / 1440;
  const rw = RW_SIGMA * Math.sqrt(days);
  const atMs = Date.parse(c.at);
  const targetMs = atMs + c.horizon_minutes * 60_000;
  const xTarget = Math.log(1 + (targetMs - LAUNCH_MS) / 3600_000);

  // No history at all: no data to forecast from. Emit a neutral, valid
  // forecast rather than crashing (a crash is an invalid case).
  if (c.history.length === 0) {
    const spread = 2;
    return {
      trophies: c.trophies.map((t) => ({
        id: t.id,
        q: zs.map((z) => round4(clampPct(invLogit(logit(1) + z * spread)))),
      })),
    };
  }

  const last = c.history[c.history.length - 1];
  // Last non-null rate per trophy (a read may lack a trophy).
  const lastRate = (i: number): number => {
    for (let k = c.history.length - 1; k >= 0; k--) {
      const v = c.history[k].rates[i];
      if (v !== null && v !== undefined) return v;
    }
    return 0;
  };

  // Is the history a single unchanged page? (One distinct state.)
  const states = new Set(
    c.history.map((r) => `${r.players_tracked}|${r.rates.join(',')}`),
  );
  if (states.size <= 1) {
    // Flat history: the counts carry no trend, and the page is stale, so a
    // refresh is due. Dilute the median by the calibrated factor and widen
    // the spread to cover the level shift a refresh brings.
    const horizonHours = c.horizon_minutes / 60;
    const dilute = Math.min(
      DILUTE_CAP,
      DILUTE_BASE * (horizonHours / DILUTE_HOURS) ** DILUTE_EXP,
    );
    const spread = Math.sqrt(rw * rw + FLAT_SHOCK * FLAT_SHOCK);
    return {
      trophies: c.trophies.map((t, i) => {
        const l = logit(lastRate(i)) - Math.log(dilute);
        return {
          id: t.id,
          q: zs.map((z) => round4(clampPct(invLogit(l + z * spread)))),
        };
      }),
    };
  }

  // Rich history: two-count extrapolation over the last 48 hours of reads.
  const window = c.history.filter((r) => atMs - Date.parse(r.at) <= WINDOW_MS);
  const fit = window.length >= 3 ? window : c.history;

  // Fit ln N(t) linear in ln(1 + hours since launch).
  const xsN: number[] = [];
  const ysN: number[] = [];
  for (const r of fit) {
    xsN.push(Math.log(1 + hoursSinceLaunch(r.at)));
    ysN.push(Math.log(Math.max(r.players_tracked, 1)));
  }
  const [aN, bN] = linfit(xsN, ysN);
  const nTarget = Math.max(Math.exp(aN + bN * xTarget), last.players_tracked);

  const trophies = c.trophies.map((t, i) => {
    const xs: number[] = [];
    const ysE: number[] = [];
    const ysL: number[] = [];
    for (const r of fit) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const e = Math.max((v / 100) * r.players_tracked, 0.5);
      xs.push(Math.log(1 + hoursSinceLaunch(r.at)));
      ysE.push(Math.log(e));
      ysL.push(logit(v));
    }
    const lastE = Math.max((lastRate(i) / 100) * last.players_tracked, 0.5);
    let median: number;
    let resid = 0;
    if (xs.length >= 3 && new Set(xs).size >= 2) {
      const [aE, bE] = linfit(xs, ysE);
      const eTarget = Math.min(Math.max(Math.exp(aE + bE * xTarget), lastE), nTarget);
      median = (100 * eTarget) / nTarget;
      [, , resid] = linfit(xs, ysL);
    } else {
      // Too little signal for this trophy: persistence.
      median = lastRate(i);
    }
    const spread = Math.sqrt(rw * rw + resid * resid + RICH_FLOOR * RICH_FLOOR);
    const l = logit(median);
    return {
      id: t.id,
      q: zs.map((z) => round4(clampPct(invLogit(l + z * spread)))),
    };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
