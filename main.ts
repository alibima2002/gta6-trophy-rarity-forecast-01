// gta6-trophy-rarity-forecast-01
//
// Forecast every GTA VI PlayStation 5 trophy's Exophase earn rate at the
// target a case names, from the house's reads of the trophy page up to the
// case checkpoint. Node 22, standard library only, no network, no clock,
// no randomness: the same input bytes always produce the same output bytes.
//
// The model, in one paragraph: a trophy's rate is a ratio, E(t) / N(t),
// where N(t) is the players Exophase tracks and E(t) the tracked players
// who have earned the trophy. The program rebuilds both counts from each
// read (E = rate x N), fits ln N and ln E as straight lines in
// ln(1 + hours since launch) over the last 48 hours of reads, extrapolates
// both to the target, and reports E / N. Splitting the ratio matters: new
// players dilute a rate while progressing players raise it, and the two
// pulls need their own slopes. The band around the median is a random
// walk (0.5 logit per root-day, as the house baseline has it) combined
// with the trend fit's residual and a floor, so the quantiles widen where
// the trend is unsure. When the history is one unchanged page — the
// launch-night practice cases — there is no trend to fit and the page is
// stale, so the median is diluted by a factor calibrated on the house's
// documented first refresh and on a live read of the page, and the band
// is widened to cover the level shift a refresh brings. The full method,
// calibration and validation are in METHOD.md.
//
// Usage: the grader runs `node main.ts` once per case with the case file
// in $MAKEFUN_INPUT_DIR (default /input) and reads the forecast from
// $MAKEFUN_OUTPUT_DIR (default /output). Every *.in file in the input
// directory gets one *.out file with the same stem.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Launch: the zero of the house curve baseline's clock, 2026-11-19T00:00:00Z.
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
// Random-walk volatility in logit units per square-root day (house baseline).
const RW_PER_ROOT_DAY = 0.5;
// Spread floor on a rich history: refresh risk the residuals may not show.
const RICH_SPREAD_FLOOR = 0.2;
// Extra spread when the history is one unchanged page: a refresh is due.
const STALE_PAGE_SHOCK = 1.0;
// How far back the rich-history fits look, in milliseconds.
const FIT_WINDOW_MS = 48 * 3600_000;
// Dilution prior for a flat history, calibrated on the house's documented
// first refresh (638 -> 3,868 players over the first ~4.5 h) and the live
// read at 2026-11-19T17:51Z (METHOD.md): D = 6 * (H / 1.5) ^ 0.226, capped.
const DILUTE_SCALE = 6;
const DILUTE_REF_HOURS = 1.5;
const DILUTE_POWER = 0.226;
const DILUTE_MAX = 20;

type TrophyRow = { id: number };
type PageRead = { at: string; players_tracked: number; rates: Array<number | null> };
type CaseFile = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyRow[];
  history: PageRead[];
};

/** Clamp a rate to the range the logit can represent (the page prints 2 dp). */
function clampRate(pct: number): number {
  return pct < 0.01 ? 0.01 : pct > 99.99 ? 99.99 : pct;
}

/** logit of a rate in percent. */
function toLogit(pct: number): number {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
}

/** Inverse logit, back to a rate in percent. */
function fromLogit(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

/** Round to 4 decimal places, as the house baseline does. */
function round4dp(x: number): number {
  return Math.round(x * 10_000) / 10_000;
}

/**
 * Inverse standard normal CDF by Acklam's rational approximation
 * (accurate to about 1e-9), so the program honours whatever levels a case
 * asks for instead of hard-coding the five z-values.
 */
function invNorm(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
    138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
    66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
    3.754408661907416];
  const tail = 0.02425;
  if (p < tail) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - tail) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Ordinary least squares for y = a + b*x; also the residual std. dev. */
function fitLine(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  const meanX = xs.reduce((s, v) => s + v, 0) / n;
  const meanY = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - meanX) * (xs[i] - meanX);
    sxy += (xs[i] - meanX) * (ys[i] - meanY);
  }
  if (sxx === 0) return [meanY, 0, 0];
  const slope = sxy / sxx;
  const intercept = meanY - slope * meanX;
  let resid = 0;
  if (n > 2) {
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (ys[i] - (intercept + slope * xs[i])) ** 2;
    resid = Math.sqrt(ss / (n - 2));
  }
  return [intercept, slope, resid];
}

/** Hours from launch to an ISO timestamp. */
function hoursAfterLaunch(iso: string): number {
  return (Date.parse(iso) - LAUNCH_MS) / 3600_000;
}

/** Forecast one case: { trophies: [{ id, q: [q05, q25, q50, q75, q95] }] }. */
function forecast(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  const zs = c.levels.map((t) => invNorm(t));
  const days = c.horizon_minutes / 1440;
  const rw = RW_PER_ROOT_DAY * Math.sqrt(days);
  const atMs = Date.parse(c.at);
  const targetMs = atMs + c.horizon_minutes * 60_000;
  const xTarget = Math.log(1 + (targetMs - LAUNCH_MS) / 3600_000);

  // No reads at all: nothing to forecast from. Emit a neutral, valid
  // forecast rather than crashing (a crash is an invalid case).
  if (c.history.length === 0) {
    const spread = 2;
    return {
      trophies: c.trophies.map((t) => ({
        id: t.id,
        q: zs.map((z) => round4dp(clampRate(fromLogit(toLogit(1) + z * spread)))),
      })),
    };
  }

  const last = c.history[c.history.length - 1];

  // The most recent non-null rate for trophy i (a read may lack a trophy).
  const lastRate = (i: number): number => {
    for (let k = c.history.length - 1; k >= 0; k--) {
      const v = c.history[k].rates[i];
      if (v !== null && v !== undefined) return v;
    }
    return 0;
  };

  // How many distinct pages does the history hold? One distinct state
  // means the page never refreshed during the case.
  const distinctPages = new Set(
    c.history.map((r) => `${r.players_tracked}|${r.rates.join(',')}`),
  ).size;

  if (distinctPages <= 1) {
    // Flat history: the counts carry no trend and the page is stale, so a
    // refresh is due. Dilute the median by the calibrated factor and widen
    // the band to cover the level shift a refresh brings.
    const horizonHours = c.horizon_minutes / 60;
    const dilute = Math.min(
      DILUTE_MAX,
      DILUTE_SCALE * (horizonHours / DILUTE_REF_HOURS) ** DILUTE_POWER,
    );
    const spread = Math.sqrt(rw * rw + STALE_PAGE_SHOCK * STALE_PAGE_SHOCK);
    return {
      trophies: c.trophies.map((t, i) => {
        const centre = toLogit(lastRate(i)) - Math.log(dilute);
        return {
          id: t.id,
          q: zs.map((z) => round4dp(clampRate(fromLogit(centre + z * spread)))),
        };
      }),
    };
  }

  // Rich history: two-count extrapolation over the last 48 hours of reads.
  const window = c.history.filter((r) => atMs - Date.parse(r.at) <= FIT_WINDOW_MS);
  const fit = window.length >= 3 ? window : c.history;

  // Fit ln N(t) linear in ln(1 + hours since launch).
  const xsN: number[] = [];
  const ysN: number[] = [];
  for (const r of fit) {
    xsN.push(Math.log(1 + hoursAfterLaunch(r.at)));
    ysN.push(Math.log(Math.max(r.players_tracked, 1)));
  }
  const [aN, bN] = fitLine(xsN, ysN);
  const nTarget = Math.max(Math.exp(aN + bN * xTarget), last.players_tracked);

  const trophies = c.trophies.map((t, i) => {
    const xs: number[] = [];
    const ysE: number[] = [];
    const ysL: number[] = [];
    for (const r of fit) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const earners = Math.max((v / 100) * r.players_tracked, 0.5);
      xs.push(Math.log(1 + hoursAfterLaunch(r.at)));
      ysE.push(Math.log(earners));
      ysL.push(toLogit(v));
    }
    const lastEarners = Math.max((lastRate(i) / 100) * last.players_tracked, 0.5);
    let median: number;
    let resid = 0;
    if (xs.length >= 3 && new Set(xs).size >= 2) {
      const [aE, bE] = fitLine(xs, ysE);
      const eTarget = Math.min(Math.max(Math.exp(aE + bE * xTarget), lastEarners), nTarget);
      median = (100 * eTarget) / nTarget;
      [, , resid] = fitLine(xs, ysL);
    } else {
      // Too little signal for this trophy: hold its last rate.
      median = lastRate(i);
    }
    const spread = Math.sqrt(rw * rw + resid * resid + RICH_SPREAD_FLOOR * RICH_SPREAD_FLOOR);
    const centre = toLogit(median);
    return {
      id: t.id,
      q: zs.map((z) => round4dp(clampRate(fromLogit(centre + z * spread)))),
    };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
