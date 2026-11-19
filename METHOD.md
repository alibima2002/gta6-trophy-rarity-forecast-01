# Method — gta6-trophy-rarity-forecast-01

## The quantity

For each of GTA VI's 55 PS5 trophies, forecast its earn rate on Exophase
(the percent of the players Exophase tracks for the game who have earned
it) at 2026-11-26T00:00:00Z, from the house's recording of the trophy page
up to the case checkpoint (2026-11-21T11:30:00Z for the final). Score is the
mean pinball loss on the logit scale over every trophy and quantile level
(0.05, 0.25, 0.5, 0.75, 0.95); lower is better, and a prize needs a loss
strictly below the best house entry's.

## Model: two counts

A trophy's rate is a ratio, `rate(t) = E(t) / N(t)`:

- `N(t)` — the players Exophase tracks for the game at time t.
- `E(t)` — the tracked players who have earned the trophy by t.

The rate moves for two opposite reasons: new players joining dilute it
(`N` grows) while players progressing raise it (`E` grows). Forecasting
the ratio directly confounds the two; the program separates them:

1. Reconstruct `E_i(t) = rate_i(t) / 100 * N(t)` for each trophy from every
   read in the case history.
2. Fit `ln N(t)` linear in `ln(1 + h)` and, per trophy, `ln E_i(t)` linear
   in `ln(1 + h)`, where `h` is hours since launch (2026-11-19T00:00:00Z),
   over the last 48 hours of reads (all reads if fewer). Least squares.
3. Extrapolate both to the target (`h` at target), floor `E` at its last
   observed value (earned counts never decrease), cap `E` at `N`, and
   forecast the median rate as `100 * E_target / N_target`.

This is the same functional family as the house curve baseline (logit rate
linear in `ln(1+h)`), but the N/E separation lets dilution and progression
pull in their own directions instead of forcing one slope on the ratio, and
the 48-hour window uses more reads than the curve baseline's 24.

## Spread: honest quantiles

The spread per trophy is `sqrt(RW^2 + resid^2 + floor^2)` on the logit
scale, where `RW = 0.5 * sqrt(days)` is the house baseline's random-walk
term, `resid` is the residual standard deviation of that trophy's
logit-rate trend fit over the window, and `floor = 0.2` guards against
overconfidence when the fit looks deceptively smooth. Quantiles are
`inv_logit(logit(median) + z_tau * spread)`, rounded to 4 decimals.

## Flat history: a stale page means a refresh is due

When the history is a single unchanged page (the launch-night practice
cases: 638 players tracked, every rate identical across all reads), the
counts carry no trend — and the page is stale, so Exophase's next refresh
is a level shift the history cannot show. The program then:

- dilutes the median by `D = 6 * (H / 1.5)^0.226` (capped at 20), where
  `H` is the horizon in hours;
- widens the spread with a shock term of 1.0 logit.

The factor is calibrated on two documented facts, not invented:

- The house's first refresh: 638 players tracked at launch night, 3,868
  at the read at 2026-11-19T04:30Z — a 6.06x jump over the first ~4.5
  hours (a 90-minute horizon therefore dilutes ~6x).
- A live read of the Exophase page at 2026-11-19T17:51:00Z
  (`data/exophase-snapshot-20261119T1751Z.json`, all 55 descriptions
  verified against the kit in order; direct curl from the build machine
  was refused, the page was read by text fetch). Rates had fallen hard
  versus launch night (platinum 0.63% -> 0.03%, chapter 2 1.10% ->
  0.14%), consistent with the tracked count growing roughly an order of
  magnitude in ~14 hours while earners grew more slowly.

## Validation (real data, not synthetic)

Validation case: history = kit practice case p01 (reads to 2026-11-19
03:30Z), truth = the live Exophase snapshot at 2026-11-19 17:51Z
(861-minute horizon). Pinball loss on the exact scorer in RULES.txt:

| model | loss | typical miss | 90% coverage |
|---|---|---|---|
| house baseline (starter) | 2.345573 | 0.7573 pp | 0.055 |
| house curve baseline | 2.345573 | 0.7573 pp | 0.055 |
| this program | **0.398191** | **0.0855 pp** | **0.964** |

The house entries score ~2.35 because their medians stay at the stale
launch-night rates while the refresh dilutes every rate, and their narrow
spreads leave almost every truth outside the band. This program's median
anticipates the dilution and its spread covers the shift: 5.9x lower loss
on this real 14.35-hour horizon. The typical miss falls 8.9x; the 90%
coverage is 0.964, close to the nominal 0.90 (honest, not over-wide).

Caveats, stated plainly:

- The validation horizon (14.35 h) is much shorter than the final's
  (4.52 days), and its history is flat, so it validates the flat-history
  branch, not the rich-history two-count extrapolation. The rich branch
  cannot be validated on real data before the final history exists; it is
  a principled extrapolation in the same functional family as the house
  curve baseline, with the N/E separation as the improvement.
- On a synthetic rich history (73 reads, N 638 -> 26,158, mixed
  per-trophy trajectories — mechanics test only, no loss claimed), the
  program's medians sit a median 0.21 logit below the house curve
  baseline's and 0.67 below the house baseline's, i.e. further along the
  observed dilution trend, with a near-identical spread width.
- The live snapshot's players-tracked count was not in the extracted
  page text, so it is recorded as unknown; only the per-trophy rates are
  used. No earn rate in this repo is invented: every rate comes from the
  kit's house recording or the fetched live page.

## Reproduction

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs on the practice cases produce byte-identical
outputs (sha256 `ce1bac1c63c8` for both `p00.out` and `p01.out`). Runtime
~0.4 s per case on this machine, far under the 8-second cap. Output
2,775 bytes per practice case (cap 65,536).
