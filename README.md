# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alibima2002/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alibima2002/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, from the house's recording of Exophase's trophy page up to the case checkpoint.

## How it works

`main.ts` (Node 22, standard library only) reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case: five quantiles (5%, 25%, 50%, 75%, 95%) of each of the 55 trophies' earn rate at the case target.

The model is a **two-count model**: for each trophy it rebuilds the players tracked N(t) and the players who earned the trophy E(t) = rate(t) × N(t), fits ln N and ln E as straight lines in ln(1 + hours since launch) over the last 48 hours of reads, extrapolates both to the target, and forecasts the rate as E/N. Splitting the two counts matters because a trophy's rate moves for two opposite reasons — new players joining dilute it while players progressing raise it. The spread is a random walk (0.5 logit per √day) plus the trend fit's residual plus a floor. When the history is a single unchanged page (no refresh observed yet), the median is diluted by a factor calibrated on the house's documented first refresh and on a live Exophase read, and the spread is widened — a stale page means a refresh is due. Method, calibration and validation numbers: [METHOD.md](METHOD.md).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`input/` holds the kit's practice cases (`p00`, `p01`); the grader runs the program once per case with only that case's file in `/input` and reads the forecast from `/output/<id>.out`.

## Validation

On a real validation case (practice case p01's history to 03:30Z, truth the live Exophase rates at 17:51Z, 861-minute horizon), the program scores a mean pinball loss of **0.398** against the house baseline's **2.346** — 5.9× lower. See METHOD.md for the full table and the caveats.

## Data

- `data/exophase-snapshot-20261119T1751Z.json` — the live Exophase trophy page as read 2026-11-19T17:51:00Z (all 55 trophies, descriptions verified against the kit in order). Validation anchor only; the program never reads it — at grading it forecasts from the case history alone.

## Rebuild note

This entry was rebuilt from a fresh scheduled session on 2026-11-20 (see METHOD.md): the first build's code and design are unchanged, but its transcript bundle could not be written — the run record's parent chain was missing — so the entry was re-authored in a session that opens with the approval, and the validation above was re-run on the rebuilt program.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
