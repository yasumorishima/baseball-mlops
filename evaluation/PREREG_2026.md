# PREREG 2026 — scoring the frozen 2026 predictions (DRAFT, not frozen)

Status: DRAFT. Fill the TBD items using 2025 rows only, then commit this file alone with its md5 in `evaluation/MD5SUMS_prereg.txt` **before any 2026 outcome is loaded**. After that, changes go only into `evaluation/AMENDMENTS.md`, and only before the run.

## Inputs (frozen, not to be regenerated)
- Set A: `master:predictions/batter_predictions.csv`, `pitcher_predictions.csv` (commit `4a6bae1`, 2026-03-16 UTC). md5 (checked 2026-10-23): batter `013c9ffa10ef321eee310fd781924676`, pitcher `a5c9445207e8a5338ba497573b593b91`.
- Set B: `develop@decbd23:predictions/batter_predictions.csv`, `pitcher_predictions.csv` (2026-03-21 UTC), copied into `evaluation/inputs/setB/` unchanged. md5 (checked 2026-10-23): batter `baf5c7d35d279223135e74b2e74ec596`, pitcher `c3636ec8ae03146b1507ea4b9fc899ae`.
- Both precede the 2026 regular season (2026-03-25 – 2026-09-27, statsapi `/seasons/2026`).

## Outcome
- Batters: 2026 wOBA; pitchers: 2026 xFIP, from HF `yasumorishima/mlb-stats` `statsapi_batting` (`woba`) / `statsapi_pitching` (`xfip`), `season == 2026`, `is_partial == False`, fetched after 2026-09-27.
- Input revision (40-hex HF commit, never `main`): TBD. Record here, from metadata only, that at this revision the 2026 rows have `is_partial == False` and `fetched_at` after 2026-09-27.
- Scale rule (decided from the 2025 comparison of FanGraphs `wOBA_last`/`xFIP_last` vs statsapi 2025 `woba`/`xfip`): TBD — identity, or a mapping fitted on 2025 only. Record bias / sd / correlation here.

## Population
- Players in the prediction file, matched to an MLBAM id via 2025 rows (match rule and unmatched count: TBD), with 2026 `plateAppearances >= 100` (batters) / `outs >= 90`, i.e. IP ≥ 30 (pitchers) — same filters as `src/train.py` training targets. `inningsPitched` is baseball notation (`42.1` = 42⅓) and is not used as a number. Traded players: season totals across teams.

## Statistics
- Primary (one per role): set A, paired difference MAE(`ensemble_*`) − MAE(`marcel_*`); 95% percentile bootstrap over players (10,000 resamples, seed fixed: TBD). "Beats Marcel" only if the upper bound < 0.
- Secondary (reported, no claim): every model column in sets A and B vs Marcel; floors = last season (`*_last`) and the 2026 population mean; 80% interval coverage of `ci_lo80/ci_hi80` (nominal 0.80); comparison with the 2025 holdout figures in README (different season, quoted not compared formally).
- No subgroup searching after the run.
