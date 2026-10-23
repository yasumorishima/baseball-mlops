# CLAUDE.md — baseball-mlops

Working rules for Claude Code (cloud or local). This file is the source of truth for this project; the owner's local notes are not visible to cloud sessions.

## Session protocol (cloud)
This project is worked on in Claude Code cloud. The owner starts a session with just 「続き」/「つづきから」.
1. Read this file first, then do the first unfinished item in "Next tasks".
2. Before ending, record what you did and what comes next **in this file** (update "Next tasks" and add a dated line to "Progress log"). This file is the only memory that survives between cloud sessions.
3. Cloud can push only its session branch: open a PR to `master` and merge it yourself (own repo) so the next session sees the update. Report the merged commit.
4. Write replies to the owner in Japanese (です/ます). Do not claim anything you did not run in this session. Numbers in this file that you did not re-measure are quoted, not verified.

## Progress log
- 2026-10-23: CLAUDE.md added for cloud work (written from the owner's PC, read-only inspection). State found: training pipeline idle since 2026-04; the frozen 2026 predictions in `predictions/` have never been scored against the 2026 season (no scoring file exists in the repo).

## What this is
Next-season prediction of MLB batter **wOBA** and pitcher **xFIP** from Statcast + FanGraphs features (LightGBM, CatBoost, ElasticNet "Bayes" on the Marcel residual, PECOTA-style components, inverse-MAE ensemble), compared with the Marcel method. Streamlit app (production, `master`): https://baseball-mlops.streamlit.app/ — it reads the CSVs in `predictions/` directly and does not call W&B; it also tries `GET {API_URL}/model/info` (3 s timeout, `streamlit/app.py` `get_model_info`) and shows nothing when that fails, which is always the case now because the API is not deployed.

Current state (checked 2026-10-23):
- **Retraining is retired.** `weekly_retrain.yml` / `gcp_deploy.yml` / BigQuery / BQML / Cloud Run were removed on 2026-04-19 (PR #9). The training scripts in `src/` remain but `src/fetch_statcast.py` pulls FanGraphs through pybaseball, and FanGraphs returns 403 to GitHub-hosted runners (measured 2026-09-23 in `mlb-data-pipeline`). `backtest.yml` therefore is expected to fail at its fetch step (not run since 2026-03). HF `fg_*` tables are frozen at seasons 2015–2025.
- **API (`api/main.py`) is not deployed anywhere.** It is the only runtime piece that loads models from the W&B registry.
- `spring_update.yml` is disabled (manually), `codeql.yml` is disabled for inactivity and its push/PR triggers name `main`, but the default branch is `master`.
- `develop` is 42 commits ahead / 19 behind `master` (pre-retirement BigQuery-era work); do not merge it.
- The published accuracy table in README (2025 strict holdout: batter wOBA ML 0.0291 vs Marcel 0.0331; pitcher xFIP 0.4837 vs 0.5038) comes from `predictions/backtest/` (2026-03-06).

## The evaluation that is pending
Two prediction sets for **2026** were committed **before Opening Day (2026-03-25; regular season ended 2026-09-27, per statsapi `/seasons/2026`)**:

| set | where | commit time (UTC) | columns |
|---|---|---|---|
| A (served by production app) | `master:predictions/{batter,pitcher}_predictions.csv` | `4a6bae1` 2026-03-16 06:03 | `pred_*` (LightGBM), `marcel_*`, `*_last` (2025), `bayes_*`, `ci_lo80`/`ci_hi80`, `ensemble_*` |
| B (v8 5-model) | `develop@decbd23:predictions/...` | 2026-03-21 18:13 | adds `cat_*`, `component_*` |

Rows: set A 551 batters / 475 pitchers; set B 791 batters / 475 pitchers. Rows are keyed by `player` name + `Team` + `season_last=2025` (no ids). Commit times are git metadata (bot commits from the retired Weekly Retrain); state that caveat when you cite them as proof of timing. **Do not modify or regenerate these files** — they are the record being scored.

## Non-negotiable discipline
1. **Pre-register, then score once.** Write `evaluation/PREREG_2026.md` (a draft is in this repo) and commit it, with its md5 in `evaluation/MD5SUMS_prereg.txt`, **before** any 2026 outcome value is loaded. Only 2025 rows may be read while preparing it (id matching, scale check). A gap found before the run goes into `evaluation/AMENDMENTS.md` before the run. Never change definitions after seeing results.
2. The scoring script asserts the prereg md5 and the md5 of the four prediction files first.
3. Floors are part of the result: Marcel, last-season value, and the league mean of the scored population. "Beats Marcel" needs a paired interval, not a point estimate.
4. Report nulls as facts; no "ceiling"/"limit" framing. Do not rewrite the old README accuracy claims to look better or worse — add the 2026 result next to them, labelled as the first true out-of-sample season.
5. Code that produces a result gets an independent review before it runs; verify every review claim against the code/data before accepting it.
6. **Free only.** No paid service, no paid API key. W&B stays on the free personal plan; nothing in the evaluation needs W&B.
7. Public repo: no secrets, keys, IPs or personal data in commits. Do not commit raw downloaded data (HF parquet, statsapi dumps).

## Next tasks (in order)
1. **Score the frozen 2026 predictions** (no GHA, no secrets needed).
   a. Build the id map using 2025 rows only: HF `fg_batting.parquet` / `fg_pitching.parquet` carry `fg_id`, `player_id`, `name`, `Team`, `season`, `PA`/`IP`, `wOBA`/`xFIP`. Read with a season filter so no 2026 row is loaded, e.g. `pd.read_parquet(url, filters=[("season", "==", 2025)])`. Join prediction rows on (`player`, `Team`, `season=2025`) → `player_id` (verify `player_id` is the MLBAM id by matching a few rows to HF `statsapi_batting`/`statsapi_pitching` 2025 by name; if it is not, use the Chadwick register `key_fangraphs → key_mlbam` from raw.githubusercontent.com). Report unmatched/duplicate rows.
   b. Scale check on 2025 (no 2026 read): compare FanGraphs 2025 `wOBA`/`xFIP` (= `wOBA_last`/`xFIP_last` in the prediction files) with statsapi 2025 `woba`/`xfip` for the same players (statsapi read with the 2025 filter as in a) (bias, sd of difference, correlation). Write the outcome-definition rule into the prereg from this (identity if they agree; otherwise a declared mapping fitted on 2025 only).
   c. Fill in and freeze `evaluation/PREREG_2026.md` (population: 2026 PA ≥ 100 / IP ≥ 30 = the training filters in `src/train.py`, written in statsapi columns: `plateAppearances >= 100` / `outs >= 90` — statsapi has no `PA`/`IP` columns and `inningsPitched` is baseball notation, `42.1` = 42⅓, so never compare it as a decimal; the input revision pinned as a 40-hex HF commit (see Data access); primary = set A `ensemble_*` vs `marcel_*` paired MAE difference with a bootstrap interval; secondary = each column, set B, the 80% interval coverage of `ci_lo80/ci_hi80`, last-season and league-mean floors). Commit it alone first.
   d. Then load 2026 statsapi rows at the pinned revision (`season==2026`; the prereg already recorded, from metadata only — read with `columns=["season","is_partial","fetched_at"]`, never the outcome columns — that `is_partial` is False and `fetched_at` is after 2026-09-27 at that revision), score once with `evaluation/score_2026.py`, commit outputs under `evaluation/results/`, and add a "2026 out-of-sample" section to README (JA, short) next to the old backtest table.
2. **README corrections (after task 1, same or next PR):** it still says the W&B-registry runtime is "running" (only the undeployed API uses it), describes an "RPi5 Parquet → Tailscale rsync" redesign and a "RPi5 Parquet" data section heading, lists `statcast_pitches` as a table, and says park factors come from FanGraphs (HF `park_factors` now comes from Savant). Fix these to the measured state; no new claims.
3. **CodeQL:** change `codeql.yml` branch filters from `main` to `master` (the push trigger has never fired). Re-enabling the disabled schedule needs the web UI/API (owner), so just note it.
Nothing else is pending. Retraining for 2027 is **not** a task: it needs a feature rebuild without FanGraphs (statsapi `sabermetrics` + Savant tables on HF) and the owner's go-ahead first.

## Data access (cloud network allowlist)
- `huggingface.co` + its CDN hosts (`us.aws.cdn.hf.co` — `resolve/` URLs redirect there, measured 2026-10-23 —, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) — dataset `yasumorishima/mlb-stats` (refreshed weekly by `mlb-data-pipeline`, so `main` moves). **Always read a pinned revision:** `https://huggingface.co/datasets/yasumorishima/mlb-stats/resolve/<40-hex>/<table>.parquet`, with the sha from `https://huggingface.co/api/datasets/yasumorishima/mlb-stats/revision/main` recorded in the prereg (`deaeb85b8601c64981cd2082d06941f2d20f5f3f` was `main` on 2026-10-23). Columns (checked at that revision): `statsapi_batting` `player_id, season, name, plateAppearances, woba, is_partial, fetched_at`; `statsapi_pitching` `player_id, season, name, inningsPitched, battersFaced, outs, xfip, is_partial, fetched_at`; `fg_batting` `fg_id, season, name, Team, PA, wOBA, player_id`; `fg_pitching` `… IP, xFIP, player_id`; `fg_*` stop at 2025.
- `statsapi.mlb.com` (schedule/season dates, spot checks), `raw.githubusercontent.com` (Chadwick register), `baseballsavant.mlb.com` if needed.
- Not needed for the evaluation: `api.wandb.ai` (only for retraining/API), FanGraphs (blocked from datacenter IPs; do not work around it).
- GitHub Actions: no workflow runs on push to `master`. Cloud cannot dispatch workflows; none is needed for the tasks above. Repo secrets (names only): `WANDB_API_KEY`, `WANDB_ENTITY`, `DISCORD_WEBHOOK_URL` — used only by `backtest.yml`.

## Credits (keep in README)
Baseball Savant (MLBAM), MLB Stats API (via HF `yasumorishima/mlb-stats`), FanGraphs (historical features via pybaseball), Chadwick Register (ODC-By 1.0) if used.
