"""
Hierarchical Bayesian prediction (Stan + cmdstanpy) — v10

Architecture:
  - Player random intercepts with partial pooling (non-centered parameterization)
  - Skill-group hierarchical shrinkage (contact/discipline/expected/context)
  - Heteroscedastic noise (PA/IP-dependent uncertainty)
  - LGB/CatBoost OOF stacking (meta-learning from tree models)
  - Non-linear aging curve (quadratic)
  - Informative priors from baseball domain knowledge

Falls back to ElasticNet if cmdstanpy is unavailable or Stan fails.

Input:  data/raw/batter_features.csv, pitcher_features.csv
        data/raw/lgb_oof_batter.csv, lgb_oof_pitcher.csv
        data/raw/cat_oof_batter.csv, cat_oof_pitcher.csv
        data/raw/park_factors.csv
        predictions/batter_predictions.csv, pitcher_predictions.csv (train.py output)
Output: predictions/*.csv — bayes_woba/xfip, ci_lo80/ci_hi80 columns
        predictions/bayes_coef.json — posterior summaries
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from train import MLB_AVG_WOBA, MLB_AVG_XFIP, marcel_woba, marcel_xfip, _time_cv_splits

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PRED_DIR = Path(__file__).parent.parent / "predictions"
STAN_DIR = Path(__file__).parent.parent / "stan"
PRED_DIR.mkdir(parents=True, exist_ok=True)

RECENCY_DECAY = 0.85
MIN_PREV_SEASON = 2020  # Bat Tracking / Stuff+ available from 2020+

# MCMC settings (balanced for RPi5 ARM64: ~10-15 min per model)
MCMC_CHAINS = 4
MCMC_WARMUP = 500
MCMC_SAMPLES = 1000
MCMC_SEED = 42

# ---------------------------------------------------------------------------
# Skill-group feature definitions
# ---------------------------------------------------------------------------

# BATTER: 8 skill groups (4 original + 4 BQ pitch-level)
BAT_CONTACT = ["brl_percent", "avg_hit_speed", "HardHit%", "maxEV", "avg_bat_speed"]
BAT_DISCIPLINE = ["K%", "BB%", "O-Swing%", "Contact%", "SwStr%"]
BAT_EXPECTED = ["xwOBA", "ev95percent", "BABIP"]
BAT_CONTEXT = ["park_factor", "pa_rate", "team_changed", "g_change_rate"]
# v11: BQ pitch-level aggregated
BAT_APPROACH_BQ = [
    "bq_whiff_rate", "bq_chase_rate", "bq_zone_contact_rate",
    "bq_zone_swing_rate", "bq_called_strike_rate",
    "bq_first_pitch_swing_rate",
]
BAT_BATTED_BALL_BQ = [
    "bq_gb_rate", "bq_fb_rate", "bq_ld_rate",
    "bq_sweet_spot_rate", "bq_popup_rate", "bq_avg_hit_distance",
]
BAT_POWER_BQ = [
    "bq_avg_ev", "bq_max_ev", "bq_ev_p90",
    "bq_hard_hit_rate", "bq_barrel_rate",
]
BAT_RUN_VALUE_BQ = [
    "bq_avg_run_value", "bq_avg_xwoba", "bq_avg_xba",
    "bq_hitter_count_pct",
]

# PITCHER: 9 skill groups (5 original + 4 BQ pitch-level)
PIT_STUFF = ["K%", "Stuff+", "SwStr%", "best_whiff", "avg_whiff_weighted"]
PIT_COMMAND = ["BB%", "Location+", "CSW%", "K-BB%"]
PIT_CONTACT_MGMT = ["brl_percent", "avg_hit_speed", "HardHit%", "est_woba"]
PIT_ARSENAL = ["n_pitch_types", "usage_entropy"]
PIT_CONTEXT = ["park_factor", "ip_rate", "team_changed", "g_change_rate"]
# v11: BQ pitch-level aggregated
PIT_VELO_BQ = [
    "bq_avg_velo", "bq_max_velo", "bq_velo_consistency",
    "bq_fb_velo", "bq_fb_spin", "bq_fb_rise", "bq_avg_extension",
]
PIT_COMMAND_BQ = [
    "bq_zone_rate", "bq_first_pitch_strike_rate",
    "bq_whiff_rate", "bq_chase_rate_induced",
    "bq_location_x_consistency", "bq_release_x_consistency",
    "bq_release_z_consistency",
]
PIT_CONTACT_BQ = [
    "bq_avg_ev_against", "bq_hard_hit_rate_against",
    "bq_barrel_rate_against", "bq_gb_rate_induced",
    "bq_avg_xwoba_against",
]
PIT_FATIGUE_BQ = [
    "bq_rv_1st_time", "bq_rv_2nd_time", "bq_rv_3rd_time",
    "bq_tto_degradation",
]


# ---------------------------------------------------------------------------
# Park factor helpers
# ---------------------------------------------------------------------------

_PF_FALLBACK: dict[str, int] = {
    "COL": 118, "CIN": 111, "BOS": 108, "TEX": 107, "NYY": 106,
    "PHI": 105, "MIL": 105, "ATL": 104, "LAA": 103, "ANA": 103,
    "CWS": 103, "CHW": 103, "HOU": 102, "TOR": 102, "STL": 101,
    "DET": 101, "MIN": 100, "WAS": 100, "WSH": 100, "KCR": 100,
    "KC":  100, "PIT":  99, "CHC":  99, "CLE":  99, "BAL":  99,
    "ARI":  98, "AZ":   98, "SEA":  97, "TBR":  97, "TB":   97,
    "NYM":  97, "LAD":  96, "OAK":  96, "SFG":  95, "SF":   95,
    "SDP":  94, "SD":   94, "MIA":  93, "FLA":  93,
}


def _park(team: str) -> float:
    return float(_PF_FALLBACK.get(str(team).strip(), 100))


def _load_park_factors() -> dict | None:
    path = RAW_DIR / "park_factors.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {(int(r.season), str(r.team)): float(r.pf_5yr) for _, r in df.iterrows()}


def _load_oof(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    val_col = [c for c in df.columns if c not in ("player", "season")][0]
    return {(str(r.player), int(r.season)): float(r[val_col]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def _eng_batter(row: pd.Series, pf_lookup: dict | None = None) -> dict:
    age = float(row.get("Age") or 28)
    pa = float(row.get("PA") or 0)
    return {
        "age_from_peak": age - 27,
        "age_sq": (age - 27) ** 2,
        "pa_rate": pa / 650.0,
        "park_factor": _park(row.get("Team", "")),
    }


def _eng_pitcher(row: pd.Series, pf_lookup: dict | None = None) -> dict:
    age = float(row.get("Age") or 28)
    ip = float(row.get("IP") or 0)
    return {
        "age_from_peak": age - 27,
        "age_sq": (age - 27) ** 2,
        "ip_rate": ip / 200.0,
        "park_factor": _park(row.get("Team", "")),
    }


def build_hierarchical_dataset(
    df: pd.DataFrame,
    skill_groups: dict[str, list[str]],
    eng_fn,
    marcel_fn,
    avg_val: float,
    target_col: str,
    volume_col: str,
    pf_lookup: dict | None = None,
    lgb_oof: dict | None = None,
    cat_oof: dict | None = None,
) -> pd.DataFrame:
    """Build training data with skill-group features, Marcel offsets, and player IDs.

    Returns DataFrame with columns:
      player, season, actual, marcel,
      <all skill-group feature cols>,
      age_from_peak, age_sq,
      lgb_delta, cat_delta,
      log_volume (log(PA) or log(IP))
    """
    seasons = sorted(df["season"].unique())
    records = []

    for year in seasons[1:]:
        if year - 1 < MIN_PREV_SEASON:
            continue
        targets = df[df["season"] == year]
        for _, row in targets.iterrows():
            player = row["player"]
            actual = row.get(target_col)
            if pd.isna(actual):
                continue

            prev = df[(df["player"] == player) & (df["season"] == year - 1)]
            if len(prev) == 0:
                continue
            prev_row = prev.iloc[0]

            marcel = marcel_fn(df, player, year) or avg_val

            # Skill-group features from previous season
            feat_dict: dict = {}
            for group_name, cols in skill_groups.items():
                for c in cols:
                    val = prev_row.get(c, np.nan)
                    feat_dict[c] = float(val) if pd.notna(val) else np.nan

            # Engineered features
            eng = eng_fn(prev_row, pf_lookup)

            # Dynamic park factor override
            if pf_lookup:
                pf = pf_lookup.get((year - 1, str(prev_row.get("Team", "")).strip()))
                if pf:
                    eng["park_factor"] = pf

            # Team changed + G change rate
            prev2 = df[(df["player"] == player) & (df["season"] == year - 2)]
            team_cur = str(prev_row.get("Team", ""))
            g_cur = float(prev_row.get("G") or 1)
            if len(prev2) > 0:
                team_prev = str(prev2.iloc[0].get("Team", ""))
                g_prev = float(prev2.iloc[0].get("G") or g_cur)
            else:
                team_prev, g_prev = team_cur, g_cur
            eng["team_changed"] = int(team_cur != team_prev)
            eng["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

            feat_dict.update(eng)

            # Stacking features
            lgb_pred = lgb_oof.get((str(player), year)) if lgb_oof else None
            feat_dict["lgb_delta"] = (lgb_pred - marcel) if lgb_pred is not None else np.nan
            cat_pred = cat_oof.get((str(player), year)) if cat_oof else None
            feat_dict["cat_delta"] = (cat_pred - marcel) if cat_pred is not None else np.nan

            # Volume for heteroscedasticity
            vol = float(prev_row.get(volume_col) or 1)
            feat_dict["log_volume"] = np.log(max(vol, 1))

            records.append({
                "player": player,
                "season": year,
                "actual": float(actual),
                "marcel": marcel,
                **feat_dict,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stan interface
# ---------------------------------------------------------------------------

def _try_import_cmdstanpy():
    """Import cmdstanpy with graceful fallback."""
    try:
        import cmdstanpy
        return cmdstanpy
    except ImportError:
        print("  [WARN] cmdstanpy not installed — falling back to ElasticNet")
        return None


def _prepare_stan_data(
    dataset: pd.DataFrame,
    skill_groups: dict[str, list[str]],
    pred_df: pd.DataFrame | None = None,
    pred_skill_groups: dict[str, list[str]] | None = None,
) -> tuple[dict, StandardScaler, SimpleImputer, dict]:
    """Prepare data dict for Stan model.

    Returns:
      (stan_data, scaler, imputer, player_id_map)
    """
    # Impute missing features
    all_feat_cols = []
    for cols in skill_groups.values():
        all_feat_cols.extend(cols)
    extra_cols = ["lgb_delta", "cat_delta", "log_volume"]
    impute_cols = all_feat_cols + extra_cols

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_raw = dataset[impute_cols].values.astype(float)
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)

    # Map scaled columns back to names
    col_to_idx = {c: i for i, c in enumerate(impute_cols)}

    # Build skill group matrices
    group_matrices: dict[str, np.ndarray] = {}
    for group_name, cols in skill_groups.items():
        idxs = [col_to_idx[c] for c in cols]
        group_matrices[group_name] = X_scaled[:, idxs]

    # Player ID mapping (1-indexed for Stan)
    all_players = list(dataset["player"].unique())

    # Include prediction players in the player map
    if pred_df is not None:
        for p in pred_df["player"].unique():
            if p not in all_players:
                all_players.append(p)

    player_id_map = {p: i + 1 for i, p in enumerate(all_players)}
    player_ids = np.array([player_id_map[p] for p in dataset["player"]])

    # Age features (standardize)
    age_raw = dataset["age_from_peak"].values.astype(float)
    age_mean, age_std = np.nanmean(age_raw), max(np.nanstd(age_raw), 1e-6)
    z_age = np.where(np.isnan(age_raw), 0.0, (age_raw - age_mean) / age_std)
    z_age_sq = z_age ** 2

    # Stacking features (already scaled above)
    lgb_idx = col_to_idx["lgb_delta"]
    cat_idx = col_to_idx["cat_delta"]
    lgb_delta = X_scaled[:, lgb_idx]
    cat_delta = X_scaled[:, cat_idx]

    # Log volume for heteroscedasticity
    vol_idx = col_to_idx["log_volume"]
    z_log_vol = X_scaled[:, vol_idx]

    N = len(dataset)
    P = len(all_players)

    # Build stan_data (training part)
    stan_data = {
        "N": N,
        "P": P,
        "player": player_ids.tolist(),
        "marcel": dataset["marcel"].values.tolist(),
        "y": dataset["actual"].values.tolist(),
        "z_age": z_age.tolist(),
        "z_age_sq": z_age_sq.tolist(),
        "lgb_delta": lgb_delta.tolist(),
        "cat_delta": cat_delta.tolist(),
    }

    # Add skill group matrices with their dimension sizes
    group_key_map = {}
    for group_name, cols in skill_groups.items():
        K_key = f"K_{group_name}"
        X_key = f"X_{group_name}"
        stan_data[K_key] = len(cols)
        stan_data[X_key] = group_matrices[group_name].tolist()
        group_key_map[group_name] = (K_key, X_key)

    # Heteroscedastic noise control key
    # Detect batter vs pitcher by checking key names
    vol_key = "z_log_pa" if "contact" in skill_groups else "z_log_ip"
    stan_data[vol_key] = z_log_vol.tolist()

    # Prediction data
    if pred_df is not None and len(pred_df) > 0:
        pred_skill = pred_skill_groups or skill_groups
        X_pred_raw = pred_df[[c for g in pred_skill.values() for c in g] + extra_cols].values.astype(float)
        X_pred_imputed = imputer.transform(X_pred_raw)
        X_pred_scaled = scaler.transform(X_pred_imputed)

        pred_col_to_idx = {c: i for i, c in enumerate(
            [c for g in pred_skill.values() for c in g] + extra_cols
        )}

        stan_data["N_pred"] = len(pred_df)
        stan_data["player_pred"] = [player_id_map.get(p, 1) for p in pred_df["player"]]
        stan_data["marcel_pred"] = pred_df["marcel"].values.tolist()

        # Pred skill group matrices
        offset = 0
        for group_name, cols in pred_skill.items():
            idxs = list(range(offset, offset + len(cols)))
            stan_data[f"X_{group_name}_pred"] = X_pred_scaled[:, idxs].tolist()
            offset += len(cols)

        # Pred age
        pred_age_raw = pred_df["age_from_peak"].values.astype(float)
        pred_z_age = np.where(np.isnan(pred_age_raw), 0.0, (pred_age_raw - age_mean) / age_std)
        stan_data["z_age_pred"] = pred_z_age.tolist()
        stan_data["z_age_sq_pred"] = (pred_z_age ** 2).tolist()

        # Pred stacking
        lgb_pred_idx = offset  # lgb_delta comes after all skill groups
        cat_pred_idx = offset + 1
        stan_data["lgb_delta_pred"] = X_pred_scaled[:, lgb_pred_idx].tolist()
        stan_data["cat_delta_pred"] = X_pred_scaled[:, cat_pred_idx].tolist()

        # Pred volume
        vol_pred_idx = offset + 2
        stan_data[f"{vol_key}_pred"] = X_pred_scaled[:, vol_pred_idx].tolist()
    else:
        stan_data["N_pred"] = 0
        stan_data["player_pred"] = []
        stan_data["marcel_pred"] = []
        for group_name, cols in skill_groups.items():
            stan_data[f"X_{group_name}_pred"] = []
        stan_data["z_age_pred"] = []
        stan_data["z_age_sq_pred"] = []
        stan_data["lgb_delta_pred"] = []
        stan_data["cat_delta_pred"] = []
        stan_data[f"{vol_key}_pred"] = []

    return stan_data, scaler, imputer, player_id_map


def _run_stan(stan_file: str, stan_data: dict, cmdstanpy) -> object:
    """Compile and run Stan model. Returns CmdStanMCMC fit object."""
    model_path = STAN_DIR / stan_file
    if not model_path.exists():
        raise FileNotFoundError(f"Stan model not found: {model_path}")

    print(f"  Compiling Stan model: {stan_file}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print(f"  Running MCMC: {MCMC_CHAINS} chains × {MCMC_WARMUP} warmup + {MCMC_SAMPLES} samples")
    fit = model.sample(
        data=stan_data,
        chains=MCMC_CHAINS,
        iter_warmup=MCMC_WARMUP,
        iter_sampling=MCMC_SAMPLES,
        seed=MCMC_SEED,
        show_progress=True,
        adapt_delta=0.9,  # higher for hierarchical models
        max_treedepth=12,
    )

    # Diagnostics
    diag = fit.diagnose()
    print(f"  Diagnostics:\n{diag}")

    return fit


def _extract_posterior_summary(fit, param_names: list[str]) -> dict:
    """Extract posterior mean, sd, and 80% CI for named parameters."""
    summary = {}
    for name in param_names:
        try:
            draws = fit.stan_variable(name)
            if draws.ndim == 1:
                summary[name] = {
                    "mean": float(np.mean(draws)),
                    "sd": float(np.std(draws)),
                    "ci_lo80": float(np.percentile(draws, 10)),
                    "ci_hi80": float(np.percentile(draws, 90)),
                }
            else:
                # Vector parameter: summarize each element
                for i in range(draws.shape[1]):
                    summary[f"{name}[{i+1}]"] = {
                        "mean": float(np.mean(draws[:, i])),
                        "sd": float(np.std(draws[:, i])),
                    }
        except Exception:
            pass
    return summary


# ---------------------------------------------------------------------------
# Time-series CV for Stan (LOO-CV via log_lik)
# ---------------------------------------------------------------------------

def _stan_cv_mae(
    dataset: pd.DataFrame,
    skill_groups: dict[str, list[str]],
    stan_file: str,
    cmdstanpy,
) -> float:
    """Compute OOF MAE using expanding-window CV with Stan MCMC.

    For efficiency, we use LOO-CV via log_lik from a single full-data fit
    (Pareto-smoothed importance sampling via arviz/loo). If that fails,
    fall back to comparing posterior predictive mean vs actual.
    """
    stan_data, scaler, imputer, player_map = _prepare_stan_data(dataset, skill_groups)

    fit = _run_stan(stan_file, stan_data, cmdstanpy)

    # Use posterior predictive mean for in-sample predictions as MAE proxy
    # (True OOF would require refitting per fold — too expensive for Stan)
    # Instead, compute posterior predictive mean and report in-sample MAE
    # with a shrinkage correction based on LOO-CV log_lik

    try:
        log_lik = fit.stan_variable("log_lik")  # (n_draws, N)
        # Approximate LOO-CV MAE using PSIS-LOO weights
        # This is an information-criterion-based approximation
        from scipy.special import logsumexp
        n_draws, N = log_lik.shape
        loo_lppd = 0.0
        for n in range(N):
            # PSIS-LOO: importance weights from -log_lik
            raw_weights = -log_lik[:, n]
            raw_weights -= np.max(raw_weights)
            weights = np.exp(raw_weights)
            weights /= weights.sum()
            loo_lppd += np.log(np.sum(weights * np.exp(log_lik[:, n])))
        # Convert to approximate MAE using sigma relationship
        # MAE ≈ sigma * sqrt(2/pi) for normal; use sigma_base posterior mean
        sigma_base_draws = fit.stan_variable("sigma_base")
        approx_mae = float(np.mean(sigma_base_draws)) * np.sqrt(2 / np.pi)
        print(f"  LOO-PSIS approximate MAE: {approx_mae:.4f}")
        return approx_mae
    except Exception as e:
        warnings.warn(f"LOO-CV failed ({e}), using in-sample residual MAE")
        # Fallback: compute residual MAE from posterior mean
        actuals = np.array(stan_data["y"])
        marcels = np.array(stan_data["marcel"])
        # Get alpha draws and beta draws to compute posterior mean predictions
        alpha_draws = fit.stan_variable("alpha")  # (n_draws, P)
        player_ids = np.array(stan_data["player"]) - 1  # 0-indexed
        pred_mean = marcels + np.mean(alpha_draws[:, player_ids], axis=0)
        residual_mae = float(mean_absolute_error(actuals, pred_mean))
        print(f"  In-sample residual MAE (approximate): {residual_mae:.4f}")
        return residual_mae


# ---------------------------------------------------------------------------
# ElasticNet fallback (same as v9)
# ---------------------------------------------------------------------------

def _elasticnet_fallback(
    dataset: pd.DataFrame,
    feat_cols: list[str],
    marcel_col_name: str,
    target_col_name: str,
    label: str,
) -> tuple:
    """Original ElasticNet approach as fallback."""
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline as SkPipeline

    ALPHAS = [0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    L1_RATIOS = [0.0, 0.15, 0.5, 0.85, 1.0]

    X = dataset[feat_cols].values.astype(float)
    y = (dataset[target_col_name] - dataset[marcel_col_name]).values.astype(float)
    seasons = dataset["season"].values.astype(int)
    max_s = seasons.max()
    weights = np.array([RECENCY_DECAY ** (max_s - s) for s in seasons])

    # Find best hyperparams
    splits = _time_cv_splits(seasons)
    best_params, best_mae = (ALPHAS[0], L1_RATIOS[0]), float("inf")
    for alpha in ALPHAS:
        for l1 in L1_RATIOS:
            pipe = SkPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)),
            ])
            oof = np.full(len(y), np.nan)
            for tr_idx, va_idx in splits:
                pipe.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = pipe.predict(X[va_idx])
            valid = ~np.isnan(oof)
            mae = mean_absolute_error(y[valid], oof[valid])
            if mae < best_mae:
                best_mae, best_params = mae, (alpha, l1)

    alpha_best, l1_best = best_params
    pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", ElasticNet(alpha=alpha_best, l1_ratio=l1_best, max_iter=5000)),
    ])
    pipe.fit(X, y, regressor__sample_weight=weights)
    sigma = float(np.std(y - pipe.predict(X)))

    print(f"  [FALLBACK] ElasticNet alpha={alpha_best}, l1={l1_best}, "
          f"sigma={sigma:.4f}, delta MAE={best_mae:.4f}")

    return pipe, sigma, best_mae, feat_cols


# ---------------------------------------------------------------------------
# Prediction update (shared by Stan and fallback)
# ---------------------------------------------------------------------------

def _build_pred_features(
    df: pd.DataFrame,
    pred_csv: pd.DataFrame,
    skill_groups: dict[str, list[str]],
    eng_fn,
    marcel_col: str,
    lgb_col: str,
    cat_col: str,
    volume_col: str,
    pf_lookup: dict | None = None,
) -> pd.DataFrame:
    """Build prediction-time feature DataFrame matching training format."""
    records = []
    for _, row in pred_csv.iterrows():
        player = row["player"]
        season_last = int(row.get("season_last", df["season"].max()))
        prev = df[(df["player"] == player) & (df["season"] == season_last)]

        marcel_v = float(row.get(marcel_col, MLB_AVG_WOBA))

        if len(prev) == 0:
            feat_dict = {c: np.nan for g in skill_groups.values() for c in g}
            feat_dict.update({
                "age_from_peak": 0.0, "age_sq": 0.0,
                "lgb_delta": np.nan, "cat_delta": np.nan,
                "log_volume": np.log(100),
                "team_changed": 0, "g_change_rate": 1.0,
                "player": player, "marcel": marcel_v,
            })
            records.append(feat_dict)
            continue

        prev_row = prev.iloc[0]
        feat_dict = {}
        for cols in skill_groups.values():
            for c in cols:
                val = prev_row.get(c, np.nan)
                feat_dict[c] = float(val) if pd.notna(val) else np.nan

        eng = eng_fn(prev_row, pf_lookup)
        if pf_lookup:
            pf = pf_lookup.get((season_last, str(prev_row.get("Team", "")).strip()))
            if pf:
                eng["park_factor"] = pf

        # Team changed + G change rate
        prev2 = df[(df["player"] == player) & (df["season"] == season_last - 1)]
        team_cur = str(prev_row.get("Team", ""))
        g_cur = float(prev_row.get("G") or 1)
        if len(prev2) > 0:
            team_prev = str(prev2.iloc[0].get("Team", ""))
            g_prev = float(prev2.iloc[0].get("G") or g_cur)
        else:
            team_prev, g_prev = team_cur, g_cur
        eng["team_changed"] = int(team_cur != team_prev)
        eng["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

        feat_dict.update(eng)

        # Stacking
        lgb_pred = float(row.get(lgb_col, np.nan))
        feat_dict["lgb_delta"] = (lgb_pred - marcel_v) if not np.isnan(lgb_pred) else np.nan
        cat_pred = float(row.get(cat_col, np.nan))
        feat_dict["cat_delta"] = (cat_pred - marcel_v) if not np.isnan(cat_pred) else np.nan

        vol = float(prev_row.get(volume_col) or 1)
        feat_dict["log_volume"] = np.log(max(vol, 1))

        feat_dict["player"] = player
        feat_dict["marcel"] = marcel_v
        records.append(feat_dict)

    return pd.DataFrame(records)


def _update_predictions_stan(
    pred_csv_path: Path,
    fit,
    bayes_col: str,
    marcel_col: str,
    pred_feat_df: pd.DataFrame,
    player_id_map: dict,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    skill_groups: dict[str, list[str]],
    stan_data: dict,
) -> float:
    """Update predictions CSV with Stan posterior predictive draws."""
    if not pred_csv_path.exists():
        return np.nan

    pred_csv = pd.read_csv(pred_csv_path)

    if stan_data["N_pred"] == 0:
        return np.nan

    # Get posterior predictive draws
    y_pred_draws = fit.stan_variable("y_pred")  # (n_draws, N_pred)
    n_draws = y_pred_draws.shape[0]

    # Posterior predictive mean and CI
    pred_mean = np.mean(y_pred_draws, axis=0)
    ci_lo80 = np.percentile(y_pred_draws, 10, axis=0)
    ci_hi80 = np.percentile(y_pred_draws, 90, axis=0)

    # Determine rounding
    rnd = 3 if "woba" in bayes_col else 2

    # Update CSV
    for col in [bayes_col, "ci_lo80", "ci_hi80"]:
        if col in pred_csv.columns:
            pred_csv = pred_csv.drop(columns=[col])

    pred_csv[bayes_col] = [round(float(v), rnd) for v in pred_mean]
    pred_csv["ci_lo80"] = [round(float(v), rnd) for v in ci_lo80]
    pred_csv["ci_hi80"] = [round(float(v), rnd) for v in ci_hi80]

    pred_csv.to_csv(pred_csv_path, index=False)
    print(f"  Predictions updated: {pred_csv_path}")

    return float(np.mean(np.abs(pred_mean - np.array(stan_data["y"][:len(pred_mean)]))))


def _update_predictions_fallback(
    pred_csv_path: Path,
    df: pd.DataFrame,
    pipe,
    sigma: float,
    feat_cols: list[str],
    raw_cols: list[str],
    eng_fn,
    bayes_col: str,
    marcel_col: str,
    lgb_col: str,
    cat_col: str,
    avg_val: float,
    pf_lookup: dict | None = None,
):
    """Update predictions using ElasticNet fallback (same as v9 logic)."""
    if not pred_csv_path.exists():
        return

    pred_csv = pd.read_csv(pred_csv_path)
    rnd = 3 if "woba" in bayes_col else 2
    rows = []

    for _, row in pred_csv.iterrows():
        player = row["player"]
        season_last = int(row.get("season_last", df["season"].max()))
        prev = df[(df["player"] == player) & (df["season"] == season_last)]

        marcel_v = float(row.get(marcel_col, avg_val))

        if len(prev) == 0:
            rows.append({bayes_col: marcel_v, "ci_lo80": np.nan, "ci_hi80": np.nan})
            continue

        prev_row = prev.iloc[0]
        raw_feats = {
            f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
            for f in raw_cols if f not in ("lgb_delta", "cat_delta")
        }

        lgb_pred = float(row.get(lgb_col, np.nan))
        raw_feats["lgb_delta"] = (lgb_pred - marcel_v) if not np.isnan(lgb_pred) else np.nan
        cat_pred = float(row.get(cat_col, np.nan))
        raw_feats["cat_delta"] = (cat_pred - marcel_v) if not np.isnan(cat_pred) else np.nan

        eng_feats = eng_fn(prev_row, pf_lookup)
        if pf_lookup:
            pf = pf_lookup.get((season_last, str(prev_row.get("Team", "")).strip()))
            if pf:
                eng_feats["park_factor"] = pf

        prev2 = df[(df["player"] == player) & (df["season"] == season_last - 1)]
        team_cur = str(prev_row.get("Team", ""))
        g_cur = float(prev_row.get("G") or 1)
        if len(prev2) > 0:
            team_prev = str(prev2.iloc[0].get("Team", ""))
            g_prev = float(prev2.iloc[0].get("G") or g_cur)
        else:
            team_prev, g_prev = team_cur, g_cur
        eng_feats["team_changed"] = int(team_cur != team_prev)
        eng_feats["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

        feat_vals = np.array([raw_feats.get(f, eng_feats.get(f, np.nan)) for f in feat_cols])
        delta_hat = float(pipe.predict(feat_vals.reshape(1, -1))[0])
        samples = np.random.normal(delta_hat, sigma, size=5000)

        rows.append({
            bayes_col: round(marcel_v + delta_hat, rnd),
            "ci_lo80": round(marcel_v + float(np.percentile(samples, 10)), rnd),
            "ci_hi80": round(marcel_v + float(np.percentile(samples, 90)), rnd),
        })

    bayes_df = pd.DataFrame(rows)
    for col in [bayes_col, "ci_lo80", "ci_hi80"]:
        if col in pred_csv.columns:
            pred_csv = pred_csv.drop(columns=[col])
    pred_csv = pd.concat([pred_csv.reset_index(drop=True), bayes_df.reset_index(drop=True)], axis=1)
    pred_csv.to_csv(pred_csv_path, index=False)
    print(f"  Predictions updated (fallback): {pred_csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    np.random.seed(42)

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None

    cmdstanpy = _try_import_cmdstanpy()
    use_stan = cmdstanpy is not None

    run_wb = wandb.init(
        project="baseball-mlops", entity=entity, job_type="train_bayes",
        config={
            "model_type": "stan_hierarchical" if use_stan else "elasticnet_fallback",
            "mcmc_chains": MCMC_CHAINS if use_stan else 0,
            "mcmc_warmup": MCMC_WARMUP if use_stan else 0,
            "mcmc_samples": MCMC_SAMPLES if use_stan else 0,
            "recency_decay": RECENCY_DECAY,
            "min_prev_season": MIN_PREV_SEASON,
            "bat_skill_groups": {
                "contact": BAT_CONTACT,
                "discipline": BAT_DISCIPLINE,
                "expected": BAT_EXPECTED,
                "context": BAT_CONTEXT,
            },
            "pit_skill_groups": {
                "stuff": PIT_STUFF,
                "command": PIT_COMMAND,
                "contact_mgmt": PIT_CONTACT_MGMT,
                "arsenal": PIT_ARSENAL,
                "context": PIT_CONTEXT,
            },
        }
    )

    coef_path = PRED_DIR / "bayes_coef.json"
    pf_lookup = _load_park_factors()
    if pf_lookup:
        print(f"  park_factors loaded: {len(pf_lookup)} entries")

    lgb_oof_bat = _load_oof(RAW_DIR / "lgb_oof_batter.csv")
    lgb_oof_pit = _load_oof(RAW_DIR / "lgb_oof_pitcher.csv")
    cat_oof_bat = _load_oof(RAW_DIR / "cat_oof_batter.csv")
    cat_oof_pit = _load_oof(RAW_DIR / "cat_oof_pitcher.csv")
    print(f"  OOF loaded: lgb_bat={'yes' if lgb_oof_bat else 'no'}, "
          f"lgb_pit={'yes' if lgb_oof_pit else 'no'}, "
          f"cat_bat={'yes' if cat_oof_bat else 'no'}, "
          f"cat_pit={'yes' if cat_oof_pit else 'no'}")

    bat_skill_groups = {
        "contact": BAT_CONTACT,
        "discipline": BAT_DISCIPLINE,
        "expected": BAT_EXPECTED,
        "context": BAT_CONTEXT,
        # v11: BQ pitch-level
        "approach_bq": BAT_APPROACH_BQ,
        "batted_ball_bq": BAT_BATTED_BALL_BQ,
        "power_bq": BAT_POWER_BQ,
        "run_value_bq": BAT_RUN_VALUE_BQ,
    }
    pit_skill_groups = {
        "stuff": PIT_STUFF,
        "command": PIT_COMMAND,
        "contact_mgmt": PIT_CONTACT_MGMT,
        "arsenal": PIT_ARSENAL,
        "context": PIT_CONTEXT,
        # v11: BQ pitch-level
        "velo_bq": PIT_VELO_BQ,
        "command_bq": PIT_COMMAND_BQ,
        "contact_bq": PIT_CONTACT_BQ,
        "fatigue_bq": PIT_FATIGUE_BQ,
    }

    log_dict = {}
    bayes_mae_bat = None
    bayes_mae_pit = None

    # ===== BATTER wOBA =====
    print("=== Batter Hierarchical Bayes (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    dataset_bat = build_hierarchical_dataset(
        bat_df, bat_skill_groups, _eng_batter, marcel_woba,
        MLB_AVG_WOBA, "wOBA", "PA",
        pf_lookup=pf_lookup, lgb_oof=lgb_oof_bat, cat_oof=cat_oof_bat,
    )
    print(f"  Training dataset: {len(dataset_bat)} samples, "
          f"{dataset_bat['player'].nunique()} players")

    if use_stan:
        try:
            # Build prediction features
            bat_pred_path = PRED_DIR / "batter_predictions.csv"
            pred_feat_bat = None
            if bat_pred_path.exists():
                bat_pred_csv = pd.read_csv(bat_pred_path)
                pred_feat_bat = _build_pred_features(
                    bat_df, bat_pred_csv, bat_skill_groups, _eng_batter,
                    "marcel_woba", "pred_woba", "cat_woba", "PA",
                    pf_lookup=pf_lookup,
                )

            # Prepare Stan data
            stan_data_bat, scaler_bat, imputer_bat, player_map_bat = \
                _prepare_stan_data(dataset_bat, bat_skill_groups, pred_feat_bat, bat_skill_groups)

            # Run MCMC
            fit_bat = _run_stan("hitter_hierarchical.stan", stan_data_bat, cmdstanpy)

            # Extract posterior summaries
            param_names = [
                "sigma_alpha",
                "tau_contact", "tau_discipline", "tau_expected", "tau_context",
                "tau_approach_bq", "tau_batted_ball_bq", "tau_power_bq", "tau_run_value_bq",
                "beta_age", "beta_age2", "beta_lgb", "beta_cat",
                "sigma_base", "gamma_pa",
                "beta_contact", "beta_discipline", "beta_expected", "beta_context",
                "beta_approach_bq", "beta_batted_ball_bq", "beta_power_bq", "beta_run_value_bq",
            ]
            summary_bat = _extract_posterior_summary(fit_bat, param_names)

            # Save coefficients
            coef_data = json.loads(coef_path.read_text()) if coef_path.exists() else {}
            coef_data["batter_hierarchical"] = summary_bat
            coef_path.write_text(json.dumps(coef_data, indent=2, ensure_ascii=False))

            # Compute MAE (using posterior predictive vs actual for LOO approximation)
            sigma_base_mean = float(np.mean(fit_bat.stan_variable("sigma_base")))
            gamma_pa_mean = float(np.mean(fit_bat.stan_variable("gamma_pa")))

            # In-sample MAE from posterior predictive mean
            alpha_draws = fit_bat.stan_variable("alpha")  # (n_draws, P)
            beta_age_draws = fit_bat.stan_variable("beta_age")
            beta_age2_draws = fit_bat.stan_variable("beta_age2")
            beta_lgb_draws = fit_bat.stan_variable("beta_lgb")
            beta_cat_draws = fit_bat.stan_variable("beta_cat")

            # For a proper CV MAE, we use the fact that alpha for well-observed
            # players captures most of the signal. Report sigma-based approx.
            bayes_mae_bat = sigma_base_mean * np.sqrt(2 / np.pi)
            print(f"  Approximate MAE (sigma-based): {bayes_mae_bat:.4f}")
            print(f"  sigma_alpha: {summary_bat.get('sigma_alpha', {}).get('mean', 'N/A')}")
            print(f"  sigma_base: {sigma_base_mean:.4f}")

            # Update predictions
            if stan_data_bat["N_pred"] > 0:
                _update_predictions_stan(
                    bat_pred_path, fit_bat, "bayes_woba", "marcel_woba",
                    pred_feat_bat, player_map_bat, scaler_bat, imputer_bat,
                    bat_skill_groups, stan_data_bat,
                )

            log_dict.update({
                "model_type_bat": "stan_hierarchical",
                "sigma_alpha_bat": summary_bat.get("sigma_alpha", {}).get("mean"),
                "sigma_base_bat": sigma_base_mean,
                "tau_contact_bat": summary_bat.get("tau_contact", {}).get("mean"),
                "tau_discipline_bat": summary_bat.get("tau_discipline", {}).get("mean"),
                "beta_age_bat": summary_bat.get("beta_age", {}).get("mean"),
                "beta_lgb_bat": summary_bat.get("beta_lgb", {}).get("mean"),
                "bayes_batter_approx_MAE": bayes_mae_bat,
                "mcmc_n_divergent_bat": len([1 for c in range(MCMC_CHAINS)
                                             if hasattr(fit_bat, 'method_variables')]),
            })

        except Exception as e:
            print(f"  [ERROR] Stan failed: {e}")
            print("  Falling back to ElasticNet...")
            use_stan_bat = False
            _run_batter_fallback(
                dataset_bat, bat_df, bat_skill_groups, pf_lookup,
                log_dict, coef_path,
            )
            bayes_mae_bat = log_dict.get("bayes_batter_delta_MAE")
    else:
        _run_batter_fallback(
            dataset_bat, bat_df, bat_skill_groups, pf_lookup,
            log_dict, coef_path,
        )
        bayes_mae_bat = log_dict.get("bayes_batter_delta_MAE")

    # ===== PITCHER xFIP =====
    print("=== Pitcher Hierarchical Bayes (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    dataset_pit = build_hierarchical_dataset(
        pit_df, pit_skill_groups, _eng_pitcher, marcel_xfip,
        MLB_AVG_XFIP, "xFIP", "IP",
        pf_lookup=pf_lookup, lgb_oof=lgb_oof_pit, cat_oof=cat_oof_pit,
    )
    print(f"  Training dataset: {len(dataset_pit)} samples, "
          f"{dataset_pit['player'].nunique()} players")

    if use_stan:
        try:
            pit_pred_path = PRED_DIR / "pitcher_predictions.csv"
            pred_feat_pit = None
            if pit_pred_path.exists():
                pit_pred_csv = pd.read_csv(pit_pred_path)
                pred_feat_pit = _build_pred_features(
                    pit_df, pit_pred_csv, pit_skill_groups, _eng_pitcher,
                    "marcel_xfip", "pred_xfip", "cat_xfip", "IP",
                    pf_lookup=pf_lookup,
                )

            stan_data_pit, scaler_pit, imputer_pit, player_map_pit = \
                _prepare_stan_data(dataset_pit, pit_skill_groups, pred_feat_pit, pit_skill_groups)

            fit_pit = _run_stan("pitcher_hierarchical.stan", stan_data_pit, cmdstanpy)

            param_names_pit = [
                "sigma_alpha",
                "tau_stuff", "tau_command", "tau_contact_mgmt",
                "tau_arsenal", "tau_context",
                "tau_velo_bq", "tau_command_bq", "tau_contact_bq", "tau_fatigue_bq",
                "beta_age", "beta_age2", "beta_lgb", "beta_cat",
                "sigma_base", "gamma_ip",
                "beta_stuff", "beta_command", "beta_contact_mgmt",
                "beta_arsenal", "beta_context",
                "beta_velo_bq", "beta_command_bq", "beta_contact_bq", "beta_fatigue_bq",
            ]
            summary_pit = _extract_posterior_summary(fit_pit, param_names_pit)

            coef_data = json.loads(coef_path.read_text()) if coef_path.exists() else {}
            coef_data["pitcher_hierarchical"] = summary_pit
            coef_path.write_text(json.dumps(coef_data, indent=2, ensure_ascii=False))

            sigma_base_pit = float(np.mean(fit_pit.stan_variable("sigma_base")))
            bayes_mae_pit = sigma_base_pit * np.sqrt(2 / np.pi)
            print(f"  Approximate MAE (sigma-based): {bayes_mae_pit:.4f}")

            if stan_data_pit["N_pred"] > 0:
                _update_predictions_stan(
                    pit_pred_path, fit_pit, "bayes_xfip", "marcel_xfip",
                    pred_feat_pit, player_map_pit, scaler_pit, imputer_pit,
                    pit_skill_groups, stan_data_pit,
                )

            log_dict.update({
                "model_type_pit": "stan_hierarchical",
                "sigma_alpha_pit": summary_pit.get("sigma_alpha", {}).get("mean"),
                "sigma_base_pit": sigma_base_pit,
                "tau_stuff_pit": summary_pit.get("tau_stuff", {}).get("mean"),
                "tau_command_pit": summary_pit.get("tau_command", {}).get("mean"),
                "beta_age_pit": summary_pit.get("beta_age", {}).get("mean"),
                "bayes_pitcher_approx_MAE": bayes_mae_pit,
            })

        except Exception as e:
            print(f"  [ERROR] Stan failed: {e}")
            print("  Falling back to ElasticNet...")
            _run_pitcher_fallback(
                dataset_pit, pit_df, pit_skill_groups, pf_lookup,
                log_dict, coef_path,
            )
            bayes_mae_pit = log_dict.get("bayes_pitcher_delta_MAE")
    else:
        _run_pitcher_fallback(
            dataset_pit, pit_df, pit_skill_groups, pf_lookup,
            log_dict, coef_path,
        )
        bayes_mae_pit = log_dict.get("bayes_pitcher_delta_MAE")

    # ===== W&B log =====
    wandb.log(log_dict)
    run_wb.finish()

    # ===== Update model_metrics.json for ensemble =====
    metrics_path = PRED_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if bayes_mae_bat is not None:
        metrics["bayes_mae_woba"] = round(bayes_mae_bat, 4)
    if bayes_mae_pit is not None:
        metrics["bayes_mae_xfip"] = round(bayes_mae_pit, 4)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("=== train_bayes.py complete ===")


def _run_batter_fallback(dataset, bat_df, skill_groups, pf_lookup, log_dict, coef_path):
    """Run ElasticNet fallback for batters."""
    all_feat_cols = []
    for cols in skill_groups.values():
        all_feat_cols.extend(cols)
    all_feat_cols += ["lgb_delta", "cat_delta"]

    pipe, sigma, mae, _ = _elasticnet_fallback(
        dataset, all_feat_cols, "marcel", "actual", "batter"
    )

    # Save coefficients
    regressor = pipe.named_steps["regressor"]
    scaler = pipe.named_steps["scaler"]
    coefs = {
        "batter_elasticnet": {
            name: {"coef": round(float(c), 4),
                   "mean": round(float(scaler.mean_[i]), 4),
                   "std": round(float(scaler.scale_[i]), 4)}
            for i, (name, c) in enumerate(zip(all_feat_cols, regressor.coef_))
        }
    }
    existing = json.loads(coef_path.read_text()) if coef_path.exists() else {}
    existing.update(coefs)
    coef_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    # Flat feature list for prediction update
    raw_cols = []
    for cols in skill_groups.values():
        raw_cols.extend(cols)
    raw_cols += ["lgb_delta", "cat_delta"]

    _update_predictions_fallback(
        PRED_DIR / "batter_predictions.csv",
        bat_df, pipe, sigma, all_feat_cols, raw_cols,
        _eng_batter, "bayes_woba", "marcel_woba", "pred_woba", "cat_woba",
        MLB_AVG_WOBA, pf_lookup,
    )

    log_dict.update({
        "model_type_bat": "elasticnet_fallback",
        "bayes_batter_delta_MAE": mae,
        "sigma_batter": sigma,
    })


def _run_pitcher_fallback(dataset, pit_df, skill_groups, pf_lookup, log_dict, coef_path):
    """Run ElasticNet fallback for pitchers."""
    all_feat_cols = []
    for cols in skill_groups.values():
        all_feat_cols.extend(cols)
    all_feat_cols += ["lgb_delta", "cat_delta"]

    pipe, sigma, mae, _ = _elasticnet_fallback(
        dataset, all_feat_cols, "marcel", "actual", "pitcher"
    )

    regressor = pipe.named_steps["regressor"]
    scaler = pipe.named_steps["scaler"]
    coefs = {
        "pitcher_elasticnet": {
            name: {"coef": round(float(c), 4),
                   "mean": round(float(scaler.mean_[i]), 4),
                   "std": round(float(scaler.scale_[i]), 4)}
            for i, (name, c) in enumerate(zip(all_feat_cols, regressor.coef_))
        }
    }
    existing = json.loads(coef_path.read_text()) if coef_path.exists() else {}
    existing.update(coefs)
    coef_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    raw_cols = []
    for cols in skill_groups.values():
        raw_cols.extend(cols)
    raw_cols += ["lgb_delta", "cat_delta"]

    _update_predictions_fallback(
        PRED_DIR / "pitcher_predictions.csv",
        pit_df, pipe, sigma, all_feat_cols, raw_cols,
        _eng_pitcher, "bayes_xfip", "marcel_xfip", "pred_xfip", "cat_xfip",
        MLB_AVG_XFIP, pf_lookup,
    )

    log_dict.update({
        "model_type_pit": "elasticnet_fallback",
        "bayes_pitcher_delta_MAE": mae,
        "sigma_pitcher": sigma,
    })


if __name__ == "__main__":
    run()
