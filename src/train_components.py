"""
Component-level prediction (PECOTA approach, v9)

Instead of predicting wOBA directly, predict K%, BB%, BABIP, ISO separately
and reconstruct wOBA from components. Different components have different
predictability: K% is very stable, BABIP is noisy.

For pitchers, predict K%, BB%, HR/9 → reconstruct xFIP.

xFIP formula: xFIP = ((13 × (league_HR/FB × FB)) + (3 × BB) - (2 × K)) / IP + FIP_constant
Simplified: we predict components and use a regression to reconstruct xFIP.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import wandb
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    build_train_data_batters, build_train_data_pitchers,
    MLB_AVG_WOBA, MLB_AVG_XFIP, _time_cv_splits,
    RAW_DIR, PRED_DIR,
)

# Component targets for batters
BATTER_COMPONENTS = ["K%", "BB%", "BABIP", "ISO"]
# Component targets for pitchers
PITCHER_COMPONENTS = ["K%", "BB%", "HR/9"]

COMPONENT_OPTUNA_TRIALS = 40  # Per component (7 models × 40 = 280 total, ARM64 optimized)
_EARLY_STOPPING = 50
RECENCY_DECAY = 0.85


def _log_elapsed(label: str, start: float, budget_min: int = 60):
    """経過時間をログし、budget の 80% 超過で警告"""
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  ⚠️ WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) — timeout risk!")


def _recency_weights(seasons: np.ndarray) -> np.ndarray:
    max_s = seasons.max()
    return np.array([RECENCY_DECAY ** (max_s - s) for s in seasons])


def _tune_component(X, y, seasons, n_trials=COMPONENT_OPTUNA_TRIALS):
    """LightGBM hyperparameter tuning for a single component"""
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        }
        splits = _time_cv_splits(seasons)
        weights = _recency_weights(seasons)
        oof = np.full(len(y), np.nan)
        for tr_idx, va_idx in splits:
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      sample_weight=weights[tr_idx],
                      eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                                 lgb.log_evaluation(-1)])
            oof[va_idx] = model.predict(X.iloc[va_idx])
        valid = ~np.isnan(oof)
        return float(mean_absolute_error(y[valid], oof[valid]))

    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"objective": "regression", "metric": "mae", "verbosity": -1,
                 "n_estimators": 500, "bagging_freq": 5})
    return best


def _train_component(X, y, seasons, params):
    """Train single component model, return OOF predictions"""
    splits = _time_cv_splits(seasons)
    weights = _recency_weights(seasons)
    oof = np.full(len(y), np.nan)

    for tr_idx, va_idx in splits:
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  sample_weight=weights[tr_idx],
                  eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict(X.iloc[va_idx])

    # Final model on all data
    final = lgb.LGBMRegressor(**params)
    final.fit(X, y, sample_weight=weights)

    valid = ~np.isnan(oof)
    mae = mean_absolute_error(y[valid], oof[valid])
    return final, mae, oof


def _reconstruct_woba(k_pct, bb_pct, babip, iso):
    """
    Approximate wOBA from components using linear relationship:
    wOBA ≈ c0 + c1*BB% + c2*(1-K%)*BABIP + c3*ISO

    Coefficients estimated from MLB 2015-2025 data:
    Higher BB% → higher wOBA (walks are value)
    Lower K% → more balls in play → BABIP matters more
    Higher ISO → more extra bases → higher wOBA
    """
    # Linear approximation (fitted on historical MLB data)
    # These are approximate coefficients; the regression model will learn exact weights
    return 0.180 + 0.50 * bb_pct + 0.42 * (1 - k_pct) * babip + 0.65 * iso


def _reconstruct_xfip(k_pct, bb_pct, hr9, ip_rate=1.0):
    """
    Approximate xFIP from components.
    xFIP ≈ constant - K_effect + BB_effect + HR_effect
    """
    # FIP constant ≈ 3.10 (varies by year, use average)
    fip_constant = 3.10
    # Approximate: K → reduces xFIP, BB → increases, HR/9 → increases
    return fip_constant - 2.0 * k_pct * 6.5 + 3.0 * bb_pct * 6.5 + 13.0 * hr9 / 9.0


def run_component_prediction():
    """Train component models and generate reconstructed predictions"""
    t0 = time.time()
    print("=" * 60)
    print("COMPONENT-LEVEL PREDICTION (PECOTA approach)")
    print("=" * 60)

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None

    # ===== BATTER COMPONENTS =====
    print("\n=== Batter Components ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    train_bat = build_train_data_batters(bat_df)

    feat_cols_bat = [c for c in train_bat.columns
                     if c not in ("player", "season", "target_woba")]
    X_bat = train_bat[feat_cols_bat].apply(pd.to_numeric, errors="coerce")
    seasons_bat = train_bat["season"].values

    # Build component targets from the raw data
    component_oof_bat = {}
    component_models_bat = {}
    component_mae_bat = {}

    for comp in BATTER_COMPONENTS:
        # Component target: next-year value of this component
        target_col = f"target_{comp.replace('%', '_pct').replace('/', '_')}"

        # Build target via vectorized merge (much faster than iterrows)
        train_comp = train_bat.copy()
        if comp in bat_df.columns:
            lookup = bat_df[["player", "season", comp]].drop_duplicates(
                subset=["player", "season"], keep="first"
            ).rename(columns={comp: target_col})
            train_comp = train_comp.merge(lookup, on=["player", "season"], how="left")
        else:
            train_comp[target_col] = np.nan
        train_comp = train_comp.dropna(subset=[target_col])

        if len(train_comp) < 50:
            print(f"  {comp}: insufficient data ({len(train_comp)}), skipping")
            continue

        X_comp = train_comp[feat_cols_bat].apply(pd.to_numeric, errors="coerce")
        y_comp = train_comp[target_col]
        seasons_comp = train_comp["season"].values

        print(f"  {comp}: tuning ({COMPONENT_OPTUNA_TRIALS} trials, {len(y_comp)} samples) ...")
        best_params = _tune_component(X_comp, y_comp, seasons_comp)
        model, mae, oof = _train_component(X_comp, y_comp, seasons_comp, best_params)

        component_models_bat[comp] = model
        component_mae_bat[comp] = mae
        component_oof_bat[comp] = (train_comp["player"].values, train_comp["season"].values, oof)
        print(f"  {comp}: MAE = {mae:.4f}")
        _log_elapsed(f"bat_{comp}", t0)

    # Reconstruct wOBA from component OOF predictions
    if all(c in component_oof_bat for c in BATTER_COMPONENTS):
        # Align OOF predictions by (player, season)
        base = train_bat.dropna(subset=["target_woba"]).copy()
        for comp in BATTER_COMPONENTS:
            players, seasons, oof = component_oof_bat[comp]
            oof_df = pd.DataFrame({"player": players, "season": seasons,
                                   f"oof_{comp}": oof})
            base = base.merge(oof_df, on=["player", "season"], how="left")

        # Simple reconstruction: linear regression from components → wOBA
        from sklearn.linear_model import Ridge
        comp_cols = [f"oof_{c}" for c in BATTER_COMPONENTS]
        valid = base[comp_cols + ["target_woba"]].dropna()
        X_recon = valid[comp_cols].values
        y_recon = valid["target_woba"].values

        reg = Ridge(alpha=1.0)
        reg.fit(X_recon, y_recon)
        recon_pred = reg.predict(X_recon)
        recon_mae = mean_absolute_error(y_recon, recon_pred)
        print(f"\n  Reconstructed wOBA MAE (in-sample): {recon_mae:.4f}")

        # Time-series CV for true OOF reconstruction MAE
        comp_oof_all = base[comp_cols].values
        target_all = base["target_woba"].values
        seasons_all = base["season"].values
        splits = _time_cv_splits(seasons_all)
        recon_oof = np.full(len(target_all), np.nan)
        for tr_idx, va_idx in splits:
            valid_tr = ~np.any(np.isnan(comp_oof_all[tr_idx]), axis=1)
            valid_va = ~np.any(np.isnan(comp_oof_all[va_idx]), axis=1)
            if valid_tr.sum() < 10 or valid_va.sum() == 0:
                continue
            reg_cv = Ridge(alpha=1.0)
            reg_cv.fit(comp_oof_all[tr_idx][valid_tr], target_all[tr_idx][valid_tr])
            recon_oof[va_idx[valid_va]] = reg_cv.predict(comp_oof_all[va_idx][valid_va])
        valid_recon = ~np.isnan(recon_oof)
        if valid_recon.sum() > 0:
            recon_cv_mae = mean_absolute_error(target_all[valid_recon], recon_oof[valid_recon])
            print(f"  Reconstructed wOBA MAE (CV OOF): {recon_cv_mae:.4f}")
            component_mae_bat["recon_woba"] = recon_cv_mae
        else:
            recon_cv_mae = None

        # Save reconstruction model
        import joblib
        MODELS_DIR = Path(__file__).parent.parent / "models"
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(reg, MODELS_DIR / "component_recon_batter.pkl")
        for comp, model in component_models_bat.items():
            safe_name = comp.replace("%", "pct").replace("/", "_")
            joblib.dump(model, MODELS_DIR / f"component_{safe_name}_batter.pkl")

    _log_elapsed("batter_components_total", t0)

    # ===== PITCHER COMPONENTS =====
    print("\n=== Pitcher Components ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    train_pit = build_train_data_pitchers(pit_df)

    feat_cols_pit = [c for c in train_pit.columns
                     if c not in ("player", "season", "target_xfip")]
    X_pit = train_pit[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
    seasons_pit = train_pit["season"].values

    component_oof_pit = {}
    component_models_pit = {}
    component_mae_pit = {}

    for comp in PITCHER_COMPONENTS:
        target_col = f"target_{comp.replace('%', '_pct').replace('/', '_')}"

        # Build target via vectorized merge (much faster than iterrows)
        train_comp = train_pit.copy()
        if comp in pit_df.columns:
            lookup = pit_df[["player", "season", comp]].drop_duplicates(
                subset=["player", "season"], keep="first"
            ).rename(columns={comp: target_col})
            train_comp = train_comp.merge(lookup, on=["player", "season"], how="left")
        else:
            train_comp[target_col] = np.nan
        train_comp = train_comp.dropna(subset=[target_col])

        if len(train_comp) < 50:
            print(f"  {comp}: insufficient data ({len(train_comp)}), skipping")
            continue

        X_comp = train_comp[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
        y_comp = train_comp[target_col]
        seasons_comp = train_comp["season"].values

        print(f"  {comp}: tuning ({COMPONENT_OPTUNA_TRIALS} trials, {len(y_comp)} samples) ...")
        best_params = _tune_component(X_comp, y_comp, seasons_comp)
        model, mae, oof = _train_component(X_comp, y_comp, seasons_comp, best_params)

        component_models_pit[comp] = model
        component_mae_pit[comp] = mae
        component_oof_pit[comp] = (train_comp["player"].values, train_comp["season"].values, oof)
        print(f"  {comp}: MAE = {mae:.4f}")
        _log_elapsed(f"pit_{comp}", t0)

    # Reconstruct xFIP
    if all(c in component_oof_pit for c in PITCHER_COMPONENTS):
        base = train_pit.dropna(subset=["target_xfip"]).copy()
        for comp in PITCHER_COMPONENTS:
            players, seasons, oof = component_oof_pit[comp]
            oof_df = pd.DataFrame({"player": players, "season": seasons,
                                   f"oof_{comp}": oof})
            base = base.merge(oof_df, on=["player", "season"], how="left")

        from sklearn.linear_model import Ridge
        comp_cols = [f"oof_{c}" for c in PITCHER_COMPONENTS]
        valid = base[comp_cols + ["target_xfip"]].dropna()
        X_recon = valid[comp_cols].values
        y_recon = valid["target_xfip"].values

        reg = Ridge(alpha=1.0)
        reg.fit(X_recon, y_recon)

        # CV OOF reconstruction
        comp_oof_all = base[comp_cols].values
        target_all = base["target_xfip"].values
        seasons_all = base["season"].values
        splits = _time_cv_splits(seasons_all)
        recon_oof = np.full(len(target_all), np.nan)
        for tr_idx, va_idx in splits:
            valid_tr = ~np.any(np.isnan(comp_oof_all[tr_idx]), axis=1)
            valid_va = ~np.any(np.isnan(comp_oof_all[va_idx]), axis=1)
            if valid_tr.sum() < 10 or valid_va.sum() == 0:
                continue
            reg_cv = Ridge(alpha=1.0)
            reg_cv.fit(comp_oof_all[tr_idx][valid_tr], target_all[tr_idx][valid_tr])
            recon_oof[va_idx[valid_va]] = reg_cv.predict(comp_oof_all[va_idx][valid_va])
        valid_recon = ~np.isnan(recon_oof)
        if valid_recon.sum() > 0:
            recon_cv_mae = mean_absolute_error(target_all[valid_recon], recon_oof[valid_recon])
            print(f"\n  Reconstructed xFIP MAE (CV OOF): {recon_cv_mae:.4f}")
            component_mae_pit["recon_xfip"] = recon_cv_mae

        import joblib
        MODELS_DIR = Path(__file__).parent.parent / "models"
        joblib.dump(reg, MODELS_DIR / "component_recon_pitcher.pkl")
        for comp, model in component_models_pit.items():
            safe_name = comp.replace("%", "pct").replace("/", "_")
            joblib.dump(model, MODELS_DIR / f"component_{safe_name}_pitcher.pkl")

    _log_elapsed("pitcher_components_total", t0)

    # ===== W&B Logging =====
    run_wb = wandb.init(project="baseball-mlops", entity=entity, job_type="component_prediction",
                        config={"batter_components": BATTER_COMPONENTS,
                                "pitcher_components": PITCHER_COMPONENTS,
                                "optuna_trials_per_component": COMPONENT_OPTUNA_TRIALS})
    log_dict = {}
    for comp, mae in component_mae_bat.items():
        safe = comp.replace("%", "pct").replace("/", "_")
        log_dict[f"component_bat_{safe}_MAE"] = round(mae, 4)
    for comp, mae in component_mae_pit.items():
        safe = comp.replace("%", "pct").replace("/", "_")
        log_dict[f"component_pit_{safe}_MAE"] = round(mae, 4)
    wandb.log(log_dict)
    run_wb.finish()
    _log_elapsed("wandb_sync", t0)

    # Save component MAEs to model_metrics.json
    metrics_path = PRED_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if "recon_woba" in component_mae_bat:
        metrics["component_mae_woba"] = round(component_mae_bat["recon_woba"], 4)
    if "recon_xfip" in component_mae_pit:
        metrics["component_mae_xfip"] = round(component_mae_pit["recon_xfip"], 4)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Generate component predictions for latest season
    _generate_component_predictions(bat_df, pit_df, component_models_bat,
                                     component_models_pit, feat_cols_bat, feat_cols_pit)

    _log_elapsed("total", t0)
    print("\n=== train_components.py complete ===")


def _generate_component_predictions(bat_df, pit_df, bat_models, pit_models,
                                     feat_cols_bat, feat_cols_pit):
    """Generate component-reconstructed predictions for ensemble"""
    import joblib
    MODELS_DIR = Path(__file__).parent.parent / "models"

    # Batter
    pred_path = PRED_DIR / "batter_predictions.csv"
    if pred_path.exists() and all(c in bat_models for c in BATTER_COMPONENTS):
        preds = pd.read_csv(pred_path)
        recon_model = joblib.load(MODELS_DIR / "component_recon_batter.pkl")

        # We need to predict each component for each player, then reconstruct
        # For now, use component model predictions from the CSV's existing features
        comp_preds = []
        for _, row in preds.iterrows():
            # Component predictions need the feature vector
            # Since we don't have the full feature extraction here,
            # we'll mark this as requiring the full pipeline
            comp_preds.append(np.nan)

        preds["component_woba"] = comp_preds
        preds.to_csv(pred_path, index=False)
        print(f"  Batter component predictions saved (placeholder)")

    # Pitcher
    pred_path = PRED_DIR / "pitcher_predictions.csv"
    if pred_path.exists() and all(c in pit_models for c in PITCHER_COMPONENTS):
        preds = pd.read_csv(pred_path)
        preds["component_xfip"] = np.nan  # Placeholder
        preds.to_csv(pred_path, index=False)
        print(f"  Pitcher component predictions saved (placeholder)")


if __name__ == "__main__":
    run_component_prediction()
