"""
CatBoost 学習 + W&B 記録 (v8)

LightGBM とは異なる分割戦略を持つ CatBoost をアンサンブル多様性のために追加。
同じ特徴量・CV戦略を使い、OOF予測をスタッキング用に出力する。
Optuna 200 trials + MedianPruner でハイパーパラメータ最適化（RPi5で約1h）。
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import wandb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    build_train_data_batters, build_train_data_pitchers,
    MLB_AVG_WOBA, MLB_AVG_XFIP, _time_cv_splits,
    RAW_DIR, PRED_DIR, MODELS_DIR, _predict_next_season,
    marcel_woba, marcel_xfip,
    _bat_delta_features, _pit_delta_features,
    OPTUNA_TRIALS,
)

CATBOOST_TRIALS = 200  # LGBの1/5（CatBoostは1trial≒18s、200trials≒1h）
_EARLY_STOPPING = 50
RECENCY_DECAY = 0.85


def _recency_weights(seasons: np.ndarray) -> np.ndarray:
    max_s = seasons.max()
    return np.array([RECENCY_DECAY ** (max_s - s) for s in seasons])


def _cv_mae_cat(params: dict, X: pd.DataFrame, y: pd.Series,
                seasons: np.ndarray,
                trial: optuna.Trial | None = None) -> float:
    """時系列 CV で OOF MAE を計算（trial渡しでpruning対応）"""
    splits = _time_cv_splits(seasons)
    weights = _recency_weights(seasons)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        pool_tr = Pool(X.iloc[tr_idx], y.iloc[tr_idx], weight=weights[tr_idx])
        pool_va = Pool(X.iloc[va_idx], y.iloc[va_idx])
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=_EARLY_STOPPING)
        oof[va_idx] = model.predict(X.iloc[va_idx])
        # fold完了ごとに中間MAEを報告→MedianPrunerが劣悪trialを早期打切り
        if trial is not None:
            valid_so_far = ~np.isnan(oof)
            trial.report(float(mean_absolute_error(y[valid_so_far], oof[valid_so_far])), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    valid = ~np.isnan(oof)
    return float(mean_absolute_error(y[valid], oof[valid]))


def tune_catboost(X: pd.DataFrame, y: pd.Series, seasons: np.ndarray,
                  n_trials: int = CATBOOST_TRIALS) -> dict:
    """Optuna で CatBoost ハイパーパラメータを最適化"""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "MAE",
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
        }
        return _cv_mae_cat(params, X, y, seasons, trial=trial)

    sampler = optuna.samplers.TPESampler(n_startup_trials=15, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "loss_function": "MAE",
        "iterations": 1000,
        "random_seed": 42,
    })
    return best


def train_catboost(X: pd.DataFrame, y: pd.Series, params: dict,
                   seasons: np.ndarray) -> tuple:
    """時系列 CV で CatBoost を学習し MAE・最終モデル・OOF を返す。"""
    splits = _time_cv_splits(seasons)
    weights = _recency_weights(seasons)
    oof = np.full(len(y), np.nan)
    models = []

    for tr_idx, va_idx in splits:
        pool_tr = Pool(X.iloc[tr_idx], y.iloc[tr_idx], weight=weights[tr_idx])
        pool_va = Pool(X.iloc[va_idx], y.iloc[va_idx])
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=_EARLY_STOPPING)
        oof[va_idx] = model.predict(X.iloc[va_idx])
        models.append(model)

    valid = ~np.isnan(oof)
    mae = mean_absolute_error(y[valid], oof[valid])

    # 全データで再学習
    pool_all = Pool(X, y, weight=weights)
    final = CatBoostRegressor(**params, verbose=0)
    final.fit(pool_all)
    return final, mae, models, oof


def save_catboost_to_wandb(model, mae: float, target: str, feature_names: list, config: dict):
    """CatBoost モデルを W&B に記録"""
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run = wandb.init(project="baseball-mlops", entity=entity, job_type="train_catboost",
                     config={**config, "target": target, "model": "catboost"})

    wandb.log({f"catboost_MAE_{target}": mae})

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    wandb.log({f"catboost_feature_importance_{target}": wandb.Table(dataframe=importance.head(20))})

    # モデル保存
    artifact = wandb.Artifact(f"catboost-{target}-model", type="model",
                               description=f"CatBoost {target} predictor, MAE={mae:.4f}",
                               metadata={"mae": mae})
    model_path = MODELS_DIR / f"catboost_{target}_model.pkl"
    joblib.dump(model, model_path)
    artifact.add_file(str(model_path))
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()


def run_catboost_training():
    """CatBoost 打者・投手モデルを学習して W&B に記録"""
    # 打者
    print("=== CatBoost Batter (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    train_bat = build_train_data_batters(bat_df)
    train_bat = train_bat.dropna(subset=["target_woba"])

    feat_cols_bat = [c for c in train_bat.columns
                     if c not in ("player", "season", "target_woba")]
    X_bat = train_bat[feat_cols_bat].apply(pd.to_numeric, errors="coerce")
    y_bat = train_bat["target_woba"]
    seasons_bat = train_bat["season"].values

    print(f"  Optuna tuning ({CATBOOST_TRIALS} trials) ...")
    best_params_bat = tune_catboost(X_bat, y_bat, seasons_bat, n_trials=CATBOOST_TRIALS)
    print(f"  best: lr={best_params_bat['learning_rate']:.4f}, "
          f"depth={best_params_bat['depth']}")

    model_bat, mae_bat, _, oof_bat = train_catboost(
        X_bat, y_bat, best_params_bat, seasons=seasons_bat
    )
    print(f"  CatBoost MAE wOBA: {mae_bat:.4f}")

    # OOF 保存
    oof_mask_bat = ~np.isnan(oof_bat)
    pd.DataFrame({
        "player": train_bat["player"].values[oof_mask_bat],
        "season": train_bat["season"].values[oof_mask_bat],
        "cat_woba_oof": oof_bat[oof_mask_bat],
    }).to_csv(RAW_DIR / "cat_oof_batter.csv", index=False)

    save_catboost_to_wandb(model_bat, mae_bat, "woba", feat_cols_bat,
                           {f"cat_{k}": v for k, v in best_params_bat.items()
                            if k in ("learning_rate", "depth", "l2_leaf_reg")})

    # 投手
    print("=== CatBoost Pitcher (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    train_pit = build_train_data_pitchers(pit_df)
    train_pit = train_pit.dropna(subset=["target_xfip"])

    feat_cols_pit = [c for c in train_pit.columns
                     if c not in ("player", "season", "target_xfip")]
    X_pit = train_pit[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
    y_pit = train_pit["target_xfip"]
    seasons_pit = train_pit["season"].values

    print(f"  Optuna tuning ({CATBOOST_TRIALS} trials) ...")
    best_params_pit = tune_catboost(X_pit, y_pit, seasons_pit, n_trials=CATBOOST_TRIALS)

    model_pit, mae_pit, _, oof_pit = train_catboost(
        X_pit, y_pit, best_params_pit, seasons=seasons_pit
    )
    print(f"  CatBoost MAE xFIP: {mae_pit:.4f}")

    # OOF 保存
    oof_mask_pit = ~np.isnan(oof_pit)
    pd.DataFrame({
        "player": train_pit["player"].values[oof_mask_pit],
        "season": train_pit["season"].values[oof_mask_pit],
        "cat_xfip_oof": oof_pit[oof_mask_pit],
    }).to_csv(RAW_DIR / "cat_oof_pitcher.csv", index=False)

    save_catboost_to_wandb(model_pit, mae_pit, "xfip", feat_cols_pit,
                           {f"cat_{k}": v for k, v in best_params_pit.items()
                            if k in ("learning_rate", "depth", "l2_leaf_reg")})

    # MAE をメトリクスに追記
    metrics_path = PRED_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({
        "cat_mae_woba": round(mae_bat, 4),
        "cat_mae_xfip": round(mae_pit, 4),
    })
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # 予測結果を predictions CSV に追記
    _update_catboost_predictions(model_bat, bat_df, feat_cols_bat, "woba", "batter")
    _update_catboost_predictions(model_pit, pit_df, feat_cols_pit, "xfip", "pitcher")

    print("=== train_catboost.py complete ===")


def _update_catboost_predictions(model, full_df, feat_cols, target_name, kind):
    """CatBoost predictions を CSV に追記"""
    pred_path = PRED_DIR / f"{kind}_predictions.csv"
    if not pred_path.exists():
        return
    preds = pd.read_csv(pred_path)
    col_name = f"cat_{target_name}"

    latest_season = full_df["season"].max()
    next_year = latest_season + 1

    cat_preds = []
    for _, row in preds.iterrows():
        player = row["player"]
        season_last = int(row.get("season_last", latest_season))
        feats = {}

        from train import BATTER_FEATURES, PITCHER_FEATURES, _get_park_factors
        BASE_FEATURES = BATTER_FEATURES if kind == "batter" else PITCHER_FEATURES
        delta_fn = _bat_delta_features if kind == "batter" else _pit_delta_features
        pf = _get_park_factors()

        teams_by_lag = {}
        for lag in range(1, 4):
            y = next_year - 1 - lag
            prev = full_df[(full_df["player"] == player) & (full_df["season"] == y)]
            suffix = f"_y{lag}"
            if lag == 1:
                cur = full_df[(full_df["player"] == player) & (full_df["season"] == season_last)]
                if len(cur) > 0:
                    for f in BASE_FEATURES:
                        feats[f"_y1"] = cur.iloc[0].get(f, np.nan)
                    teams_by_lag[1] = str(cur.iloc[0].get("Team", ""))
                if len(prev) > 0:
                    teams_by_lag[2] = str(prev.iloc[0].get("Team", ""))
            for f in [c for c in feat_cols if c.endswith(suffix)]:
                base = f.replace(suffix, "")
                feats[f] = prev.iloc[0].get(base, np.nan) if len(prev) > 0 else np.nan
            if len(prev) > 0:
                teams_by_lag[lag] = str(prev.iloc[0].get("Team", ""))

        # Build feature vector using same columns as training
        feat_df = pd.DataFrame([feats])[feat_cols].apply(pd.to_numeric, errors="coerce") if all(c in feats for c in feat_cols[:3]) else None

        if feat_df is not None:
            try:
                cat_preds.append(round(float(model.predict(feat_df)[0]), 3))
            except Exception:
                cat_preds.append(np.nan)
        else:
            cat_preds.append(np.nan)

    preds[col_name] = cat_preds
    preds.to_csv(pred_path, index=False)
    print(f"  {kind} CatBoost predictions updated: {pred_path}")


if __name__ == "__main__":
    run_catboost_training()
