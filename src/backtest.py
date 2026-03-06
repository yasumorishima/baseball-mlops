"""
Year-by-year backtest with strict 2025 holdout.

Design:
  - Optuna hyperparameter tuning uses ONLY data < 2025
  - 2025 is a TRUE holdout: never seen during tuning or CV
  - Expanding-window CV within 2020-2024 for year-by-year MAE
  - Final model trained on all data < 2025, evaluated once on 2025
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from train import (
    build_train_data_batters,
    build_train_data_pitchers,
    tune_hyperparams,
    train_model,
    MLB_AVG_WOBA,
    MLB_AVG_XFIP,
    OPTUNA_TRIALS,
    RAW_DIR,
    PRED_DIR,
)

BACKTEST_DIR = PRED_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_YEAR = 2025


def _yearly_mae(oof: np.ndarray, actual: pd.Series, marcel: pd.Series,
                seasons: np.ndarray, avg_val: float) -> pd.DataFrame:
    rows = []
    for year in sorted(np.unique(seasons)):
        mask = (seasons == year) & ~np.isnan(oof)
        if mask.sum() == 0:
            continue
        y_true = actual.values[mask]
        y_ml = oof[mask]
        y_mar = marcel.fillna(avg_val).values[mask]
        ml_mae = mean_absolute_error(y_true, y_ml)
        mar_mae = mean_absolute_error(y_true, y_mar)
        rows.append({
            "year": int(year),
            "n_players": int(mask.sum()),
            "ml_mae": round(ml_mae, 4),
            "marcel_mae": round(mar_mae, 4),
            "ml_wins": ml_mae < mar_mae,
            "improvement_pct": round((mar_mae - ml_mae) / mar_mae * 100, 1),
        })
    return pd.DataFrame(rows)


def _outlier_analysis(oof: np.ndarray, actual: pd.Series, marcel: pd.Series,
                      players: pd.Series, seasons: np.ndarray,
                      avg_val: float, top_n: int = 30) -> pd.DataFrame:
    valid = ~np.isnan(oof)
    df = pd.DataFrame({
        "player": players.values[valid],
        "season": seasons[valid].astype(int),
        "actual": actual.values[valid],
        "ml_pred": oof[valid],
        "marcel_pred": marcel.fillna(avg_val).values[valid],
    })
    df["ml_error"] = df["actual"] - df["ml_pred"]
    df["ml_abs_error"] = df["ml_error"].abs()
    df["marcel_error"] = df["actual"] - df["marcel_pred"]
    df["marcel_abs_error"] = df["marcel_error"].abs()
    df["ml_worse_than_marcel"] = df["ml_abs_error"] > df["marcel_abs_error"]
    return df.nlargest(top_n, "ml_abs_error").round(4).reset_index(drop=True)


def _era_split(yearly_df: pd.DataFrame) -> dict:
    pre = yearly_df[yearly_df["year"] < 2023]
    post = yearly_df[yearly_df["year"] >= 2023]
    result = {}
    for label, subset in [("pre_2023", pre), ("post_2023", post)]:
        if subset.empty:
            continue
        total_n = subset["n_players"].sum()
        ml_weighted = (subset["ml_mae"] * subset["n_players"]).sum() / total_n
        mar_weighted = (subset["marcel_mae"] * subset["n_players"]).sum() / total_n
        result[label] = {
            "years": subset["year"].tolist(),
            "n_players_total": int(total_n),
            "ml_mae_weighted": round(ml_weighted, 4),
            "marcel_mae_weighted": round(mar_weighted, 4),
            "ml_wins_all_years": bool(subset["ml_wins"].all()),
            "improvement_pct": round((mar_weighted - ml_weighted) / mar_weighted * 100, 1),
        }
    return result


def _run_one(label: str, full_df: pd.DataFrame, build_fn, target_col: str,
             marcel_col: str, avg_val: float) -> dict:
    print(f"\n=== {label} ===")
    train_all = build_fn(full_df).dropna(subset=[target_col])

    feat_cols = [c for c in train_all.columns
                 if c not in ("player", "season", target_col)]
    seasons_all = train_all["season"].values

    # --- Split: CV data (< HOLDOUT_YEAR) vs holdout (== HOLDOUT_YEAR) ---
    cv_mask = seasons_all < HOLDOUT_YEAR
    ho_mask = seasons_all == HOLDOUT_YEAR

    X_cv = train_all.loc[cv_mask, feat_cols].apply(pd.to_numeric, errors="coerce")
    y_cv = train_all.loc[cv_mask, target_col]
    seasons_cv = seasons_all[cv_mask]

    X_ho = train_all.loc[ho_mask, feat_cols].apply(pd.to_numeric, errors="coerce")
    y_ho = train_all.loc[ho_mask, target_col]

    print(f"  CV data: {len(y_cv)} samples (< {HOLDOUT_YEAR})")
    print(f"  Holdout: {len(y_ho)} samples (== {HOLDOUT_YEAR})")

    # --- Optuna tuning on CV data ONLY ---
    print(f"  Optuna tuning ({OPTUNA_TRIALS} trials, CV data only) ...")
    best_params = tune_hyperparams(X_cv, y_cv, seasons_cv, n_trials=OPTUNA_TRIALS)
    print(f"  best: lr={best_params['learning_rate']:.4f}, "
          f"leaves={best_params['num_leaves']}, "
          f"min_child={best_params['min_child_samples']}")

    # --- CV OOF predictions (2020-2024) ---
    _, _, _, oof_cv = train_model(X_cv, y_cv, best_params, seasons=seasons_cv)

    yearly_cv = _yearly_mae(oof_cv, y_cv, train_all.loc[cv_mask, marcel_col],
                            seasons_cv, avg_val)
    yearly_cv.to_csv(BACKTEST_DIR / f"yearly_mae_{label.lower()}.csv", index=False)

    print(f"\n  [CV Year-by-Year MAE (tuned on < {HOLDOUT_YEAR})]")
    for _, r in yearly_cv.iterrows():
        win = "ML" if r["ml_wins"] else "Marcel"
        print(f"    {int(r['year'])}: ML={r['ml_mae']:.4f}  Marcel={r['marcel_mae']:.4f}"
              f"  ({win} wins, {r['improvement_pct']:+.1f}%)")

    era_split = _era_split(yearly_cv)
    print(f"\n  [Pre/Post 2023 Split]")
    for k, v in era_split.items():
        print(f"    {k}: ML={v['ml_mae_weighted']:.4f}  Marcel={v['marcel_mae_weighted']:.4f}"
              f"  (improvement {v['improvement_pct']:+.1f}%)")

    # --- TRUE HOLDOUT: train on all CV data, predict holdout ---
    print(f"\n  [True Holdout {HOLDOUT_YEAR} (never seen during Optuna)]")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_cv, y_cv)
    ho_preds = final_model.predict(X_ho)

    ho_ml_mae = mean_absolute_error(y_ho, ho_preds)
    ho_marcel = train_all.loc[ho_mask, marcel_col].fillna(avg_val)
    ho_mar_mae = mean_absolute_error(y_ho, ho_marcel)
    ho_ml_wins = ho_ml_mae < ho_mar_mae
    ho_improvement = round((ho_mar_mae - ho_ml_mae) / ho_mar_mae * 100, 1)

    print(f"    ML  MAE: {ho_ml_mae:.4f}")
    print(f"    Marcel MAE: {ho_mar_mae:.4f}")
    print(f"    {'ML wins' if ho_ml_wins else 'Marcel wins'} ({ho_improvement:+.1f}%)")

    holdout_result = {
        "year": HOLDOUT_YEAR,
        "n_players": int(len(y_ho)),
        "ml_mae": round(ho_ml_mae, 4),
        "marcel_mae": round(ho_mar_mae, 4),
        "ml_wins": ho_ml_wins,
        "improvement_pct": ho_improvement,
        "is_true_holdout": True,
    }

    # --- Outliers (CV + holdout) ---
    oof_all = np.full(len(train_all), np.nan)
    oof_all[np.where(cv_mask)[0]] = oof_cv
    oof_all[np.where(ho_mask)[0]] = ho_preds

    outliers = _outlier_analysis(oof_all, train_all[target_col],
                                 train_all[marcel_col],
                                 train_all["player"], seasons_all, avg_val)
    outliers.to_csv(BACKTEST_DIR / f"outliers_{label.lower()}.csv", index=False)

    print(f"\n  [Top 10 Outliers]")
    for _, r in outliers.head(10).iterrows():
        worse = " (ML worse)" if r["ml_worse_than_marcel"] else ""
        fmt = ".3f" if "woba" in target_col.lower() else ".2f"
        print(f"    {r['player']} ({int(r['season'])}): actual={r['actual']:{fmt}}"
              f"  ML={r['ml_pred']:{fmt}}  Marcel={r['marcel_pred']:{fmt}}"
              f"  err={r['ml_error']:+{fmt}}{worse}")

    return {
        "cv_yearly": yearly_cv.to_dict(orient="records"),
        "era_split": era_split,
        "holdout": holdout_result,
        "cv_ml_wins_all_years": bool(yearly_cv["ml_wins"].all()),
        "cv_total_years": len(yearly_cv),
        "cv_ml_win_years": int(yearly_cv["ml_wins"].sum()),
    }


def run_backtest():
    print("=" * 60)
    print("BACKTEST: Strict Holdout Design")
    print(f"  Optuna + CV: all years < {HOLDOUT_YEAR}")
    print(f"  True holdout: {HOLDOUT_YEAR}")
    print("=" * 60)

    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")

    bat_result = _run_one("Batter", bat_df, build_train_data_batters,
                          "target_woba", "marcel_woba", MLB_AVG_WOBA)
    pit_result = _run_one("Pitcher", pit_df, build_train_data_pitchers,
                          "target_xfip", "marcel_xfip", MLB_AVG_XFIP)

    summary = {"batter": bat_result, "pitcher": pit_result}
    (BACKTEST_DIR / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    bat_cv = f"{bat_result['cv_ml_win_years']}/{bat_result['cv_total_years']}"
    pit_cv = f"{pit_result['cv_ml_win_years']}/{pit_result['cv_total_years']}"
    bat_ho = "ML" if bat_result["holdout"]["ml_wins"] else "Marcel"
    pit_ho = "ML" if pit_result["holdout"]["ml_wins"] else "Marcel"
    print(f"CV (< {HOLDOUT_YEAR}): Batter ML wins {bat_cv}, Pitcher ML wins {pit_cv}")
    print(f"HOLDOUT ({HOLDOUT_YEAR}): Batter={bat_ho} ({bat_result['holdout']['improvement_pct']:+.1f}%), "
          f"Pitcher={pit_ho} ({pit_result['holdout']['improvement_pct']:+.1f}%)")
    print(f"Results saved to {BACKTEST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_backtest()
