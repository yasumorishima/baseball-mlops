"""
年度別バックテスト分析

1. 年度別安定性: ML vs Marcel の MAE を各年で比較
2. 2023年以降の精度維持: ルール変更(ピッチクロック/シフト制限等)の影響
3. 大外れ分析: 予測が大きく外れた選手をリストアップ
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def _yearly_mae(oof: np.ndarray, actual: pd.Series, marcel: pd.Series,
                seasons: np.ndarray, avg_val: float) -> pd.DataFrame:
    """年度ごとに ML MAE と Marcel MAE を計算"""
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
    """予測が大きく外れた選手をリストアップ"""
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
    # ML がより大きく外した選手を優先的に表示
    df["ml_worse_than_marcel"] = df["ml_abs_error"] > df["marcel_abs_error"]
    return df.nlargest(top_n, "ml_abs_error").round(4).reset_index(drop=True)


def _era_split(yearly_df: pd.DataFrame) -> dict:
    """2023年前後で精度を比較（ルール変更影響の定量化）"""
    pre = yearly_df[yearly_df["year"] < 2023]
    post = yearly_df[yearly_df["year"] >= 2023]
    result = {}
    for label, subset in [("pre_2023", pre), ("post_2023", post)]:
        if subset.empty:
            continue
        total_n = subset["n_players"].sum()
        # 選手数加重平均 MAE
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


def run_backtest():
    print("=" * 60)
    print("BACKTEST: Year-by-Year Stability Analysis")
    print("=" * 60)

    # --- 打者 ---
    print("\n=== Batter (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    train_bat = build_train_data_batters(bat_df).dropna(subset=["target_woba"])

    feat_cols_bat = [c for c in train_bat.columns
                     if c not in ("player", "season", "target_woba")]
    X_bat = train_bat[feat_cols_bat].apply(pd.to_numeric, errors="coerce")
    y_bat = train_bat["target_woba"]
    seasons_bat = train_bat["season"].values

    print(f"  Optuna tuning ({OPTUNA_TRIALS} trials) ...")
    best_bat = tune_hyperparams(X_bat, y_bat, seasons_bat, n_trials=OPTUNA_TRIALS)
    _, _, _, oof_bat = train_model(X_bat, y_bat, best_bat, seasons=seasons_bat)

    yearly_bat = _yearly_mae(oof_bat, y_bat, train_bat["marcel_woba"],
                             seasons_bat, MLB_AVG_WOBA)
    yearly_bat.to_csv(BACKTEST_DIR / "yearly_mae_batter.csv", index=False)

    print("\n  [Year-by-Year wOBA MAE]")
    for _, r in yearly_bat.iterrows():
        win = "ML" if r["ml_wins"] else "Marcel"
        print(f"    {int(r['year'])}: ML={r['ml_mae']:.4f}  Marcel={r['marcel_mae']:.4f}"
              f"  ({win} wins, {r['improvement_pct']:+.1f}%)")

    era_bat = _era_split(yearly_bat)
    print(f"\n  [Pre/Post 2023 Split]")
    for k, v in era_bat.items():
        print(f"    {k}: ML={v['ml_mae_weighted']:.4f}  Marcel={v['marcel_mae_weighted']:.4f}"
              f"  (improvement {v['improvement_pct']:+.1f}%)")

    outliers_bat = _outlier_analysis(oof_bat, y_bat, train_bat["marcel_woba"],
                                     train_bat["player"], seasons_bat, MLB_AVG_WOBA)
    outliers_bat.to_csv(BACKTEST_DIR / "outliers_batter.csv", index=False)
    print(f"\n  [Top 10 Outliers - Batter]")
    for _, r in outliers_bat.head(10).iterrows():
        worse = " (ML worse)" if r["ml_worse_than_marcel"] else ""
        print(f"    {r['player']} ({int(r['season'])}): actual={r['actual']:.3f}"
              f"  ML={r['ml_pred']:.3f}  Marcel={r['marcel_pred']:.3f}"
              f"  err={r['ml_error']:+.3f}{worse}")

    # --- 投手 ---
    print("\n=== Pitcher (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    train_pit = build_train_data_pitchers(pit_df).dropna(subset=["target_xfip"])

    feat_cols_pit = [c for c in train_pit.columns
                     if c not in ("player", "season", "target_xfip")]
    X_pit = train_pit[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
    y_pit = train_pit["target_xfip"]
    seasons_pit = train_pit["season"].values

    print(f"  Optuna tuning ({OPTUNA_TRIALS} trials) ...")
    best_pit = tune_hyperparams(X_pit, y_pit, seasons_pit, n_trials=OPTUNA_TRIALS)
    _, _, _, oof_pit = train_model(X_pit, y_pit, best_pit, seasons=seasons_pit)

    yearly_pit = _yearly_mae(oof_pit, y_pit, train_pit["marcel_xfip"],
                             seasons_pit, MLB_AVG_XFIP)
    yearly_pit.to_csv(BACKTEST_DIR / "yearly_mae_pitcher.csv", index=False)

    print("\n  [Year-by-Year xFIP MAE]")
    for _, r in yearly_pit.iterrows():
        win = "ML" if r["ml_wins"] else "Marcel"
        print(f"    {int(r['year'])}: ML={r['ml_mae']:.4f}  Marcel={r['marcel_mae']:.4f}"
              f"  ({win} wins, {r['improvement_pct']:+.1f}%)")

    era_pit = _era_split(yearly_pit)
    print(f"\n  [Pre/Post 2023 Split]")
    for k, v in era_pit.items():
        print(f"    {k}: ML={v['ml_mae_weighted']:.4f}  Marcel={v['marcel_mae_weighted']:.4f}"
              f"  (improvement {v['improvement_pct']:+.1f}%)")

    outliers_pit = _outlier_analysis(oof_pit, y_pit, train_pit["marcel_xfip"],
                                     train_pit["player"], seasons_pit, MLB_AVG_XFIP)
    outliers_pit.to_csv(BACKTEST_DIR / "outliers_pitcher.csv", index=False)
    print(f"\n  [Top 10 Outliers - Pitcher]")
    for _, r in outliers_pit.head(10).iterrows():
        worse = " (ML worse)" if r["ml_worse_than_marcel"] else ""
        print(f"    {r['player']} ({int(r['season'])}): actual={r['actual']:.2f}"
              f"  ML={r['ml_pred']:.2f}  Marcel={r['marcel_pred']:.2f}"
              f"  err={r['ml_error']:+.2f}{worse}")

    # --- サマリー JSON ---
    summary = {
        "batter": {
            "yearly": yearly_bat.to_dict(orient="records"),
            "era_split": era_bat,
            "ml_wins_all_years": bool(yearly_bat["ml_wins"].all()),
            "total_years": len(yearly_bat),
            "ml_win_years": int(yearly_bat["ml_wins"].sum()),
        },
        "pitcher": {
            "yearly": yearly_pit.to_dict(orient="records"),
            "era_split": era_pit,
            "ml_wins_all_years": bool(yearly_pit["ml_wins"].all()),
            "total_years": len(yearly_pit),
            "ml_win_years": int(yearly_pit["ml_wins"].sum()),
        },
    }
    (BACKTEST_DIR / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    bat_wins = f"{summary['batter']['ml_win_years']}/{summary['batter']['total_years']}"
    pit_wins = f"{summary['pitcher']['ml_win_years']}/{summary['pitcher']['total_years']}"
    print(f"SUMMARY: Batter ML wins {bat_wins} years, Pitcher ML wins {pit_wins} years")
    print(f"Results saved to {BACKTEST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_backtest()
