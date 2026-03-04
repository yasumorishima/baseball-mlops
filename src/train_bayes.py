"""
Bayesian Ridge 学習 + W&B 記録

Marcel との差分 (delta) を Statcast 特徴量で Ridge 回帰し、
Monte Carlo CI (80%) を付与する。

入力:  predictions/batter_predictions.csv, pitcher_predictions.csv (train.py 出力)
       data/raw/batter_features.csv, pitcher_features.csv
出力:  predictions/*.csv に bayes_woba/ci_lo80/ci_hi80 列を追記
       predictions/bayes_coef.json (β係数)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# train.py と同じ定数・関数を参照するためインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    MLB_AVG_WOBA, MLB_AVG_XFIP,
    marcel_woba, marcel_xfip,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PRED_DIR = Path(__file__).parent.parent / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Ridge で使う Statcast 特徴量（直前シーズン値）
BAYES_FEAT_H = ["K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "xwOBA", "sprint_speed"]
BAYES_FEAT_P = ["K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "est_woba"]

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


# ---------------------------------------------------------------------------
# データ構築
# ---------------------------------------------------------------------------

def build_delta_dataset(df: pd.DataFrame, feat_cols: list,
                         marcel_fn, avg_val: float,
                         target_col: str) -> pd.DataFrame:
    """
    delta = actual(t+1) - marcel_pred(t+1) を y として、
    t 時点の Statcast 特徴量を X とするデータセットを構築。

    df        : batter_features.csv or pitcher_features.csv
    feat_cols : BAYES_FEAT_H or BAYES_FEAT_P
    target_col: "wOBA" or "xFIP"
    """
    seasons = sorted(df["season"].unique())
    records = []

    for year in seasons[1:]:  # t+1 が存在する年から
        targets = df[df["season"] == year]
        for _, row in targets.iterrows():
            player = row["player"]
            actual = row.get(target_col, np.nan)
            if pd.isna(actual):
                continue

            # Marcel 予測 (= t+1 の予測)
            marcel = marcel_fn(df, player, year) or avg_val

            # t 時点 (= year-1) の特徴量
            prev = df[(df["player"] == player) & (df["season"] == year - 1)]
            if len(prev) == 0:
                continue

            feats = {}
            for f in feat_cols:
                feats[f] = prev.iloc[0].get(f, np.nan)

            if any(pd.isna(v) for v in feats.values()):
                continue

            record = {"player": player, "season": year,
                      "delta": actual - marcel, **feats}
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def find_alpha(X: np.ndarray, y: np.ndarray, alphas: list) -> float:
    """5-fold CV MAE で最良 alpha を選択"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha, best_mae = alphas[0], float("inf")
    for alpha in alphas:
        oof = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X):
            m = Ridge(alpha=alpha)
            m.fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict(X[va_idx])
        mae = mean_absolute_error(y, oof)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
    return best_alpha


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    """StandardScaler + Ridge を全データで学習、残差 sigma も返す"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    sigma = float(np.std(y - model.predict(Xs)))
    return scaler, model, sigma


# ---------------------------------------------------------------------------
# 予測 + CI
# ---------------------------------------------------------------------------

def predict_with_ci(feat_vals: np.ndarray, scaler: StandardScaler,
                    model: Ridge, sigma: float,
                    n_samples: int = 5000) -> tuple[float, float, float]:
    """
    Returns (bayes_delta, ci_lo80, ci_hi80)
    bayes_delta = point estimate of delta
    CI: Monte Carlo N(delta_i, sigma) × n_samples → 10th/90th percentile
    """
    Xs = scaler.transform(feat_vals.reshape(1, -1))
    delta_hat = float(model.predict(Xs)[0])
    samples = np.random.normal(delta_hat, sigma, size=n_samples)
    ci_lo = float(np.percentile(samples, 10))
    ci_hi = float(np.percentile(samples, 90))
    return delta_hat, ci_lo, ci_hi


def save_coef(model: Ridge, scaler: StandardScaler,
              feat_names: list, path: Path, label: str):
    """β係数（z-score スケール）を JSON に保存"""
    coefs = {
        label: {
            name: {
                "coef": round(float(c), 4),
                "mean": round(float(scaler.mean_[i]), 4),
                "std":  round(float(scaler.scale_[i]), 4),
            }
            for i, (name, c) in enumerate(zip(feat_names, model.coef_))
        }
    }
    if path.exists():
        existing = json.loads(path.read_text())
        existing.update(coefs)
        coefs = existing
    path.write_text(json.dumps(coefs, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(42)
    np.random.seed(42)

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run_wb = wandb.init(
        project="baseball-mlops", entity=entity, job_type="train_bayes",
        config={
            "bayes_feat_h": BAYES_FEAT_H,
            "bayes_feat_p": BAYES_FEAT_P,
            "alphas_grid": ALPHAS,
        }
    )

    coef_path = PRED_DIR / "bayes_coef.json"

    # ===== 打者 wOBA =====
    print("=== Batter Bayes (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    delta_bat = build_delta_dataset(
        bat_df, BAYES_FEAT_H, marcel_woba, MLB_AVG_WOBA, "wOBA"
    )
    print(f"  delta dataset: {len(delta_bat)} samples")

    X_bat = delta_bat[BAYES_FEAT_H].values.astype(float)
    y_bat = delta_bat["delta"].values.astype(float)

    alpha_bat = find_alpha(X_bat, y_bat, ALPHAS)
    scaler_bat, model_bat, sigma_bat = fit_ridge(X_bat, y_bat, alpha_bat)

    # CV MAE (bayes)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_bat = np.zeros(len(y_bat))
    for tr, va in kf.split(X_bat):
        sc = StandardScaler().fit(X_bat[tr])
        m = Ridge(alpha=alpha_bat).fit(sc.transform(X_bat[tr]), y_bat[tr])
        oof_bat[va] = m.predict(sc.transform(X_bat[va]))
    bayes_mae_bat = float(mean_absolute_error(y_bat, oof_bat))
    print(f"  alpha={alpha_bat}, sigma={sigma_bat:.4f}, Bayes delta MAE={bayes_mae_bat:.4f}")

    save_coef(model_bat, scaler_bat, BAYES_FEAT_H, coef_path, "batter")

    # ===== 投手 xFIP =====
    print("=== Pitcher Bayes (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    delta_pit = build_delta_dataset(
        pit_df, BAYES_FEAT_P, marcel_xfip, MLB_AVG_XFIP, "xFIP"
    )
    print(f"  delta dataset: {len(delta_pit)} samples")

    X_pit = delta_pit[BAYES_FEAT_P].values.astype(float)
    y_pit = delta_pit["delta"].values.astype(float)

    alpha_pit = find_alpha(X_pit, y_pit, ALPHAS)
    scaler_pit, model_pit, sigma_pit = fit_ridge(X_pit, y_pit, alpha_pit)

    oof_pit = np.zeros(len(y_pit))
    for tr, va in kf.split(X_pit):
        sc = StandardScaler().fit(X_pit[tr])
        m = Ridge(alpha=alpha_pit).fit(sc.transform(X_pit[tr]), y_pit[tr])
        oof_pit[va] = m.predict(sc.transform(X_pit[va]))
    bayes_mae_pit = float(mean_absolute_error(y_pit, oof_pit))
    print(f"  alpha={alpha_pit}, sigma={sigma_pit:.4f}, Bayes delta MAE={bayes_mae_pit:.4f}")

    save_coef(model_pit, scaler_pit, BAYES_FEAT_P, coef_path, "pitcher")

    # ===== W&B ログ =====
    top5_bat = sorted(
        zip(BAYES_FEAT_H, model_bat.coef_), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    top5_pit = sorted(
        zip(BAYES_FEAT_P, model_pit.coef_), key=lambda x: abs(x[1]), reverse=True
    )[:5]

    log_dict = {
        "alpha_batter": alpha_bat,
        "alpha_pitcher": alpha_pit,
        "sigma_batter": sigma_bat,
        "sigma_pitcher": sigma_pit,
        "bayes_batter_delta_MAE": bayes_mae_bat,
        "bayes_pitcher_delta_MAE": bayes_mae_pit,
    }
    for feat, coef in top5_bat:
        log_dict[f"coef_bat_{feat}"] = round(coef, 4)
    for feat, coef in top5_pit:
        log_dict[f"coef_pit_{feat}"] = round(coef, 4)
    wandb.log(log_dict)
    run_wb.finish()

    # ===== predictions CSV に bayes 列を追記 =====
    # 打者
    bat_pred_path = PRED_DIR / "batter_predictions.csv"
    if bat_pred_path.exists():
        bat_pred = pd.read_csv(bat_pred_path)
        bayes_rows = []
        for _, row in bat_pred.iterrows():
            player = row["player"]
            season_last = int(row.get("season_last", bat_df["season"].max()))
            prev = bat_df[
                (bat_df["player"] == player) & (bat_df["season"] == season_last)
            ]
            if len(prev) == 0 or any(
                pd.isna(prev.iloc[0].get(f)) for f in BAYES_FEAT_H
            ):
                bayes_rows.append({
                    "bayes_woba": row.get("marcel_woba", np.nan),
                    "ci_lo80": np.nan, "ci_hi80": np.nan,
                })
                continue
            feat_vals = np.array([prev.iloc[0][f] for f in BAYES_FEAT_H], dtype=float)
            delta_hat, ci_lo, ci_hi = predict_with_ci(
                feat_vals, scaler_bat, model_bat, sigma_bat
            )
            marcel_val = row.get("marcel_woba", MLB_AVG_WOBA)
            bayes_rows.append({
                "bayes_woba": round(float(marcel_val) + delta_hat, 3),
                "ci_lo80": round(float(marcel_val) + ci_lo, 3),
                "ci_hi80": round(float(marcel_val) + ci_hi, 3),
            })

        bayes_df = pd.DataFrame(bayes_rows)
        # 既存列を上書きしないように drop してから concat
        for col in ["bayes_woba", "ci_lo80", "ci_hi80"]:
            if col in bat_pred.columns:
                bat_pred = bat_pred.drop(columns=[col])
        bat_pred = pd.concat([bat_pred.reset_index(drop=True),
                               bayes_df.reset_index(drop=True)], axis=1)
        bat_pred.to_csv(bat_pred_path, index=False)
        print(f"Batter predictions updated: {bat_pred_path}")

    # 投手
    pit_pred_path = PRED_DIR / "pitcher_predictions.csv"
    if pit_pred_path.exists():
        pit_pred = pd.read_csv(pit_pred_path)
        bayes_rows = []
        for _, row in pit_pred.iterrows():
            player = row["player"]
            season_last = int(row.get("season_last", pit_df["season"].max()))
            prev = pit_df[
                (pit_df["player"] == player) & (pit_df["season"] == season_last)
            ]
            if len(prev) == 0 or any(
                pd.isna(prev.iloc[0].get(f)) for f in BAYES_FEAT_P
            ):
                bayes_rows.append({
                    "bayes_xfip": row.get("marcel_xfip", np.nan),
                    "ci_lo80": np.nan, "ci_hi80": np.nan,
                })
                continue
            feat_vals = np.array([prev.iloc[0][f] for f in BAYES_FEAT_P], dtype=float)
            delta_hat, ci_lo, ci_hi = predict_with_ci(
                feat_vals, scaler_pit, model_pit, sigma_pit
            )
            marcel_val = row.get("marcel_xfip", MLB_AVG_XFIP)
            bayes_rows.append({
                "bayes_xfip": round(float(marcel_val) + delta_hat, 2),
                "ci_lo80": round(float(marcel_val) + ci_lo, 2),
                "ci_hi80": round(float(marcel_val) + ci_hi, 2),
            })

        bayes_df = pd.DataFrame(bayes_rows)
        for col in ["bayes_xfip", "ci_lo80", "ci_hi80"]:
            if col in pit_pred.columns:
                pit_pred = pit_pred.drop(columns=[col])
        pit_pred = pd.concat([pit_pred.reset_index(drop=True),
                               bayes_df.reset_index(drop=True)], axis=1)
        pit_pred.to_csv(pit_pred_path, index=False)
        print(f"Pitcher predictions updated: {pit_pred_path}")

    print("=== train_bayes.py complete ===")


if __name__ == "__main__":
    run()
