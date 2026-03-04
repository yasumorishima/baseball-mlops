"""
Bayesian Ridge 学習 + W&B 記録

Marcel との差分 (delta) を Statcast + FanGraphs 特徴量で Ridge 回帰し、
Monte Carlo CI (80%) を付与する。

主な改良点 (v2):
  - 特徴量拡充: 打球角度/ハードヒット%/球場補正/年齢カーブ/出場信頼度/ラック指標
  - SimpleImputer: NaN行を落とさず median で補完
  - Recency weight: 直近シーズンに高い重み (decay=0.85/yr)

入力:  data/raw/batter_features.csv, pitcher_features.csv
       predictions/batter_predictions.csv, pitcher_predictions.csv (train.py 出力)
出力:  predictions/*.csv に bayes_*/ci_lo80/ci_hi80 列を追記
       predictions/bayes_coef.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import MLB_AVG_WOBA, MLB_AVG_XFIP, marcel_woba, marcel_xfip

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PRED_DIR = Path(__file__).parent.parent / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS       = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RECENCY_DECAY = 0.85   # 1年遡るごとに 0.85 倍

# ---------------------------------------------------------------------------
# 球場補正辞書（FanGraphs Basic Park Factor, 2022-2024 平均, 100=中立）
# >100 = 打者有利, <100 = 投手有利
# ---------------------------------------------------------------------------

PARK_FACTORS: dict[str, int] = {
    "COL": 118, "CIN": 111, "BOS": 108, "TEX": 107, "NYY": 106,
    "PHI": 105, "MIL": 105, "ATL": 104, "LAA": 103, "ANA": 103,
    "CWS": 103, "CHW": 103, "HOU": 102, "TOR": 102, "STL": 101,
    "DET": 101, "MIN": 100, "WAS": 100, "WSH": 100, "KCR": 100,
    "KC":  100, "PIT":  99, "CHC":  99, "CLE":  99, "BAL":  99,
    "ARI":  98, "AZ":   98, "SEA":  97, "TBR":  97, "TB":   97,
    "NYM":  97, "LAD":  96, "OAK":  96, "SFG":  95, "SF":   95,
    "SDP":  94, "SD":   94, "MIA":  93, "FLA":  93,
}

# ---------------------------------------------------------------------------
# 特徴量定義（RAW=CSVから直接読む列名, ENG=計算して追加する列名）
# ---------------------------------------------------------------------------

_BAT_RAW = [
    # 既存
    "K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "xwOBA", "sprint_speed",
    # 新規: Statcast
    "avg_hit_angle", "ev95percent",
    # 新規: FanGraphs
    "HardHit%", "Contact%", "O-Swing%", "PA", "G",
]
_BAT_ENG = ["age_from_peak", "age_sq", "pa_rate", "xwoba_luck", "park_factor"]
BAYES_FEAT_H = _BAT_RAW + _BAT_ENG

_PIT_RAW = [
    # 既存
    "K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "est_woba",
    # 新規: Statcast
    "avg_hit_angle", "ev95percent",
    # 新規: FanGraphs
    "HardHit%", "K-BB%", "CSW%", "IP", "G",
]
_PIT_ENG = ["age_from_peak", "age_sq", "ip_rate", "park_factor"]
BAYES_FEAT_P = _PIT_RAW + _PIT_ENG


# ---------------------------------------------------------------------------
# 特徴量エンジニアリングヘルパー
# ---------------------------------------------------------------------------

def _park(team: str) -> float:
    return float(PARK_FACTORS.get(str(team).strip(), 100))


def _eng_batter(row: pd.Series) -> dict:
    age   = float(row.get("Age") or 28)
    pa    = float(row.get("PA") or 0)
    xwoba = float(row.get("est_woba") or np.nan) if pd.notna(row.get("est_woba")) else np.nan
    woba  = float(row.get("wOBA")  or np.nan) if pd.notna(row.get("wOBA"))  else np.nan
    return {
        "age_from_peak": age - 27,
        "age_sq":        (age - 27) ** 2,
        "pa_rate":       pa / 650.0,
        "xwoba_luck":    (xwoba - woba) if (not np.isnan(xwoba) and not np.isnan(woba)) else np.nan,
        "park_factor":   _park(row.get("Team", "")),
    }


def _eng_pitcher(row: pd.Series) -> dict:
    age = float(row.get("Age") or 28)
    ip  = float(row.get("IP") or 0)
    return {
        "age_from_peak": age - 27,
        "age_sq":        (age - 27) ** 2,
        "ip_rate":       ip / 200.0,
        "park_factor":   _park(row.get("Team", "")),
    }


# ---------------------------------------------------------------------------
# データ構築
# ---------------------------------------------------------------------------

def build_delta_dataset(df: pd.DataFrame, raw_cols: list, eng_fn,
                         marcel_fn, avg_val: float,
                         target_col: str) -> pd.DataFrame:
    """
    y = actual(t+1) - marcel_pred(t+1)
    X = t 時点の raw + engineered 特徴量
    """
    seasons = sorted(df["season"].unique())
    records = []

    for year in seasons[1:]:
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
            delta  = float(actual) - marcel

            # raw features
            raw_feats = {f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
                         for f in raw_cols}
            # engineered features
            eng_feats = eng_fn(prev_row)

            records.append({
                "player": player, "season": year, "delta": delta,
                **raw_feats, **eng_feats,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def _make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ridge",   Ridge(alpha=alpha)),
    ])


def find_alpha(X: np.ndarray, y: np.ndarray, alphas: list) -> float:
    """5-fold CV MAE で最良 alpha を選択"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha, best_mae = alphas[0], float("inf")
    for alpha in alphas:
        oof = cross_val_predict(_make_pipeline(alpha), X, y, cv=kf)
        mae = mean_absolute_error(y, oof)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
    return best_alpha


def fit_pipeline(X: np.ndarray, y: np.ndarray, alpha: float,
                 sample_weight: np.ndarray | None = None):
    """Pipeline を全データで学習し、sigma (残差標準偏差) も返す"""
    pipe = _make_pipeline(alpha)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["ridge__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_kwargs)
    sigma = float(np.std(y - pipe.predict(X)))
    return pipe, sigma


def recency_weights(seasons: np.ndarray) -> np.ndarray:
    """直近シーズンほど重くなるサンプルウェイト"""
    max_s = seasons.max()
    return np.array([RECENCY_DECAY ** (max_s - s) for s in seasons])


# ---------------------------------------------------------------------------
# 予測 + CI
# ---------------------------------------------------------------------------

def predict_with_ci(feat_vals: np.ndarray, pipe: Pipeline, sigma: float,
                    n_samples: int = 5000) -> tuple[float, float, float]:
    """(delta_hat, ci_lo80, ci_hi80)"""
    delta_hat = float(pipe.predict(feat_vals.reshape(1, -1))[0])
    samples   = np.random.normal(delta_hat, sigma, size=n_samples)
    return delta_hat, float(np.percentile(samples, 10)), float(np.percentile(samples, 90))


def save_coef(pipe: Pipeline, feat_names: list, path: Path, label: str):
    """β係数を JSON に保存（imputer → scaler → ridge のパラメータを取り出す）"""
    scaler: StandardScaler = pipe.named_steps["scaler"]
    ridge:  Ridge          = pipe.named_steps["ridge"]
    coefs = {
        label: {
            name: {"coef": round(float(c), 4),
                   "mean": round(float(scaler.mean_[i]), 4),
                   "std":  round(float(scaler.scale_[i]), 4)}
            for i, (name, c) in enumerate(zip(feat_names, ridge.coef_))
        }
    }
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(coefs)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run():
    np.random.seed(42)

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run_wb = wandb.init(
        project="baseball-mlops", entity=entity, job_type="train_bayes",
        config={
            "bayes_feat_h": BAYES_FEAT_H,
            "bayes_feat_p": BAYES_FEAT_P,
            "alphas_grid":  ALPHAS,
            "recency_decay": RECENCY_DECAY,
        }
    )

    coef_path = PRED_DIR / "bayes_coef.json"

    # ===== 打者 wOBA =====
    print("=== Batter Bayes (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    delta_bat = build_delta_dataset(
        bat_df, _BAT_RAW, _eng_batter, marcel_woba, MLB_AVG_WOBA, "wOBA"
    )
    print(f"  delta dataset: {len(delta_bat)} samples")

    feat_cols_bat = BAYES_FEAT_H
    X_bat = delta_bat[feat_cols_bat].values.astype(float)
    y_bat = delta_bat["delta"].values.astype(float)
    w_bat = recency_weights(delta_bat["season"].values)

    alpha_bat          = find_alpha(X_bat, y_bat, ALPHAS)
    pipe_bat, sigma_bat = fit_pipeline(X_bat, y_bat, alpha_bat, w_bat)
    oof_bat            = cross_val_predict(_make_pipeline(alpha_bat), X_bat, y_bat,
                                           cv=KFold(5, shuffle=True, random_state=42))
    bayes_mae_bat = float(mean_absolute_error(y_bat, oof_bat))
    print(f"  alpha={alpha_bat}, sigma={sigma_bat:.4f}, delta MAE={bayes_mae_bat:.4f}")
    save_coef(pipe_bat, feat_cols_bat, coef_path, "batter")

    # ===== 投手 xFIP =====
    print("=== Pitcher Bayes (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    delta_pit = build_delta_dataset(
        pit_df, _PIT_RAW, _eng_pitcher, marcel_xfip, MLB_AVG_XFIP, "xFIP"
    )
    print(f"  delta dataset: {len(delta_pit)} samples")

    feat_cols_pit = BAYES_FEAT_P
    X_pit = delta_pit[feat_cols_pit].values.astype(float)
    y_pit = delta_pit["delta"].values.astype(float)
    w_pit = recency_weights(delta_pit["season"].values)

    alpha_pit          = find_alpha(X_pit, y_pit, ALPHAS)
    pipe_pit, sigma_pit = fit_pipeline(X_pit, y_pit, alpha_pit, w_pit)
    oof_pit            = cross_val_predict(_make_pipeline(alpha_pit), X_pit, y_pit,
                                           cv=KFold(5, shuffle=True, random_state=42))
    bayes_mae_pit = float(mean_absolute_error(y_pit, oof_pit))
    print(f"  alpha={alpha_pit}, sigma={sigma_pit:.4f}, delta MAE={bayes_mae_pit:.4f}")
    save_coef(pipe_pit, feat_cols_pit, coef_path, "pitcher")

    # ===== W&B ログ =====
    ridge_bat: Ridge = pipe_bat.named_steps["ridge"]
    ridge_pit: Ridge = pipe_pit.named_steps["ridge"]
    top5_bat = sorted(zip(feat_cols_bat, ridge_bat.coef_),
                      key=lambda x: abs(x[1]), reverse=True)[:5]
    top5_pit = sorted(zip(feat_cols_pit, ridge_pit.coef_),
                      key=lambda x: abs(x[1]), reverse=True)[:5]

    log_dict = {
        "alpha_batter":            alpha_bat,
        "alpha_pitcher":           alpha_pit,
        "sigma_batter":            sigma_bat,
        "sigma_pitcher":           sigma_pit,
        "bayes_batter_delta_MAE":  bayes_mae_bat,
        "bayes_pitcher_delta_MAE": bayes_mae_pit,
        "n_samples_batter":        len(delta_bat),
        "n_samples_pitcher":       len(delta_pit),
    }
    for feat, coef in top5_bat:
        log_dict[f"coef_bat_{feat}"] = round(coef, 4)
    for feat, coef in top5_pit:
        log_dict[f"coef_pit_{feat}"] = round(coef, 4)
    wandb.log(log_dict)
    run_wb.finish()

    # ===== predictions CSV に bayes 列を追記 =====
    _update_batter_predictions(bat_df, pipe_bat, sigma_bat)
    _update_pitcher_predictions(pit_df, pipe_pit, sigma_pit)

    print("=== train_bayes.py complete ===")


def _update_batter_predictions(bat_df: pd.DataFrame, pipe: Pipeline, sigma: float):
    pred_path = PRED_DIR / "batter_predictions.csv"
    if not pred_path.exists():
        return
    bat_pred = pd.read_csv(pred_path)
    rows = []
    for _, row in bat_pred.iterrows():
        player     = row["player"]
        season_last = int(row.get("season_last", bat_df["season"].max()))
        prev = bat_df[(bat_df["player"] == player) & (bat_df["season"] == season_last)]

        if len(prev) == 0:
            rows.append({"bayes_woba": row.get("marcel_woba"), "ci_lo80": np.nan, "ci_hi80": np.nan})
            continue

        prev_row  = prev.iloc[0]
        raw_feats = {f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
                     for f in _BAT_RAW}
        eng_feats = _eng_batter(prev_row)
        feat_vals = np.array([raw_feats.get(f, eng_feats.get(f, np.nan)) for f in BAYES_FEAT_H])

        delta_hat, ci_lo, ci_hi = predict_with_ci(feat_vals, pipe, sigma)
        marcel_val = float(row.get("marcel_woba", MLB_AVG_WOBA))
        rows.append({
            "bayes_woba": round(marcel_val + delta_hat, 3),
            "ci_lo80":    round(marcel_val + ci_lo, 3),
            "ci_hi80":    round(marcel_val + ci_hi, 3),
        })

    bayes_df = pd.DataFrame(rows)
    for col in ["bayes_woba", "ci_lo80", "ci_hi80"]:
        if col in bat_pred.columns:
            bat_pred = bat_pred.drop(columns=[col])
    bat_pred = pd.concat([bat_pred.reset_index(drop=True), bayes_df.reset_index(drop=True)], axis=1)
    bat_pred.to_csv(pred_path, index=False)
    print(f"Batter predictions updated: {pred_path}")


def _update_pitcher_predictions(pit_df: pd.DataFrame, pipe: Pipeline, sigma: float):
    pred_path = PRED_DIR / "pitcher_predictions.csv"
    if not pred_path.exists():
        return
    pit_pred = pd.read_csv(pred_path)
    rows = []
    for _, row in pit_pred.iterrows():
        player      = row["player"]
        season_last = int(row.get("season_last", pit_df["season"].max()))
        prev = pit_df[(pit_df["player"] == player) & (pit_df["season"] == season_last)]

        if len(prev) == 0:
            rows.append({"bayes_xfip": row.get("marcel_xfip"), "ci_lo80": np.nan, "ci_hi80": np.nan})
            continue

        prev_row  = prev.iloc[0]
        raw_feats = {f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
                     for f in _PIT_RAW}
        eng_feats = _eng_pitcher(prev_row)
        feat_vals = np.array([raw_feats.get(f, eng_feats.get(f, np.nan)) for f in BAYES_FEAT_P])

        delta_hat, ci_lo, ci_hi = predict_with_ci(feat_vals, pipe, sigma)
        marcel_val = float(row.get("marcel_xfip", MLB_AVG_XFIP))
        rows.append({
            "bayes_xfip": round(marcel_val + delta_hat, 2),
            "ci_lo80":    round(marcel_val + ci_lo, 2),
            "ci_hi80":    round(marcel_val + ci_hi, 2),
        })

    bayes_df = pd.DataFrame(rows)
    for col in ["bayes_xfip", "ci_lo80", "ci_hi80"]:
        if col in pit_pred.columns:
            pit_pred = pit_pred.drop(columns=[col])
    pit_pred = pd.concat([pit_pred.reset_index(drop=True), bayes_df.reset_index(drop=True)], axis=1)
    pit_pred.to_csv(pred_path, index=False)
    print(f"Pitcher predictions updated: {pred_path}")


if __name__ == "__main__":
    run()
