"""
LightGBM 学習 + W&B 記録 + Model Registry 保存

打者: 翌年 wOBA 予測
投手: 翌年 xFIP 予測
ベースライン: MLB Marcel法（加重平均 + 平均回帰 + 年齢調整）との比較
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROJ_DIR = DATA_DIR / "projections"
MODELS_DIR = Path(__file__).parent.parent / "models"
PROJ_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MLB 平均値（平均回帰の基準）
MLB_AVG_WOBA = 0.320
MLB_AVG_XFIP = 4.20
REGRESS_PA = 1000   # 打者回帰用 PA 相当
REGRESS_IP = 200    # 投手回帰用 IP 相当

# 年齢調整（ピーク 27 歳）
def age_adj_bat(age: float) -> float:
    return 0.003 * (27 - age)

def age_adj_pit(age: float) -> float:
    return -0.006 * (27 - age)  # xFIP は低い方が良いので符号逆


# ---------------------------------------------------------------------------
# Marcel ベースライン
# ---------------------------------------------------------------------------

def marcel_woba(df: pd.DataFrame, player: str, year: int) -> float | None:
    """過去 3 年の PA 加重平均 + 平均回帰 + 年齢調整"""
    rows = []
    weights = [5, 4, 3]
    for i, w in enumerate(weights):
        y = year - 1 - i
        row = df[(df["player"] == player) & (df["season"] == y)]
        if len(row) == 0:
            continue
        rows.append((row.iloc[0]["wOBA"], row.iloc[0].get("PA", 500), w))

    if not rows:
        return None

    total_w = sum(r[2] for r in rows)
    woba_w = sum(r[0] * r[2] for r in rows) / total_w
    pa_w = sum(r[1] * r[2] for r in rows) / total_w

    # 平均回帰
    woba_reg = (woba_w * pa_w + MLB_AVG_WOBA * REGRESS_PA) / (pa_w + REGRESS_PA)

    # 年齢調整
    row_cur = df[(df["player"] == player) & (df["season"] == year - 1)]
    age = float(row_cur.iloc[0]["Age"]) if len(row_cur) > 0 else 28.0
    return round(woba_reg + age_adj_bat(age), 3)


def marcel_xfip(df: pd.DataFrame, player: str, year: int) -> float | None:
    """過去 3 年の IP 加重平均 + 平均回帰 + 年齢調整"""
    rows = []
    weights = [5, 4, 3]
    for i, w in enumerate(weights):
        y = year - 1 - i
        row = df[(df["player"] == player) & (df["season"] == y)]
        if len(row) == 0:
            continue
        rows.append((row.iloc[0]["xFIP"], row.iloc[0].get("IP", 100), w))

    if not rows:
        return None

    total_w = sum(r[2] for r in rows)
    xfip_w = sum(r[0] * r[2] for r in rows) / total_w
    ip_w = sum(r[1] * r[2] for r in rows) / total_w

    xfip_reg = (xfip_w * ip_w + MLB_AVG_XFIP * REGRESS_IP) / (ip_w + REGRESS_IP)

    row_cur = df[(df["player"] == player) & (df["season"] == year - 1)]
    age = float(row_cur.iloc[0]["Age"]) if len(row_cur) > 0 else 28.0
    return round(xfip_reg + age_adj_pit(age), 2)


# ---------------------------------------------------------------------------
# 特徴量構築
# ---------------------------------------------------------------------------

BATTER_FEATURES = [
    "wOBA", "xwoba", "xba", "xslg",
    "K%", "BB%", "ISO", "BABIP", "OBP", "SLG",
    "exit_velocity_avg", "launch_angle_avg", "barrel_batted_rate",
    "hard_hit_percent", "sweet_spot_percent", "sprint_speed",
    "Age", "PA",
]

PITCHER_FEATURES = [
    "xFIP", "FIP", "ERA", "xera",
    "K%", "BB%", "HR/9", "WHIP", "BABIP", "LOB%",
    "exit_velocity_avg", "launch_angle_avg", "barrel_batted_rate",
    "hard_hit_percent", "xwoba",
    "Age", "IP",
]


def build_train_data_batters(df: pd.DataFrame, min_pa: int = 100):
    """翌年 wOBA をターゲットとした学習データを構築"""
    records = []
    seasons = sorted(df["season"].unique())
    for year in seasons[3:]:  # 過去3年分の特徴量が必要
        targets = df[(df["season"] == year) & (df["PA"] >= min_pa)]
        for _, row in targets.iterrows():
            player = row["player"]
            feats = {}
            # 過去 3 年の特徴量（y1=直前, y2=2年前, y3=3年前）
            for lag in range(1, 4):
                prev = df[(df["player"] == player) & (df["season"] == year - lag)]
                suffix = f"_y{lag}"
                if len(prev) == 0:
                    for f in BATTER_FEATURES:
                        feats[f + suffix] = np.nan
                else:
                    for f in BATTER_FEATURES:
                        feats[f + suffix] = prev.iloc[0].get(f, np.nan)

            # Marcel ベースライン
            feats["marcel_woba"] = marcel_woba(df, player, year) or np.nan

            feats["target_woba"] = row["wOBA"]
            feats["player"] = player
            feats["season"] = year
            records.append(feats)

    return pd.DataFrame(records)


def build_train_data_pitchers(df: pd.DataFrame, min_ip: int = 30):
    """翌年 xFIP をターゲットとした学習データを構築"""
    records = []
    seasons = sorted(df["season"].unique())
    for year in seasons[3:]:
        targets = df[(df["season"] == year) & (df["IP"] >= min_ip)]
        for _, row in targets.iterrows():
            player = row["player"]
            feats = {}
            for lag in range(1, 4):
                prev = df[(df["player"] == player) & (df["season"] == year - lag)]
                suffix = f"_y{lag}"
                if len(prev) == 0:
                    for f in PITCHER_FEATURES:
                        feats[f + suffix] = np.nan
                else:
                    for f in PITCHER_FEATURES:
                        feats[f + suffix] = prev.iloc[0].get(f, np.nan)

            feats["marcel_xfip"] = marcel_xfip(df, player, year) or np.nan
            feats["target_xfip"] = row["xFIP"]
            feats["player"] = player
            feats["season"] = year
            records.append(feats)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "n_estimators": 300,
}


def train_model(X: pd.DataFrame, y: pd.Series, params: dict) -> tuple:
    """5-fold CV で LightGBM を学習、MAE と最終モデルを返す"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict(X_va)
        models.append(model)

    mae = mean_absolute_error(y, oof)
    # 全データで再学習（最終モデル）
    final = lgb.LGBMRegressor(**params)
    final.fit(X, y)
    return final, mae, models


def save_to_wandb(model, mae: float, target: str, feature_names: list, config: dict):
    """W&B にモデル・メトリクス・特徴量重要度を記録"""
    run = wandb.init(project="baseball-mlops", job_type="train",
                     config={**config, "target": target})

    wandb.log({f"MAE_{target}": mae})

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    wandb.log({f"feature_importance_{target}": wandb.Table(dataframe=importance.head(20))})

    # モデルを Artifact として保存
    artifact = wandb.Artifact(f"{target}-model", type="model",
                               description=f"LightGBM {target} predictor, MAE={mae:.4f}")
    model_path = MODELS_DIR / f"{target}_model.pkl"
    joblib.dump(model, model_path)
    artifact.add_file(str(model_path))
    run.log_artifact(artifact, aliases=["latest"])

    run.finish()
    return artifact


def run_training():
    """打者・投手モデルを学習して W&B に記録"""
    # 打者
    print("=== Batter (wOBA) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    train_bat = build_train_data_batters(bat_df)
    train_bat = train_bat.dropna(subset=["target_woba"])

    feat_cols_bat = [c for c in train_bat.columns
                     if c not in ("player", "season", "target_woba")]
    X_bat = train_bat[feat_cols_bat].apply(pd.to_numeric, errors="coerce")
    y_bat = train_bat["target_woba"]

    model_bat, mae_bat, _ = train_model(X_bat, y_bat, LGB_PARAMS)
    print(f"  ML  MAE wOBA: {mae_bat:.4f}")

    # Marcel ベースライン MAE
    marcel_mae_bat = mean_absolute_error(y_bat, train_bat["marcel_woba"].fillna(MLB_AVG_WOBA))
    print(f"  Marcel MAE wOBA: {marcel_mae_bat:.4f}")

    save_to_wandb(model_bat, mae_bat, "woba", feat_cols_bat,
                  {"marcel_mae": marcel_mae_bat, "n_samples": len(X_bat)})

    # 投手
    print("=== Pitcher (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    train_pit = build_train_data_pitchers(pit_df)
    train_pit = train_pit.dropna(subset=["target_xfip"])

    feat_cols_pit = [c for c in train_pit.columns
                     if c not in ("player", "season", "target_xfip")]
    X_pit = train_pit[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
    y_pit = train_pit["target_xfip"]

    model_pit, mae_pit, _ = train_model(X_pit, y_pit, LGB_PARAMS)
    print(f"  ML  MAE xFIP: {mae_pit:.4f}")

    marcel_mae_pit = mean_absolute_error(y_pit, train_pit["marcel_xfip"].fillna(MLB_AVG_XFIP))
    print(f"  Marcel MAE xFIP: {marcel_mae_pit:.4f}")

    save_to_wandb(model_pit, mae_pit, "xfip", feat_cols_pit,
                  {"marcel_mae": marcel_mae_pit, "n_samples": len(X_pit)})

    # 予測結果を保存
    bat_df_latest = bat_df[bat_df["season"] == bat_df["season"].max()]
    pit_df_latest = pit_df[pit_df["season"] == pit_df["season"].max()]

    bat_preds = _predict_next_season(model_bat, bat_df, bat_df_latest, feat_cols_bat,
                                     "wOBA", "pred_woba", marcel_woba, MLB_AVG_WOBA)
    pit_preds = _predict_next_season(model_pit, pit_df, pit_df_latest, feat_cols_pit,
                                     "xFIP", "pred_xfip", marcel_xfip, MLB_AVG_XFIP)

    bat_preds.to_csv(PROJ_DIR / "batter_predictions.csv", index=False)
    pit_preds.to_csv(PROJ_DIR / "pitcher_predictions.csv", index=False)
    print(f"Predictions saved: {len(bat_preds)} batters, {len(pit_preds)} pitchers")


def _predict_next_season(model, full_df, latest_df, feat_cols,
                          target_col, pred_col, marcel_fn, avg_val) -> pd.DataFrame:
    """最新シーズンの選手について翌年予測を生成"""
    next_year = int(latest_df["season"].max()) + 1
    records = []
    for _, row in latest_df.iterrows():
        player = row["player"]
        feats = {}
        for lag in range(1, 4):
            y = next_year - 1 - lag
            prev = full_df[(full_df["player"] == player) & (full_df["season"] == y)]
            suffix = f"_y{lag}"
            if lag == 1:
                for f in feat_cols:
                    base = f.replace("_y1", "").replace("_y2", "").replace("_y3", "")
                    if f.endswith("_y1"):
                        feats[f] = row.get(base, np.nan)
                    elif f.endswith("_y2") and len(prev) > 0:
                        feats[f] = prev.iloc[0].get(base, np.nan)
                    elif f.endswith(suffix):
                        feats[f] = np.nan
            else:
                for f in [c for c in feat_cols if c.endswith(suffix)]:
                    base = f.replace(suffix, "")
                    feats[f] = prev.iloc[0].get(base, np.nan) if len(prev) > 0 else np.nan

        marcel_val = (marcel_fn(full_df, player, next_year) or avg_val)
        feats["marcel_" + target_col.lower().replace("/", "")] = marcel_val

        feats_df = pd.DataFrame([feats])[feat_cols].apply(pd.to_numeric, errors="coerce")
        pred = model.predict(feats_df)[0]

        records.append({
            "player": player,
            "Team": row.get("Team", ""),
            "Age": row.get("Age", ""),
            "season_last": next_year - 1,
            "pred_year": next_year,
            pred_col: round(pred, 3),
            f"marcel_{target_col.lower().replace('/', '')}": round(marcel_val, 3),
            target_col + "_last": round(row.get(target_col, np.nan), 3),
        })

    return pd.DataFrame(records).sort_values(pred_col)


if __name__ == "__main__":
    run_training()
