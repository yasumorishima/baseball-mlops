"""
ElasticNet (Bayesian-style) 学習 + LightGBM スタッキング + W&B 記録 (v5)

Marcel との差分 (delta) を Statcast + FanGraphs 特徴量 + LightGBM OOF で
ElasticNet 回帰し、Monte Carlo CI (80%) を付与する。

v5 改良点:
  - 打者のデータ期間を 2015以降に拡張（Stuff+ 不要のため MIN_PREV_SEASON_BAT=2015）
    投手は Stuff+ が 2020以降のみ → MIN_PREV_SEASON_PIT=2020 を維持
  - LightGBM OOF をスタッキング特徴量として追加（lgb_delta = lgb_oof - marcel）
    train.py の 5-fold OOF を data/raw/lgb_oof_{batter,pitcher}.csv から読み込む
  - Ridge → ElasticNet に切り替え（alpha × l1_ratio グリッドサーチ）
    l1_ratio=0.0: Ridge / l1_ratio=1.0: Lasso / 中間: ElasticNet
    → L1 正則化による自動特徴選択

入力:  data/raw/batter_features.csv, pitcher_features.csv
       data/raw/lgb_oof_batter.csv, lgb_oof_pitcher.csv  (train.py 出力)
       data/raw/park_factors.csv                          (fetch_statcast.py 出力)
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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import MLB_AVG_WOBA, MLB_AVG_XFIP, marcel_woba, marcel_xfip, _time_cv_splits

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PRED_DIR = Path(__file__).parent.parent / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# ハイパーパラメータ探索グリッド
ALPHAS     = [0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
L1_RATIOS  = [0.0, 0.15, 0.5, 0.85, 1.0]  # 0.0=Ridge, 1.0=Lasso
RECENCY_DECAY = 0.85

# 打者: Statcast 特徴量（brl_percent/xwOBA 等）が 2020 以降で揃うため 2020 を維持
# 投手: Stuff+ は 2020 以降のみ → 同じく 2020
MIN_PREV_SEASON_BAT = 2020
MIN_PREV_SEASON_PIT = 2020

# ---------------------------------------------------------------------------
# 球場補正辞書（静的フォールバック）
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

# ---------------------------------------------------------------------------
# 特徴量定義
# ---------------------------------------------------------------------------

_BAT_RAW = [
    # Statcast
    "K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "xwOBA", "sprint_speed",
    "avg_hit_angle", "ev95percent",
    # FanGraphs
    "HardHit%", "Contact%", "O-Swing%", "PA", "G", "SwStr%", "maxEV",
    # スタッキング特徴量: LightGBM OOF delta
    "lgb_delta",
]
_BAT_ENG = [
    "age_from_peak", "age_sq", "pa_rate", "xwoba_luck", "park_factor",
    "team_changed",   # 移籍フラグ
    "g_change_rate",  # G(t)/G(t-1)
]
BAYES_FEAT_H = _BAT_RAW + _BAT_ENG

_PIT_RAW = [
    # Statcast
    "K%", "BB%", "BABIP", "brl_percent", "avg_hit_speed", "est_woba",
    "avg_hit_angle", "ev95percent",
    # FanGraphs
    "HardHit%", "K-BB%", "CSW%", "IP", "G", "SwStr%",
    # FanGraphs (2020+ のみ、SimpleImputer が NaN を補完)
    "Stuff+", "Location+", "Pitching+",
    # スタッキング特徴量: LightGBM OOF delta
    "lgb_delta",
]
_PIT_ENG = [
    "age_from_peak", "age_sq", "ip_rate", "park_factor",
    "team_changed",
    "g_change_rate",
]
BAYES_FEAT_P = _PIT_RAW + _PIT_ENG


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _park(team: str) -> float:
    return float(_PF_FALLBACK.get(str(team).strip(), 100))


def _load_park_factors() -> dict | None:
    """park_factors.csv → {(season, team): pf_5yr}"""
    path = RAW_DIR / "park_factors.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {(int(r.season), str(r.team)): float(r.pf_5yr) for _, r in df.iterrows()}


def _load_lgb_oof(path: Path) -> dict | None:
    """lgb_oof_*.csv → {(player_str, season_int): lgb_pred}"""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    val_col = [c for c in df.columns if c not in ("player", "season")][0]
    return {(str(r.player), int(r.season)): float(r[val_col]) for _, r in df.iterrows()}


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
                         target_col: str,
                         min_prev_season: int,
                         pf_lookup: dict | None = None,
                         lgb_oof: dict | None = None) -> pd.DataFrame:
    """
    y = actual(t+1) - marcel_pred(t+1)
    X = t 時点の raw + engineered 特徴量 + lgb_delta（スタッキング）

    lgb_delta = lgb_oof(t+1) - marcel(t+1)
    """
    seasons = sorted(df["season"].unique())
    records = []

    for year in seasons[1:]:
        if year - 1 < min_prev_season:
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
            delta  = float(actual) - marcel

            # raw features（lgb_delta を除いて通常取得）
            raw_feats = {
                f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
                for f in raw_cols if f != "lgb_delta"
            }

            # LightGBM スタッキング特徴量
            if lgb_oof:
                lgb_pred = lgb_oof.get((str(player), year))
                raw_feats["lgb_delta"] = (lgb_pred - marcel) if lgb_pred is not None else np.nan
            else:
                raw_feats["lgb_delta"] = np.nan

            # engineered features
            eng_feats = eng_fn(prev_row)

            # 動的 park_factor で上書き
            if pf_lookup:
                pf = pf_lookup.get((year - 1, str(prev_row.get("Team", "")).strip()))
                if pf:
                    eng_feats["park_factor"] = pf

            # チーム変更フラグ・G前年比
            prev2 = df[(df["player"] == player) & (df["season"] == year - 2)]
            team_cur  = str(prev_row.get("Team", ""))
            g_cur     = float(prev_row.get("G") or 1)
            if len(prev2) > 0:
                team_prev = str(prev2.iloc[0].get("Team", ""))
                g_prev    = float(prev2.iloc[0].get("G") or g_cur)
            else:
                team_prev, g_prev = team_cur, g_cur
            eng_feats["team_changed"]  = int(team_cur != team_prev)
            eng_feats["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

            records.append({
                "player": player, "season": year, "delta": delta,
                **raw_feats, **eng_feats,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# モデル (ElasticNet)
# ---------------------------------------------------------------------------

def _make_pipeline(alpha: float, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("scaler",    StandardScaler()),
        ("regressor", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
    ])


def find_hyperparams(X: np.ndarray, y: np.ndarray, seasons: np.ndarray,
                     alphas: list, l1_ratios: list) -> tuple[float, float]:
    """時系列 expanding-window CV MAE で最良 (alpha, l1_ratio) を選択。

    seasons に基づく walk-forward splits を使用（未来リーク防止）。
    """
    splits = _time_cv_splits(seasons)
    best_params, best_mae = (alphas[0], l1_ratios[0]), float("inf")
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            pipe = _make_pipeline(alpha, l1_ratio)
            oof = np.full(len(y), np.nan)
            for tr_idx, va_idx in splits:
                pipe.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = pipe.predict(X[va_idx])
            valid = ~np.isnan(oof)
            mae = mean_absolute_error(y[valid], oof[valid])
            if mae < best_mae:
                best_mae, best_params = mae, (alpha, l1_ratio)
    return best_params


def fit_pipeline(X: np.ndarray, y: np.ndarray,
                 alpha: float, l1_ratio: float,
                 sample_weight: np.ndarray | None = None):
    pipe = _make_pipeline(alpha, l1_ratio)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["regressor__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_kwargs)
    sigma = float(np.std(y - pipe.predict(X)))
    return pipe, sigma


def recency_weights(seasons: np.ndarray) -> np.ndarray:
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
    """係数を JSON に保存"""
    scaler    = pipe.named_steps["scaler"]
    regressor = pipe.named_steps["regressor"]
    coefs = {
        label: {
            name: {"coef": round(float(c), 4),
                   "mean": round(float(scaler.mean_[i]), 4),
                   "std":  round(float(scaler.scale_[i]), 4)}
            for i, (name, c) in enumerate(zip(feat_names, regressor.coef_))
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
            "bayes_feat_h":        BAYES_FEAT_H,
            "bayes_feat_p":        BAYES_FEAT_P,
            "alphas_grid":         ALPHAS,
            "l1_ratios_grid":      L1_RATIOS,
            "recency_decay":       RECENCY_DECAY,
            "min_prev_season_bat": MIN_PREV_SEASON_BAT,
            "min_prev_season_pit": MIN_PREV_SEASON_PIT,
        }
    )

    coef_path = PRED_DIR / "bayes_coef.json"

    # 動的 park_factor ロード
    pf_lookup = _load_park_factors()
    if pf_lookup:
        print(f"  park_factors loaded: {len(pf_lookup)} entries (dynamic)")
    else:
        print("  park_factors.csv not found — using static _PF_FALLBACK")

    # LightGBM OOF ロード（スタッキング用）
    lgb_oof_bat = _load_lgb_oof(RAW_DIR / "lgb_oof_batter.csv")
    lgb_oof_pit = _load_lgb_oof(RAW_DIR / "lgb_oof_pitcher.csv")
    print(f"  lgb_oof_batter:  {'loaded' if lgb_oof_bat else 'not found (lgb_delta=NaN)'}")
    print(f"  lgb_oof_pitcher: {'loaded' if lgb_oof_pit else 'not found (lgb_delta=NaN)'}")

    # ===== 打者 wOBA (min_prev={MIN_PREV_SEASON_BAT}) =====
    print(f"=== Batter Bayes (wOBA, min_prev={MIN_PREV_SEASON_BAT}) ===")
    bat_df = pd.read_csv(RAW_DIR / "batter_features.csv")
    delta_bat = build_delta_dataset(
        bat_df, _BAT_RAW, _eng_batter, marcel_woba, MLB_AVG_WOBA, "wOBA",
        min_prev_season=MIN_PREV_SEASON_BAT,
        pf_lookup=pf_lookup,
        lgb_oof=lgb_oof_bat,
    )
    print(f"  delta dataset: {len(delta_bat)} samples")

    feat_cols_bat = BAYES_FEAT_H
    X_bat = delta_bat[feat_cols_bat].values.astype(float)
    y_bat = delta_bat["delta"].values.astype(float)
    w_bat = recency_weights(delta_bat["season"].values)

    seasons_bat = delta_bat["season"].values.astype(int)
    alpha_bat, l1_bat   = find_hyperparams(X_bat, y_bat, seasons_bat, ALPHAS, L1_RATIOS)
    pipe_bat, sigma_bat = fit_pipeline(X_bat, y_bat, alpha_bat, l1_bat, w_bat)
    # 計測も同じ時系列 splits で（find_hyperparams 内部と一貫）
    oof_bat_cv = np.full(len(y_bat), np.nan)
    for tr_i, va_i in _time_cv_splits(seasons_bat):
        _p = _make_pipeline(alpha_bat, l1_bat)
        _p.fit(X_bat[tr_i], y_bat[tr_i])
        oof_bat_cv[va_i] = _p.predict(X_bat[va_i])
    valid_bat = ~np.isnan(oof_bat_cv)
    bayes_mae_bat = float(mean_absolute_error(y_bat[valid_bat], oof_bat_cv[valid_bat]))
    print(f"  alpha={alpha_bat}, l1={l1_bat}, sigma={sigma_bat:.4f}, delta MAE={bayes_mae_bat:.4f}")
    save_coef(pipe_bat, feat_cols_bat, coef_path, "batter")

    # ===== 投手 xFIP (min_prev={MIN_PREV_SEASON_PIT}) =====
    print(f"=== Pitcher Bayes (xFIP, min_prev={MIN_PREV_SEASON_PIT}) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    delta_pit = build_delta_dataset(
        pit_df, _PIT_RAW, _eng_pitcher, marcel_xfip, MLB_AVG_XFIP, "xFIP",
        min_prev_season=MIN_PREV_SEASON_PIT,
        pf_lookup=pf_lookup,
        lgb_oof=lgb_oof_pit,
    )
    print(f"  delta dataset: {len(delta_pit)} samples")

    feat_cols_pit = BAYES_FEAT_P
    X_pit = delta_pit[feat_cols_pit].values.astype(float)
    y_pit = delta_pit["delta"].values.astype(float)
    w_pit = recency_weights(delta_pit["season"].values)

    seasons_pit = delta_pit["season"].values.astype(int)
    alpha_pit, l1_pit   = find_hyperparams(X_pit, y_pit, seasons_pit, ALPHAS, L1_RATIOS)
    pipe_pit, sigma_pit = fit_pipeline(X_pit, y_pit, alpha_pit, l1_pit, w_pit)
    oof_pit_cv = np.full(len(y_pit), np.nan)
    for tr_i, va_i in _time_cv_splits(seasons_pit):
        _p = _make_pipeline(alpha_pit, l1_pit)
        _p.fit(X_pit[tr_i], y_pit[tr_i])
        oof_pit_cv[va_i] = _p.predict(X_pit[va_i])
    valid_pit = ~np.isnan(oof_pit_cv)
    bayes_mae_pit = float(mean_absolute_error(y_pit[valid_pit], oof_pit_cv[valid_pit]))
    print(f"  alpha={alpha_pit}, l1={l1_pit}, sigma={sigma_pit:.4f}, delta MAE={bayes_mae_pit:.4f}")
    save_coef(pipe_pit, feat_cols_pit, coef_path, "pitcher")

    # ===== W&B ログ =====
    regressor_bat = pipe_bat.named_steps["regressor"]
    regressor_pit = pipe_pit.named_steps["regressor"]
    top5_bat = sorted(zip(feat_cols_bat, regressor_bat.coef_),
                      key=lambda x: abs(x[1]), reverse=True)[:5]
    top5_pit = sorted(zip(feat_cols_pit, regressor_pit.coef_),
                      key=lambda x: abs(x[1]), reverse=True)[:5]

    log_dict = {
        "alpha_batter":            alpha_bat,
        "l1_ratio_batter":         l1_bat,
        "alpha_pitcher":           alpha_pit,
        "l1_ratio_pitcher":        l1_pit,
        "sigma_batter":            sigma_bat,
        "sigma_pitcher":           sigma_pit,
        "bayes_batter_delta_MAE":  bayes_mae_bat,
        "bayes_pitcher_delta_MAE": bayes_mae_pit,
        "n_samples_batter":        len(delta_bat),
        "n_samples_pitcher":       len(delta_pit),
        "lgb_stacking_bat":        lgb_oof_bat is not None,
        "lgb_stacking_pit":        lgb_oof_pit is not None,
    }
    for feat, coef in top5_bat:
        log_dict[f"coef_bat_{feat}"] = round(coef, 4)
    for feat, coef in top5_pit:
        log_dict[f"coef_pit_{feat}"] = round(coef, 4)
    wandb.log(log_dict)
    run_wb.finish()

    # ===== predictions CSV に bayes 列を追記 =====
    _update_batter_predictions(bat_df, pipe_bat, sigma_bat, pf_lookup=pf_lookup)
    _update_pitcher_predictions(pit_df, pipe_pit, sigma_pit, pf_lookup=pf_lookup)

    print("=== train_bayes.py complete ===")


def _update_batter_predictions(bat_df: pd.DataFrame, pipe: Pipeline, sigma: float,
                                pf_lookup: dict | None = None):
    pred_path = PRED_DIR / "batter_predictions.csv"
    if not pred_path.exists():
        return
    bat_pred = pd.read_csv(pred_path)
    rows = []
    for _, row in bat_pred.iterrows():
        player      = row["player"]
        season_last = int(row.get("season_last", bat_df["season"].max()))
        prev = bat_df[(bat_df["player"] == player) & (bat_df["season"] == season_last)]

        if len(prev) == 0:
            rows.append({"bayes_woba": row.get("marcel_woba"), "ci_lo80": np.nan, "ci_hi80": np.nan})
            continue

        prev_row  = prev.iloc[0]
        raw_feats = {
            f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
            for f in _BAT_RAW if f != "lgb_delta"
        }

        # LightGBM delta (pred_woba は train.py が出力済み)
        lgb_pred = float(row.get("pred_woba", np.nan))
        marcel_v = float(row.get("marcel_woba", MLB_AVG_WOBA))
        raw_feats["lgb_delta"] = (lgb_pred - marcel_v) if not np.isnan(lgb_pred) else np.nan

        eng_feats = _eng_batter(prev_row)

        # 動的 park_factor
        if pf_lookup:
            pf = pf_lookup.get((season_last, str(prev_row.get("Team", "")).strip()))
            if pf:
                eng_feats["park_factor"] = pf

        # チーム変更フラグ・G前年比
        prev2     = bat_df[(bat_df["player"] == player) & (bat_df["season"] == season_last - 1)]
        team_cur  = str(prev_row.get("Team", ""))
        g_cur     = float(prev_row.get("G") or 1)
        if len(prev2) > 0:
            team_prev = str(prev2.iloc[0].get("Team", ""))
            g_prev    = float(prev2.iloc[0].get("G") or g_cur)
        else:
            team_prev, g_prev = team_cur, g_cur
        eng_feats["team_changed"]  = int(team_cur != team_prev)
        eng_feats["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

        feat_vals = np.array([raw_feats.get(f, eng_feats.get(f, np.nan)) for f in BAYES_FEAT_H])

        delta_hat, ci_lo, ci_hi = predict_with_ci(feat_vals, pipe, sigma)
        rows.append({
            "bayes_woba": round(marcel_v + delta_hat, 3),
            "ci_lo80":    round(marcel_v + ci_lo, 3),
            "ci_hi80":    round(marcel_v + ci_hi, 3),
        })

    bayes_df = pd.DataFrame(rows)
    for col in ["bayes_woba", "ci_lo80", "ci_hi80"]:
        if col in bat_pred.columns:
            bat_pred = bat_pred.drop(columns=[col])
    bat_pred = pd.concat([bat_pred.reset_index(drop=True), bayes_df.reset_index(drop=True)], axis=1)
    bat_pred.to_csv(pred_path, index=False)
    print(f"Batter predictions updated: {pred_path}")


def _update_pitcher_predictions(pit_df: pd.DataFrame, pipe: Pipeline, sigma: float,
                                 pf_lookup: dict | None = None):
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
        raw_feats = {
            f: (float(prev_row[f]) if f in prev_row.index and pd.notna(prev_row.get(f)) else np.nan)
            for f in _PIT_RAW if f != "lgb_delta"
        }

        # LightGBM delta
        lgb_pred = float(row.get("pred_xfip", np.nan))
        marcel_v = float(row.get("marcel_xfip", MLB_AVG_XFIP))
        raw_feats["lgb_delta"] = (lgb_pred - marcel_v) if not np.isnan(lgb_pred) else np.nan

        eng_feats = _eng_pitcher(prev_row)

        # 動的 park_factor
        if pf_lookup:
            pf = pf_lookup.get((season_last, str(prev_row.get("Team", "")).strip()))
            if pf:
                eng_feats["park_factor"] = pf

        # チーム変更フラグ・G前年比
        prev2     = pit_df[(pit_df["player"] == player) & (pit_df["season"] == season_last - 1)]
        team_cur  = str(prev_row.get("Team", ""))
        g_cur     = float(prev_row.get("G") or 1)
        if len(prev2) > 0:
            team_prev = str(prev2.iloc[0].get("Team", ""))
            g_prev    = float(prev2.iloc[0].get("G") or g_cur)
        else:
            team_prev, g_prev = team_cur, g_cur
        eng_feats["team_changed"]  = int(team_cur != team_prev)
        eng_feats["g_change_rate"] = round(g_cur / max(g_prev, 1), 3)

        feat_vals = np.array([raw_feats.get(f, eng_feats.get(f, np.nan)) for f in BAYES_FEAT_P])

        delta_hat, ci_lo, ci_hi = predict_with_ci(feat_vals, pipe, sigma)
        rows.append({
            "bayes_xfip": round(marcel_v + delta_hat, 2),
            "ci_lo80":    round(marcel_v + ci_lo, 2),
            "ci_hi80":    round(marcel_v + ci_hi, 2),
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
