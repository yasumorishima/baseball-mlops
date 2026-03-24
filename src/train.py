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
import optuna
import wandb
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROJ_DIR = DATA_DIR / "projections"
MODELS_DIR = Path(__file__).parent.parent / "models"
# Streamlit Cloud 用（git 管理対象）
PRED_DIR = Path(__file__).parent.parent / "predictions"
PROJ_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

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

# FanGraphs カラム名 (大文字) + Statcast カラム名 (小文字 / snake_case)
BATTER_FEATURES = [
    # FanGraphs (既存)
    "wOBA", "xwOBA", "K%", "BB%", "ISO", "BABIP", "OBP", "SLG",
    # FanGraphs (追加 v5)
    "SwStr%", "HardHit%", "Contact%", "O-Swing%", "G",
    # v11: FanGraphs 主要指標（全年利用可能）
    "wRC+", "WAR", "Off", "Def", "BsR", "Spd",
    "AVG", "OPS", "wRAA",
    "HR/FB",
    # v11: FanGraphs 打球タイプ（全年利用可能）
    "GB%", "FB%", "LD%", "IFFB%",
    "Pull%", "Cent%", "Oppo%",
    "Soft%", "Med%", "Hard%",
    # v11: FanGraphs ゾーン別スイング・コンタクト
    "O-Contact%", "Z-Contact%", "Z-Swing%",
    # v11: FanGraphs 球種別打撃価値
    "wFB/C", "wSL/C", "wCH/C",
    # Statcast expected stats
    "est_ba", "est_slg", "est_woba",
    # Statcast exit velo / barrels
    "avg_hit_speed", "avg_hit_angle", "brl_percent", "ev95percent",
    "anglesweetspotpercent",
    # Statcast sprint speed
    "sprint_speed",
    # v8: Bat Tracking (2024+ only, NaN for earlier years — LightGBM handles natively)
    "avg_bat_speed", "swing_tilt", "attack_angle",
    "ideal_attack_angle_rate",
    # v8: Batted ball direction
    "pull_rate", "oppo_rate",
    # ====================================================================
    # v11: BQ pitch-level aggregated features (from mlb_wp.statcast_pitches)
    # ====================================================================
    # Plate discipline (pitch selection quality)
    "bq_whiff_rate", "bq_chase_rate", "bq_zone_contact_rate",
    "bq_zone_swing_rate", "bq_called_strike_rate",
    "bq_first_pitch_swing_rate", "bq_swing_rate",
    "bq_two_strike_whiff_rate",
    # Batted ball profile
    "bq_gb_rate", "bq_fb_rate", "bq_ld_rate", "bq_popup_rate",
    "bq_sweet_spot_rate", "bq_avg_hit_distance",
    # Power / exit velocity quality
    "bq_avg_ev", "bq_max_ev", "bq_ev_p90", "bq_ev_consistency",
    "bq_avg_la", "bq_hard_hit_rate", "bq_barrel_rate",
    # Expected stats (pitch-level)
    "bq_avg_xwoba", "bq_avg_xba", "bq_avg_woba_value",
    "bq_avg_babip_value", "bq_avg_iso_value",
    # Bat tracking (2024+)
    "bq_avg_bat_speed", "bq_avg_swing_length", "bq_avg_attack_angle",
    "bq_bat_speed_consistency", "bq_max_bat_speed",
    # Run values
    "bq_avg_run_value", "bq_total_run_value",
    # Pitch mix faced
    "bq_avg_velo_faced", "bq_fastball_faced_pct",
    "bq_breaking_faced_pct", "bq_offspeed_faced_pct",
    # Count leverage
    "bq_hitter_count_pct", "bq_pitcher_count_pct",
    # Baserunning
    "bq_sb_success_rate", "bq_sb_attempt_rate",
    # 基本情報
    "Age", "PA",
]

PITCHER_FEATURES = [
    # FanGraphs (既存)
    "xFIP", "FIP", "ERA", "K%", "BB%", "HR/9", "WHIP", "BABIP", "LOB%",
    # FanGraphs (追加 v5)
    "SwStr%", "K-BB%", "CSW%", "G",
    # FanGraphs Stuff+/Location+/Pitching+ (2020+, NaN for earlier)
    "Stuff+", "Location+", "Pitching+",
    # v11: FanGraphs 主要指標
    "WAR", "SIERA", "ERA-", "FIP-", "xFIP-",
    "K/9", "BB/9", "K/BB", "HR/FB",
    # v11: FanGraphs 打球タイプ
    "GB%", "FB%", "LD%", "IFFB%",
    "Pull%", "Cent%", "Oppo%",
    "Soft%", "Med%", "Hard%",
    # v11: FanGraphs ゾーン別スイング・コンタクト
    "O-Swing%", "Z-Swing%", "O-Contact%", "Z-Contact%", "Zone%",
    # v11: FanGraphs 球種別投球価値
    "wFB/C", "wSL/C", "wCH/C",
    # v11: FanGraphs 先発/リリーフ
    "GS", "Start-IP", "Relief-IP",
    # Statcast expected stats
    "est_ba", "est_slg", "est_woba", "xera",
    # Statcast exit velo (被打球)
    "avg_hit_speed", "avg_hit_angle", "brl_percent", "ev95percent",
    # v8: Pitch-level arsenal features (aggregated per pitcher)
    "n_pitch_types", "primary_usage", "best_whiff",
    "avg_whiff_weighted", "best_rv100", "usage_entropy",
    # ====================================================================
    # v11: BQ pitch-level aggregated features (from mlb_wp.statcast_pitches)
    # ====================================================================
    # Stuff (velocity, spin, movement)
    "bq_avg_velo", "bq_max_velo", "bq_velo_consistency",
    "bq_avg_spin", "bq_avg_h_break", "bq_avg_v_break",
    "bq_total_movement", "bq_avg_extension", "bq_avg_arm_angle",
    # Fastball detail
    "bq_fb_velo", "bq_fb_spin", "bq_fb_rise", "bq_fb_h_break",
    # Breaking ball detail
    "bq_brk_velo", "bq_brk_spin", "bq_brk_h_break", "bq_brk_v_break",
    # Offspeed detail
    "bq_ch_velo", "bq_ch_drop", "bq_fb_ch_velo_diff",
    # Command (location, zone, consistency)
    "bq_zone_rate", "bq_edge_rate", "bq_first_pitch_strike_rate",
    "bq_location_x_consistency", "bq_location_z_consistency",
    "bq_release_x_consistency", "bq_release_z_consistency",
    # Whiff / Chase
    "bq_whiff_rate", "bq_chase_rate_induced", "bq_csw_rate",
    # Contact management
    "bq_avg_ev_against", "bq_avg_la_against",
    "bq_hard_hit_rate_against", "bq_barrel_rate_against",
    "bq_gb_rate_induced", "bq_fb_rate_induced",
    "bq_avg_xwoba_against", "bq_avg_xba_against",
    "bq_contact_rate_against",
    # Arsenal detail
    "bq_n_pitch_types", "bq_fastball_pct",
    "bq_breaking_pct", "bq_offspeed_pct",
    # Fatigue / Times Through Order
    "bq_rv_1st_time", "bq_rv_2nd_time", "bq_rv_3rd_time",
    "bq_tto_degradation",
    # Run values
    "bq_avg_pitcher_run_value",
    "bq_rv_fastball", "bq_rv_breaking", "bq_rv_offspeed",
    # 基本情報
    "Age", "IP",
]


def _safe(val):
    """NaN-safe check"""
    try:
        return not np.isnan(val)
    except (TypeError, ValueError):
        return val is not None


def _bat_delta_features(feats: dict) -> None:
    """打者ラグ差分・交互作用項・エンジニアリング特徴量をインプレースで追加"""
    def _d(a, b):
        v = feats.get(a, np.nan), feats.get(b, np.nan)
        return v[0] - v[1] if (_safe(v[0]) and _safe(v[1])) else np.nan

    # --- v6: 1年ラグ差分 ---
    feats["wOBA_delta_1"]   = _d("wOBA_y1",        "wOBA_y2")
    feats["xwOBA_delta_1"]  = _d("xwOBA_y1",       "xwOBA_y2")
    feats["K_pct_delta_1"]  = _d("K%_y1",          "K%_y2")
    feats["BB_pct_delta_1"] = _d("BB%_y1",         "BB%_y2")
    feats["brl_delta_1"]    = _d("brl_percent_y1",  "brl_percent_y2")

    # --- v7: 2年ラグ差分（長期トレンド） ---
    feats["wOBA_delta_2"]   = _d("wOBA_y2",        "wOBA_y3")

    # --- v8: Bat tracking deltas (2024+ only) ---
    feats["bat_speed_delta_1"] = _d("avg_bat_speed_y1", "avg_bat_speed_y2")

    # --- v11: BQ pitch-level deltas ---
    feats["bq_whiff_delta_1"]  = _d("bq_whiff_rate_y1", "bq_whiff_rate_y2")
    feats["bq_chase_delta_1"]  = _d("bq_chase_rate_y1", "bq_chase_rate_y2")
    feats["bq_ev_delta_1"]     = _d("bq_avg_ev_y1", "bq_avg_ev_y2")
    feats["bq_barrel_delta_1"] = _d("bq_barrel_rate_y1", "bq_barrel_rate_y2")
    feats["bq_ld_delta_1"]     = _d("bq_ld_rate_y1", "bq_ld_rate_y2")

    # --- v6: 交互作用 ---
    xwoba = feats.get("xwOBA_y1", np.nan)
    woba  = feats.get("wOBA_y1",  np.nan)
    age   = feats.get("Age_y1",   np.nan)
    luck  = (xwoba - woba) if (_safe(xwoba) and _safe(woba)) else np.nan
    feats["age_x_luck"] = age * luck if (_safe(age) and _safe(luck)) else np.nan

    # --- v7: エンジニアリング特徴量 ---
    # 年齢曲線（ピーク27歳）
    if _safe(age):
        feats["age_from_peak"] = age - 27
        feats["age_sq"] = (age - 27) ** 2
    else:
        feats["age_from_peak"] = np.nan
        feats["age_sq"] = np.nan

    # 出場率（健康・レギュラー度合い）
    pa = feats.get("PA_y1", np.nan)
    feats["pa_rate"] = pa / 650.0 if _safe(pa) else np.nan

    # 運の乖離（翌年回帰の強い予測因子）
    feats["xwoba_luck"] = luck if _safe(luck) else np.nan


def _pit_delta_features(feats: dict) -> None:
    """投手ラグ差分・交互作用項・エンジニアリング特徴量をインプレースで追加"""
    def _d(a, b):
        v = feats.get(a, np.nan), feats.get(b, np.nan)
        return v[0] - v[1] if (_safe(v[0]) and _safe(v[1])) else np.nan

    # --- v6: 1年ラグ差分 ---
    feats["xFIP_delta_1"]   = _d("xFIP_y1",   "xFIP_y2")
    feats["K_pct_delta_1"]  = _d("K%_y1",     "K%_y2")
    feats["BB_pct_delta_1"] = _d("BB%_y1",    "BB%_y2")
    feats["KBB_delta_1"]    = _d("K-BB%_y1",  "K-BB%_y2")

    # --- v7: 2年ラグ差分（長期トレンド） ---
    feats["xFIP_delta_2"]   = _d("xFIP_y2",   "xFIP_y3")

    # --- v11: Stuff+/Pitching+ deltas (2020+) ---
    feats["stuff_plus_delta_1"]    = _d("Stuff+_y1", "Stuff+_y2")
    feats["pitching_plus_delta_1"] = _d("Pitching+_y1", "Pitching+_y2")

    # --- v8: Arsenal deltas ---
    feats["whiff_delta_1"]  = _d("best_whiff_y1",   "best_whiff_y2")
    feats["usage_entropy_delta_1"] = _d("usage_entropy_y1", "usage_entropy_y2")

    # --- v11: BQ pitch-level deltas ---
    feats["bq_velo_delta_1"]    = _d("bq_avg_velo_y1", "bq_avg_velo_y2")
    feats["bq_spin_delta_1"]    = _d("bq_avg_spin_y1", "bq_avg_spin_y2")
    feats["bq_whiff_delta_1"]   = _d("bq_whiff_rate_y1", "bq_whiff_rate_y2")
    feats["bq_zone_delta_1"]    = _d("bq_zone_rate_y1", "bq_zone_rate_y2")
    feats["bq_ev_ag_delta_1"]   = _d("bq_avg_ev_against_y1", "bq_avg_ev_against_y2")
    feats["bq_gb_delta_1"]      = _d("bq_gb_rate_induced_y1", "bq_gb_rate_induced_y2")

    # --- v6: 交互作用 ---
    kbb = feats.get("K-BB%_y1", np.nan)
    age = feats.get("Age_y1",   np.nan)
    feats["age_x_kbb"] = age * kbb if (_safe(age) and _safe(kbb)) else np.nan

    # --- v7: エンジニアリング特徴量 ---
    if _safe(age):
        feats["age_from_peak"] = age - 27
        feats["age_sq"] = (age - 27) ** 2
    else:
        feats["age_from_peak"] = np.nan
        feats["age_sq"] = np.nan

    ip = feats.get("IP_y1", np.nan)
    feats["ip_rate"] = ip / 200.0 if _safe(ip) else np.nan

    era = feats.get("ERA_y1", np.nan)
    fip = feats.get("FIP_y1", np.nan)
    feats["fip_era_gap"] = (era - fip) if (_safe(era) and _safe(fip)) else np.nan


def _load_park_factors() -> dict[tuple[str, int], float]:
    """Park factors CSV を (team, season) → pf_5yr の辞書に変換"""
    pf_path = RAW_DIR / "park_factors.csv"
    if not pf_path.exists():
        return {}
    pf_df = pd.read_csv(pf_path)
    # カラム名が異なる場合に対応
    team_col = "team" if "team" in pf_df.columns else "Team"
    season_col = "season" if "season" in pf_df.columns else "Season"
    pf_col = "pf_5yr" if "pf_5yr" in pf_df.columns else "basic_5yr"
    if pf_col not in pf_df.columns:
        # フォールバック: 最初の数値カラムを使う
        for c in pf_df.columns:
            if pf_df[c].dtype in ("float64", "int64") and c not in (season_col,):
                pf_col = c
                break
    result = {}
    for _, r in pf_df.iterrows():
        result[(str(r[team_col]), int(r[season_col]))] = float(r[pf_col])
    return result


_PARK_FACTORS: dict[tuple[str, int], float] | None = None


def _get_park_factors() -> dict[tuple[str, int], float]:
    global _PARK_FACTORS
    if _PARK_FACTORS is None:
        _PARK_FACTORS = _load_park_factors()
    return _PARK_FACTORS


def build_train_data_batters(df: pd.DataFrame, min_pa: int = 100):
    """翌年 wOBA をターゲットとした学習データを構築"""
    pf = _get_park_factors()
    records = []
    seasons = sorted(df["season"].unique())
    for year in seasons[3:]:  # 過去3年分の特徴量が必要
        targets = df[(df["season"] == year) & (df["PA"] >= min_pa)]
        for _, row in targets.iterrows():
            player = row["player"]
            feats = {}
            teams_by_lag = {}
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
                    teams_by_lag[lag] = str(prev.iloc[0].get("Team", ""))

            _bat_delta_features(feats)

            # v7: team_changed（y1 → y2 でチーム変更）
            t1, t2 = teams_by_lag.get(1, ""), teams_by_lag.get(2, "")
            feats["team_changed"] = int(t1 != t2 and t1 != "" and t2 != "")

            # v7: park_factor（直前シーズンのチームの球場補正）
            feats["park_factor"] = pf.get((t1, year - 1), np.nan) if t1 else np.nan

            # Marcel ベースライン
            feats["marcel_woba"] = marcel_woba(df, player, year) or np.nan

            feats["target_woba"] = row["wOBA"]
            feats["player"] = player
            feats["season"] = year
            records.append(feats)

    return pd.DataFrame(records)


def build_train_data_pitchers(df: pd.DataFrame, min_ip: int = 30):
    """翌年 xFIP をターゲットとした学習データを構築"""
    pf = _get_park_factors()
    records = []
    seasons = sorted(df["season"].unique())
    for year in seasons[3:]:
        targets = df[(df["season"] == year) & (df["IP"] >= min_ip)]
        for _, row in targets.iterrows():
            player = row["player"]
            feats = {}
            teams_by_lag = {}
            for lag in range(1, 4):
                prev = df[(df["player"] == player) & (df["season"] == year - lag)]
                suffix = f"_y{lag}"
                if len(prev) == 0:
                    for f in PITCHER_FEATURES:
                        feats[f + suffix] = np.nan
                else:
                    for f in PITCHER_FEATURES:
                        feats[f + suffix] = prev.iloc[0].get(f, np.nan)
                    teams_by_lag[lag] = str(prev.iloc[0].get("Team", ""))

            _pit_delta_features(feats)

            # v7: team_changed / park_factor
            t1, t2 = teams_by_lag.get(1, ""), teams_by_lag.get(2, "")
            feats["team_changed"] = int(t1 != t2 and t1 != "" and t2 != "")
            feats["park_factor"] = pf.get((t1, year - 1), np.nan) if t1 else np.nan

            feats["marcel_xfip"] = marcel_xfip(df, player, year) or np.nan
            feats["target_xfip"] = row["xFIP"]
            feats["player"] = player
            feats["season"] = year
            records.append(feats)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def _time_cv_splits(seasons: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """年単位 expanding window splits（最低2年の学習データを確保）。

    fold i: train = seasons < val_year, val = seasons == val_year
    unique_years の先頭2年はtraining onlyに使い、3年目以降をval年とする。
    """
    unique_years = sorted(np.unique(seasons))
    splits = []
    for val_year in unique_years[2:]:
        train_idx = np.where(seasons < val_year)[0]
        val_idx   = np.where(seasons == val_year)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# デフォルトパラメータ（Optuna 未実行時のフォールバック）
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
    "n_estimators": 500,
}

# Optuna で探索する最大木数（early_stopping で自動打ち切り）
_MAX_ESTIMATORS = 1000
_EARLY_STOPPING = 50
RECENCY_DECAY = 0.85  # v8: 近年データを重み付け（Bayesと同じ decay）


def _recency_weights(seasons: np.ndarray) -> np.ndarray:
    """v8: LightGBM学習にも recency weighting を適用"""
    max_s = seasons.max()
    return np.array([RECENCY_DECAY ** (max_s - s) for s in seasons])


def _cv_mae(params: dict, X: pd.DataFrame, y: pd.Series,
            seasons: np.ndarray) -> float:
    """時系列 CV で OOF MAE を計算（Optuna objective から呼ぶ）"""
    splits = _time_cv_splits(seasons)
    weights = _recency_weights(seasons)  # v8: recency weighting
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in splits:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            sample_weight=weights[tr_idx],  # v8
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx])
    valid = ~np.isnan(oof)
    return float(mean_absolute_error(y[valid], oof[valid]))


def tune_hyperparams(X: pd.DataFrame, y: pd.Series, seasons: np.ndarray,
                     n_trials: int = 100) -> dict:
    """Optuna で LightGBM ハイパーパラメータを最適化し、最良パラメータを返す。

    MedianPruner を使って見込みのないトライアルを早期打ち切りすることで
    100 トライアルで 150〜200 相当の探索効果を得る。
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":          "regression",
            "metric":             "mae",
            "verbosity":          -1,
            "n_estimators":       _MAX_ESTIMATORS,
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":       5,
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        return _cv_mae(params, X, y, seasons)

    sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective":   "regression",
        "metric":      "mae",
        "verbosity":   -1,
        "n_estimators": _MAX_ESTIMATORS,
        "bagging_freq": 5,
    })
    return best


def train_model(X: pd.DataFrame, y: pd.Series, params: dict,
                seasons: np.ndarray | None = None) -> tuple:
    """時系列 expanding-window CV で LightGBM を学習し MAE・最終モデル・OOF を返す。"""
    if seasons is not None:
        splits = _time_cv_splits(seasons)
    else:
        from sklearn.model_selection import KFold
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))

    weights = _recency_weights(seasons) if seasons is not None else None  # v8
    oof = np.full(len(y), np.nan)
    models = []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr = weights[tr_idx] if weights is not None else None  # v8
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,  # v8: recency weighting
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                              lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict(X_va)
        models.append(model)

    valid = ~np.isnan(oof)
    mae = mean_absolute_error(y[valid], oof[valid])
    # 全データで再学習（最終モデル）
    final = lgb.LGBMRegressor(**params)
    final.fit(X, y, sample_weight=weights)  # v8: recency weighting
    return final, mae, models, oof


def save_to_wandb(model, mae: float, target: str, feature_names: list, config: dict):
    """W&B にモデル・メトリクス・特徴量重要度を記録し、MAE が改善したら production タグを昇格"""
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run = wandb.init(project="baseball-mlops", entity=entity, job_type="train",
                     config={**config, "target": target})

    wandb.log({f"MAE_{target}": mae})

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    wandb.log({f"feature_importance_{target}": wandb.Table(dataframe=importance.head(20))})

    # モデルを Artifact として保存（MAE をメタデータに記録）
    artifact = wandb.Artifact(f"{target}-model", type="model",
                               description=f"LightGBM {target} predictor, MAE={mae:.4f}",
                               metadata={"mae": mae})
    model_path = MODELS_DIR / f"{target}_model.pkl"
    joblib.dump(model, model_path)
    artifact.add_file(str(model_path))
    run.log_artifact(artifact, aliases=["latest"])

    run.finish()

    # production タグ: 既存 production より MAE が小さければ自動昇格
    api = wandb.Api()
    prefix = f"{entity}/" if entity else ""
    art_path = f"{prefix}baseball-mlops/{target}-model:latest"
    try:
        prod = api.artifact(f"{prefix}baseball-mlops/{target}-model:production")
        prod_mae = prod.metadata.get("mae", float("inf"))
        if mae < prod_mae:
            art = api.artifact(art_path)
            if "production" not in art.aliases:
                art.aliases.append("production")
                art.save()
            print(f"  → production 昇格 (MAE {mae:.4f} < {prod_mae:.4f})")
        else:
            print(f"  → production 据え置き (MAE {mae:.4f} >= {prod_mae:.4f})")
    except Exception:
        # production が未存在 → 初回は必ず昇格
        art = api.artifact(art_path)
        if "production" not in art.aliases:
            art.aliases.append("production")
            art.save()
        print(f"  → production 初回登録 (MAE {mae:.4f})")

    return artifact


OPTUNA_TRIALS = 1000


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
    seasons_bat = train_bat["season"].values

    print(f"  Optuna tuning ({OPTUNA_TRIALS} trials) ...")
    best_params_bat = tune_hyperparams(X_bat, y_bat, seasons_bat, n_trials=OPTUNA_TRIALS)
    print(f"  best: lr={best_params_bat['learning_rate']:.4f}, "
          f"leaves={best_params_bat['num_leaves']}, "
          f"min_child={best_params_bat['min_child_samples']}")

    model_bat, mae_bat, _, oof_bat = train_model(
        X_bat, y_bat, best_params_bat, seasons=seasons_bat
    )
    print(f"  ML  MAE wOBA: {mae_bat:.4f}")

    # Marcel ベースライン MAE
    marcel_mae_bat = mean_absolute_error(y_bat, train_bat["marcel_woba"].fillna(MLB_AVG_WOBA))
    print(f"  Marcel MAE wOBA: {marcel_mae_bat:.4f}")

    # OOF 保存（train_bayes.py のスタッキング用）
    # time-series CV では先頭2年分は val に入らないため NaN → 除外
    oof_mask_bat = ~np.isnan(oof_bat)
    pd.DataFrame({
        "player": train_bat["player"].values[oof_mask_bat],
        "season": train_bat["season"].values[oof_mask_bat],
        "lgb_woba_oof": oof_bat[oof_mask_bat],
    }).to_csv(RAW_DIR / "lgb_oof_batter.csv", index=False)

    save_to_wandb(model_bat, mae_bat, "woba", feat_cols_bat,
                  {"marcel_mae": marcel_mae_bat, "n_samples": len(X_bat),
                   **{f"bat_{k}": v for k, v in best_params_bat.items()
                      if k in ("learning_rate", "num_leaves", "min_child_samples")}})

    # 投手
    print("=== Pitcher (xFIP) ===")
    pit_df = pd.read_csv(RAW_DIR / "pitcher_features.csv")
    train_pit = build_train_data_pitchers(pit_df)
    train_pit = train_pit.dropna(subset=["target_xfip"])

    feat_cols_pit = [c for c in train_pit.columns
                     if c not in ("player", "season", "target_xfip")]
    X_pit = train_pit[feat_cols_pit].apply(pd.to_numeric, errors="coerce")
    y_pit = train_pit["target_xfip"]
    seasons_pit = train_pit["season"].values

    print(f"  Optuna tuning ({OPTUNA_TRIALS} trials) ...")
    best_params_pit = tune_hyperparams(X_pit, y_pit, seasons_pit, n_trials=OPTUNA_TRIALS)
    print(f"  best: lr={best_params_pit['learning_rate']:.4f}, "
          f"leaves={best_params_pit['num_leaves']}, "
          f"min_child={best_params_pit['min_child_samples']}")

    model_pit, mae_pit, _, oof_pit = train_model(
        X_pit, y_pit, best_params_pit, seasons=seasons_pit
    )
    print(f"  ML  MAE xFIP: {mae_pit:.4f}")

    marcel_mae_pit = mean_absolute_error(y_pit, train_pit["marcel_xfip"].fillna(MLB_AVG_XFIP))
    print(f"  Marcel MAE xFIP: {marcel_mae_pit:.4f}")

    # OOF 保存（time-series CV: 先頭2年分は NaN → 除外）
    oof_mask_pit = ~np.isnan(oof_pit)
    pd.DataFrame({
        "player": train_pit["player"].values[oof_mask_pit],
        "season": train_pit["season"].values[oof_mask_pit],
        "lgb_xfip_oof": oof_pit[oof_mask_pit],
    }).to_csv(RAW_DIR / "lgb_oof_pitcher.csv", index=False)

    save_to_wandb(model_pit, mae_pit, "xfip", feat_cols_pit,
                  {"marcel_mae": marcel_mae_pit, "n_samples": len(X_pit),
                   **{f"pit_{k}": v for k, v in best_params_pit.items()
                      if k in ("learning_rate", "num_leaves", "min_child_samples")}})

    # 予測結果を保存
    bat_df_latest = bat_df[bat_df["season"] == bat_df["season"].max()]
    pit_df_latest = pit_df[pit_df["season"] == pit_df["season"].max()]

    bat_preds = _predict_next_season(model_bat, bat_df, bat_df_latest, feat_cols_bat,
                                     "wOBA", "pred_woba", marcel_woba, MLB_AVG_WOBA,
                                     delta_fn=_bat_delta_features)
    pit_preds = _predict_next_season(model_pit, pit_df, pit_df_latest, feat_cols_pit,
                                     "xFIP", "pred_xfip", marcel_xfip, MLB_AVG_XFIP,
                                     delta_fn=_pit_delta_features)

    bat_preds.to_csv(PROJ_DIR / "batter_predictions.csv", index=False)
    pit_preds.to_csv(PROJ_DIR / "pitcher_predictions.csv", index=False)
    # Streamlit Cloud 用（git 管理対象）にもコピー
    bat_preds.to_csv(PRED_DIR / "batter_predictions.csv", index=False)
    pit_preds.to_csv(PRED_DIR / "pitcher_predictions.csv", index=False)
    print(f"Predictions saved: {len(bat_preds)} batters, {len(pit_preds)} pitchers")

    # アンサンブル用に MAE を保存
    metrics = {"lgb_mae_woba": round(mae_bat, 4), "lgb_mae_xfip": round(mae_pit, 4),
               "marcel_mae_woba": round(marcel_mae_bat, 4), "marcel_mae_xfip": round(marcel_mae_pit, 4)}
    (PRED_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))


def _predict_next_season(model, full_df, latest_df, feat_cols,
                          target_col, pred_col, marcel_fn, avg_val,
                          delta_fn=None) -> pd.DataFrame:
    """最新シーズンの選手について翌年予測を生成"""
    pf = _get_park_factors()
    next_year = int(latest_df["season"].max()) + 1
    records = []
    for _, row in latest_df.iterrows():
        player = row["player"]
        feats = {}
        teams_by_lag = {}
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
                teams_by_lag[1] = str(row.get("Team", ""))
                if len(prev) > 0:
                    teams_by_lag[2] = str(prev.iloc[0].get("Team", ""))
            else:
                for f in [c for c in feat_cols if c.endswith(suffix)]:
                    base = f.replace(suffix, "")
                    feats[f] = prev.iloc[0].get(base, np.nan) if len(prev) > 0 else np.nan
                if len(prev) > 0:
                    teams_by_lag[lag] = str(prev.iloc[0].get("Team", ""))

        if delta_fn is not None:
            delta_fn(feats)

        # v7: team_changed / park_factor
        t1, t2 = teams_by_lag.get(1, ""), teams_by_lag.get(2, "")
        feats["team_changed"] = int(t1 != t2 and t1 != "" and t2 != "")
        feats["park_factor"] = pf.get((t1, next_year - 1), np.nan) if t1 else np.nan

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
