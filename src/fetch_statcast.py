"""
MLB Statcast + FanGraphs データ取得

FanGraphs Leaderboards (wOBA/xFIP直接提供) +
Statcast (EV/Barrel%/xwOBA/sprint speed等) をマージして特徴量を構築する。

Data sources:
- FanGraphs via pybaseball (batting_stats / pitching_stats)
- Baseball Savant via pybaseball (statcast_batter_exitvelo_barrels 等)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball as pb

pb.cache.enable()

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_SEASON = 2015
END_SEASON = 2025

MAX_RETRIES = 3
RETRY_DELAY = 5


def _fetch_with_retry(func, *args, **kwargs):
    """Savant API呼び出しをリトライ付きで実行（不正CSV対策）"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    retry {attempt}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# 打者
# ---------------------------------------------------------------------------

def fetch_batting_fangraphs(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """FanGraphs 打者成績（wOBA/xwOBA/K%/BB%/ISO/BABIP/WAR等）"""
    print(f"FanGraphs batting {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.batting_stats, year, year, qual=50)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  FG batting {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "fg_batting.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_batting_exitvelo(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 打者 打球速度・バレル率"""
    print(f"Statcast batter exit velo {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.statcast_batter_exitvelo_barrels, year, minBBE=50)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  exit velo {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_batter_exitvelo.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_batting_expected(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 打者 期待値指標 (xBA/xSLG/xwOBA)"""
    print(f"Statcast batter expected stats {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.statcast_batter_expected_stats, year, minPA=50)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  expected stats {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_batter_expected.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_batted_ball(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast batted ball direction stats (pull/oppo rates)"""
    print(f"Statcast batted ball {start}-{end} ...")
    from savant_extras import batted_ball
    frames = []
    for year in range(start, end + 1):
        try:
            df = batted_ball(year, player_type="batter", min_bbe="q")
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  batted ball {year} skipped: {e}")
    if not frames:
        print("  → no batted ball data")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_batted_ball.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_sprint_speed(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast スプリントスピード"""
    print(f"Statcast sprint speed {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.statcast_sprint_speed, year)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  sprint speed {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_sprint_speed.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_bat_tracking(start: int = 2024, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast bat tracking (Hawk-Eye, 2024+): bat_speed, swing_tilt, attack_angle等"""
    print(f"Statcast bat tracking {start}-{end} ...")
    from savant_extras import bat_tracking
    frames = []
    for year in range(max(start, 2024), end + 1):
        try:
            df = bat_tracking(f"{year}-03-20", f"{year}-11-05", player_type="batter", min_swings="q")
            df["Season"] = year
            frames.append(df)
            time.sleep(2)
        except Exception as e:
            print(f"  bat tracking {year} skipped: {e}")
    if not frames:
        print("  → no bat tracking data")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_bat_tracking.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


# ---------------------------------------------------------------------------
# 投手
# ---------------------------------------------------------------------------

def fetch_pitching_fangraphs(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """FanGraphs 投手成績（xFIP/K%/BB%/WHIP/BABIP/WAR等）"""
    print(f"FanGraphs pitching {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.pitching_stats, year, year, qual=30)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  FG pitching {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "fg_pitching.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_pitching_exitvelo(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 投手 被打球速度・バレル率"""
    print(f"Statcast pitcher exit velo {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.statcast_pitcher_exitvelo_barrels, year, minBBE=50)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  pitcher exit velo {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_pitcher_exitvelo.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_pitching_expected(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 投手 期待値指標 (xERA/xwOBA被打球)"""
    print(f"Statcast pitcher expected stats {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = _fetch_with_retry(pb.statcast_pitcher_expected_stats, year, minPA=50)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  pitcher expected {year} skipped after {MAX_RETRIES} retries: {e}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_pitcher_expected.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_pitcher_arsenal(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast pitch-level arsenal stats (per pitch type per pitcher)"""
    print(f"Statcast pitcher arsenal {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = pb.statcast_pitcher_arsenal_stats(year, minPA=25)
            df["Season"] = year
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  arsenal {year} skipped: {e}")
    if not frames:
        print("  → no arsenal data")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_pitcher_arsenal.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


# ---------------------------------------------------------------------------
# マージ
# ---------------------------------------------------------------------------

def _sc_player_col(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast CSV の選手名カラムを統一し season カラムを生成する"""
    # player 名の構築（複数の命名パターンに対応）
    if "last_name, first_name" in df.columns:
        df["player"] = df["last_name, first_name"].str.split(", ").apply(
            lambda x: f"{x[1]} {x[0]}" if len(x) == 2 else x[0]
        )
    elif "last_name" in df.columns and "first_name" in df.columns:
        df["player"] = df["first_name"] + " " + df["last_name"]
    elif "name" in df.columns and "player" not in df.columns:
        # savant_extras 形式: "Last, First" → "First Last"
        df["player"] = df["name"].apply(
            lambda x: f"{x.split(', ')[1]} {x.split(', ')[0]}" if ", " in str(x) else str(x)
        )
    # season カラム統一（year / Season → season）
    for src in ("year", "Season"):
        if src in df.columns and "season" not in df.columns:
            df = df.rename(columns={src: "season"})
    return df


def build_batter_features() -> pd.DataFrame:
    """FanGraphs + Statcast をマージして打者特徴量テーブルを構築"""
    fg = pd.read_csv(DATA_DIR / "fg_batting.csv")
    ev = pd.read_csv(DATA_DIR / "sc_batter_exitvelo.csv")
    exp = pd.read_csv(DATA_DIR / "sc_batter_expected.csv")
    spd = pd.read_csv(DATA_DIR / "sc_sprint_speed.csv")

    fg = fg.rename(columns={"playerid": "fg_id", "Name": "player", "Season": "season"})
    ev = _sc_player_col(ev)
    exp = _sc_player_col(exp)
    spd = _sc_player_col(spd)

    # pybaseball expected stats の実カラム名: est_ba / est_slg / est_woba
    exp_cols = ["player", "season"] + [
        c for c in ("est_ba", "est_slg", "est_woba")
        if c in exp.columns
    ]
    sc = ev.merge(exp[exp_cols], on=["player", "season"], how="left")
    if "sprint_speed" in spd.columns:
        sc = sc.merge(spd[["player", "season", "sprint_speed"]],
                      on=["player", "season"], how="left")

    # Bat tracking (2024+)
    bt_path = DATA_DIR / "sc_bat_tracking.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        bt = _sc_player_col(bt)
        bt_cols = ["player", "season"] + [c for c in bt.columns
                   if c in ("avg_bat_speed", "swing_tilt", "attack_angle",
                            "ideal_attack_angle_rate", "competitive_swings")]
        bt_cols = [c for c in bt_cols if c in bt.columns]
        if len(bt_cols) > 2:
            sc = sc.merge(bt[bt_cols], on=["player", "season"], how="left")

    # Batted ball (pull/oppo rates)
    bb_path = DATA_DIR / "sc_batted_ball.csv"
    if bb_path.exists():
        bb = pd.read_csv(bb_path)
        bb = _sc_player_col(bb)
        bb_cols = ["player", "season"] + [c for c in bb.columns
                   if c in ("pull_rate", "oppo_rate")]
        bb_cols = [c for c in bb_cols if c in bb.columns]
        if len(bb_cols) > 2:
            sc = sc.merge(bb[bb_cols], on=["player", "season"], how="left")

    merged = fg.merge(sc, on=["player", "season"], how="left")

    # BQ pitch-level aggregated features (from fetch_bq_features.py)
    bq_path = DATA_DIR / "bq_batter_features.csv"
    if bq_path.exists():
        bq = pd.read_csv(bq_path)
        bq_cols = ["player", "season"] + [c for c in bq.columns if c.startswith("bq_")]
        bq_cols = [c for c in bq_cols if c in bq.columns]
        merged = merged.merge(bq[bq_cols], on=["player", "season"], how="left")
        n_bq = merged[[c for c in merged.columns if c.startswith("bq_")]].notna().any(axis=1).sum()
        print(f"  BQ features merged: {n_bq}/{len(merged)} rows matched")

    out_path = DATA_DIR / "batter_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Batter features: {len(merged)} rows, {len(merged.columns)} cols")
    return merged


def _aggregate_pitcher_arsenal() -> pd.DataFrame:
    """pitch-type別データを投手×シーズンに集約"""
    path = DATA_DIR / "sc_pitcher_arsenal.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _sc_player_col(df)
    if "Season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"Season": "season"})

    records = []
    for (player, season), grp in df.groupby(["player", "season"]):
        n_types = len(grp)
        usage = grp["pitch_usage"].values if "pitch_usage" in grp.columns else np.array([1/n_types]*n_types)
        whiff = grp["whiff_percent"].values if "whiff_percent" in grp.columns else np.array([np.nan]*n_types)
        rv100 = grp["run_value_per_100"].values if "run_value_per_100" in grp.columns else np.array([np.nan]*n_types)

        records.append({
            "player": player,
            "season": season,
            "n_pitch_types": n_types,
            "primary_usage": float(np.nanmax(usage)) if len(usage) > 0 else np.nan,
            "best_whiff": float(np.nanmax(whiff)) if len(whiff) > 0 else np.nan,
            "avg_whiff_weighted": float(np.nansum(usage * whiff) / np.nansum(usage)) if np.nansum(usage) > 0 else np.nan,
            "best_rv100": float(np.nanmin(rv100)) if len(rv100) > 0 else np.nan,  # lower = better
            "usage_entropy": float(-np.nansum(usage * np.log2(np.clip(usage, 1e-10, 1)))) if len(usage) > 0 else np.nan,
        })
    return pd.DataFrame(records)


def build_pitcher_features() -> pd.DataFrame:
    """FanGraphs + Statcast をマージして投手特徴量テーブルを構築"""
    fg = pd.read_csv(DATA_DIR / "fg_pitching.csv")
    ev = pd.read_csv(DATA_DIR / "sc_pitcher_exitvelo.csv")
    exp = pd.read_csv(DATA_DIR / "sc_pitcher_expected.csv")

    fg = fg.rename(columns={"playerid": "fg_id", "Name": "player", "Season": "season"})
    ev = _sc_player_col(ev)
    exp = _sc_player_col(exp)

    exp_cols = ["player", "season"] + [
        c for c in ("est_ba", "est_slg", "est_woba", "xera")
        if c in exp.columns
    ]
    sc = ev.merge(exp[exp_cols], on=["player", "season"], how="left")

    # Arsenal features (aggregated)
    arsenal_agg = _aggregate_pitcher_arsenal()
    if len(arsenal_agg) > 0:
        sc = sc.merge(arsenal_agg, on=["player", "season"], how="left")

    merged = fg.merge(sc, on=["player", "season"], how="left")

    # BQ pitch-level aggregated features (from fetch_bq_features.py)
    bq_path = DATA_DIR / "bq_pitcher_features.csv"
    if bq_path.exists():
        bq = pd.read_csv(bq_path)
        bq_cols = ["player", "season"] + [c for c in bq.columns if c.startswith("bq_")]
        bq_cols = [c for c in bq_cols if c in bq.columns]
        merged = merged.merge(bq[bq_cols], on=["player", "season"], how="left")
        n_bq = merged[[c for c in merged.columns if c.startswith("bq_")]].notna().any(axis=1).sum()
        print(f"  BQ features merged: {n_bq}/{len(merged)} rows matched")

    out_path = DATA_DIR / "pitcher_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Pitcher features: {len(merged)} rows, {len(merged.columns)} cols")
    return merged


def fetch_park_factors():
    from savant_extras import park_factors_range
    df = park_factors_range(START_SEASON, END_SEASON)
    out_path = DATA_DIR / "park_factors.csv"
    df.to_csv(out_path, index=False)
    print(f"Park factors: {len(df)} rows ({df['season'].min()}-{df['season'].max()}), saved to {out_path}")
    return df


if __name__ == "__main__":
    # 全データ取得（GitHub Actions で実行）
    fetch_park_factors()
    fetch_batting_fangraphs()
    fetch_batting_exitvelo()
    fetch_batting_expected()
    fetch_batted_ball()        # NEW
    fetch_sprint_speed()
    fetch_bat_tracking()       # NEW
    fetch_pitching_fangraphs()
    fetch_pitching_exitvelo()
    fetch_pitching_expected()
    fetch_pitcher_arsenal()    # NEW

    # BQ pitch-level features (optional — requires GCP credentials)
    try:
        from fetch_bq_features import fetch_batter_bq_features, fetch_pitcher_bq_features
        fetch_batter_bq_features()
        fetch_pitcher_bq_features()
    except Exception as e:
        print(f"BQ feature fetch skipped: {e}")

    build_batter_features()
    build_pitcher_features()
    print("All done.")
