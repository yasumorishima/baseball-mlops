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

import pandas as pd
import pybaseball as pb

pb.cache.enable()

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_SEASON = 2015
END_SEASON = 2024


# ---------------------------------------------------------------------------
# 打者
# ---------------------------------------------------------------------------

def fetch_batting_fangraphs(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """FanGraphs 打者成績（wOBA/xwOBA/K%/BB%/ISO/BABIP/WAR等）"""
    print(f"FanGraphs batting {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.batting_stats(year, year, qual=50)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "fg_batting.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_batting_exitvelo(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 打者 打球速度・バレル率"""
    print(f"Statcast batter exit velo {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.statcast_batter_exitvelo_barrels(year, minBBE=50)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_batter_exitvelo.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_batting_expected(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 打者 期待値指標 (xBA/xSLG/xwOBA)"""
    print(f"Statcast batter expected stats {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.statcast_batter_expected_stats(year, minPA=50)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_batter_expected.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_sprint_speed(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast スプリントスピード"""
    print(f"Statcast sprint speed {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.statcast_sprint_speed(year)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_sprint_speed.csv", index=False)
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
        df = pb.pitching_stats(year, year, qual=30)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "fg_pitching.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_pitching_exitvelo(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 投手 被打球速度・バレル率"""
    print(f"Statcast pitcher exit velo {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.statcast_pitcher_exitvelo_barrels(year, minBBE=50)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_pitcher_exitvelo.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


def fetch_pitching_expected(start: int = START_SEASON, end: int = END_SEASON) -> pd.DataFrame:
    """Statcast 投手 期待値指標 (xERA/xwOBA被打球)"""
    print(f"Statcast pitcher expected stats {start}-{end} ...")
    frames = []
    for year in range(start, end + 1):
        df = pb.statcast_pitcher_expected_stats(year, minPA=50)
        df["Season"] = year
        frames.append(df)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DIR / "sc_pitcher_expected.csv", index=False)
    print(f"  → {len(out)} rows saved")
    return out


# ---------------------------------------------------------------------------
# マージ
# ---------------------------------------------------------------------------

def _sc_player_col(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast CSV の選手名カラムを統一し season カラムを生成する"""
    # player 名の構築（"last_name, first_name" 結合型 or 別カラム型どちらにも対応）
    if "last_name, first_name" in df.columns:
        df["player"] = df["last_name, first_name"].str.split(", ").apply(
            lambda x: f"{x[1]} {x[0]}" if len(x) == 2 else x[0]
        )
    elif "last_name" in df.columns and "first_name" in df.columns:
        df["player"] = df["first_name"] + " " + df["last_name"]
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

    merged = fg.merge(sc, on=["player", "season"], how="left")
    out_path = DATA_DIR / "batter_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Batter features: {len(merged)} rows, cols sample: {list(merged.columns[:8])}")
    return merged


def build_pitcher_features() -> pd.DataFrame:
    """FanGraphs + Statcast をマージして投手特徴量テーブルを構築"""
    fg = pd.read_csv(DATA_DIR / "fg_pitching.csv")
    ev = pd.read_csv(DATA_DIR / "sc_pitcher_exitvelo.csv")
    exp = pd.read_csv(DATA_DIR / "sc_pitcher_expected.csv")

    fg = fg.rename(columns={"playerid": "fg_id", "Name": "player", "Season": "season"})
    ev = _sc_player_col(ev)
    exp = _sc_player_col(exp)

    exp_cols = ["player", "season"] + [
        c for c in ("est_ba", "est_slg", "est_woba", "est_era")
        if c in exp.columns
    ]
    sc = ev.merge(exp[exp_cols], on=["player", "season"], how="left")

    merged = fg.merge(sc, on=["player", "season"], how="left")
    out_path = DATA_DIR / "pitcher_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Pitcher features: {len(merged)} rows, cols sample: {list(merged.columns[:8])}")
    return merged


if __name__ == "__main__":
    # 全データ取得（GitHub Actions で実行）
    fetch_batting_fangraphs()
    fetch_batting_exitvelo()
    fetch_batting_expected()
    fetch_sprint_speed()
    fetch_pitching_fangraphs()
    fetch_pitching_exitvelo()
    fetch_pitching_expected()
    build_batter_features()
    build_pitcher_features()
    print("All done.")
