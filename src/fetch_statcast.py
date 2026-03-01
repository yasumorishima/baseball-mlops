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

def build_batter_features() -> pd.DataFrame:
    """FanGraphs + Statcast をマージして打者特徴量テーブルを構築"""
    fg = pd.read_csv(DATA_DIR / "fg_batting.csv")
    ev = pd.read_csv(DATA_DIR / "sc_batter_exitvelo.csv")
    exp = pd.read_csv(DATA_DIR / "sc_batter_expected.csv")
    spd = pd.read_csv(DATA_DIR / "sc_sprint_speed.csv")

    # FanGraphs の選手IDカラム名を統一
    fg = fg.rename(columns={"playerid": "fg_id", "Name": "player", "Season": "season"})

    # Statcast は last_name, first_name → full_name を作る
    for df in [ev, exp]:
        df["player"] = df["last_name, first_name"].str.split(", ").apply(
            lambda x: f"{x[1]} {x[0]}" if len(x) == 2 else x[0]
        )
        df.rename(columns={"Season": "season"}, inplace=True)

    spd = spd.rename(columns={"last_name, first_name": "player_raw", "Season": "season"})
    spd["player"] = spd["player_raw"].str.split(", ").apply(
        lambda x: f"{x[1]} {x[0]}" if len(x) == 2 else x[0]
    )

    # Statcast 同士をマージ
    sc = ev.merge(
        exp[["player", "season", "xba", "xslg", "xwoba", "xwobacon"]],
        on=["player", "season"], how="left"
    ).merge(
        spd[["player", "season", "sprint_speed"]],
        on=["player", "season"], how="left"
    )

    # FanGraphs とマージ（選手名 + シーズンで結合）
    merged = fg.merge(sc, on=["player", "season"], how="left")

    out_path = DATA_DIR / "batter_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Batter features: {len(merged)} rows → {out_path}")
    return merged


def build_pitcher_features() -> pd.DataFrame:
    """FanGraphs + Statcast をマージして投手特徴量テーブルを構築"""
    fg = pd.read_csv(DATA_DIR / "fg_pitching.csv")
    ev = pd.read_csv(DATA_DIR / "sc_pitcher_exitvelo.csv")
    exp = pd.read_csv(DATA_DIR / "sc_pitcher_expected.csv")

    fg = fg.rename(columns={"playerid": "fg_id", "Name": "player", "Season": "season"})

    for df in [ev, exp]:
        df["player"] = df["last_name, first_name"].str.split(", ").apply(
            lambda x: f"{x[1]} {x[0]}" if len(x) == 2 else x[0]
        )
        df.rename(columns={"Season": "season"}, inplace=True)

    sc = ev.merge(
        exp[["player", "season", "xba", "xslg", "xwoba", "xera"]],
        on=["player", "season"], how="left"
    )

    merged = fg.merge(sc, on=["player", "season"], how="left")

    out_path = DATA_DIR / "pitcher_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Pitcher features: {len(merged)} rows → {out_path}")
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
