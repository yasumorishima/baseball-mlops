"""
オープン戦 2026 実績データ取得

FanGraphs の 2026 スプリングトレーニング成績を pybaseball で取得し、
predictions/ に保存する（Streamlit Cloud 用）。
"""

import time
from pathlib import Path

import pandas as pd
from pybaseball import batting_stats, pitching_stats

PRED_DIR = Path(__file__).parent.parent / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEASON = 2026
MIN_PA = 5    # 打者: 最低 5 打席
MIN_IP = 2    # 投手: 最低 2 投球回


def fetch_spring_batting():
    print("Fetching spring training batting stats (2026)...")
    try:
        df = batting_stats(SEASON, qual=MIN_PA)
        if df is None or df.empty:
            print("  No batting data available yet.")
            return pd.DataFrame()
        # 必要カラムだけ残す
        cols = ["Name", "Team", "Age", "PA", "wOBA", "xwOBA", "AVG", "OBP", "SLG",
                "HR", "BB%", "K%", "EV", "Barrel%", "HardHit%"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].rename(columns={"Name": "player"})
        print(f"  {len(df)} batters fetched.")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


def fetch_spring_pitching():
    print("Fetching spring training pitching stats (2026)...")
    try:
        df = pitching_stats(SEASON, qual=MIN_IP)
        if df is None or df.empty:
            print("  No pitching data available yet.")
            return pd.DataFrame()
        cols = ["Name", "Team", "Age", "IP", "ERA", "xERA", "xFIP", "FIP",
                "K%", "BB%", "Whiff%", "CSW%", "vFA (pi)"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].rename(columns={"Name": "player"})
        print(f"  {len(df)} pitchers fetched.")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    bat = fetch_spring_batting()
    time.sleep(1)
    pit = fetch_spring_pitching()

    if not bat.empty:
        bat.to_csv(PRED_DIR / "spring_batting_2026.csv", index=False)
        print(f"Saved: spring_batting_2026.csv ({len(bat)} rows)")
    if not pit.empty:
        pit.to_csv(PRED_DIR / "spring_pitching_2026.csv", index=False)
        print(f"Saved: spring_pitching_2026.csv ({len(pit)} rows)")
    print("Done.")
