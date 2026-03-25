"""
BigQuery pitch-level Statcast → player×season aggregated features.

Reads from mlb_shared.statcast_pitches (6.8M rows, 122 columns, 2015-2024)
and aggregates ALL usable pitch-level columns to player×season level.

Feature domains:
  BATTER: plate_discipline, batted_ball_profile, power_quality,
          bat_tracking, run_values, pitch_mix_faced
  PITCHER: stuff (velo/spin/movement), command (zone/location/release),
           whiff_chase, contact_management, arsenal_detail, fatigue

Output: data/raw/bq_batter_features.csv, data/raw/bq_pitcher_features.csv

Usage:
  python src/fetch_bq_features.py
  python src/fetch_bq_features.py --batter-only
  python src/fetch_bq_features.py --pitcher-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT = "data-platform-490901"
DATASET = "mlb_shared"
TABLE = "statcast_pitches"
TABLE_REF = f"`{PROJECT}.{DATASET}.{TABLE}`"

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimum sample sizes for reliable aggregation
MIN_PITCHES_BATTER = 200   # ~50 PA
MIN_PITCHES_PITCHER = 400  # ~100 batters faced


def _get_bq_client():
    """Get authenticated BigQuery client."""
    from google.cloud import bigquery

    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        local_key = Path(r"C:\Users\fw_ya\.claude\gcp-sa-key.json")
        if local_key.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_key)

    return bigquery.Client(project=PROJECT)


# ---------------------------------------------------------------------------
# Swing descriptions (for whiff/contact/chase calculations)
# ---------------------------------------------------------------------------
_SWING_DESCS = (
    "'swinging_strike', 'swinging_strike_blocked', 'foul_tip', "
    "'foul', 'foul_bunt', 'foul_pitchout', "
    "'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'"
)
_WHIFF_DESCS = (
    "'swinging_strike', 'swinging_strike_blocked', 'foul_tip'"
)
_CONTACT_DESCS = (
    "'foul', 'foul_bunt', 'foul_pitchout', "
    "'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'"
)
_BIP_DESCS = (
    "'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'"
)


# ---------------------------------------------------------------------------
# Batter aggregation SQL
# ---------------------------------------------------------------------------
BATTER_QUERY = f"""
SELECT
  batter AS mlb_id,
  game_year AS season,
  COUNT(*) AS bq_total_pitches,

  -- ======================================================================
  -- PLATE DISCIPLINE (pitch selection quality)
  -- ======================================================================

  -- Whiff rate: swinging strikes / total swings
  SAFE_DIVIDE(
    SUM(CASE WHEN description IN ({_WHIFF_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END)
  ) AS bq_whiff_rate,

  -- Chase rate: swing at pitches outside zone
  SAFE_DIVIDE(
    SUM(CASE WHEN zone > 9
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN zone > 9 THEN 1 ELSE 0 END)
  ) AS bq_chase_rate,

  -- Zone contact rate: contact when swinging in zone
  SAFE_DIVIDE(
    SUM(CASE WHEN zone BETWEEN 1 AND 9
             AND description IN ({_CONTACT_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN zone BETWEEN 1 AND 9
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END)
  ) AS bq_zone_contact_rate,

  -- Zone swing rate: swing rate on pitches in zone
  SAFE_DIVIDE(
    SUM(CASE WHEN zone BETWEEN 1 AND 9
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END)
  ) AS bq_zone_swing_rate,

  -- Called strike rate: takes on strikes
  SAFE_DIVIDE(
    SUM(CASE WHEN description = 'called_strike' THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_called_strike_rate,

  -- First pitch swing rate
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_number = 1
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_number = 1 THEN 1 ELSE 0 END)
  ) AS bq_first_pitch_swing_rate,

  -- Overall swing rate
  SAFE_DIVIDE(
    SUM(CASE WHEN description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_swing_rate,

  -- Two-strike whiff rate (K vulnerability)
  SAFE_DIVIDE(
    SUM(CASE WHEN strikes = 2
             AND description IN ({_WHIFF_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN strikes = 2
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END)
  ) AS bq_two_strike_whiff_rate,

  -- ======================================================================
  -- BATTED BALL PROFILE
  -- ======================================================================

  -- GB/FB/LD/Popup rates
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'ground_ball' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_gb_rate,
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_fb_rate,
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'line_drive' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_ld_rate,
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'popup' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_popup_rate,

  -- Sweet spot rate (launch angle 8-32 degrees)
  SAFE_DIVIDE(
    SUM(CASE WHEN launch_angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END),
    SUM(CASE WHEN launch_angle IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_sweet_spot_rate,

  -- Average hit distance
  AVG(CASE WHEN hit_distance_sc IS NOT NULL AND hit_distance_sc > 0
       THEN hit_distance_sc END) AS bq_avg_hit_distance,

  -- ======================================================================
  -- POWER / EXIT VELOCITY QUALITY
  -- ======================================================================

  -- Exit velocity stats
  AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END) AS bq_avg_ev,
  MAX(launch_speed) AS bq_max_ev,
  APPROX_QUANTILES(
    CASE WHEN launch_speed IS NOT NULL THEN launch_speed END, 100
  )[OFFSET(90)] AS bq_ev_p90,
  STDDEV(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END) AS bq_ev_consistency,

  -- Average launch angle
  AVG(CASE WHEN launch_angle IS NOT NULL THEN launch_angle END) AS bq_avg_la,

  -- Hard hit rate (>=95 mph)
  SAFE_DIVIDE(
    SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END),
    SUM(CASE WHEN launch_speed IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_hard_hit_rate,

  -- Barrel rate (98+ mph, 26-30 deg — Statcast definition)
  SAFE_DIVIDE(
    SUM(CASE WHEN launch_speed >= 98 AND launch_angle BETWEEN 26 AND 30
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN launch_speed IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_barrel_rate,

  -- ======================================================================
  -- EXPECTED STATS (pitch-level xwOBA/xBA)
  -- ======================================================================

  AVG(estimated_woba_using_speedangle) AS bq_avg_xwoba,
  AVG(estimated_ba_using_speedangle) AS bq_avg_xba,
  AVG(woba_value) AS bq_avg_woba_value,
  AVG(babip_value) AS bq_avg_babip_value,
  AVG(iso_value) AS bq_avg_iso_value,

  -- ======================================================================
  -- BAT TRACKING (2024+ only — will be NULL for earlier years)
  -- ======================================================================

  AVG(bat_speed) AS bq_avg_bat_speed,
  AVG(swing_length) AS bq_avg_swing_length,
  AVG(attack_angle) AS bq_avg_attack_angle,
  STDDEV(bat_speed) AS bq_bat_speed_consistency,
  MAX(bat_speed) AS bq_max_bat_speed,

  -- ======================================================================
  -- RUN VALUES
  -- ======================================================================

  AVG(delta_run_exp) AS bq_avg_run_value,
  SUM(delta_run_exp) AS bq_total_run_value,

  -- ======================================================================
  -- PITCH MIX FACED (what does this batter see?)
  -- ======================================================================

  AVG(release_speed) AS bq_avg_velo_faced,

  -- Fastball faced %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA') THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_fastball_faced_pct,

  -- Breaking ball faced %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_breaking_faced_pct,

  -- Offspeed faced %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC', 'KN')
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_offspeed_faced_pct,

  -- ======================================================================
  -- COUNT LEVERAGE / APPROACH
  -- ======================================================================

  -- Hitter's count frequency (2-0, 3-0, 3-1)
  SAFE_DIVIDE(
    SUM(CASE WHEN (balls = 2 AND strikes = 0)
              OR (balls = 3 AND strikes = 0)
              OR (balls = 3 AND strikes = 1) THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_hitter_count_pct,

  -- Pitcher's count frequency (0-2, 1-2)
  SAFE_DIVIDE(
    SUM(CASE WHEN (balls = 0 AND strikes = 2)
              OR (balls = 1 AND strikes = 2) THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_pitcher_count_pct,

  -- ======================================================================
  -- BASERUNNING (from events)
  -- ======================================================================

  SUM(CASE WHEN events LIKE '%stolen_base%' THEN 1 ELSE 0 END)
    AS bq_sb_events,
  SUM(CASE WHEN events LIKE '%caught_stealing%' THEN 1 ELSE 0 END)
    AS bq_cs_events

FROM {TABLE_REF}
WHERE game_type = 'R'
GROUP BY batter, game_year
HAVING COUNT(*) >= {MIN_PITCHES_BATTER}
ORDER BY game_year, batter
"""


# ---------------------------------------------------------------------------
# Pitcher aggregation SQL
# ---------------------------------------------------------------------------
PITCHER_QUERY = f"""
SELECT
  pitcher AS mlb_id,
  game_year AS season,
  -- player_name is pitcher name in Statcast
  ANY_VALUE(player_name) AS pitcher_name,
  COUNT(*) AS bq_total_pitches_thrown,

  -- ======================================================================
  -- STUFF (velocity, spin, movement)
  -- ======================================================================

  -- Overall velocity
  AVG(release_speed) AS bq_avg_velo,
  MAX(release_speed) AS bq_max_velo,
  STDDEV(release_speed) AS bq_velo_consistency,
  AVG(effective_speed) AS bq_avg_effective_speed,

  -- Overall spin & movement
  AVG(release_spin_rate) AS bq_avg_spin,
  AVG(ABS(pfx_x)) AS bq_avg_h_break,
  AVG(pfx_z) AS bq_avg_v_break,
  AVG(SQRT(pfx_x * pfx_x + pfx_z * pfx_z)) AS bq_total_movement,
  AVG(spin_axis) AS bq_avg_spin_axis,

  -- Release
  AVG(release_extension) AS bq_avg_extension,
  AVG(arm_angle) AS bq_avg_arm_angle,
  STDDEV(release_pos_x) AS bq_release_x_consistency,
  STDDEV(release_pos_z) AS bq_release_z_consistency,

  -- Fastball specific (FF/SI/FC)
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN release_speed END) AS bq_fb_velo,
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN release_spin_rate END) AS bq_fb_spin,
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN pfx_z END) AS bq_fb_rise,
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN ABS(pfx_x) END) AS bq_fb_h_break,

  -- Breaking ball specific (SL/CU/KC/SV/CS/ST)
  AVG(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
       THEN release_speed END) AS bq_brk_velo,
  AVG(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
       THEN release_spin_rate END) AS bq_brk_spin,
  AVG(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
       THEN ABS(pfx_x) END) AS bq_brk_h_break,
  AVG(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
       THEN pfx_z END) AS bq_brk_v_break,

  -- Offspeed specific (CH/FS/FO/SC)
  AVG(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC')
       THEN release_speed END) AS bq_ch_velo,
  AVG(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC')
       THEN pfx_z END) AS bq_ch_drop,

  -- Velo differential (fastball - changeup = tunneling effectiveness)
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN release_speed END)
  - AVG(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC')
       THEN release_speed END) AS bq_fb_ch_velo_diff,

  -- ======================================================================
  -- COMMAND (location, zone, consistency)
  -- ======================================================================

  -- Zone rate
  SAFE_DIVIDE(
    SUM(CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_zone_rate,

  -- Edge rate (zones 11-14 in Statcast = shadow/edge)
  SAFE_DIVIDE(
    SUM(CASE WHEN zone BETWEEN 11 AND 14 THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_edge_rate,

  -- First pitch strike rate
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_number = 1
             AND (description = 'called_strike'
                  OR description IN ({_WHIFF_DESCS})
                  OR description IN ('foul', 'foul_bunt'))
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_number = 1 THEN 1 ELSE 0 END)
  ) AS bq_first_pitch_strike_rate,

  -- Location consistency (lower = more precise)
  STDDEV(plate_x) AS bq_location_x_consistency,
  STDDEV(plate_z) AS bq_location_z_consistency,

  -- ======================================================================
  -- WHIFF / CHASE (pitcher-induced swing decisions)
  -- ======================================================================

  -- Whiff rate
  SAFE_DIVIDE(
    SUM(CASE WHEN description IN ({_WHIFF_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END)
  ) AS bq_whiff_rate,

  -- Chase rate induced
  SAFE_DIVIDE(
    SUM(CASE WHEN zone > 9
             AND description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN zone > 9 THEN 1 ELSE 0 END)
  ) AS bq_chase_rate_induced,

  -- Called strike + whiff rate (CSW proxy)
  SAFE_DIVIDE(
    SUM(CASE WHEN description IN ('called_strike',
         'swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END),
    COUNT(*)
  ) AS bq_csw_rate,

  -- ======================================================================
  -- CONTACT MANAGEMENT (batted ball quality allowed)
  -- ======================================================================

  AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END)
    AS bq_avg_ev_against,
  AVG(CASE WHEN launch_angle IS NOT NULL THEN launch_angle END)
    AS bq_avg_la_against,
  SAFE_DIVIDE(
    SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END),
    SUM(CASE WHEN launch_speed IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_hard_hit_rate_against,
  SAFE_DIVIDE(
    SUM(CASE WHEN launch_speed >= 98 AND launch_angle BETWEEN 26 AND 30
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN launch_speed IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_barrel_rate_against,

  -- GB rate induced (ground ball pitchers)
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'ground_ball' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_gb_rate_induced,

  -- Fly ball rate
  SAFE_DIVIDE(
    SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END),
    SUM(CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_fb_rate_induced,

  -- Expected stats against
  AVG(estimated_woba_using_speedangle) AS bq_avg_xwoba_against,
  AVG(estimated_ba_using_speedangle) AS bq_avg_xba_against,

  -- Contact rate against (lower = more whiffs)
  SAFE_DIVIDE(
    SUM(CASE WHEN description IN ({_CONTACT_DESCS}) THEN 1 ELSE 0 END),
    SUM(CASE WHEN description IN ({_SWING_DESCS}) THEN 1 ELSE 0 END)
  ) AS bq_contact_rate_against,

  -- ======================================================================
  -- ARSENAL DETAIL (pitch mix & type-specific quality)
  -- ======================================================================

  COUNT(DISTINCT pitch_type) AS bq_n_pitch_types,

  -- Fastball usage %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA') THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_fastball_pct,

  -- Breaking ball usage %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_breaking_pct,

  -- Offspeed usage %
  SAFE_DIVIDE(
    SUM(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC')
         THEN 1 ELSE 0 END),
    SUM(CASE WHEN pitch_type IS NOT NULL THEN 1 ELSE 0 END)
  ) AS bq_offspeed_pct,

  -- ======================================================================
  -- FATIGUE / TIMES THROUGH ORDER
  -- ======================================================================

  AVG(CASE WHEN n_thruorder_pitcher = 1
       THEN delta_pitcher_run_exp END) AS bq_rv_1st_time,
  AVG(CASE WHEN n_thruorder_pitcher = 2
       THEN delta_pitcher_run_exp END) AS bq_rv_2nd_time,
  AVG(CASE WHEN n_thruorder_pitcher >= 3
       THEN delta_pitcher_run_exp END) AS bq_rv_3rd_time,

  -- Performance drop 3rd time through
  AVG(CASE WHEN n_thruorder_pitcher >= 3
       THEN delta_pitcher_run_exp END)
  - AVG(CASE WHEN n_thruorder_pitcher = 1
       THEN delta_pitcher_run_exp END) AS bq_tto_degradation,

  -- ======================================================================
  -- RUN VALUES
  -- ======================================================================

  AVG(delta_pitcher_run_exp) AS bq_avg_pitcher_run_value,

  -- Run value on fastballs vs breaking vs offspeed
  AVG(CASE WHEN pitch_type IN ('FF', 'SI', 'FC', 'FA')
       THEN delta_pitcher_run_exp END) AS bq_rv_fastball,
  AVG(CASE WHEN pitch_type IN ('SL', 'CU', 'KC', 'SV', 'CS', 'ST')
       THEN delta_pitcher_run_exp END) AS bq_rv_breaking,
  AVG(CASE WHEN pitch_type IN ('CH', 'FS', 'FO', 'SC')
       THEN delta_pitcher_run_exp END) AS bq_rv_offspeed

FROM {TABLE_REF}
WHERE game_type = 'R'
GROUP BY pitcher, game_year
HAVING COUNT(*) >= {MIN_PITCHES_PITCHER}
ORDER BY game_year, pitcher
"""


# ---------------------------------------------------------------------------
# Player ID → Name mapping
# ---------------------------------------------------------------------------

def _build_player_id_map(data_dir: Path) -> dict[int, str]:
    """Build MLB ID → player name map from existing Savant CSV files.

    Uses sc_batter_exitvelo.csv (has player_id + last_name, first_name)
    and sc_pitcher_exitvelo.csv as sources.
    """
    id_map: dict[int, str] = {}

    for csv_name in ("sc_batter_exitvelo.csv", "sc_pitcher_exitvelo.csv"):
        path = data_dir / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)

        # Build name from available columns
        if "last_name, first_name" in df.columns:
            for _, row in df.iterrows():
                pid = row.get("player_id")
                name_raw = row.get("last_name, first_name", "")
                if pd.notna(pid) and pd.notna(name_raw):
                    parts = str(name_raw).split(", ")
                    name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else str(name_raw)
                    id_map[int(pid)] = name
        elif "last_name" in df.columns and "first_name" in df.columns:
            for _, row in df.iterrows():
                pid = row.get("player_id")
                if pd.notna(pid) and pd.notna(row.get("first_name")):
                    id_map[int(pid)] = f"{row['first_name']} {row['last_name']}"

    return id_map


def _map_pitcher_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map pitcher names from BQ player_name column (format: 'Last, First')."""
    if "pitcher_name" in df.columns:
        df["player"] = df["pitcher_name"].apply(
            lambda x: (f"{x.split(', ')[1]} {x.split(', ')[0]}"
                       if pd.notna(x) and ", " in str(x) else str(x))
        )
        df = df.drop(columns=["pitcher_name"])
    return df


# ---------------------------------------------------------------------------
# Main fetch functions
# ---------------------------------------------------------------------------

def fetch_batter_bq_features() -> pd.DataFrame:
    """Query BQ for batter pitch-level aggregations."""
    client = _get_bq_client()
    print("Fetching batter BQ features...")
    df = client.query(BATTER_QUERY).to_dataframe()
    print(f"  Raw: {len(df):,} batter-seasons")

    # Map MLB ID → player name
    id_map = _build_player_id_map(DATA_DIR)
    if id_map:
        df["player"] = df["mlb_id"].map(id_map)
        n_mapped = df["player"].notna().sum()
        print(f"  Mapped {n_mapped:,}/{len(df):,} batter IDs to names")
    else:
        print("  WARNING: No player ID map available — using pybaseball lookup")
        try:
            from pybaseball import playerid_reverse_lookup
            unique_ids = df["mlb_id"].unique().tolist()
            lookup = playerid_reverse_lookup(unique_ids, key_type="mlbam")
            if "name_first" in lookup.columns and "name_last" in lookup.columns:
                lookup["player"] = lookup["name_first"] + " " + lookup["name_last"]
                lookup_map = dict(zip(lookup["key_mlbam"], lookup["player"]))
                df["player"] = df["mlb_id"].map(lookup_map)
        except Exception as e:
            print(f"  pybaseball lookup failed: {e}")
            df["player"] = df["mlb_id"].astype(str)

    # Computed features
    sb = df["bq_sb_events"].fillna(0)
    cs = df["bq_cs_events"].fillna(0)
    df["bq_sb_success_rate"] = (sb / (sb + cs)).where(sb + cs > 0)
    df["bq_sb_attempt_rate"] = ((sb + cs) / df["bq_total_pitches"] * 100)

    out_path = DATA_DIR / "bq_batter_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df):,} rows, {len(df.columns)} cols)")

    # Coverage report
    bq_cols = [c for c in df.columns if c.startswith("bq_")]
    print(f"  BQ feature columns: {len(bq_cols)}")
    for col in bq_cols[:5]:
        null_pct = df[col].isna().mean() * 100
        print(f"    {col}: {null_pct:.1f}% null")
    if len(bq_cols) > 5:
        print(f"    ... and {len(bq_cols) - 5} more")

    return df


def fetch_pitcher_bq_features() -> pd.DataFrame:
    """Query BQ for pitcher pitch-level aggregations."""
    client = _get_bq_client()
    print("Fetching pitcher BQ features...")
    df = client.query(PITCHER_QUERY).to_dataframe()
    print(f"  Raw: {len(df):,} pitcher-seasons")

    # Map pitcher names from BQ (player_name column)
    df = _map_pitcher_names(df)
    n_named = df["player"].notna().sum()
    print(f"  Named: {n_named:,}/{len(df):,}")

    # Fallback for any unmapped: use ID-based lookup
    unmapped = df["player"].isna() | (df["player"] == "nan")
    if unmapped.any():
        id_map = _build_player_id_map(DATA_DIR)
        if id_map:
            df.loc[unmapped, "player"] = df.loc[unmapped, "mlb_id"].map(id_map)

    out_path = DATA_DIR / "bq_pitcher_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df):,} rows, {len(df.columns)} cols)")

    # Coverage report
    bq_cols = [c for c in df.columns if c.startswith("bq_")]
    print(f"  BQ feature columns: {len(bq_cols)}")
    for col in bq_cols[:5]:
        null_pct = df[col].isna().mean() * 100
        print(f"    {col}: {null_pct:.1f}% null")
    if len(bq_cols) > 5:
        print(f"    ... and {len(bq_cols) - 5} more")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch BQ pitch-level aggregated features")
    parser.add_argument("--batter-only", action="store_true")
    parser.add_argument("--pitcher-only", action="store_true")
    args = parser.parse_args()

    if args.pitcher_only:
        fetch_pitcher_bq_features()
    elif args.batter_only:
        fetch_batter_bq_features()
    else:
        fetch_batter_bq_features()
        fetch_pitcher_bq_features()
    print("BQ feature fetch complete.")


if __name__ == "__main__":
    main()
