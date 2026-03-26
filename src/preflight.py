"""Preflight checks before training.

Validates BQ connectivity, table existence, data availability, and
pybaseball API accessibility. Fails fast so 2+ hour training runs
don't waste time on preventable errors.

Designed for maximum debuggability: every failure prints exactly what
was expected, what was found, and enough context to fix the issue.

Usage:
  python src/preflight.py              # full check
  python src/preflight.py --bq-only    # BQ checks only (skip pybaseball)

Exit code 0 = all checks passed, 1 = failure.
"""

from __future__ import annotations

import os
import sys
import traceback
from difflib import get_close_matches
from pathlib import Path

PROJECT = "data-platform-490901"
DATASET = "mlb_shared"
TABLE = "statcast_pitches"
TABLE_REF = f"`{PROJECT}.{DATASET}.{TABLE}`"

# Minimum expected rows per year (conservative — 50% of typical)
MIN_ROWS_PER_YEAR = {
    2015: 350_000, 2016: 350_000, 2017: 350_000, 2018: 350_000,
    2019: 350_000, 2020: 125_000, 2021: 350_000, 2022: 350_000,
    2023: 350_000, 2024: 350_000,
}

# Key columns that must exist in statcast_pitches
REQUIRED_COLUMNS = [
    "game_pk", "pitcher", "batter", "events", "game_year",
    "release_speed", "launch_speed", "home_team", "away_team",
    "game_type", "on_1b", "on_2b", "on_3b",
    "bat_speed", "swing_length", "attack_angle",  # bat tracking (2024+)
    "estimated_woba_using_speedangle", "delta_run_exp",
]

errors: list[str] = []
warnings: list[str] = []


def _get_bq_client():
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

    cred_used = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "(default)")
    print(f"  Credentials: {cred_used}")
    return bigquery.Client(project=PROJECT)


def _find_similar(name: str, candidates: set[str], n: int = 5) -> list[str]:
    """Find similar column names using fuzzy matching."""
    # Exact substring match first
    substr = [c for c in sorted(candidates) if name.lower() in c.lower() or c.lower() in name.lower()]
    # Then difflib close matches
    fuzzy = get_close_matches(name, sorted(candidates), n=n, cutoff=0.5)
    # Merge, deduplicate, preserve order
    seen = set()
    result = []
    for c in substr + fuzzy:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:n]


def check_bq_connection():
    """Check BQ is reachable and table exists."""
    print("=" * 60)
    print("1. BQ Connection & Table Existence")
    print("=" * 60)
    try:
        client = _get_bq_client()
        print(f"  Project: {PROJECT}")
        print(f"  Dataset: {DATASET}")
        print(f"  Table:   {TABLE}")

        # List all tables in dataset
        tables = list(client.list_tables(f"{PROJECT}.{DATASET}"))
        print(f"  Tables in {DATASET}: {len(tables)}")
        for t in tables:
            print(f"    - {t.table_id}")

        table = client.get_table(f"{PROJECT}.{DATASET}.{TABLE}")
        print(f"\n  {DATASET}.{TABLE}:")
        print(f"    Rows:    {table.num_rows:,}")
        print(f"    Columns: {len(table.schema)}")
        print(f"    Size:    {table.num_bytes / 1024**3:.2f} GB")
        print(f"    Created: {table.created}")
        print(f"    Modified:{table.modified}")

        if table.num_rows < 5_000_000:
            errors.append(
                f"statcast_pitches: {table.num_rows:,} rows (expected 6M+). "
                f"Table may be incomplete or truncated."
            )
        return client
    except Exception as e:
        errors.append(
            f"BQ connection failed.\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error:      {e}\n"
            f"  GOOGLE_APPLICATION_CREDENTIALS={os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return None


def check_year_coverage(client):
    """Check each year has enough rows."""
    print("\n" + "=" * 60)
    print("2. Year Coverage (regular season only)")
    print("=" * 60)
    q = f"""
        SELECT CAST(game_year AS INT64) AS yr, COUNT(*) AS n,
               COUNTIF(events IS NOT NULL) AS ab_outcomes,
               COUNT(DISTINCT game_pk) AS games,
               COUNT(DISTINCT pitcher) AS pitchers,
               COUNT(DISTINCT batter) AS batters
        FROM {TABLE_REF}
        WHERE game_type = 'R'
        GROUP BY yr ORDER BY yr
    """
    print(f"  SQL: {q.strip()[:120]}...")
    try:
        rows = list(client.query(q).result())
        if not rows:
            errors.append("Year coverage query returned 0 rows — table may be empty")
            return

        total = sum(r.n for r in rows)
        years_found = {r.yr for r in rows}

        print(f"\n  {'Year':<6} {'Pitches':>12} {'ABs':>10} {'Games':>7} {'P':>5} {'B':>5}  Status")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*7} {'-'*5} {'-'*5}  {'-'*6}")
        for row in rows:
            min_expected = MIN_ROWS_PER_YEAR.get(row.yr, 300_000)
            status = "OK" if row.n >= min_expected else "LOW"
            print(f"  {row.yr:<6} {row.n:>12,} {row.ab_outcomes:>10,} "
                  f"{row.games:>7,} {row.pitchers:>5,} {row.batters:>5,}  [{status}]")
            if row.n < min_expected:
                errors.append(
                    f"Year {row.yr}: {row.n:,} rows (expected {min_expected:,}+). "
                    f"Games={row.games}, Pitchers={row.pitchers}, Batters={row.batters}. "
                    f"Data may be incomplete."
                )
        print(f"  {'TOTAL':<6} {total:>12,}")

        expected_years = set(range(2015, 2025))
        missing = expected_years - years_found
        if missing:
            errors.append(
                f"Missing years: {sorted(missing)}. "
                f"Years found: {sorted(years_found)}. "
                f"Re-run fetch_statcast_pitches.py for missing years."
            )
    except Exception as e:
        errors.append(f"Year coverage query failed.\n  SQL: {q}\n  Error: {e}")


def check_required_columns(client):
    """Check key columns exist in statcast_pitches."""
    print("\n" + "=" * 60)
    print("3. Required Columns")
    print("=" * 60)
    try:
        table = client.get_table(f"{PROJECT}.{DATASET}.{TABLE}")
        schema_cols = {f.name for f in table.schema}
        schema_types = {f.name: f.field_type for f in table.schema}

        missing = [c for c in REQUIRED_COLUMNS if c not in schema_cols]
        if missing:
            print(f"  FAIL: {len(missing)} missing columns:")
            for mc in missing:
                similar = _find_similar(mc, schema_cols)
                print(f"    Expected: '{mc}'")
                if similar:
                    print(f"    Similar columns found: {similar}")
                else:
                    print(f"    No similar columns found in schema")
            errors.append(
                f"Missing {len(missing)} columns in statcast_pitches: {missing}. "
                f"Schema has {len(schema_cols)} columns total."
            )
            # Print full schema for debugging
            print(f"\n  Full schema ({len(schema_cols)} columns):")
            for col in sorted(schema_cols):
                print(f"    {col} ({schema_types[col]})")
        else:
            print(f"  OK: all {len(REQUIRED_COLUMNS)} required columns present")
            # Show types for key columns
            for c in ["game_pk", "pitcher", "batter", "game_year", "bat_speed"]:
                if c in schema_types:
                    print(f"    {c}: {schema_types[c]}")

        # Null rates by year for critical columns
        null_check_cols = ["release_speed", "launch_speed", "events",
                          "bat_speed", "estimated_woba_using_speedangle"]
        present_cols = [c for c in null_check_cols if c in schema_cols]
        if present_cols:
            cols_str = ", ".join(
                f"ROUND(COUNTIF({c} IS NULL) * 100.0 / COUNT(*), 1) AS null_{c}"
                for c in present_cols
            )
            q = f"""
                SELECT CAST(game_year AS INT64) AS yr, COUNT(*) AS n, {cols_str}
                FROM {TABLE_REF}
                WHERE game_type = 'R'
                GROUP BY yr ORDER BY yr
            """
            print(f"\n  Null rates by year:")
            header = f"  {'Year':<6} {'Rows':>10}"
            for c in present_cols:
                header += f"  {c[:18]:>18}"
            print(header)
            for row in client.query(q).result():
                line = f"  {row.yr:<6} {row.n:>10,}"
                for c in present_cols:
                    val = getattr(row, f"null_{c}", None)
                    flag = ""
                    if val is not None:
                        # bat_speed is expected ~90% null (only 2024+)
                        if c == "bat_speed" and row.yr < 2024 and val > 80:
                            flag = " (ok:pre-2024)"
                        elif c == "bat_speed" and row.yr >= 2024 and val > 50:
                            flag = " WARNING"
                        elif c != "bat_speed" and c != "events" and val > 30:
                            flag = " WARNING"
                        line += f"  {val:>14.1f}%{flag:>3}"
                    else:
                        line += f"  {'N/A':>18}"
                print(line)
    except Exception as e:
        errors.append(f"Column check failed.\n  Error: {e}\n  Traceback:\n{traceback.format_exc()}")


def check_bq_feature_aggregation(client):
    """Spot-check that BQ aggregation queries produce expected values."""
    print("\n" + "=" * 60)
    print("4. BQ Feature Aggregation (spot check)")
    print("=" * 60)

    # Batter spot check
    batter_q = f"""
        SELECT
            batter AS player_id,
            CAST(game_year AS INT64) AS season,
            SAFE_DIVIDE(COUNTIF(description = 'swinging_strike'),
                        COUNTIF(description LIKE '%swing%')) AS bq_whiff_rate,
            AVG(launch_speed) AS bq_avg_ev,
            AVG(bat_speed) AS bq_avg_bat_speed,
            AVG(delta_run_exp) AS bq_avg_run_value,
            COUNT(*) AS n_pitches
        FROM {TABLE_REF}
        WHERE game_type = 'R' AND game_year = 2024
        GROUP BY batter, game_year
        HAVING COUNT(*) >= 200
        LIMIT 3
    """
    print(f"  Batter query:")
    print(f"    {batter_q.strip()[:200]}...")
    try:
        rows = list(client.query(batter_q).result())
        if rows:
            print(f"  OK: batter aggregation returned {len(rows)} rows")
            for i, r in enumerate(rows):
                print(f"    [{i}] player={r.player_id}, pitches={r.n_pitches:,}, "
                      f"whiff={r.bq_whiff_rate:.3f}, avg_ev={r.bq_avg_ev:.1f}, "
                      f"bat_speed={r.bq_avg_bat_speed}, run_value={r.bq_avg_run_value:.4f}")
                # Sanity checks on values
                if r.bq_whiff_rate > 1.0 or r.bq_whiff_rate < 0:
                    warnings.append(f"Batter whiff_rate={r.bq_whiff_rate} out of [0,1] range")
                if r.bq_avg_ev and (r.bq_avg_ev < 50 or r.bq_avg_ev > 120):
                    warnings.append(f"Batter avg_ev={r.bq_avg_ev} outside normal range [50-120]")
        else:
            errors.append(
                "Batter aggregation returned 0 rows for 2024 with >= 200 pitches. "
                "Check if game_type='R' data exists for 2024."
            )
    except Exception as e:
        errors.append(f"Batter aggregation failed.\n  SQL: {batter_q}\n  Error: {e}")

    # Pitcher spot check
    pitcher_q = f"""
        SELECT
            pitcher AS player_id,
            CAST(game_year AS INT64) AS season,
            AVG(release_speed) AS bq_avg_velo,
            SAFE_DIVIDE(COUNTIF(description = 'swinging_strike'),
                        COUNTIF(description LIKE '%swing%')) AS bq_whiff_rate,
            SAFE_DIVIDE(COUNTIF(zone BETWEEN 1 AND 9), COUNT(*)) AS bq_zone_rate,
            COUNT(*) AS n_pitches
        FROM {TABLE_REF}
        WHERE game_type = 'R' AND game_year = 2024
        GROUP BY pitcher, game_year
        HAVING COUNT(*) >= 400
        LIMIT 3
    """
    print(f"\n  Pitcher query:")
    print(f"    {pitcher_q.strip()[:200]}...")
    try:
        rows = list(client.query(pitcher_q).result())
        if rows:
            print(f"  OK: pitcher aggregation returned {len(rows)} rows")
            for i, r in enumerate(rows):
                print(f"    [{i}] player={r.player_id}, pitches={r.n_pitches:,}, "
                      f"velo={r.bq_avg_velo:.1f}, whiff={r.bq_whiff_rate:.3f}, "
                      f"zone={r.bq_zone_rate:.3f}")
                if r.bq_avg_velo and (r.bq_avg_velo < 70 or r.bq_avg_velo > 105):
                    warnings.append(f"Pitcher avg_velo={r.bq_avg_velo} outside normal range [70-105]")
        else:
            errors.append(
                "Pitcher aggregation returned 0 rows for 2024 with >= 400 pitches. "
                "Check if game_type='R' data exists for 2024."
            )
    except Exception as e:
        errors.append(f"Pitcher aggregation failed.\n  SQL: {pitcher_q}\n  Error: {e}")


def check_pybaseball_api():
    """Quick smoke test that pybaseball can reach Savant/FanGraphs."""
    print("\n" + "=" * 60)
    print("5. pybaseball API Smoke Test")
    print("=" * 60)
    try:
        from pybaseball import batting_stats
        df = batting_stats(2024, qual=502)
        if len(df) < 10:
            errors.append(
                f"pybaseball batting_stats(2024, qual=502) returned {len(df)} rows "
                f"(expected 30+). FanGraphs API may be down or data format changed. "
                f"Columns returned: {list(df.columns[:20])}"
            )
        else:
            print(f"  OK: batting_stats(2024) returned {len(df)} qualified batters")
            expected = {"wOBA", "K%", "BB%", "ISO", "BABIP"}
            actual = set(df.columns)
            missing = expected - actual
            if missing:
                similar_info = []
                for m in missing:
                    sim = _find_similar(m, actual)
                    similar_info.append(f"'{m}' (similar: {sim})")
                warnings.append(
                    f"pybaseball missing columns: {', '.join(similar_info)}. "
                    f"First 30 columns: {list(df.columns[:30])}"
                )
            else:
                print(f"  OK: key FanGraphs columns present ({sorted(expected)})")
    except Exception as e:
        errors.append(
            f"pybaseball batting_stats API failed.\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error: {e}\n"
            f"  This means FanGraphs data cannot be fetched. Training will fail.\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )

    try:
        from savant_extras import bat_tracking
        df = bat_tracking(2024)
        if len(df) < 10:
            warnings.append(
                f"savant_extras bat_tracking(2024) returned {len(df)} rows (expected 100+). "
                f"Columns: {list(df.columns[:15])}"
            )
        else:
            print(f"  OK: savant_extras bat_tracking(2024) returned {len(df)} rows")
    except Exception as e:
        warnings.append(
            f"savant_extras bat_tracking failed: {type(e).__name__}: {e}. "
            f"Bat tracking features will be NaN for all years."
        )


def check_existing_csvs():
    """Check if any cached CSVs exist from previous runs."""
    print("\n" + "=" * 60)
    print("6. Cached Data Files")
    print("=" * 60)
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    print(f"  Directory: {raw_dir}")
    print(f"  Exists: {raw_dir.exists()}")

    key_files = [
        "batter_features.csv", "pitcher_features.csv",
        "bq_batter_features.csv", "bq_pitcher_features.csv",
        "fg_batting.csv", "fg_pitching.csv", "park_factors.csv",
    ]
    for f in key_files:
        p = raw_dir / f
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p, nrows=3)
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {f}: {size_mb:.1f}MB, {len(df.columns)} cols")
            print(f"    Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        else:
            print(f"  {f}: NOT FOUND (will be created during fetch)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preflight checks")
    parser.add_argument("--bq-only", action="store_true",
                       help="Skip pybaseball API check")
    args = parser.parse_args()

    print("=" * 60)
    print("baseball-mlops Preflight Check")
    print("=" * 60)

    client = check_bq_connection()
    if client:
        check_year_coverage(client)
        check_required_columns(client)
        check_bq_feature_aggregation(client)

    if not args.bq_only:
        check_pybaseball_api()

    check_existing_csvs()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for i, w in enumerate(warnings, 1):
            print(f"    [{i}] {w}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for i, e in enumerate(errors, 1):
            print(f"    [{i}] {e}")
        print(f"\n{'='*60}")
        print(f"PREFLIGHT FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
        print(f"Fix the errors above before running training.")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print(f"\n  All checks passed ({len(warnings)} warnings)")
        print(f"\n{'='*60}")
        print(f"PREFLIGHT OK — ready to train")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
