"""
BigQuery データロードスクリプト

fetch_statcast.py の全出力 CSV を BigQuery にロードする。
生データ13テーブルは WRITE_TRUNCATE（フルリプレース）。
model_metrics_history のみ WRITE_APPEND（時系列蓄積）。

Usage:
    python src/load_to_bq.py                    # 全テーブルロード
    python src/load_to_bq.py --table raw_fg_batting  # 単テーブル
    python src/load_to_bq.py --metrics           # model_metrics.json を履歴に追記
    python src/load_to_bq.py --predictions       # predictions/*.csv を上書き
    python src/load_to_bq.py --all --metrics     # 全テーブル + メトリクス履歴
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "data-platform-490901"
DATASET_ID = "mlb_statcast"
FULL_DATASET = f"{PROJECT_ID}.{DATASET_ID}"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PRED_DIR = Path(__file__).parent.parent / "predictions"

# CSV ファイル名 → BQ テーブル名のマッピング（生データ13テーブル）
RAW_TABLE_MAP = {
    "fg_batting.csv": "raw_fg_batting",
    "fg_pitching.csv": "raw_fg_pitching",
    "sc_batter_exitvelo.csv": "raw_sc_batter_exitvelo",
    "sc_batter_expected.csv": "raw_sc_batter_expected",
    "sc_sprint_speed.csv": "raw_sc_sprint_speed",
    "sc_batted_ball.csv": "raw_sc_batted_ball",
    "sc_bat_tracking.csv": "raw_sc_bat_tracking",
    "sc_pitcher_exitvelo.csv": "raw_sc_pitcher_exitvelo",
    "sc_pitcher_expected.csv": "raw_sc_pitcher_expected",
    "sc_pitcher_arsenal.csv": "raw_sc_pitcher_arsenal",
    "park_factors.csv": "raw_park_factors",
    "batter_features.csv": "raw_batter_features",
    "pitcher_features.csv": "raw_pitcher_features",
}

# predictions CSV → BQ テーブル名（既存テーブルを上書き）
PRED_TABLE_MAP = {
    "batter_predictions.csv": "batter_predictions",
    "pitcher_predictions.csv": "pitcher_predictions",
}


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """BQ 非互換カラム名を修正する（% → _pct、特殊文字除去）

    BigQuery カラム名に使えるのは英数字・アンダースコアのみ。
    重複カラム名が発生した場合は連番サフィックスを付与する。
    """
    import re

    rename = {}
    for col in df.columns:
        new = col
        # 意味のある置換を先に行う
        new = new.replace("%", "_pct")
        new = new.replace("/", "_per_")
        new = new.replace("+", "_plus")
        # 残りの非互換文字（括弧、ハイフン、スペース、ドット、#等）をアンダースコアに
        new = re.sub(r"[^a-zA-Z0-9_]", "_", new)
        # 連続アンダースコアを1つに
        new = re.sub(r"_+", "_", new)
        # 末尾アンダースコアを除去
        new = new.strip("_")
        # 先頭が数字の場合は _ を付加
        if new and new[0].isdigit():
            new = f"_{new}"
        if new != col:
            rename[col] = new
    if rename:
        df = df.rename(columns=rename)

    # 重複カラム名を検出して連番サフィックスで解消
    # BQ はカラム名が case-insensitive なので小文字で比較する
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        key = col.lower()
        if key in seen:
            seen[key] += 1
            deduped = f"{col}_{seen[key]}"
            print(f"    WARN: duplicate column '{col}' → '{deduped}'")
            new_cols.append(deduped)
        else:
            seen[key] = 0
            new_cols.append(col)
    if new_cols != list(df.columns):
        df.columns = new_cols

    return df


def _get_client() -> bigquery.Client:
    """BigQuery クライアントを取得"""
    return bigquery.Client(project=PROJECT_ID)


def load_csv_to_bq(
    csv_path: Path,
    table_name: str,
    write_disposition: str = "WRITE_TRUNCATE",
) -> int:
    """CSV ファイルを BigQuery テーブルにロードする。

    Returns:
        ロードした行数
    """
    if not csv_path.exists():
        print(f"  SKIP: {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  SKIP: {csv_path.name} is empty")
        return 0

    df = _sanitize_columns(df)

    client = _get_client()
    table_id = f"{FULL_DATASET}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # 完了を待機
    except Exception as e:
        print(f"  ERROR: {table_name} ← {csv_path.name}: {type(e).__name__}: {e!r}")
        print(f"    columns ({len(df.columns)}): {list(df.columns)[:10]}...")
        print(f"    dtypes: {dict(df.dtypes.value_counts())}")
        return 0

    table = client.get_table(table_id)
    print(f"  OK: {table_name} ← {csv_path.name} ({table.num_rows} rows)")
    return table.num_rows


def load_all_raw() -> dict:
    """全生データ CSV を BigQuery にロード（WRITE_TRUNCATE）

    Returns:
        {テーブル名: 行数} の辞書
    """
    print("=== Loading raw Statcast data to BigQuery ===")
    results = {}
    for csv_name, table_name in RAW_TABLE_MAP.items():
        csv_path = RAW_DIR / csv_name
        rows = load_csv_to_bq(csv_path, table_name, "WRITE_TRUNCATE")
        results[table_name] = rows
    return results


def load_predictions() -> dict:
    """predictions/*.csv を BigQuery に上書き（WRITE_TRUNCATE）"""
    print("=== Loading predictions to BigQuery ===")
    results = {}
    for csv_name, table_name in PRED_TABLE_MAP.items():
        csv_path = PRED_DIR / csv_name
        rows = load_csv_to_bq(csv_path, table_name, "WRITE_TRUNCATE")
        results[table_name] = rows
    return results


def append_metrics() -> None:
    """model_metrics.json を model_metrics_history テーブルに追記"""
    metrics_path = PRED_DIR / "model_metrics.json"
    if not metrics_path.exists():
        print("  SKIP: model_metrics.json not found")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # タイムスタンプと run metadata を追加
    row = {
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row.update(metrics)

    df = pd.DataFrame([row])

    client = _get_client()
    table_id = f"{FULL_DATASET}.model_metrics_history"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        ],
    )

    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
    except Exception as e:
        print(f"  ERROR: model_metrics_history append failed: {e}")
        print("  Attempting WRITE_TRUNCATE fallback...")
        job_config_fallback = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            autodetect=True,
        )
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config_fallback)
        job.result()

    table = client.get_table(table_id)
    print(f"  OK: model_metrics_history appended (total {table.num_rows} rows)")


def validate_bq_compat() -> bool:
    """学習前にCSVのBQ互換性を検証する（カラム名・型・重複）

    Returns:
        True: 全CSV OK, False: 問題あり（学習を中断すべき）
    """
    import re

    print("=== Pre-training BQ compatibility check ===")
    errors = []

    for csv_name, table_name in {**RAW_TABLE_MAP, **PRED_TABLE_MAP}.items():
        if csv_name in RAW_TABLE_MAP:
            csv_path = RAW_DIR / csv_name
        else:
            csv_path = PRED_DIR / csv_name

        if not csv_path.exists():
            # predictions は学習後に生成されるのでスキップ
            if csv_name in PRED_TABLE_MAP:
                continue
            print(f"  WARN: {csv_name} not found (will be skipped)")
            continue

        df = pd.read_csv(csv_path, nrows=5)  # ヘッダーと型だけ確認
        sanitized = _sanitize_columns(df)

        # 重複カラムチェック（BQ は case-insensitive）
        lower_cols = sanitized.columns.str.lower()
        dupes = sanitized.columns[lower_cols.duplicated()].tolist()
        if dupes:
            errors.append(f"{table_name}: duplicate columns after sanitize: {dupes}")

        # 空カラム名チェック
        empty_cols = [c for c in sanitized.columns if not c or c.isspace()]
        if empty_cols:
            errors.append(f"{table_name}: empty column names found")

        # pyarrow 変換テスト
        try:
            import pyarrow as pa
            pa.Table.from_pandas(sanitized)
        except Exception as e:
            errors.append(f"{table_name}: pyarrow conversion failed: {e!r}")

        print(f"  OK: {table_name} ({len(sanitized.columns)} cols)")

    # metrics_history スキーマ互換チェック
    try:
        client = _get_client()
        table_id = f"{FULL_DATASET}.model_metrics_history"
        table = client.get_table(table_id)
        existing_cols = {f.name for f in table.schema}
        print(f"  OK: model_metrics_history schema ({len(existing_cols)} cols)")
    except Exception:
        print("  WARN: model_metrics_history not found (will be created)")

    if errors:
        print(f"\n  FAIL: {len(errors)} error(s) found:")
        for err in errors:
            print(f"    - {err}")
        return False

    print("\n  All checks passed.")
    return True


def print_summary(results: dict) -> None:
    """ロード結果のサマリーを出力"""
    loaded = {k: v for k, v in results.items() if v > 0}
    skipped = {k: v for k, v in results.items() if v == 0}
    total_rows = sum(loaded.values())

    print(f"\n=== Summary ===")
    print(f"  Loaded: {len(loaded)} tables, {total_rows:,} total rows")
    if skipped:
        print(f"  Skipped: {len(skipped)} tables ({', '.join(skipped.keys())})")


def main():
    parser = argparse.ArgumentParser(description="Load Statcast data to BigQuery")
    parser.add_argument("--table", type=str, help="Load a single table by BQ name")
    parser.add_argument("--all", action="store_true", help="Load all raw + predictions")
    parser.add_argument("--raw", action="store_true", help="Load raw data only")
    parser.add_argument("--predictions", action="store_true", help="Load predictions only")
    parser.add_argument("--metrics", action="store_true", help="Append model metrics history")
    parser.add_argument("--validate", action="store_true", help="Pre-training BQ compatibility check")
    args = parser.parse_args()

    # バリデーションモード
    if args.validate:
        ok = validate_bq_compat()
        sys.exit(0 if ok else 1)

    # デフォルト: 引数なしなら全テーブル
    if not any([args.table, args.all, args.raw, args.predictions, args.metrics]):
        args.all = True

    results = {}

    if args.table:
        # 単テーブル指定
        csv_name = None
        for k, v in {**RAW_TABLE_MAP, **PRED_TABLE_MAP}.items():
            if v == args.table:
                csv_name = k
                break
        if csv_name is None:
            print(f"ERROR: Unknown table '{args.table}'")
            print(f"Available: {', '.join(list(RAW_TABLE_MAP.values()) + list(PRED_TABLE_MAP.values()))}")
            sys.exit(1)
        # raw か predictions か判定
        if csv_name in RAW_TABLE_MAP:
            csv_path = RAW_DIR / csv_name
        else:
            csv_path = PRED_DIR / csv_name
        rows = load_csv_to_bq(csv_path, args.table, "WRITE_TRUNCATE")
        results[args.table] = rows

    if args.all or args.raw:
        results.update(load_all_raw())

    if args.all or args.predictions:
        results.update(load_predictions())

    if args.all or args.metrics:
        append_metrics()

    if results:
        print_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
