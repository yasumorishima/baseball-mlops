"""
BigQuery ML 学習・評価・予測ラッパー

sql/ ディレクトリの SQL ファイルを読み込んで BigQuery client で実行する。
各 SQL ファイルはセミコロン区切りで複数ステートメントを含む。

Usage:
    python src/bqml_train.py --train      # CREATE MODEL 実行（打者+投手）
    python src/bqml_train.py --evaluate   # ML.EVALUATE → 結果表示 + metrics 追記
    python src/bqml_train.py --predict    # ML.PREDICT → 予測結果テーブルに保存
    python src/bqml_train.py --views      # 分析用ビュー作成
    python src/bqml_train.py --all        # 全て実行
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

PROJECT_ID = "data-platform-490901"
DATASET_ID = "mlb_shared"
SQL_DIR = Path(__file__).parent.parent / "sql"
PRED_DIR = Path(__file__).parent.parent / "predictions"


def _get_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def _execute_sql_file(client: bigquery.Client, sql_path: Path) -> list:
    """SQL ファイルを読み込んでステートメント単位で実行する。

    セミコロンで分割し、空ステートメントとコメント行をスキップする。
    Returns: 各クエリの結果リスト
    """
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = sql_text.split(";")

    results = []
    for i, stmt in enumerate(statements):
        # コメント行と空行を除去して実質的な内容があるか判定
        lines = [
            line for line in stmt.strip().split("\n")
            if line.strip() and not line.strip().startswith("--")
        ]
        if not lines:
            continue

        stmt_clean = stmt.strip()
        # ステートメントの概要を抽出（最初の非コメント行）
        summary = lines[0][:80] if lines else "?"
        print(f"  [{i+1}] {summary} ...")

        try:
            job = client.query(stmt_clean)
            result = job.result()
            rows = list(result) if result.total_rows else []
            results.append(rows)

            if rows:
                # 結果を表示
                for row in rows:
                    print(f"      {dict(row)}")
            else:
                print(f"      OK (no rows returned)")
        except Exception as e:
            print(f"      ERROR: {e}")
            results.append(None)

    return results


def train(client: bigquery.Client) -> None:
    """CREATE MODEL を実行（打者+投手、Boosted Tree + 線形回帰）"""
    print("=== Training BQML models ===")

    print("\n--- Batter wOBA models ---")
    _execute_sql_file(client, SQL_DIR / "bqml_train_batter.sql")

    print("\n--- Pitcher xFIP models ---")
    _execute_sql_file(client, SQL_DIR / "bqml_train_pitcher.sql")

    print("\nBQML training complete.")


def evaluate(client: bigquery.Client) -> dict:
    """ML.EVALUATE を実行して結果を返す + model_metrics.json に追記"""
    print("=== Evaluating BQML models ===")

    results = _execute_sql_file(client, SQL_DIR / "bqml_evaluate.sql")

    # 結果を model_metrics.json に追記
    metrics_path = PRED_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    bqml_metrics = {}

    # ML.EVALUATE 結果から MAE を抽出
    for result_rows in results:
        if result_rows is None:
            continue
        for row in result_rows:
            row_dict = dict(row)
            model_name = row_dict.get("model", "")

            # ML.EVALUATE は mean_absolute_error を返す
            mae = row_dict.get("mean_absolute_error")
            if mae is not None:
                if "batter_boosted" in model_name:
                    bqml_metrics["bqml_bt_mae_woba"] = round(float(mae), 4)
                elif "batter_linear" in model_name:
                    bqml_metrics["bqml_linear_mae_woba"] = round(float(mae), 4)
                elif "pitcher_boosted" in model_name:
                    bqml_metrics["bqml_bt_mae_xfip"] = round(float(mae), 4)
                elif "pitcher_linear" in model_name:
                    bqml_metrics["bqml_linear_mae_xfip"] = round(float(mae), 4)

            # 比較クエリの結果
            comparison = row_dict.get("comparison", "")
            if "batter" in comparison:
                for k in ["bqml_mae", "python_ensemble_mae", "marcel_mae"]:
                    if k in row_dict and row_dict[k] is not None:
                        bqml_metrics[f"comparison_batter_{k}"] = round(float(row_dict[k]), 4)
            elif "pitcher" in comparison:
                for k in ["bqml_mae", "python_ensemble_mae", "marcel_mae"]:
                    if k in row_dict and row_dict[k] is not None:
                        bqml_metrics[f"comparison_pitcher_{k}"] = round(float(row_dict[k]), 4)

    # metrics ファイルに追記
    if bqml_metrics:
        metrics.update(bqml_metrics)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nBQML metrics saved to model_metrics.json: {bqml_metrics}")

    return bqml_metrics


def predict(client: bigquery.Client) -> None:
    """ML.PREDICT を実行して予測結果テーブルに保存"""
    print("=== Running BQML predictions ===")
    _execute_sql_file(client, SQL_DIR / "bqml_predict.sql")

    # 結果行数を確認
    for table_name in ["bqml_predictions_batter", "bqml_predictions_pitcher"]:
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
        try:
            table = client.get_table(table_id)
            print(f"  {table_name}: {table.num_rows} rows")
        except Exception as e:
            print(f"  {table_name}: ERROR - {e}")

    print("\nBQML predictions complete.")


def create_views(client: bigquery.Client) -> None:
    """分析用ビューを作成"""
    print("=== Creating analysis views ===")
    _execute_sql_file(client, SQL_DIR / "views.sql")
    print("\nViews created.")


def main():
    parser = argparse.ArgumentParser(description="BigQuery ML training & evaluation")
    parser.add_argument("--train", action="store_true", help="Train BQML models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate BQML models")
    parser.add_argument("--predict", action="store_true", help="Run BQML predictions")
    parser.add_argument("--views", action="store_true", help="Create analysis views")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.predict, args.views, args.all]):
        args.all = True

    client = _get_client()

    if args.all or args.views:
        create_views(client)

    if args.all or args.train:
        train(client)

    if args.all or args.evaluate:
        evaluate(client)

    if args.all or args.predict:
        predict(client)

    print("\nDone.")


if __name__ == "__main__":
    main()
