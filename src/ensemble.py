"""
逆MAE重み付きアンサンブル v8 (Marcel + LGB + CatBoost + Bayes + Component)

predictions/model_metrics.json から各モデルの MAE を読み込み、
逆 MAE 比で重み付き平均を計算して ensemble_woba / ensemble_xfip 列を追記する。

v8: 3モデル → 5モデルに拡張（CatBoost + Component予測追加）
    利用可能なモデルだけで動的にアンサンブルを構築（フォールバック対応）
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

PRED_DIR = Path(__file__).parent.parent / "predictions"

# フォールバック（全モデルが揃わない場合の基準値）
_FALLBACK_METRICS = {
    "marcel_mae_woba": 0.0326,
    "lgb_mae_woba":    0.0294,
    "bayes_mae_woba":  0.0286,
    "marcel_mae_xfip": 0.5576,
    "lgb_mae_xfip":    0.5329,
    "bayes_mae_xfip":  0.4846,
}


def _load_metrics() -> dict:
    path = PRED_DIR / "model_metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        # 最低限 marcel + lgb があれば動作
        if "marcel_mae_woba" in m and "lgb_mae_woba" in m:
            return m
    return _FALLBACK_METRICS


def _inv_mae_weights(*maes: float) -> list[float]:
    """N モデルの逆MAEを正規化して重みを返す。"""
    w = np.array([1 / m for m in maes])
    w = w / w.sum()
    return [float(x) for x in w]


def _theoretical_mae(weights: list, maes: list) -> float:
    return float(np.sqrt(sum(w**2 * m**2 for w, m in zip(weights, maes))))


def _ensemble_col(df: pd.DataFrame, model_specs: list[tuple[str, str, float]],
                  fallback_val: float) -> pd.Series:
    """
    model_specs: [(col_name, fallback_col, weight), ...]
    動的にアンサンブルを計算。
    """
    result = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for col, fallback_col, w in model_specs:
        if col in df.columns:
            vals = df[col].fillna(df.get(fallback_col, fallback_val))
            result += w * vals
            total_weight += w
    if total_weight > 0:
        result = result / total_weight * sum(w for _, _, w in model_specs if _ in df.columns)
        # Re-normalize: just use proper weighted average
        result = pd.Series(0.0, index=df.index)
        norm = sum(w for col, _, w in model_specs if col in df.columns)
        for col, fallback_col, w in model_specs:
            if col in df.columns:
                vals = df[col].fillna(df.get(fallback_col, fallback_val))
                result += (w / norm) * vals
    return result


def run():
    metrics = _load_metrics()

    # ===== 打者 wOBA =====
    bat_path = PRED_DIR / "batter_predictions.csv"
    bat_weights = {}
    if bat_path.exists():
        bat = pd.read_csv(bat_path)

        # 利用可能なモデルを動的に検出
        available = []
        model_cols = [
            ("marcel_woba", "marcel_mae_woba"),
            ("pred_woba",   "lgb_mae_woba"),
            ("cat_woba",    "cat_mae_woba"),
            ("bayes_woba",  "bayes_mae_woba"),
            ("component_woba", "component_mae_woba"),
        ]
        for col, mae_key in model_cols:
            if col in bat.columns and mae_key in metrics:
                available.append((col, metrics[mae_key]))

        if len(available) >= 2:
            names = [a[0] for a in available]
            maes = [a[1] for a in available]
            weights = _inv_mae_weights(*maes)

            ensemble = pd.Series(0.0, index=bat.index)
            for (col, _), w in zip(available, weights):
                ensemble += w * bat[col].fillna(bat.get("marcel_woba", 0.320))
            bat["ensemble_woba"] = ensemble.round(3)
            bat.to_csv(bat_path, index=False)

            weight_str = ", ".join(f"{col}={w:.3f}" for (col, _), w in zip(available, weights))
            print(f"Batter ensemble ({len(available)} models): {weight_str}")
            bat_weights = dict(zip(names, weights))

            theo = _theoretical_mae(weights, maes)
            print(f"  wOBA theoretical MAE: {theo:.4f} (best single: {min(maes):.4f})")
        else:
            print("Batter: insufficient models for ensemble")

    # ===== 投手 xFIP =====
    pit_path = PRED_DIR / "pitcher_predictions.csv"
    pit_weights = {}
    if pit_path.exists():
        pit = pd.read_csv(pit_path)

        available = []
        model_cols = [
            ("marcel_xfip", "marcel_mae_xfip"),
            ("pred_xfip",   "lgb_mae_xfip"),
            ("cat_xfip",    "cat_mae_xfip"),
            ("bayes_xfip",  "bayes_mae_xfip"),
            ("component_xfip", "component_mae_xfip"),
        ]
        for col, mae_key in model_cols:
            if col in pit.columns and mae_key in metrics:
                available.append((col, metrics[mae_key]))

        if len(available) >= 2:
            names = [a[0] for a in available]
            maes = [a[1] for a in available]
            weights = _inv_mae_weights(*maes)

            ensemble = pd.Series(0.0, index=pit.index)
            for (col, _), w in zip(available, weights):
                ensemble += w * pit[col].fillna(pit.get("marcel_xfip", 4.20))
            pit["ensemble_xfip"] = ensemble.round(2)
            pit.to_csv(pit_path, index=False)

            weight_str = ", ".join(f"{col}={w:.3f}" for (col, _), w in zip(available, weights))
            print(f"Pitcher ensemble ({len(available)} models): {weight_str}")
            pit_weights = dict(zip(names, weights))

            theo = _theoretical_mae(weights, maes)
            print(f"  xFIP theoretical MAE: {theo:.4f} (best single: {min(maes):.4f})")
        else:
            print("Pitcher: insufficient models for ensemble")

    # ===== W&B ログ =====
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run_wb = wandb.init(project="baseball-mlops", entity=entity, job_type="ensemble")

    log_dict = {k: v for k, v in metrics.items()}
    for name, w in bat_weights.items():
        log_dict[f"ensemble_w_{name}"] = round(w, 3)
    for name, w in pit_weights.items():
        log_dict[f"ensemble_w_{name}"] = round(w, 3)
    log_dict["n_models_batter"] = len(bat_weights)
    log_dict["n_models_pitcher"] = len(pit_weights)

    wandb.log(log_dict)
    run_wb.finish()
    print("=== ensemble.py complete ===")


if __name__ == "__main__":
    run()
