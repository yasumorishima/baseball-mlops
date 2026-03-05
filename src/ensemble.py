"""
逆MAE重み付きアンサンブル (Marcel + LGB + Bayes)

predictions/model_metrics.json から各モデルの MAE を読み込み、
逆 MAE 比で重み付き平均を計算して ensemble_woba / ensemble_xfip 列を追記する。

入力:  predictions/batter_predictions.csv  (pred_woba, marcel_woba, bayes_woba)
       predictions/pitcher_predictions.csv (pred_xfip, marcel_xfip, bayes_xfip)
       predictions/model_metrics.json      (train.py + train_bayes.py が出力)
出力:  同 CSV に ensemble_woba / ensemble_xfip 列を追記
       W&B に weights・理論 MAE をログ
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

PRED_DIR = Path(__file__).parent.parent / "predictions"

# model_metrics.json がない場合のフォールバック（直近 Optuna run の値）
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
        # 全キーが揃っているか確認
        if all(k in m for k in _FALLBACK_METRICS):
            return m
    return _FALLBACK_METRICS


def _inv_mae_weights(mae_a: float, mae_b: float, mae_c: float) -> tuple[float, float, float]:
    """3モデルの逆MAEを正規化して重みを返す。"""
    w = np.array([1 / mae_a, 1 / mae_b, 1 / mae_c])
    w = w / w.sum()
    return float(w[0]), float(w[1]), float(w[2])


def _theoretical_mae(w: tuple, maes: tuple) -> float:
    """
    独立予測の重み付きアンサンブル MAE の下界（誤差間の相関 = 0 を仮定）:
      MAE_ens ≈ sqrt(sum(w_i^2 * MAE_i^2))
    実際の相関は正なので真の MAE はこれより大きいが、参考値として使う。
    """
    return float(np.sqrt(sum(wi**2 * mi**2 for wi, mi in zip(w, maes))))


def run():
    metrics = _load_metrics()

    # ===== 打者 wOBA =====
    bat_path = PRED_DIR / "batter_predictions.csv"
    if bat_path.exists():
        bat = pd.read_csv(bat_path)
        if all(c in bat.columns for c in ["pred_woba", "marcel_woba", "bayes_woba"]):
            w_m, w_lgb, w_b = _inv_mae_weights(
                metrics["marcel_mae_woba"],
                metrics["lgb_mae_woba"],
                metrics["bayes_mae_woba"],
            )
            bat["ensemble_woba"] = (
                w_m   * bat["marcel_woba"].fillna(0.320) +
                w_lgb * bat["pred_woba"].fillna(bat["marcel_woba"]) +
                w_b   * bat["bayes_woba"].fillna(bat["marcel_woba"])
            ).round(3)
            bat.to_csv(bat_path, index=False)
            print(f"Batter ensemble: w_marcel={w_m:.3f}, w_lgb={w_lgb:.3f}, w_bayes={w_b:.3f}")
        else:
            w_m = w_lgb = w_b = None
            print("Batter: missing columns, skipped")
    else:
        w_m = w_lgb = w_b = None

    # ===== 投手 xFIP =====
    pit_path = PRED_DIR / "pitcher_predictions.csv"
    if pit_path.exists():
        pit = pd.read_csv(pit_path)
        if all(c in pit.columns for c in ["pred_xfip", "marcel_xfip", "bayes_xfip"]):
            w_m_p, w_lgb_p, w_b_p = _inv_mae_weights(
                metrics["marcel_mae_xfip"],
                metrics["lgb_mae_xfip"],
                metrics["bayes_mae_xfip"],
            )
            pit["ensemble_xfip"] = (
                w_m_p   * pit["marcel_xfip"].fillna(4.20) +
                w_lgb_p * pit["pred_xfip"].fillna(pit["marcel_xfip"]) +
                w_b_p   * pit["bayes_xfip"].fillna(pit["marcel_xfip"])
            ).round(2)
            pit.to_csv(pit_path, index=False)
            print(f"Pitcher ensemble: w_marcel={w_m_p:.3f}, w_lgb={w_lgb_p:.3f}, w_bayes={w_b_p:.3f}")
        else:
            w_m_p = w_lgb_p = w_b_p = None
            print("Pitcher: missing columns, skipped")
    else:
        w_m_p = w_lgb_p = w_b_p = None

    # ===== W&B ログ =====
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    entity = os.environ.get("WANDB_ENTITY") or None
    run_wb = wandb.init(project="baseball-mlops", entity=entity, job_type="ensemble")

    log_dict = {k: v for k, v in metrics.items()}
    if w_m is not None:
        theo_woba = _theoretical_mae(
            (w_m, w_lgb, w_b),
            (metrics["marcel_mae_woba"], metrics["lgb_mae_woba"], metrics["bayes_mae_woba"]),
        )
        log_dict.update({
            "ensemble_w_marcel_woba": round(w_m, 3),
            "ensemble_w_lgb_woba":    round(w_lgb, 3),
            "ensemble_w_bayes_woba":  round(w_b, 3),
            "ensemble_theoretical_mae_woba": round(theo_woba, 4),
        })
        print(f"  wOBA theoretical MAE: {theo_woba:.4f} (best single: {metrics['bayes_mae_woba']:.4f})")
    if w_m_p is not None:
        theo_xfip = _theoretical_mae(
            (w_m_p, w_lgb_p, w_b_p),
            (metrics["marcel_mae_xfip"], metrics["lgb_mae_xfip"], metrics["bayes_mae_xfip"]),
        )
        log_dict.update({
            "ensemble_w_marcel_xfip": round(w_m_p, 3),
            "ensemble_w_lgb_xfip":    round(w_lgb_p, 3),
            "ensemble_w_bayes_xfip":  round(w_b_p, 3),
            "ensemble_theoretical_mae_xfip": round(theo_xfip, 4),
        })
        print(f"  xFIP theoretical MAE: {theo_xfip:.4f} (best single: {metrics['bayes_mae_xfip']:.4f})")

    wandb.log(log_dict)
    run_wb.finish()
    print("=== ensemble.py complete ===")


if __name__ == "__main__":
    run()
