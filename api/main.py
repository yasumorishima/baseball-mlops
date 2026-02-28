"""
baseball-mlops FastAPI

W&B Model Registry から production モデルを自動ロードし、
打者 wOBA・投手 xFIP の予測を提供する。
APScheduler で 6 時間ごとに新モデルをポーリングする。
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import wandb
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException

PROJ_DIR = Path(__file__).parent.parent / "data" / "projections"

# ---------------------------------------------------------------------------
# モデル管理
# ---------------------------------------------------------------------------

state = {
    "bat_model": None,
    "pit_model": None,
    "bat_version": None,
    "pit_version": None,
    "last_updated": None,
}


def _load_from_wandb(artifact_name: str):
    """W&B から production タグのモデルをダウンロードして返す"""
    try:
        api = wandb.Api()
        artifact = api.artifact(
            f"{os.environ['WANDB_ENTITY']}/baseball-mlops/{artifact_name}:production"
        )
        download_dir = artifact.download()
        model_file = next(Path(download_dir).glob("*.pkl"))
        return joblib.load(model_file), artifact.version
    except Exception as e:
        print(f"W&B load failed ({artifact_name}): {e}")
        # フォールバック: ローカルファイル
        local = Path(__file__).parent.parent / "models" / f"{artifact_name.split('-')[0]}_model.pkl"
        if local.exists():
            return joblib.load(local), "local"
        return None, None


def reload_models():
    """W&B から最新 production モデルをロード（差分があれば更新）"""
    from datetime import datetime

    bat_model, bat_ver = _load_from_wandb("woba-model")
    pit_model, pit_ver = _load_from_wandb("xfip-model")

    if bat_model is not None:
        state["bat_model"] = bat_model
        state["bat_version"] = bat_ver
    if pit_model is not None:
        state["pit_model"] = pit_model
        state["pit_version"] = pit_ver
    state["last_updated"] = datetime.utcnow().isoformat() + "Z"
    print(f"Models reloaded: woba={bat_ver}, xfip={pit_ver}")


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    reload_models()
    scheduler.add_job(reload_models, "interval", hours=6)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(
    title="baseball-mlops API",
    description=(
        "MLB Statcast × MLOps — LightGBM による選手成績予測 API。\n\n"
        "W&B Model Registry の `production` タグモデルを 6 時間ごとに自動更新。\n\n"
        "**予測ターゲット**: 打者 wOBA / 投手 xFIP\n"
        "**ベースライン比較**: MLB Marcel 法（加重平均 + 平均回帰 + 年齢調整）"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------

def _load_pred_csv(fname: str) -> pd.DataFrame:
    path = PROJ_DIR / fname
    if not path.exists():
        raise HTTPException(status_code=503, detail="Predictions not yet generated. Run train.py first.")
    return pd.read_csv(path)


@app.get("/model/info", summary="現在のモデル情報")
def model_info():
    return {
        "woba_model_version": state["bat_version"],
        "xfip_model_version": state["pit_version"],
        "last_updated": state["last_updated"],
    }


@app.post("/model/reload", summary="W&B から最新モデルを手動リロード")
def manual_reload():
    reload_models()
    return {"status": "reloaded", "woba_version": state["bat_version"],
            "xfip_version": state["pit_version"]}


@app.get("/predict/hitter/{name}", summary="打者 翌年 wOBA 予測（Marcel vs ML）")
def predict_hitter(name: str):
    df = _load_pred_csv("batter_predictions.csv")
    match = df[df["player"].str.lower().str.contains(name.lower())]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")
    row = match.iloc[0]
    return {
        "player": row["player"],
        "team": row.get("Team", ""),
        "age": row.get("Age", ""),
        "pred_year": int(row["pred_year"]),
        "ml_woba": float(row["pred_woba"]),
        "marcel_woba": float(row["marcelwoba"]),
        "woba_last_season": float(row.get("wOBA_last", np.nan)),
        "model_version": state["bat_version"],
    }


@app.get("/predict/pitcher/{name}", summary="投手 翌年 xFIP 予測（Marcel vs ML）")
def predict_pitcher(name: str):
    df = _load_pred_csv("pitcher_predictions.csv")
    match = df[df["player"].str.lower().str.contains(name.lower())]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found")
    row = match.iloc[0]
    return {
        "player": row["player"],
        "team": row.get("Team", ""),
        "age": row.get("Age", ""),
        "pred_year": int(row["pred_year"]),
        "ml_xfip": float(row["pred_xfip"]),
        "marcel_xfip": float(row["marcelxfip"]),
        "xfip_last_season": float(row.get("xFIP_last", np.nan)),
        "model_version": state["pit_version"],
    }


@app.get("/rankings/hitters", summary="wOBA 予測ランキング")
def rankings_hitters(top: int = 20, sort_by: str = "ml_woba"):
    df = _load_pred_csv("batter_predictions.csv")
    sort_col = "pred_woba" if sort_by == "ml_woba" else "marcelwoba"
    df_sorted = df.sort_values(sort_col, ascending=False).head(top)
    return df_sorted[["player", "Team", "Age", "pred_woba", "marcelwoba", "wOBA_last"]].to_dict("records")


@app.get("/rankings/pitchers", summary="xFIP 予測ランキング（低い順）")
def rankings_pitchers(top: int = 20, sort_by: str = "ml_xfip"):
    df = _load_pred_csv("pitcher_predictions.csv")
    sort_col = "pred_xfip" if sort_by == "ml_xfip" else "marcelxfip"
    df_sorted = df.sort_values(sort_col, ascending=True).head(top)
    return df_sorted[["player", "Team", "Age", "pred_xfip", "marcelxfip", "xFIP_last"]].to_dict("records")
