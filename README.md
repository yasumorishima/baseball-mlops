# baseball-mlops

**MLB Statcast × MLOps — Weekly auto-retrained player performance prediction**

MLB Statcast のトラッキングデータ（打球速度・バレル率・xwOBA 等）を使い、
Marcel 法を上回る選手成績予測モデルを MLOps パイプラインで継続運用する。

[Marcel 法による統計予測（npb-prediction）](https://github.com/yasumorishima/npb-prediction) との役割分担:
- **npb-prediction** — NPB データ経験・Marcel 法の実証
- **baseball-mlops** — MLOps 技術力・Statcast 活用・Hawk-Eye 移植可能性

---

## 特徴

| 項目 | 内容 |
|---|---|
| 予測ターゲット | 打者: 翌年 wOBA / 投手: 翌年 xFIP |
| モデル | LightGBM（5-fold CV） |
| ベースライン | MLB Marcel 法（加重平均 + 平均回帰 + 年齢調整） |
| データ | MLB Statcast via pybaseball（EV / Barrel% / xwOBA / sprint speed 等） |
| 自動再学習 | GitHub Actions cron（毎週月曜）|
| モデル管理 | W&B Model Registry（production タグ）|
| API | FastAPI — 6 時間ごとに W&B から最新モデルを自動ロード |
| ダッシュボード | Streamlit — Marcel vs ML vs 実績 3 列比較 |

---

## アーキテクチャ

```
[GitHub Actions cron 毎週月曜 JST 11:00]
  ↓ fetch_statcast.py  （pybaseball → FanGraphs + Statcast CSV）
  ↓ train.py           （LightGBM 5-fold CV 再学習）
  ↓ W&B Artifact 保存  （MAE / 特徴量重要度 / モデルファイル）
  ↓ production タグ更新 （最良 MAE のモデルに自動付与）

[FastAPI / Docker port 8002]
  起動時 + 6 時間ごとに W&B から production モデルを自動ロード
  POST /model/reload で即時更新も可能

[Streamlit Dashboard]
  選手名   Marcel wOBA  |  ML wOBA  |  実績（開幕後）
  散布図: Marcel vs ML 乖離ハイライト
```

---

## セットアップ

### 1. GitHub Secrets 登録

| Secret | 内容 |
|---|---|
| `WANDB_API_KEY` | W&B API キー |
| `WANDB_ENTITY` | W&B ユーザー名 |
| `API_RELOAD_URL` | FastAPI の公開 URL（任意） |

### 2. ローカル / Docker

```bash
# ローカル
pip install -r requirements.txt
python src/fetch_statcast.py   # データ取得（GitHub Actions で自動実行）
WANDB_API_KEY=xxx python src/train.py
uvicorn api.main:app --port 8002
streamlit run streamlit/app.py

# Docker
docker-compose up -d
```

---

## API エンドポイント

| Endpoint | 説明 |
|---|---|
| `GET /predict/hitter/{name}` | 打者 翌年 wOBA（Marcel + ML） |
| `GET /predict/pitcher/{name}` | 投手 翌年 xFIP（Marcel + ML） |
| `GET /rankings/hitters` | wOBA 予測ランキング |
| `GET /rankings/pitchers` | xFIP 予測ランキング（低い順） |
| `GET /model/info` | 現在のモデルバージョン・更新日時 |
| `POST /model/reload` | W&B から最新モデルを手動リロード |

---

## NPB Hawk-Eye への移植

Statcast = Hawk-Eye と同じトラッキングデータ形式。
NPB Hawk-Eye データ公開後、`fetch_statcast.py` のデータソースを差し替えるだけで移植可能。

---

*Built with Claude Code / LightGBM + W&B + FastAPI + Streamlit + GitHub Actions*
