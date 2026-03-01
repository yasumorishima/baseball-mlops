# baseball-mlops

**MLB Statcast × MLOps — Weekly auto-retrained player performance prediction**

MLB Statcast のトラッキングデータ（打球速度・バレル率・xwOBA 等）を使い、
Marcel 法を上回る選手成績予測モデルを MLOps パイプラインで継続運用する。

| 環境 | URL |
|---|---|
| 本番 | https://baseball-mlops.streamlit.app/ |
| 開発（Spring Training 検証） | https://baseball-mlops-dev.streamlit.app/ |

---

## 精度（2025 バックテスト）

| | ML (LightGBM) | Marcel 法 |
|---|---|---|
| 打者 wOBA MAE | **0.0296** | 0.0325 |
| 投手 xFIP MAE | **0.5462** | 0.5661 |

NPB では Statcast 相当の特徴量が揃わず Marcel 法に届かなかったが、
MLB Statcast の豊富なトラッキング特徴量（EV / Barrel% / Whiff% 等）を使うことで ML が上回った。

---

## 特徴

| 項目 | 内容 |
|---|---|
| 予測ターゲット | 打者: 翌年 wOBA / 投手: 翌年 xFIP |
| モデル | LightGBM（5-fold CV） |
| ベースライン | MLB Marcel 法（加重平均 + 平均回帰 + 年齢調整） |
| データ | MLB Statcast via pybaseball（EV / Barrel% / xwOBA / sprint speed 等） |
| 自動再学習 | GitHub Actions cron（毎週月曜 JST 11:00） |
| モデル管理 | W&B Model Registry（production タグ自動昇格） |
| API | FastAPI（port 8002）— W&B から 6 時間ごとに最新モデルを自動ロード |
| ダッシュボード | Streamlit — Marcel vs ML 比較・Spring Training 検証 |

---

## アーキテクチャ

\`\`\`
[GitHub Actions — 毎週月曜 JST 11:00]
  ↓ fetch_statcast.py   pybaseball → FanGraphs + Statcast CSV 取得
  ↓ train.py            LightGBM 5-fold CV 再学習
  ↓ W&B Artifact 保存   MAE / 特徴量重要度 / モデルファイル
  ↓ production タグ更新  MAE 改善時のみ自動昇格
  ↓ predictions/ コミット Streamlit Cloud が直接読み込む

[FastAPI / Docker port 8002]
  起動時 + 6 時間ごとに W&B production モデルを自動ロード
  POST /model/reload で即時反映も可能

[Streamlit — 本番 / 開発 2 環境]
  本番 (master)  打者 wOBA / 投手 xFIP 予測ランキング + Marcel vs ML 散布図
  開発 (develop) 上記 + Spring Training 2026 実績 vs 予測 リアルタイム検証
\`\`\`

---

## ブランチ戦略

\`\`\`
master  ─→  baseball-mlops.streamlit.app     （本番）
  ↑ PR merge
develop ─→  baseball-mlops-dev.streamlit.app  （開発・検証）
\`\`\`

- \`develop\` で Spring Training 検証・UI 改善を試す
- 安定したら \`master\` に merge して本番反映
- Spring Training データは毎日 JST 23:00 に \`develop\` へ自動コミット

---

## GitHub Secrets

| Secret | 内容 |
|---|---|
| \`WANDB_API_KEY\` | W&B API キー |
| \`WANDB_ENTITY\` | W&B チーム名 |
| \`API_RELOAD_URL\` | FastAPI 公開 URL（任意） |

---

## API エンドポイント

| Endpoint | 説明 |
|---|---|
| \`GET /predict/hitter/{name}\` | 打者 翌年 wOBA（Marcel + ML） |
| \`GET /predict/pitcher/{name}\` | 投手 翌年 xFIP（Marcel + ML） |
| \`GET /rankings/hitters\` | wOBA 予測ランキング |
| \`GET /rankings/pitchers\` | xFIP 予測ランキング（低い順） |
| \`GET /model/info\` | 現在のモデルバージョン・更新日時・MAE |
| \`POST /model/reload\` | W&B から最新モデルを手動リロード |

---

## NPB Hawk-Eye への移植

Statcast = Hawk-Eye と同じトラッキングデータ形式。
NPB Hawk-Eye データ公開後、\`fetch_statcast.py\` のデータソースを差し替えるだけで移植可能な設計。

---

*Built with Claude Code / LightGBM + W&B + FastAPI + Streamlit + GitHub Actions*
