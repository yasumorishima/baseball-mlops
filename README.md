# baseball-mlops

**MLB Statcast × MLOps — Weekly auto-retrained player performance prediction**

MLB Statcast のトラッキングデータ（打球速度・バレル率・xwOBA 等）を使い、
Marcel 法を上回る選手成績予測モデルを MLOps パイプラインで継続運用する。

| 環境 | URL |
|---|---|
| 本番 | https://baseball-mlops.streamlit.app/ |
| 開発（Spring Training 検証） | https://baseball-mlops-dev.streamlit.app/ |

## 解説記事

- [Statcastデータで選手成績予測の精度は上がるか — Marcel法との比較](https://zenn.dev/shogaku/articles/baseball-mlops-statcast-vs-marcel)（Zenn）
- [Can Statcast Data Improve MLB Player Performance Predictions?](https://dev.to/yasumorishima/can-statcast-data-improve-mlb-player-performance-predictions-beating-marcel-with-lightgbm-1lb5)（DEV.to）

---

## 精度（時系列 CV バックテスト）

| | Marcel 法 | LightGBM | CatBoost | Bayes (ElasticNet) | Component (PECOTA) | Ensemble |
|---|---|---|---|---|---|---|
| 打者 wOBA MAE | 0.0326 | 0.0294 | TBD | **0.0287** | TBD | 逆MAE重み付き |
| 投手 xFIP MAE | 0.5576 | 0.5317 | TBD | **0.4830** | TBD | 逆MAE重み付き |

※ 未来リークなしの時系列 expanding-window CV による正直な値
※ LightGBM / CatBoost は Optuna 最適化済み（LGB 1000 trials / CatBoost 500 trials）
※ Ensemble = 最大5モデルの逆MAE重み付き平均（利用可能なモデルで動的に構築）

NPB では Statcast 相当の特徴量が揃わず Marcel 法に届かなかったが、
MLB Statcast の豊富なトラッキング特徴量（EV / Barrel% / Whiff% 等）を使うことで ML が上回った。

### Year-by-Year Backtest (ML vs Marcel)

| Year | Batter wOBA | | Pitcher xFIP | |
|------|-------------|---|-------------|---|
| | ML MAE | Marcel MAE | ML MAE | Marcel MAE |
| 2020 | 0.0359 | 0.0371 (+3.2%) | 0.595 | 0.618 (+3.7%) |
| 2021 | 0.0293 | 0.0317 (+7.6%) | 0.542 | 0.553 (+1.9%) |
| 2022 | 0.0296 | 0.0330 (+10.3%) | 0.578 | 0.569 (-1.5%) |
| 2023 | 0.0277 | 0.0303 (+8.7%) | 0.535 | 0.559 (+4.3%) |
| 2024 | 0.0280 | 0.0333 (+16.0%) | 0.509 | 0.522 (+2.5%) |
| **2025** | **0.0291** | **0.0331 (+12.1%)** | **0.484** | **0.504 (+4.0%)** |

**Batter**: ML wins all 5 CV years + 2025 holdout. Post-2023 improvement accelerating.
**Pitcher**: ML wins 4/5 CV years + 2025 holdout. 2022 loss likely due to limited training data (COVID 2020-2021 only).
**2025 holdout**: True holdout (never seen by Optuna/CV). Both batter and pitcher ML wins — no overfitting.

### 2025 Strict Holdout

2025 is evaluated as a **true holdout** — never seen during Optuna hyperparameter tuning or CV.
Optuna 1000 trials are tuned exclusively on 2020-2024 data, then the final model is trained on
all pre-2025 data and evaluated once on 2025. This eliminates any indirect data leakage concern.

| | ML MAE | Marcel MAE | ML wins | Improvement |
|---|---|---|---|---|
| Batter wOBA | **0.0291** | 0.0331 | Yes | +12.1% |
| Pitcher xFIP | **0.4837** | 0.5038 | Yes | +4.0% |

CV results (0.0281 / 0.521) and holdout results (0.0291 / 0.484) are consistent — no overfitting detected.

---

## 特徴

| 項目 | 内容 |
|---|---|
| 予測ターゲット | 打者: 翌年 wOBA / 投手: 翌年 xFIP |
| モデル (Python) | LightGBM + CatBoost + ElasticNet Bayes + Component (PECOTA方式) |
| モデル (BQML) | Boosted Tree Regressor + 線形回帰（SQL だけで ML） |
| 最適化 | Optuna（LGB 1000 / CatBoost 500 / Component 各200 trials） |
| ベースライン | MLB Marcel 法（加重平均 + 平均回帰 + 年齢調整） |
| アンサンブル | 最大5モデルの逆MAE重み付き平均（動的構築） |
| データ | MLB Statcast + Bat Tracking + Arsenal via pybaseball / savant-extras |
| データ基盤 | BigQuery — 生データ13テーブル + 予測結果 + メトリクス履歴 |
| 球場補正 | savant-extras で FanGraphs から動的取得（pf_5yr） |
| 自動再学習 | GitHub Actions cron（毎週月曜 JST 11:00） |
| モデル管理 | W&B Model Registry（production タグ自動昇格） |
| API (本番) | Cloud Run — FastAPI サーバーレスコンテナ（Artifact Registry 経由） |
| API (開発) | RPi5 Docker（port 8002）— W&B から 6 時間ごとに最新モデルを自動ロード |
| ダッシュボード | Streamlit — Marcel / ML / Bayes 3列比較・Spring Training 検証 |
| 通知 | Discord Webhook（Python + BQML 全モデルMAE詳細を自動通知） |

---

## アーキテクチャ

### GCP 分析基盤（v9）

```
[GitHub Actions — 毎週月曜 JST 11:00]
  ↓ fetch_statcast.py      pybaseball / savant-extras →
                           FanGraphs + Statcast + Bat Tracking + Arsenal + park_factors
  ↓ train.py               LightGBM — Optuna 1000 trials + Recency Decay 0.85/年
  ↓ train_catboost.py      CatBoost — Optuna 500 trials + 異なる分割戦略
  ↓ train_components.py    PECOTA方式 — K%/BB%/BABIP/ISO(HR/9) 個別予測 → Ridge再構成
  ↓ train_bayes.py         ElasticNet — Marcel残差学習 + LGB/Cat OOFスタッキング + MC CI
  ↓ ensemble.py            5モデル逆MAE重み付き平均（利用可能モデルで動的構築）
  ↓ W&B Artifact 保存      MAE / 特徴量重要度 / Optuna best_params / モデルファイル
  ↓ load_to_bq.py          BigQuery に生データ13テーブル + 予測結果をロード
  ↓ bqml_train.py          BigQuery ML — Boosted Tree + 線形回帰（SQLだけでML）
  ↓ predictions/ コミット   Streamlit Cloud が直接読み込む
  ↓ Discord 通知           Python + BQML 全モデルMAE詳細つき通知

[Cloud Run — baseball-mlops-api（本番 API）]
  FastAPI コンテナをサーバーレスデプロイ
  Artifact Registry 経由で Docker イメージを管理
  retrain 完了後に自動デプロイ (master のみ)

[RPi5 — FastAPI / Docker port 8002（開発 API）]
  起動時 + 6 時間ごとに W&B production モデルを自動ロード
  POST /model/reload で即時反映も可能

[BigQuery — data-platform-490901.mlb_statcast]
  生 Statcast データ 13テーブル + 予測結果 + BQML モデル + メトリクス履歴
  分析用ビュー 7本（打球品質リーダーボード、投手球種戦略分析 等）

[Streamlit — 本番 / 開発 2 環境]
  本番 (master)  打者 wOBA / 投手 xFIP 予測ランキング + Marcel vs ML 散布図
  開発 (develop) 上記 + Spring Training 2026 実績 vs 予測 リアルタイム検証
```

### AWS/GCP 対応表

本プロジェクトは、プロ野球球団が運用する AWS SageMaker + Airflow パイプラインと同等のアーキテクチャを GCP 上に構築している。

| 球団 AWS 基盤 | 本プロジェクト GCP 基盤 | 対応関係 |
|---|---|---|
| TrackMan / Hawk-Eye | pybaseball Statcast（同一トラッキングデータ） | データソース |
| S3 (raw data lake) | BigQuery `mlb_statcast` 生データ 13 テーブル | データレイク |
| Airflow DAG | GitHub Actions `weekly_retrain.yml` | オーケストレーション |
| SageMaker Processing Job | RPi5 self-hosted runner（Python 学習） | バッチ学習 |
| SageMaker Batch Transform | BigQuery ML `CREATE MODEL`（SQL ML） | SQL モデル学習 |
| SageMaker Endpoint | Cloud Run FastAPI コンテナ | 推論 API |
| ダッシュボード | Streamlit Cloud + BigQuery Studio | 可視化 |

---

## モデル詳細

### LightGBM（train.py）
- **CV**: 時系列 expanding-window splits（先頭2年 training only、3年目以降を val）
- **最適化**: Optuna 1000トライアル（TPESampler + MedianPruner）
  - 探索パラメータ: `learning_rate` / `num_leaves` / `min_child_samples` / `feature_fraction` / `bagging_fraction` / `reg_alpha` / `reg_lambda`
  - `n_estimators` 上限 1000 + early_stopping=50 で自動打ち切り

### CatBoost（train_catboost.py）
- **CV**: LightGBM とは異なる分割戦略でアンサンブル多様性を確保
- **最適化**: Optuna 500 トライアル
- **Recency Decay**: 0.85/年で近年サンプルを重み付け
- **OOF**: `cat_oof_batter/pitcher.csv` → Bayes スタッキングに利用

### Component Prediction — PECOTA 方式（train_components.py）
- **打者**: K% / BB% / BABIP / ISO を個別に LightGBM で予測 → Ridge で wOBA を再構成
- **投手**: K% / BB% / HR/9 を個別に LightGBM で予測 → Ridge で xFIP を再構成
- **最適化**: 各コンポーネント Optuna 200 トライアル
- 個別指標の予測を合成するため、モデルの解釈性が高い

### ElasticNet Bayes（train_bayes.py）
- **ターゲット**: `actual(t+1) − marcel(t+1)` の残差（delta）を予測
- **スタッキング**: `lgb_delta = lgb_oof − marcel`（LightGBM OOF による残差補正）
- **CV**: 同じ時系列 expanding-window splits で alpha × l1_ratio をグリッドサーチ
- **CI**: MC サンプリング N(delta_hat, σ) × 5000 → 10th/90th パーセンタイル = 80% CI
- **Recency Decay**: 0.85/年で近年サンプルを重み付け

### 特徴量（打者: 45+個 / 投手: 41+個）
| カテゴリ | 打者 | 投手 |
|---|---|---|
| Statcast | K%/BB%/BABIP/brl_percent/avg_hit_speed/xwOBA/sprint_speed/avg_hit_angle/ev95percent | K%/BB%/BABIP/brl_percent/avg_hit_speed/est_woba/avg_hit_angle/ev95percent |
| FanGraphs | HardHit%/Contact%/O-Swing%/SwStr%/G | K-BB%/CSW%/SwStr%/G/IP |
| **Bat Tracking (v8)** | **avg_bat_speed/swing_length/squared_up_rate/blast_rate/fast_swing_rate** | — |
| **Batted Ball (v8)** | **pull_percent/oppo_percent** | — |
| **Arsenal (v8)** | — | **n_pitch_types/primary_usage/best_whiff/avg_whiff_weighted/best_rv100/usage_entropy** |
| Lag delta | wOBA_delta_1/wOBA_delta_2/xwOBA_delta_1/bat_speed_delta_1 | xFIP_delta_1/xFIP_delta_2/whiff_delta_1/usage_entropy_delta_1 |
| Interaction | age_x_luck（Age × xwOBA-wOBA乖離） | age_x_kbb（Age × K-BB%） |
| Engineered | age_from_peak/age_sq/pa_rate/xwoba_luck/park_factor/team_changed | age_from_peak/age_sq/ip_rate/fip_era_gap/park_factor/team_changed |
| Stacking | lgb_delta / cat_delta | lgb_delta / cat_delta |

---

## ブランチ戦略

```
master  ─→  baseball-mlops.streamlit.app     （本番）
  ↑ PR merge
develop ─→  baseball-mlops-dev.streamlit.app  （開発・検証）
```

- `develop` で Spring Training 検証・UI 改善・モデル改善を試す
- 安定したら `master` に merge して本番反映
- Spring Training データは毎日 JST 23:00 に `develop` へ自動コミット

---

## GitHub Secrets

| Secret | 内容 |
|---|---|
| `WANDB_API_KEY` | W&B API キー |
| `WANDB_ENTITY` | W&B チーム名 |
| `API_RELOAD_URL` | FastAPI 公開 URL（RPi5、任意） |
| `GCP_SA_KEY` | GCP サービスアカウント JSON 鍵（BigQuery + Cloud Run） |
| `DISCORD_WEBHOOK_URL` | Discord 通知 Webhook URL |

---

## BigQuery Data Platform

All Statcast raw data, predictions, and BQML models are stored in BigQuery (free tier).

| Item | Value |
|---|---|
| Project | `data-platform-490901` |
| Dataset | `mlb_statcast` |

### Raw Data Tables (weekly auto-refresh)

| テーブル | ソース | 内容 |
|---|---|---|
| `raw_fg_batting` | FanGraphs | 打者成績 (wOBA/xwOBA/K%/BB% 等) |
| `raw_fg_pitching` | FanGraphs | 投手成績 (xFIP/FIP/ERA 等) |
| `raw_sc_batter_exitvelo` | Statcast | 打球速度・バレル率 |
| `raw_sc_batter_expected` | Statcast | 打者期待値 (xBA/xSLG/xwOBA) |
| `raw_sc_sprint_speed` | Statcast | スプリント速度 |
| `raw_sc_batted_ball` | Statcast | 打球方向 (pull/oppo %) |
| `raw_sc_bat_tracking` | Hawk-Eye | バット追跡 (bat speed/swing length 等) |
| `raw_sc_pitcher_exitvelo` | Statcast | 被打球速度 |
| `raw_sc_pitcher_expected` | Statcast | 投手期待値 |
| `raw_sc_pitcher_arsenal` | Statcast | 球種別統計 |
| `raw_park_factors` | FanGraphs | 球場補正係数 |
| `raw_batter_features` | 統合 | 打者全特徴量 (~95列) |
| `raw_pitcher_features` | 統合 | 投手全特徴量 (~105列) |

### Prediction & Model Tables

| テーブル | 内容 |
|---|---|
| `batter_predictions` / `pitcher_predictions` | Python 5-model ensemble 予測 |
| `bqml_predictions_batter` / `bqml_predictions_pitcher` | BQML Boosted Tree + 線形回帰 予測 |
| `model_metrics_history` | 全モデル MAE 時系列記録 |
| `backtest_outliers_*` / `backtest_yearly_mae_*` | バックテスト結果 |

### BQML Models

| モデル | タイプ | ターゲット |
|---|---|---|
| `bqml_batter_woba` | Boosted Tree Regressor | 翌年 wOBA |
| `bqml_pitcher_xfip` | Boosted Tree Regressor | 翌年 xFIP |
| `bqml_batter_woba_linear` | Linear Regression | 翌年 wOBA |
| `bqml_pitcher_xfip_linear` | Linear Regression | 翌年 xFIP |

### Analysis Views

`v_batter_trend` / `v_pitcher_trend` / `v_batted_ball_leaders` / `v_pitcher_arsenal` / `v_park_effects` / `v_model_comparison` / `v_data_coverage`

---

## API エンドポイント

| Endpoint | 説明 |
|---|---|
| `GET /predict/hitter/{name}` | 打者 翌年 wOBA（Marcel + ML + Bayes CI） |
| `GET /predict/pitcher/{name}` | 投手 翌年 xFIP（Marcel + ML + Bayes CI） |
| `GET /rankings/hitters` | wOBA 予測ランキング |
| `GET /rankings/pitchers` | xFIP 予測ランキング（低い順） |
| `GET /model/info` | 現在のモデルバージョン・更新日時・MAE |
| `POST /model/reload` | W&B から最新モデルを手動リロード |

---

## 今後の検討

### モデル改善
| 項目 | 概要 | 期待効果 |
|---|---|---|
| **Marcel 重みの MLB 最適化** | 現在 Tango 原典値（5/4/3、REG_PA=1200）を使用。NPB では最適化で MAE 1.4% 改善実績あり。MLB Statcast 期（2015〜）データでグリッドサーチ + ブートストラップ検定で再評価する | ベースライン精度向上 |
| **Neural Network（TabNet / FT-Transformer）** | テーブルデータ向け深層学習。LGB/Cat とは異なる非線形パターンを学習 | アンサンブル多様性向上 |
| **Similarity-Based Prediction** | 過去の類似選手キャリアパスから予測（PECOTA original approach） | 急激な衰退・ブレイクアウト検出 |
| **Pitch-Level Features** | 球種別 Stuff+ / Location+ をピッチレベルで集約 | 投手予測の精度向上 |
| **Platoon Splits** | 対左/対右の成績差を特徴量に追加 | プラトーン選手の精度向上 |
| **Injury / Workload Features** | IL 日数・前年投球数・WAR推移から故障リスクを加味 | 稼働率の予測 |
| **Bayesian Hyperparameter Transfer** | 前週の Optuna best_params を初期値に warm-start | 学習時間短縮 |

### データ拡張
| 項目 | 概要 |
|---|---|
| **Minor League Statcast** | MiLB Hawk-Eye データ（取得可能になり次第） |
| **Catcher Framing** | 捕手フレーミング指標の投手特徴量への反映 |
| **Defensive Metrics** | OAA / DRS をポジション別に取得 |

### インフラ・運用
| 項目 | 概要 |
|---|---|
| **A/B テスト基盤** | 新モデルと production モデルを並行評価し自動昇格 |
| **Data Drift 検出** | 入力特徴量の分布変化を W&B で監視・アラート |
| **Streamlit 選手比較機能** | 2選手の予測・特徴量を横並び比較 |
| **API レスポンスキャッシュ** | Redis で予測結果をキャッシュし応答速度向上 |

### NPB Hawk-Eye への移植

Statcast = Hawk-Eye と同じトラッキングデータ形式。
NPB Hawk-Eye データ公開後、`fetch_statcast.py` のデータソースを差し替えるだけで移植可能な設計。

---

*Built with Claude Code / LightGBM + CatBoost + Optuna + W&B + BigQuery + BigQuery ML + Cloud Run + FastAPI + Streamlit + GitHub Actions*
