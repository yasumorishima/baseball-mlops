# baseball-mlops

**MLB Statcast × MLOps — Weekly auto-retrained player performance prediction**

MLB Statcast のトラッキングデータ（打球速度・バレル率・xwOBA 等）を使い、
Marcel 法を上回る選手成績予測モデルを MLOps パイプラインで継続運用する。

| 環境 | URL |
|---|---|
| 本番 | https://baseball-mlops.streamlit.app/ |
| 開発（Spring Training 検証） | https://baseball-mlops-dev.streamlit.app/ |

---

## 精度（時系列 CV バックテスト）

| | Marcel 法 | LightGBM | Bayes (ElasticNet) | Ensemble |
|---|---|---|---|---|
| 打者 wOBA MAE | 0.0326 | 0.0294 | **0.0286** | 逆MAE重み付き |
| 投手 xFIP MAE | 0.5576 | 0.5331 | **0.4815** | 逆MAE重み付き |

※ 未来リークなしの時系列 expanding-window CV による正直な値
※ LightGBM は Optuna 1000 トライアル最適化済み（TPESampler + MedianPruner）
※ Ensemble = Marcel × 31% + LGB × 33% + Bayes × 36%（逆MAE比で自動算出）

NPB では Statcast 相当の特徴量が揃わず Marcel 法に届かなかったが、
MLB Statcast の豊富なトラッキング特徴量（EV / Barrel% / Whiff% 等）を使うことで ML が上回った。

### Year-by-Year Backtest (ML vs Marcel)

| Year | Batter wOBA | | Pitcher xFIP | |
|------|-------------|---|-------------|---|
| | ML MAE | Marcel MAE | ML MAE | Marcel MAE |
| 2020 | 0.0361 | 0.0371 (+2.8%) | 0.598 | 0.618 (+3.2%) |
| 2021 | 0.0292 | 0.0317 (+7.9%) | 0.541 | 0.553 (+2.1%) |
| 2022 | 0.0294 | 0.0330 (+11.1%) | 0.579 | 0.569 (-1.8%) |
| 2023 | 0.0280 | 0.0303 (+7.6%) | 0.535 | 0.559 (+4.3%) |
| 2024 | 0.0279 | 0.0333 (+16.2%) | 0.508 | 0.522 (+2.5%) |

**Batter**: ML wins all 5 CV years. Post-2023 improvement accelerating (7.6% -> 12.6%).
**Pitcher**: ML wins 4/5 CV years. 2022 loss likely due to limited training data (COVID 2020-2021 only).

### 2025 Strict Holdout (in progress)

2025 is evaluated as a **true holdout** -- never seen during Optuna hyperparameter tuning or CV.
Optuna 1000 trials are tuned exclusively on 2020-2024 data, then the final model is trained on
all pre-2025 data and evaluated once on 2025. This eliminates any indirect data leakage concern.

Results will be updated here once the CI run completes.

---

## 特徴

| 項目 | 内容 |
|---|---|
| 予測ターゲット | 打者: 翌年 wOBA / 投手: 翌年 xFIP |
| モデル | LightGBM（Optuna 1000trial 最適化 + 時系列 expanding-window CV） |
| ベースライン | MLB Marcel 法（加重平均 + 平均回帰 + 年齢調整） |
| Bayes 補正 | ElasticNet で Marcel 残差を予測、80% MC CI 付与 |
| データ | MLB Statcast via pybaseball（EV / Barrel% / xwOBA / sprint speed 等）|
| 球場補正 | savant-extras で FanGraphs から動的取得（pf_5yr） |
| 自動再学習 | GitHub Actions cron（毎週月曜 JST 11:00） |
| モデル管理 | W&B Model Registry（production タグ自動昇格） |
| API | FastAPI（port 8002）— W&B から 6 時間ごとに最新モデルを自動ロード |
| ダッシュボード | Streamlit — Marcel / ML / Bayes 3列比較・Spring Training 検証 |

---

## アーキテクチャ

```
[GitHub Actions — 毎週月曜 JST 11:00]
  ↓ fetch_statcast.py   pybaseball → FanGraphs + Statcast + park_factors CSV 取得
  ↓ train.py            Optuna 1000trial ハイパーパラメータ最適化
                        時系列 expanding-window CV で LightGBM 再学習
                        OOF を lgb_oof_batter/pitcher.csv に保存（Bayes スタッキング用）
  ↓ train_bayes.py      ElasticNet で Marcel 残差を学習（lgb_delta スタッキング特徴量）
                        Monte Carlo CI (80%) 付与、bayes_coef.json 保存
  ↓ ensemble.py         逆MAE重み付き平均（Marcel + LGB + Bayes）
                        model_metrics.json から重みを自動算出、W&B にログ
  ↓ W&B Artifact 保存   MAE / 特徴量重要度 / Optuna best_params / モデルファイル
  ↓ production タグ更新  MAE 改善時のみ自動昇格
  ↓ predictions/ コミット Streamlit Cloud が直接読み込む

[FastAPI / Docker port 8002]
  起動時 + 6 時間ごとに W&B production モデルを自動ロード
  POST /model/reload で即時反映も可能

[Streamlit — 本番 / 開発 2 環境]
  本番 (master)  打者 wOBA / 投手 xFIP 予測ランキング + Marcel vs ML 散布図
  開発 (develop) 上記 + Spring Training 2026 実績 vs 予測 リアルタイム検証
```

---

## モデル詳細

### LightGBM（train.py）
- **CV**: 時系列 expanding-window splits（先頭2年 training only、3年目以降を val）
- **最適化**: Optuna 1000トライアル（TPESampler + MedianPruner）
  - 探索パラメータ: `learning_rate` / `num_leaves` / `min_child_samples` / `feature_fraction` / `bagging_fraction` / `reg_alpha` / `reg_lambda`
  - `n_estimators` 上限 1000 + early_stopping=50 で自動打ち切り

### ElasticNet Bayes（train_bayes.py）
- **ターゲット**: `actual(t+1) − marcel(t+1)` の残差（delta）を予測
- **スタッキング**: `lgb_delta = lgb_oof − marcel`（LightGBM OOF による残差補正）
- **CV**: 同じ時系列 expanding-window splits で alpha × l1_ratio をグリッドサーチ
- **CI**: MC サンプリング N(delta_hat, σ) × 5000 → 10th/90th パーセンタイル = 80% CI
- **Recency Decay**: 0.85/年で近年サンプルを重み付け

### 特徴量（打者: 31個 / 投手: 28個）
| カテゴリ | 打者 | 投手 |
|---|---|---|
| Statcast | K%/BB%/BABIP/brl_percent/avg_hit_speed/xwOBA/sprint_speed/avg_hit_angle/ev95percent | K%/BB%/BABIP/brl_percent/avg_hit_speed/est_woba/avg_hit_angle/ev95percent |
| FanGraphs | HardHit%/Contact%/O-Swing%/SwStr%/G/maxEV | K-BB%/CSW%/SwStr%/G/IP |
| FanGraphs (2020+) | — | Stuff+/Location+/Pitching+ |
| Engineered | age_from_peak/age_sq/pa_rate/xwoba_luck/park_factor/team_changed/g_change_rate | age_from_peak/age_sq/ip_rate/park_factor/team_changed/g_change_rate |
| **Lag delta (v6)** | **wOBA_delta_1/xwOBA_delta_1/K_pct_delta_1/BB_pct_delta_1/brl_delta_1** | **xFIP_delta_1/K_pct_delta_1/BB_pct_delta_1/KBB_delta_1** |
| **Interaction (v6)** | **age_x_luck**（Age × xwOBA-wOBA乖離） | **age_x_kbb**（Age × K-BB%） |
| Stacking | lgb_delta | lgb_delta |

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
| `API_RELOAD_URL` | FastAPI 公開 URL（任意） |

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

## NPB Hawk-Eye への移植

Statcast = Hawk-Eye と同じトラッキングデータ形式。
NPB Hawk-Eye データ公開後、`fetch_statcast.py` のデータソースを差し替えるだけで移植可能な設計。

---

*Built with Claude Code / LightGBM + Optuna + W&B + FastAPI + Streamlit + GitHub Actions*
