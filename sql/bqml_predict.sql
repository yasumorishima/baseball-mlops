-- BigQuery ML: 翌シーズン予測を実行して結果テーブルに保存
--
-- 最新シーズンのデータを入力として、翌年の wOBA / xFIP を予測する。
-- 結果は bqml_predictions_batter / bqml_predictions_pitcher テーブルに保存。

-- ============================================================
-- 1. 打者 wOBA 予測 (Boosted Tree)
-- ============================================================
CREATE OR REPLACE TABLE `data-platform-490901.mlb_shared.bqml_predictions_batter` AS
WITH latest AS (
  SELECT MAX(season) AS max_season
  FROM `data-platform-490901.mlb_shared.raw_batter_features`
),
-- 最新シーズンの選手データで予測（ターゲットはダミー）
predict_input AS (
  SELECT t.*
  FROM `data-platform-490901.mlb_shared.v_batter_train` t, latest l
  WHERE t.season = l.max_season
),
predictions AS (
  SELECT
    player,
    season,
    season + 1 AS pred_year,
    predicted_target_woba AS bqml_woba_boosted,
    target_woba AS actual_woba
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_batter_woba`,
    (SELECT * FROM predict_input)
  )
),
linear_preds AS (
  SELECT
    player,
    predicted_target_woba AS bqml_woba_linear
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_batter_woba_linear`,
    (SELECT * FROM predict_input)
  )
)
SELECT
  p.player,
  p.season AS season_last,
  p.pred_year,
  p.bqml_woba_boosted,
  l.bqml_woba_linear,
  p.actual_woba,
  CURRENT_TIMESTAMP() AS predicted_at
FROM predictions p
LEFT JOIN linear_preds l ON p.player = l.player
ORDER BY p.bqml_woba_boosted DESC;


-- ============================================================
-- 2. 投手 xFIP 予測 (Boosted Tree)
-- ============================================================
CREATE OR REPLACE TABLE `data-platform-490901.mlb_shared.bqml_predictions_pitcher` AS
WITH latest AS (
  SELECT MAX(season) AS max_season
  FROM `data-platform-490901.mlb_shared.raw_pitcher_features`
),
predict_input AS (
  SELECT t.*
  FROM `data-platform-490901.mlb_shared.v_pitcher_train` t, latest l
  WHERE t.season = l.max_season
),
predictions AS (
  SELECT
    player,
    season,
    season + 1 AS pred_year,
    predicted_target_xfip AS bqml_xfip_boosted,
    target_xfip AS actual_xfip
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_pitcher_xfip`,
    (SELECT * FROM predict_input)
  )
),
linear_preds AS (
  SELECT
    player,
    predicted_target_xfip AS bqml_xfip_linear
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_pitcher_xfip_linear`,
    (SELECT * FROM predict_input)
  )
)
SELECT
  p.player,
  p.season AS season_last,
  p.pred_year,
  p.bqml_xfip_boosted,
  l.bqml_xfip_linear,
  p.actual_xfip,
  CURRENT_TIMESTAMP() AS predicted_at
FROM predictions p
LEFT JOIN linear_preds l ON p.player = l.player
ORDER BY p.bqml_xfip_boosted ASC;
