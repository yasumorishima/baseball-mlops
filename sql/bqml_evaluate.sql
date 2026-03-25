-- BigQuery ML: モデル評価 + Python版との精度比較
-- 各モデルの MAE を一括取得し、Python版アンサンブルと比較する

-- ============================================================
-- 1. BQML Boosted Tree 打者評価
-- ============================================================
SELECT
  'bqml_batter_boosted_tree' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.mlb_shared.bqml_batter_woba`,
  (SELECT * FROM `data-platform-490901.mlb_shared.v_batter_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 2. BQML 線形回帰 打者評価
-- ============================================================
SELECT
  'bqml_batter_linear' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.mlb_shared.bqml_batter_woba_linear`,
  (SELECT * FROM `data-platform-490901.mlb_shared.v_batter_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 3. BQML Boosted Tree 投手評価
-- ============================================================
SELECT
  'bqml_pitcher_boosted_tree' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.mlb_shared.bqml_pitcher_xfip`,
  (SELECT * FROM `data-platform-490901.mlb_shared.v_pitcher_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 4. BQML 線形回帰 投手評価
-- ============================================================
SELECT
  'bqml_pitcher_linear' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.mlb_shared.bqml_pitcher_xfip_linear`,
  (SELECT * FROM `data-platform-490901.mlb_shared.v_pitcher_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 5. BQML vs Python: 打者 MAE 比較
-- ============================================================
-- BQML 予測を取得して、Python ensemble 予測と actual を突合
WITH bqml_preds AS (
  SELECT
    player, season, predicted_target_woba AS bqml_woba
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_batter_woba`,
    (SELECT * FROM `data-platform-490901.mlb_shared.v_batter_train` WHERE is_eval = TRUE)
  )
),
actual AS (
  SELECT player, season, wOBA AS actual_woba
  FROM `data-platform-490901.mlb_shared.raw_batter_features`
  WHERE season = (SELECT MAX(season) FROM `data-platform-490901.mlb_shared.raw_batter_features`)
),
python_preds AS (
  SELECT player, ensemble_woba, marcel_woba
  FROM `data-platform-490901.mlb_shared.batter_predictions`
)
SELECT
  'batter_comparison' AS comparison,
  AVG(ABS(a.actual_woba - b.bqml_woba)) AS bqml_mae,
  AVG(ABS(a.actual_woba - p.ensemble_woba)) AS python_ensemble_mae,
  AVG(ABS(a.actual_woba - p.marcel_woba)) AS marcel_mae,
  COUNT(*) AS n_players
FROM actual a
LEFT JOIN bqml_preds b ON a.player = b.player
LEFT JOIN python_preds p ON a.player = p.player
WHERE b.bqml_woba IS NOT NULL AND p.ensemble_woba IS NOT NULL;

-- ============================================================
-- 6. BQML vs Python: 投手 MAE 比較
-- ============================================================
WITH bqml_preds AS (
  SELECT
    player, season, predicted_target_xfip AS bqml_xfip
  FROM ML.PREDICT(
    MODEL `data-platform-490901.mlb_shared.bqml_pitcher_xfip`,
    (SELECT * FROM `data-platform-490901.mlb_shared.v_pitcher_train` WHERE is_eval = TRUE)
  )
),
actual AS (
  SELECT player, season, xFIP AS actual_xfip
  FROM `data-platform-490901.mlb_shared.raw_pitcher_features`
  WHERE season = (SELECT MAX(season) FROM `data-platform-490901.mlb_shared.raw_pitcher_features`)
),
python_preds AS (
  SELECT player, ensemble_xfip, marcel_xfip
  FROM `data-platform-490901.mlb_shared.pitcher_predictions`
)
SELECT
  'pitcher_comparison' AS comparison,
  AVG(ABS(a.actual_xfip - b.bqml_xfip)) AS bqml_mae,
  AVG(ABS(a.actual_xfip - p.ensemble_xfip)) AS python_ensemble_mae,
  AVG(ABS(a.actual_xfip - p.marcel_xfip)) AS marcel_mae,
  COUNT(*) AS n_players
FROM actual a
LEFT JOIN bqml_preds b ON a.player = b.player
LEFT JOIN python_preds p ON a.player = p.player
WHERE b.bqml_xfip IS NOT NULL AND p.ensemble_xfip IS NOT NULL;
