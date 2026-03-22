-- BigQuery ML: 投手 xFIP 予測モデル
-- Boosted Tree Regressor + 線形回帰 (ベースライン比較用)
--
-- データソース: raw_pitcher_features テーブル
-- Python train.py の PITCHER_FEATURES と同じ特徴量をSQLで再現

-- ============================================================
-- Step 1: 学習用ビュー（ラグ特徴量 + ターゲット）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_pitcher_train` AS
WITH base AS (
  SELECT
    player,
    season,
    -- FanGraphs
    xFIP, FIP, ERA, `K_pct`, `BB_pct`, `HR_per_9`, WHIP, BABIP, `LOB_pct`,
    `SwStr_pct`, `K_BB_pct`, `CSW_pct`, G,
    -- Statcast expected
    est_ba, est_slg, est_woba, xera,
    -- Statcast exit velo (被打球)
    avg_hit_speed, avg_hit_angle, brl_percent, ev95percent,
    -- Arsenal (集約済み)
    n_pitch_types, primary_usage, best_whiff,
    avg_whiff_weighted, best_rv100, usage_entropy,
    -- Basic
    Age, IP, Team
  FROM `data-platform-490901.mlb_statcast.raw_pitcher_features`
  WHERE IP >= 30
),
lagged AS (
  SELECT
    player,
    season,
    -- ===== y1 (直前シーズン) =====
    LAG(xFIP, 1) OVER w AS xFIP_y1,
    LAG(FIP, 1) OVER w AS FIP_y1,
    LAG(ERA, 1) OVER w AS ERA_y1,
    LAG(`K_pct`, 1) OVER w AS K_pct_y1,
    LAG(`BB_pct`, 1) OVER w AS BB_pct_y1,
    LAG(`HR_per_9`, 1) OVER w AS HR_per_9_y1,
    LAG(WHIP, 1) OVER w AS WHIP_y1,
    LAG(BABIP, 1) OVER w AS BABIP_y1,
    LAG(`LOB_pct`, 1) OVER w AS LOB_pct_y1,
    LAG(`SwStr_pct`, 1) OVER w AS SwStr_pct_y1,
    LAG(`K_BB_pct`, 1) OVER w AS K_BB_pct_y1,
    LAG(`CSW_pct`, 1) OVER w AS CSW_pct_y1,
    LAG(G, 1) OVER w AS G_y1,
    LAG(est_ba, 1) OVER w AS est_ba_y1,
    LAG(est_slg, 1) OVER w AS est_slg_y1,
    LAG(est_woba, 1) OVER w AS est_woba_y1,
    LAG(xera, 1) OVER w AS xera_y1,
    LAG(avg_hit_speed, 1) OVER w AS avg_hit_speed_y1,
    LAG(avg_hit_angle, 1) OVER w AS avg_hit_angle_y1,
    LAG(brl_percent, 1) OVER w AS brl_percent_y1,
    LAG(ev95percent, 1) OVER w AS ev95percent_y1,
    LAG(n_pitch_types, 1) OVER w AS n_pitch_types_y1,
    LAG(primary_usage, 1) OVER w AS primary_usage_y1,
    LAG(best_whiff, 1) OVER w AS best_whiff_y1,
    LAG(avg_whiff_weighted, 1) OVER w AS avg_whiff_weighted_y1,
    LAG(best_rv100, 1) OVER w AS best_rv100_y1,
    LAG(usage_entropy, 1) OVER w AS usage_entropy_y1,
    LAG(Age, 1) OVER w AS Age_y1,
    LAG(IP, 1) OVER w AS IP_y1,

    -- ===== y2 (2年前) =====
    LAG(xFIP, 2) OVER w AS xFIP_y2,
    LAG(`K_pct`, 2) OVER w AS K_pct_y2,
    LAG(`BB_pct`, 2) OVER w AS BB_pct_y2,
    LAG(avg_hit_speed, 2) OVER w AS avg_hit_speed_y2,
    LAG(brl_percent, 2) OVER w AS brl_percent_y2,
    LAG(best_whiff, 2) OVER w AS best_whiff_y2,
    LAG(usage_entropy, 2) OVER w AS usage_entropy_y2,

    -- ===== y3 (3年前) =====
    LAG(xFIP, 3) OVER w AS xFIP_y3,

    -- ===== ターゲット =====
    xFIP AS target_xfip,

    -- ===== ラグ差分 =====
    LAG(xFIP, 1) OVER w - LAG(xFIP, 2) OVER w AS xFIP_delta_1,
    LAG(xFIP, 2) OVER w - LAG(xFIP, 3) OVER w AS xFIP_delta_2,
    LAG(`K_pct`, 1) OVER w - LAG(`K_pct`, 2) OVER w AS K_pct_delta_1,
    LAG(`BB_pct`, 1) OVER w - LAG(`BB_pct`, 2) OVER w AS BB_pct_delta_1,
    LAG(`K_BB_pct`, 1) OVER w - LAG(`K_BB_pct`, 2) OVER w AS K_BB_pct_delta_1,
    LAG(best_whiff, 1) OVER w - LAG(best_whiff, 2) OVER w AS whiff_delta_1,
    LAG(usage_entropy, 1) OVER w - LAG(usage_entropy, 2) OVER w AS usage_entropy_delta_1,

    -- ===== エンジニアリング特徴量 =====
    LAG(Age, 1) OVER w - 27 AS age_from_peak,
    POW(LAG(Age, 1) OVER w - 27, 2) AS age_sq,
    LAG(IP, 1) OVER w / 200.0 AS ip_rate,
    LAG(ERA, 1) OVER w - LAG(FIP, 1) OVER w AS fip_era_gap,

    -- age × K-BB% 交互作用
    (LAG(Age, 1) OVER w) * (LAG(`K_BB_pct`, 1) OVER w) AS age_x_kbb,

    -- チーム変更
    CASE
      WHEN LAG(Team, 1) OVER w != LAG(Team, 2) OVER w
        AND LAG(Team, 1) OVER w IS NOT NULL
        AND LAG(Team, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS team_changed,

    -- 時系列CV用: 最新年をeval
    CASE WHEN season = (SELECT MAX(season) - 1 FROM `data-platform-490901.mlb_statcast.raw_pitcher_features`)
      THEN TRUE ELSE FALSE
    END AS is_eval

  FROM base
  WINDOW w AS (PARTITION BY player ORDER BY season)
)
SELECT * FROM lagged
WHERE xFIP_y1 IS NOT NULL  -- 最低1年の過去データが必要
;


-- ============================================================
-- Step 2: Boosted Tree Regressor
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.mlb_statcast.bqml_pitcher_xfip`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['target_xfip'],
  max_iterations = 200,
  learn_rate = 0.05,
  max_tree_depth = 6,
  subsample = 0.8,
  min_tree_child_weight = 5,
  colsample_bytree = 0.8,
  l1_reg = 0.1,
  l2_reg = 1.0,
  early_stop = TRUE,
  min_rel_progress = 0.001,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  -- y1 features
  xFIP_y1, FIP_y1, ERA_y1, K_pct_y1, BB_pct_y1, HR_per_9_y1,
  WHIP_y1, BABIP_y1, LOB_pct_y1,
  SwStr_pct_y1, K_BB_pct_y1, CSW_pct_y1, G_y1,
  est_ba_y1, est_slg_y1, est_woba_y1, xera_y1,
  avg_hit_speed_y1, avg_hit_angle_y1, brl_percent_y1, ev95percent_y1,
  n_pitch_types_y1, primary_usage_y1, best_whiff_y1,
  avg_whiff_weighted_y1, best_rv100_y1, usage_entropy_y1,
  Age_y1, IP_y1,
  -- y2 features
  xFIP_y2, K_pct_y2, BB_pct_y2,
  avg_hit_speed_y2, brl_percent_y2, best_whiff_y2, usage_entropy_y2,
  -- y3 features
  xFIP_y3,
  -- delta features
  xFIP_delta_1, xFIP_delta_2, K_pct_delta_1, BB_pct_delta_1,
  K_BB_pct_delta_1, whiff_delta_1, usage_entropy_delta_1,
  -- engineered
  age_from_peak, age_sq, ip_rate, fip_era_gap, age_x_kbb, team_changed,
  -- target
  target_xfip,
  -- split
  is_eval
FROM `data-platform-490901.mlb_statcast.v_pitcher_train`
;


-- ============================================================
-- Step 3: 線形回帰 (ベースライン比較用)
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.mlb_statcast.bqml_pitcher_xfip_linear`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['target_xfip'],
  optimize_strategy = 'NORMAL_EQUATION',
  l2_reg = 1.0,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  xFIP_y1, FIP_y1, K_pct_y1, BB_pct_y1, HR_per_9_y1,
  avg_hit_speed_y1, brl_percent_y1,
  n_pitch_types_y1, best_whiff_y1, usage_entropy_y1,
  Age_y1, IP_y1,
  xFIP_delta_1, age_from_peak, age_sq, ip_rate, fip_era_gap, team_changed,
  target_xfip,
  is_eval
FROM `data-platform-490901.mlb_statcast.v_pitcher_train`
;
