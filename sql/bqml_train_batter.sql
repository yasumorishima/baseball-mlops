-- BigQuery ML: 打者 wOBA 予測モデル
-- Boosted Tree Regressor (XGBoost ベース) + 線形回帰 (ベースライン比較用)
--
-- データソース: raw_batter_features テーブル (fetch_statcast.py → load_to_bq.py)
-- Python train.py と同じ特徴量をSQLウインドウ関数で再現
-- 粒度: 選手×シーズン → 翌シーズン wOBA を予測

-- ============================================================
-- Step 1: 学習用ビュー（ラグ特徴量 + ターゲット）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_shared.v_batter_train` AS
WITH base AS (
  SELECT
    player,
    season,
    -- FanGraphs
    wOBA, xwOBA, `K_pct`, `BB_pct`, ISO, BABIP, OBP, SLG,
    `SwStr_pct`, `HardHit_pct`, `Contact_pct`, `O_Swing_pct`, G,
    -- Statcast expected
    est_ba, est_slg, est_woba,
    -- Statcast exit velo
    avg_hit_speed, avg_hit_angle, brl_percent, ev95percent, anglesweetspotpercent,
    -- Sprint speed
    sprint_speed,
    -- Bat Tracking (2024+, NULL for earlier)
    avg_bat_speed, swing_tilt, attack_angle, ideal_attack_angle_rate,
    -- Batted ball direction
    pull_rate, oppo_rate,
    -- Basic
    Age, PA, Team
  FROM `data-platform-490901.mlb_shared.raw_batter_features`
  WHERE PA >= 50
),
lagged AS (
  SELECT
    player,
    season,
    -- ===== y1 (直前シーズン) =====
    LAG(wOBA, 1) OVER w AS wOBA_y1,
    LAG(xwOBA, 1) OVER w AS xwOBA_y1,
    LAG(`K_pct`, 1) OVER w AS K_pct_y1,
    LAG(`BB_pct`, 1) OVER w AS BB_pct_y1,
    LAG(ISO, 1) OVER w AS ISO_y1,
    LAG(BABIP, 1) OVER w AS BABIP_y1,
    LAG(OBP, 1) OVER w AS OBP_y1,
    LAG(SLG, 1) OVER w AS SLG_y1,
    LAG(`SwStr_pct`, 1) OVER w AS SwStr_pct_y1,
    LAG(`HardHit_pct`, 1) OVER w AS HardHit_pct_y1,
    LAG(`Contact_pct`, 1) OVER w AS Contact_pct_y1,
    LAG(`O_Swing_pct`, 1) OVER w AS O_Swing_pct_y1,
    LAG(G, 1) OVER w AS G_y1,
    LAG(est_ba, 1) OVER w AS est_ba_y1,
    LAG(est_slg, 1) OVER w AS est_slg_y1,
    LAG(est_woba, 1) OVER w AS est_woba_y1,
    LAG(avg_hit_speed, 1) OVER w AS avg_hit_speed_y1,
    LAG(avg_hit_angle, 1) OVER w AS avg_hit_angle_y1,
    LAG(brl_percent, 1) OVER w AS brl_percent_y1,
    LAG(ev95percent, 1) OVER w AS ev95percent_y1,
    LAG(anglesweetspotpercent, 1) OVER w AS anglesweetspotpercent_y1,
    LAG(sprint_speed, 1) OVER w AS sprint_speed_y1,
    LAG(avg_bat_speed, 1) OVER w AS avg_bat_speed_y1,
    LAG(swing_tilt, 1) OVER w AS swing_tilt_y1,
    LAG(attack_angle, 1) OVER w AS attack_angle_y1,
    LAG(ideal_attack_angle_rate, 1) OVER w AS ideal_attack_angle_rate_y1,
    LAG(pull_rate, 1) OVER w AS pull_rate_y1,
    LAG(oppo_rate, 1) OVER w AS oppo_rate_y1,
    LAG(Age, 1) OVER w AS Age_y1,
    LAG(PA, 1) OVER w AS PA_y1,

    -- ===== y2 (2年前) =====
    LAG(wOBA, 2) OVER w AS wOBA_y2,
    LAG(xwOBA, 2) OVER w AS xwOBA_y2,
    LAG(`K_pct`, 2) OVER w AS K_pct_y2,
    LAG(`BB_pct`, 2) OVER w AS BB_pct_y2,
    LAG(avg_hit_speed, 2) OVER w AS avg_hit_speed_y2,
    LAG(brl_percent, 2) OVER w AS brl_percent_y2,
    LAG(avg_bat_speed, 2) OVER w AS avg_bat_speed_y2,
    LAG(swing_tilt, 2) OVER w AS swing_tilt_y2,

    -- ===== y3 (3年前) =====
    LAG(wOBA, 3) OVER w AS wOBA_y3,

    -- ===== ターゲット =====
    wOBA AS target_woba,

    -- ===== エンジニアリング特徴量 =====
    -- ラグ差分
    LAG(wOBA, 1) OVER w - LAG(wOBA, 2) OVER w AS wOBA_delta_1,
    LAG(xwOBA, 1) OVER w - LAG(xwOBA, 2) OVER w AS xwOBA_delta_1,
    LAG(`K_pct`, 1) OVER w - LAG(`K_pct`, 2) OVER w AS K_pct_delta_1,
    LAG(`BB_pct`, 1) OVER w - LAG(`BB_pct`, 2) OVER w AS BB_pct_delta_1,
    LAG(brl_percent, 1) OVER w - LAG(brl_percent, 2) OVER w AS brl_delta_1,
    LAG(wOBA, 2) OVER w - LAG(wOBA, 3) OVER w AS wOBA_delta_2,
    LAG(avg_bat_speed, 1) OVER w - LAG(avg_bat_speed, 2) OVER w AS bat_speed_delta_1,

    -- 年齢系
    LAG(Age, 1) OVER w - 27 AS age_from_peak,
    POW(LAG(Age, 1) OVER w - 27, 2) AS age_sq,

    -- 出場率
    LAG(PA, 1) OVER w / 650.0 AS pa_rate,

    -- luck (xwOBA - wOBA 乖離)
    LAG(xwOBA, 1) OVER w - LAG(wOBA, 1) OVER w AS xwoba_luck,

    -- age × luck 交互作用
    (LAG(Age, 1) OVER w) * (LAG(xwOBA, 1) OVER w - LAG(wOBA, 1) OVER w) AS age_x_luck,

    -- チーム変更
    CASE
      WHEN LAG(Team, 1) OVER w != LAG(Team, 2) OVER w
        AND LAG(Team, 1) OVER w IS NOT NULL
        AND LAG(Team, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS team_changed,

    -- 時系列CV用: 最新年をeval
    CASE WHEN season = (SELECT MAX(season) - 1 FROM `data-platform-490901.mlb_shared.raw_batter_features`)
      THEN TRUE ELSE FALSE
    END AS is_eval

  FROM base
  WINDOW w AS (PARTITION BY player ORDER BY season)
)
SELECT * FROM lagged
WHERE wOBA_y1 IS NOT NULL  -- 最低1年の過去データが必要
;


-- ============================================================
-- Step 2: Boosted Tree Regressor
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.mlb_shared.bqml_batter_woba`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['target_woba'],
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
  wOBA_y1, xwOBA_y1, K_pct_y1, BB_pct_y1, ISO_y1, BABIP_y1, OBP_y1, SLG_y1,
  SwStr_pct_y1, HardHit_pct_y1, Contact_pct_y1, O_Swing_pct_y1, G_y1,
  est_ba_y1, est_slg_y1, est_woba_y1,
  avg_hit_speed_y1, avg_hit_angle_y1, brl_percent_y1, ev95percent_y1, anglesweetspotpercent_y1,
  sprint_speed_y1,
  avg_bat_speed_y1, swing_tilt_y1, attack_angle_y1, ideal_attack_angle_rate_y1,
  pull_rate_y1, oppo_rate_y1,
  Age_y1, PA_y1,
  -- y2 features
  wOBA_y2, xwOBA_y2, K_pct_y2, BB_pct_y2,
  avg_hit_speed_y2, brl_percent_y2, avg_bat_speed_y2, swing_tilt_y2,
  -- y3 features
  wOBA_y3,
  -- delta features
  wOBA_delta_1, xwOBA_delta_1, K_pct_delta_1, BB_pct_delta_1, brl_delta_1,
  wOBA_delta_2, bat_speed_delta_1,
  -- engineered
  age_from_peak, age_sq, pa_rate, xwoba_luck, age_x_luck, team_changed,
  -- target
  target_woba,
  -- split
  is_eval
FROM `data-platform-490901.mlb_shared.v_batter_train`
;


-- ============================================================
-- Step 3: 線形回帰 (ベースライン比較用)
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.mlb_shared.bqml_batter_woba_linear`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['target_woba'],
  optimize_strategy = 'NORMAL_EQUATION',
  l2_reg = 1.0,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  wOBA_y1, xwOBA_y1, K_pct_y1, BB_pct_y1, ISO_y1, BABIP_y1,
  avg_hit_speed_y1, brl_percent_y1, sprint_speed_y1,
  avg_bat_speed_y1, swing_tilt_y1,
  Age_y1, PA_y1,
  wOBA_delta_1, age_from_peak, age_sq, pa_rate, xwoba_luck, team_changed,
  target_woba,
  is_eval
FROM `data-platform-490901.mlb_shared.v_batter_train`
;
