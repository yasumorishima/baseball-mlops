-- BigQuery 分析用ビュー
-- Statcast 生データを活用した分析クエリ集

-- ============================================================
-- 1. 選手年度別 wOBA トレンド（打者パフォーマンス推移）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_batter_trend` AS
SELECT
  player,
  season,
  wOBA,
  xwOBA,
  `K_pct`,
  `BB_pct`,
  avg_hit_speed,
  brl_percent,
  sprint_speed,
  avg_bat_speed,
  swing_tilt,
  PA,
  Age,
  Team,
  -- 前年比
  wOBA - LAG(wOBA) OVER (PARTITION BY player ORDER BY season) AS wOBA_yoy,
  xwOBA - LAG(xwOBA) OVER (PARTITION BY player ORDER BY season) AS xwOBA_yoy,
  avg_hit_speed - LAG(avg_hit_speed) OVER (PARTITION BY player ORDER BY season) AS ev_yoy,
  -- xwOBA - wOBA 乖離（運要素の指標）
  xwOBA - wOBA AS luck_factor
FROM `data-platform-490901.mlb_statcast.raw_batter_features`
ORDER BY player, season;


-- ============================================================
-- 2. 選手年度別 xFIP トレンド（投手パフォーマンス推移）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_pitcher_trend` AS
SELECT
  player,
  season,
  xFIP,
  FIP,
  ERA,
  `K_pct`,
  `BB_pct`,
  avg_hit_speed,
  brl_percent,
  n_pitch_types,
  best_whiff,
  usage_entropy,
  IP,
  Age,
  Team,
  -- 前年比
  xFIP - LAG(xFIP) OVER (PARTITION BY player ORDER BY season) AS xFIP_yoy,
  `K_pct` - LAG(`K_pct`) OVER (PARTITION BY player ORDER BY season) AS K_pct_yoy,
  -- FIP-ERA 乖離（運要素）
  ERA - FIP AS era_fip_gap
FROM `data-platform-490901.mlb_statcast.raw_pitcher_features`
ORDER BY player, season;


-- ============================================================
-- 3. Statcast 打球品質リーダーボード（シーズン別）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_batted_ball_leaders` AS
SELECT
  player,
  season,
  avg_hit_speed,
  brl_percent,
  ev95percent,
  anglesweetspotpercent,
  pull_rate,
  oppo_rate,
  avg_bat_speed,
  swing_tilt,
  attack_angle,
  ideal_attack_angle_rate,
  wOBA,
  xwOBA,
  PA
FROM `data-platform-490901.mlb_statcast.raw_batter_features`
WHERE PA >= 100
ORDER BY avg_hit_speed DESC;


-- ============================================================
-- 4. 投手球種戦略分析（Arsenal 集約）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_pitcher_arsenal` AS
SELECT
  player,
  season,
  n_pitch_types,
  primary_usage,
  best_whiff,
  avg_whiff_weighted,
  best_rv100,
  usage_entropy,
  -- 球種数とエントロピーの関係
  CASE
    WHEN n_pitch_types >= 5 AND usage_entropy > 1.2 THEN 'diverse'
    WHEN n_pitch_types <= 3 AND primary_usage > 0.6 THEN 'specialist'
    ELSE 'balanced'
  END AS arsenal_type
FROM `data-platform-490901.mlb_statcast.raw_pitcher_features`
WHERE n_pitch_types IS NOT NULL
ORDER BY usage_entropy DESC;


-- ============================================================
-- 5. 球場補正効果分析
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_park_effects` AS
SELECT
  team,
  season,
  pf_5yr,
  CASE
    WHEN pf_5yr > 105 THEN 'hitter_park'
    WHEN pf_5yr < 95 THEN 'pitcher_park'
    ELSE 'neutral'
  END AS park_type
FROM `data-platform-490901.mlb_statcast.raw_park_factors`
ORDER BY season DESC, pf_5yr DESC;


-- ============================================================
-- 6. BQML vs Python 全モデル精度比較ダッシュボード
-- ============================================================
-- model_metrics_history は append 運用のため、初回 Run で列が未定義の場合がある。
-- SELECT * で全列を取得し、カラム追加に自動対応する。
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_model_comparison` AS
SELECT *
FROM `data-platform-490901.mlb_statcast.model_metrics_history`
ORDER BY run_date DESC;


-- ============================================================
-- 7. シーズン別データ量サマリー
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.mlb_statcast.v_data_coverage` AS
SELECT
  season,
  COUNT(DISTINCT player) AS n_batters,
  AVG(PA) AS avg_pa,
  COUNT(DISTINCT CASE WHEN avg_bat_speed IS NOT NULL THEN player END) AS n_bat_tracking,
  COUNT(DISTINCT CASE WHEN pull_rate IS NOT NULL THEN player END) AS n_batted_ball
FROM `data-platform-490901.mlb_statcast.raw_batter_features`
GROUP BY season
ORDER BY season;
