// Hierarchical Bayesian model for MLB batter wOBA prediction — v11
//
// Architecture:
//   y[n] = actual wOBA
//   mu[n] = Marcel[n] + alpha[player[n]]
//           + X_contact[n] @ beta_contact
//           + X_discipline[n] @ beta_discipline
//           + X_expected[n] @ beta_expected
//           + X_context[n] @ beta_context
//           + X_approach_bq[n] @ beta_approach_bq        (NEW: BQ pitch-level)
//           + X_batted_ball_bq[n] @ beta_batted_ball_bq  (NEW: BQ pitch-level)
//           + X_power_bq[n] @ beta_power_bq              (NEW: BQ pitch-level)
//           + X_run_value_bq[n] @ beta_run_value_bq      (NEW: BQ pitch-level)
//           + beta_age * z_age[n] + beta_age2 * z_age[n]^2
//           + beta_lgb * lgb_delta[n] + beta_cat * cat_delta[n]
//   sigma[n] = sigma_base * exp(gamma_pa * z_log_pa[n])
//
// 8 skill groups total:
//   Original: contact, discipline, expected, context
//   v11 BQ:  approach_bq, batted_ball_bq, power_bq, run_value_bq

data {
  int<lower=1> N;                            // training observations
  int<lower=1> P;                            // unique players
  int<lower=1> K_contact;                    // contact quality features
  int<lower=1> K_discipline;                 // plate discipline features
  int<lower=1> K_expected;                   // expected stat features
  int<lower=1> K_context;                    // contextual features
  int<lower=1> K_approach_bq;               // BQ plate approach features
  int<lower=1> K_batted_ball_bq;            // BQ batted ball profile features
  int<lower=1> K_power_bq;                  // BQ power/EV quality features
  int<lower=1> K_run_value_bq;              // BQ run value features

  array[N] int<lower=1, upper=P> player;     // player index per obs
  vector[N] marcel;                          // Marcel baseline (offset)
  vector[N] y;                               // actual wOBA (target)

  // Feature matrices (z-scored)
  matrix[N, K_contact] X_contact;
  matrix[N, K_discipline] X_discipline;
  matrix[N, K_expected] X_expected;
  matrix[N, K_context] X_context;
  matrix[N, K_approach_bq] X_approach_bq;
  matrix[N, K_batted_ball_bq] X_batted_ball_bq;
  matrix[N, K_power_bq] X_power_bq;
  matrix[N, K_run_value_bq] X_run_value_bq;

  // Aging
  vector[N] z_age;
  vector[N] z_age_sq;

  // Stacking features (LGB/CatBoost OOF delta from Marcel)
  vector[N] lgb_delta;
  vector[N] cat_delta;

  // Heteroscedasticity control
  vector[N] z_log_pa;

  // Prediction data
  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=P> player_pred;
  vector[N_pred] marcel_pred;
  matrix[N_pred, K_contact] X_contact_pred;
  matrix[N_pred, K_discipline] X_discipline_pred;
  matrix[N_pred, K_expected] X_expected_pred;
  matrix[N_pred, K_context] X_context_pred;
  matrix[N_pred, K_approach_bq] X_approach_bq_pred;
  matrix[N_pred, K_batted_ball_bq] X_batted_ball_bq_pred;
  matrix[N_pred, K_power_bq] X_power_bq_pred;
  matrix[N_pred, K_run_value_bq] X_run_value_bq_pred;
  vector[N_pred] z_age_pred;
  vector[N_pred] z_age_sq_pred;
  vector[N_pred] lgb_delta_pred;
  vector[N_pred] cat_delta_pred;
  vector[N_pred] z_log_pa_pred;
}

parameters {
  // --- Player hierarchy ---
  real<lower=0> sigma_alpha;
  vector[P] z_alpha;

  // --- Original skill-group coefficients ---
  real<lower=0> tau_contact;
  vector[K_contact] z_beta_contact;

  real<lower=0> tau_discipline;
  vector[K_discipline] z_beta_discipline;

  real<lower=0> tau_expected;
  vector[K_expected] z_beta_expected;

  real<lower=0> tau_context;
  vector[K_context] z_beta_context;

  // --- v11 BQ skill-group coefficients ---
  // Approach: whiff, chase, zone_contact, zone_swing, called_strike, first_pitch_swing
  real<lower=0> tau_approach_bq;
  vector[K_approach_bq] z_beta_approach_bq;

  // Batted ball: GB/FB/LD/popup rates, sweet spot, distance
  real<lower=0> tau_batted_ball_bq;
  vector[K_batted_ball_bq] z_beta_batted_ball_bq;

  // Power: avg/max/p90 EV, hard hit, barrel
  real<lower=0> tau_power_bq;
  vector[K_power_bq] z_beta_power_bq;

  // Run value: avg run value, xwOBA, xBA, count leverage
  real<lower=0> tau_run_value_bq;
  vector[K_run_value_bq] z_beta_run_value_bq;

  // --- Aging ---
  real beta_age;
  real beta_age2;

  // --- Stacking ---
  real beta_lgb;
  real beta_cat;

  // --- Noise ---
  real<lower=0> sigma_base;
  real gamma_pa;
}

transformed parameters {
  vector[P] alpha = sigma_alpha * z_alpha;
  vector[K_contact] beta_contact = tau_contact * z_beta_contact;
  vector[K_discipline] beta_discipline = tau_discipline * z_beta_discipline;
  vector[K_expected] beta_expected = tau_expected * z_beta_expected;
  vector[K_context] beta_context = tau_context * z_beta_context;
  vector[K_approach_bq] beta_approach_bq = tau_approach_bq * z_beta_approach_bq;
  vector[K_batted_ball_bq] beta_batted_ball_bq = tau_batted_ball_bq * z_beta_batted_ball_bq;
  vector[K_power_bq] beta_power_bq = tau_power_bq * z_beta_power_bq;
  vector[K_run_value_bq] beta_run_value_bq = tau_run_value_bq * z_beta_run_value_bq;
}

model {
  // === Priors ===

  // Player random effects
  sigma_alpha ~ exponential(30);             // expect ~0.03 (small wOBA deviations)
  z_alpha ~ std_normal();

  // Original skill-group hierarchical scales
  tau_contact ~ normal(0, 0.04);
  tau_discipline ~ normal(0, 0.05);
  tau_expected ~ normal(0, 0.05);
  tau_context ~ normal(0, 0.02);

  // v11 BQ skill-group scales
  // Approach: pitch selection directly impacts wOBA; moderate prior
  tau_approach_bq ~ normal(0, 0.04);
  // Batted ball profile: GB/LD/FB distribution is predictive
  tau_batted_ball_bq ~ normal(0, 0.03);
  // Power quality: EV/barrel have strong wOBA signal
  tau_power_bq ~ normal(0, 0.04);
  // Run values: moderate effect (correlated with expected stats)
  tau_run_value_bq ~ normal(0, 0.03);

  // Non-centered feature coefficients
  z_beta_contact ~ std_normal();
  z_beta_discipline ~ std_normal();
  z_beta_expected ~ std_normal();
  z_beta_context ~ std_normal();
  z_beta_approach_bq ~ std_normal();
  z_beta_batted_ball_bq ~ std_normal();
  z_beta_power_bq ~ std_normal();
  z_beta_run_value_bq ~ std_normal();

  // Aging: peak at 27, ~3 wOBA points decline/year, accelerating after 30
  beta_age ~ normal(-0.003, 0.01);
  beta_age2 ~ normal(-0.001, 0.005);

  // Stacking
  beta_lgb ~ normal(0.3, 0.2);
  beta_cat ~ normal(0.2, 0.2);

  // Noise
  sigma_base ~ exponential(20);
  gamma_pa ~ normal(-0.1, 0.1);

  // === Likelihood ===
  {
    vector[N] mu = marcel
                   + alpha[player]
                   + X_contact * beta_contact
                   + X_discipline * beta_discipline
                   + X_expected * beta_expected
                   + X_context * beta_context
                   + X_approach_bq * beta_approach_bq
                   + X_batted_ball_bq * beta_batted_ball_bq
                   + X_power_bq * beta_power_bq
                   + X_run_value_bq * beta_run_value_bq
                   + beta_age * z_age
                   + beta_age2 * z_age_sq
                   + beta_lgb * lgb_delta
                   + beta_cat * cat_delta;

    vector[N] sigma;
    for (n in 1:N)
      sigma[n] = sigma_base * exp(fmin(gamma_pa * z_log_pa[n], 2.0));

    y ~ normal(mu, sigma);
  }
}

generated quantities {
  vector[N_pred] y_pred;
  vector[N] log_lik;

  // Predictions
  for (i in 1:N_pred) {
    real mu_i = marcel_pred[i]
                + alpha[player_pred[i]]
                + X_contact_pred[i] * beta_contact
                + X_discipline_pred[i] * beta_discipline
                + X_expected_pred[i] * beta_expected
                + X_context_pred[i] * beta_context
                + X_approach_bq_pred[i] * beta_approach_bq
                + X_batted_ball_bq_pred[i] * beta_batted_ball_bq
                + X_power_bq_pred[i] * beta_power_bq
                + X_run_value_bq_pred[i] * beta_run_value_bq
                + beta_age * z_age_pred[i]
                + beta_age2 * z_age_sq_pred[i]
                + beta_lgb * lgb_delta_pred[i]
                + beta_cat * cat_delta_pred[i];
    real sigma_i = sigma_base * exp(fmin(gamma_pa * z_log_pa_pred[i], 2.0));
    y_pred[i] = normal_rng(mu_i, sigma_i);
  }

  // Log-likelihood for LOO-CV
  for (n in 1:N) {
    real mu_n = marcel[n]
                + alpha[player[n]]
                + X_contact[n] * beta_contact
                + X_discipline[n] * beta_discipline
                + X_expected[n] * beta_expected
                + X_context[n] * beta_context
                + X_approach_bq[n] * beta_approach_bq
                + X_batted_ball_bq[n] * beta_batted_ball_bq
                + X_power_bq[n] * beta_power_bq
                + X_run_value_bq[n] * beta_run_value_bq
                + beta_age * z_age[n]
                + beta_age2 * z_age_sq[n]
                + beta_lgb * lgb_delta[n]
                + beta_cat * cat_delta[n];
    real sigma_n = sigma_base * exp(fmin(gamma_pa * z_log_pa[n], 2.0));
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
