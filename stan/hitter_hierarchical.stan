// Hierarchical Bayesian model for MLB batter wOBA prediction — v11
//
// 10 skill groups:
//   FG/Savant: contact, discipline, expected, context, offense, batted_ball_fg
//   BQ:       approach_bq, batted_ball_bq, power_bq, run_value_bq

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K_contact;
  int<lower=1> K_discipline;
  int<lower=1> K_expected;
  int<lower=1> K_context;
  int<lower=1> K_offense;
  int<lower=1> K_batted_ball_fg;
  int<lower=1> K_approach_bq;
  int<lower=1> K_batted_ball_bq;
  int<lower=1> K_power_bq;
  int<lower=1> K_run_value_bq;

  array[N] int<lower=1, upper=P> player;
  vector[N] marcel;
  vector[N] y;

  matrix[N, K_contact] X_contact;
  matrix[N, K_discipline] X_discipline;
  matrix[N, K_expected] X_expected;
  matrix[N, K_context] X_context;
  matrix[N, K_offense] X_offense;
  matrix[N, K_batted_ball_fg] X_batted_ball_fg;
  matrix[N, K_approach_bq] X_approach_bq;
  matrix[N, K_batted_ball_bq] X_batted_ball_bq;
  matrix[N, K_power_bq] X_power_bq;
  matrix[N, K_run_value_bq] X_run_value_bq;

  vector[N] z_age;
  vector[N] z_age_sq;
  vector[N] lgb_delta;
  vector[N] cat_delta;
  vector[N] z_log_pa;

  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=P> player_pred;
  vector[N_pred] marcel_pred;
  matrix[N_pred, K_contact] X_contact_pred;
  matrix[N_pred, K_discipline] X_discipline_pred;
  matrix[N_pred, K_expected] X_expected_pred;
  matrix[N_pred, K_context] X_context_pred;
  matrix[N_pred, K_offense] X_offense_pred;
  matrix[N_pred, K_batted_ball_fg] X_batted_ball_fg_pred;
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
  real<lower=0> sigma_alpha;
  vector[P] z_alpha;

  real<lower=0> tau_contact;
  vector[K_contact] z_beta_contact;
  real<lower=0> tau_discipline;
  vector[K_discipline] z_beta_discipline;
  real<lower=0> tau_expected;
  vector[K_expected] z_beta_expected;
  real<lower=0> tau_context;
  vector[K_context] z_beta_context;

  real<lower=0> tau_offense;
  vector[K_offense] z_beta_offense;
  real<lower=0> tau_batted_ball_fg;
  vector[K_batted_ball_fg] z_beta_batted_ball_fg;

  real<lower=0> tau_approach_bq;
  vector[K_approach_bq] z_beta_approach_bq;
  real<lower=0> tau_batted_ball_bq;
  vector[K_batted_ball_bq] z_beta_batted_ball_bq;
  real<lower=0> tau_power_bq;
  vector[K_power_bq] z_beta_power_bq;
  real<lower=0> tau_run_value_bq;
  vector[K_run_value_bq] z_beta_run_value_bq;

  real beta_age;
  real beta_age2;
  real beta_lgb;
  real beta_cat;
  real<lower=0> sigma_base;
  real gamma_pa;
}

transformed parameters {
  vector[P] alpha = sigma_alpha * z_alpha;
  vector[K_contact] beta_contact = tau_contact * z_beta_contact;
  vector[K_discipline] beta_discipline = tau_discipline * z_beta_discipline;
  vector[K_expected] beta_expected = tau_expected * z_beta_expected;
  vector[K_context] beta_context = tau_context * z_beta_context;
  vector[K_offense] beta_offense = tau_offense * z_beta_offense;
  vector[K_batted_ball_fg] beta_batted_ball_fg = tau_batted_ball_fg * z_beta_batted_ball_fg;
  vector[K_approach_bq] beta_approach_bq = tau_approach_bq * z_beta_approach_bq;
  vector[K_batted_ball_bq] beta_batted_ball_bq = tau_batted_ball_bq * z_beta_batted_ball_bq;
  vector[K_power_bq] beta_power_bq = tau_power_bq * z_beta_power_bq;
  vector[K_run_value_bq] beta_run_value_bq = tau_run_value_bq * z_beta_run_value_bq;
}

model {
  sigma_alpha ~ exponential(30);
  z_alpha ~ std_normal();

  tau_contact ~ normal(0, 0.04);
  tau_discipline ~ normal(0, 0.05);
  tau_expected ~ normal(0, 0.05);
  tau_context ~ normal(0, 0.02);
  // Offense: wRC+/WAR/OPS are strong composite predictors
  tau_offense ~ normal(0, 0.05);
  // FG batted ball: GB/FB/LD/pull/oppo distribution
  tau_batted_ball_fg ~ normal(0, 0.03);
  tau_approach_bq ~ normal(0, 0.04);
  tau_batted_ball_bq ~ normal(0, 0.03);
  tau_power_bq ~ normal(0, 0.04);
  tau_run_value_bq ~ normal(0, 0.03);

  z_beta_contact ~ std_normal();
  z_beta_discipline ~ std_normal();
  z_beta_expected ~ std_normal();
  z_beta_context ~ std_normal();
  z_beta_offense ~ std_normal();
  z_beta_batted_ball_fg ~ std_normal();
  z_beta_approach_bq ~ std_normal();
  z_beta_batted_ball_bq ~ std_normal();
  z_beta_power_bq ~ std_normal();
  z_beta_run_value_bq ~ std_normal();

  beta_age ~ normal(-0.003, 0.01);
  beta_age2 ~ normal(-0.001, 0.005);
  beta_lgb ~ normal(0.3, 0.2);
  beta_cat ~ normal(0.2, 0.2);
  sigma_base ~ exponential(20);
  gamma_pa ~ normal(-0.1, 0.1);

  {
    vector[N] mu = marcel
                   + alpha[player]
                   + X_contact * beta_contact
                   + X_discipline * beta_discipline
                   + X_expected * beta_expected
                   + X_context * beta_context
                   + X_offense * beta_offense
                   + X_batted_ball_fg * beta_batted_ball_fg
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

  for (i in 1:N_pred) {
    real mu_i = marcel_pred[i]
                + alpha[player_pred[i]]
                + X_contact_pred[i] * beta_contact
                + X_discipline_pred[i] * beta_discipline
                + X_expected_pred[i] * beta_expected
                + X_context_pred[i] * beta_context
                + X_offense_pred[i] * beta_offense
                + X_batted_ball_fg_pred[i] * beta_batted_ball_fg
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

  for (n in 1:N) {
    real mu_n = marcel[n]
                + alpha[player[n]]
                + X_contact[n] * beta_contact
                + X_discipline[n] * beta_discipline
                + X_expected[n] * beta_expected
                + X_context[n] * beta_context
                + X_offense[n] * beta_offense
                + X_batted_ball_fg[n] * beta_batted_ball_fg
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
