// Hierarchical Bayesian model for MLB batter wOBA prediction
//
// Architecture:
//   y[n] = actual wOBA
//   mu[n] = Marcel[n] + alpha[player[n]]
//           + X_contact[n] @ beta_contact
//           + X_discipline[n] @ beta_discipline
//           + X_expected[n] @ beta_expected
//           + X_context[n] @ beta_context
//           + beta_age * z_age[n] + beta_age2 * z_age[n]^2
//           + beta_lgb * lgb_delta[n] + beta_cat * cat_delta[n]
//   sigma[n] = sigma_base * exp(gamma_pa * z_log_pa[n])
//
// Key features:
//   1. Player random intercepts (partial pooling via non-centered param)
//   2. Skill-group hierarchical shrinkage (contact/discipline/expected/context)
//   3. Heteroscedastic noise (PA-dependent uncertainty)
//   4. LGB/CatBoost OOF stacking (meta-learning)
//   5. Non-linear aging curve (quadratic)
//   6. Informative priors from baseball domain knowledge

data {
  int<lower=1> N;                            // training observations
  int<lower=1> P;                            // unique players
  int<lower=1> K_contact;                    // contact quality features
  int<lower=1> K_discipline;                 // plate discipline features
  int<lower=1> K_expected;                   // expected stat features
  int<lower=1> K_context;                    // contextual features

  array[N] int<lower=1, upper=P> player;     // player index per obs
  vector[N] marcel;                          // Marcel baseline (offset)
  vector[N] y;                               // actual wOBA (target)

  // Feature matrices (z-scored)
  matrix[N, K_contact] X_contact;            // brl%, exit_velo, HardHit%, maxEV, bat_speed
  matrix[N, K_discipline] X_discipline;      // K%, BB%, O-Swing%, Contact%, SwStr%
  matrix[N, K_expected] X_expected;          // xwOBA, ev95%, BABIP
  matrix[N, K_context] X_context;            // park_factor, pa_rate, team_changed, g_change_rate

  // Aging
  vector[N] z_age;                           // (age - 27) / sd
  vector[N] z_age_sq;                        // z_age^2

  // Stacking features (LGB/CatBoost OOF delta from Marcel)
  vector[N] lgb_delta;
  vector[N] cat_delta;

  // Heteroscedasticity control
  vector[N] z_log_pa;                        // z-scored log(PA)

  // Prediction data
  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=P> player_pred;
  vector[N_pred] marcel_pred;
  matrix[N_pred, K_contact] X_contact_pred;
  matrix[N_pred, K_discipline] X_discipline_pred;
  matrix[N_pred, K_expected] X_expected_pred;
  matrix[N_pred, K_context] X_context_pred;
  vector[N_pred] z_age_pred;
  vector[N_pred] z_age_sq_pred;
  vector[N_pred] lgb_delta_pred;
  vector[N_pred] cat_delta_pred;
  vector[N_pred] z_log_pa_pred;
}

parameters {
  // --- Player hierarchy ---
  real<lower=0> sigma_alpha;                 // player-level sd
  vector[P] z_alpha;                         // non-centered player effects

  // --- Skill-group coefficients ---
  // Contact quality: brl%, exit_velo, HardHit%, maxEV, bat_speed
  real<lower=0> tau_contact;                 // group-level sd
  vector[K_contact] z_beta_contact;          // non-centered

  // Plate discipline: K%, BB%, O-Swing%, Contact%, SwStr%
  real<lower=0> tau_discipline;
  vector[K_discipline] z_beta_discipline;

  // Expected stats: xwOBA, ev95%, BABIP
  real<lower=0> tau_expected;
  vector[K_expected] z_beta_expected;

  // Context: park_factor, pa_rate, team_changed, g_change_rate
  real<lower=0> tau_context;
  vector[K_context] z_beta_context;

  // --- Aging ---
  real beta_age;                             // linear aging
  real beta_age2;                            // quadratic aging

  // --- Stacking ---
  real beta_lgb;                             // LGB meta-feature weight
  real beta_cat;                             // CatBoost meta-feature weight

  // --- Noise ---
  real<lower=0> sigma_base;                  // base observation noise
  real gamma_pa;                             // PA scaling (negative = more PA, less noise)
}

transformed parameters {
  // Non-centered parameterization for efficient sampling
  vector[P] alpha = sigma_alpha * z_alpha;
  vector[K_contact] beta_contact = tau_contact * z_beta_contact;
  vector[K_discipline] beta_discipline = tau_discipline * z_beta_discipline;
  vector[K_expected] beta_expected = tau_expected * z_beta_expected;
  vector[K_context] beta_context = tau_context * z_beta_context;
}

model {
  // === Priors ===

  // Player random effects
  sigma_alpha ~ exponential(30);             // expect ~0.03 (small wOBA deviations)
  z_alpha ~ std_normal();

  // Skill-group hierarchical scales
  tau_contact ~ normal(0, 0.04);           // contact features: moderate effect
  tau_discipline ~ normal(0, 0.05);        // discipline: strongest single-feature effects
  tau_expected ~ normal(0, 0.05);          // expected stats: strong predictors
  tau_context ~ normal(0, 0.02);           // context: smaller effects

  // Non-centered feature coefficients
  z_beta_contact ~ std_normal();
  z_beta_discipline ~ std_normal();
  z_beta_expected ~ std_normal();
  z_beta_context ~ std_normal();

  // Aging: peak at 27, ~3 wOBA points decline/year, accelerating after 30
  beta_age ~ normal(-0.003, 0.01);
  beta_age2 ~ normal(-0.001, 0.005);        // negative = accelerating decline

  // Stacking: tree models capture non-linear patterns; expected positive weights
  beta_lgb ~ normal(0.3, 0.2);
  beta_cat ~ normal(0.2, 0.2);

  // Noise
  sigma_base ~ exponential(20);              // expect ~0.05 (typical wOBA noise)
  gamma_pa ~ normal(-0.1, 0.1);             // more PA → less noise

  // === Likelihood ===
  {
    vector[N] mu = marcel
                   + alpha[player]
                   + X_contact * beta_contact
                   + X_discipline * beta_discipline
                   + X_expected * beta_expected
                   + X_context * beta_context
                   + beta_age * z_age
                   + beta_age2 * z_age_sq
                   + beta_lgb * lgb_delta
                   + beta_cat * cat_delta;

    // Heteroscedastic noise: low-PA players get wider uncertainty
    vector[N] sigma;
    for (n in 1:N)
      sigma[n] = sigma_base * exp(fmin(gamma_pa * z_log_pa[n], 2.0));

    y ~ normal(mu, sigma);
  }
}

generated quantities {
  // Posterior predictive draws for new-season predictions
  vector[N_pred] y_pred;
  // LOO-CV log-likelihood for model comparison
  vector[N] log_lik;

  // Predictions
  for (i in 1:N_pred) {
    real mu_i = marcel_pred[i]
                + alpha[player_pred[i]]
                + X_contact_pred[i] * beta_contact
                + X_discipline_pred[i] * beta_discipline
                + X_expected_pred[i] * beta_expected
                + X_context_pred[i] * beta_context
                + beta_age * z_age_pred[i]
                + beta_age2 * z_age_sq_pred[i]
                + beta_lgb * lgb_delta_pred[i]
                + beta_cat * cat_delta_pred[i];
    real sigma_i = sigma_base * exp(fmin(gamma_pa * z_log_pa_pred[i], 2.0));
    y_pred[i] = normal_rng(mu_i, sigma_i);
  }

  // Log-likelihood for LOO-CV (via loo package)
  for (n in 1:N) {
    real mu_n = marcel[n]
                + alpha[player[n]]
                + X_contact[n] * beta_contact
                + X_discipline[n] * beta_discipline
                + X_expected[n] * beta_expected
                + X_context[n] * beta_context
                + beta_age * z_age[n]
                + beta_age2 * z_age_sq[n]
                + beta_lgb * lgb_delta[n]
                + beta_cat * cat_delta[n];
    real sigma_n = sigma_base * exp(fmin(gamma_pa * z_log_pa[n], 2.0));
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
