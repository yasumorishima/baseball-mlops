// Hierarchical Bayesian model for MLB pitcher xFIP prediction
//
// Architecture mirrors hitter model but with pitcher-specific skill groups:
//   - Stuff: K%, Stuff+, SwStr%, best_whiff, avg_whiff_weighted
//   - Command: BB%, Location+, CSW%, K-BB%
//   - Contact Management: brl%, exit_velo, HardHit%, est_woba
//   - Arsenal: n_pitch_types, usage_entropy
//   - Context: park_factor, ip_rate, team_changed, g_change_rate
//
// xFIP is LOWER = BETTER, so signs are flipped vs hitter wOBA:
//   - High K% → lower xFIP (negative coefficient expected)
//   - High BB% → higher xFIP (positive coefficient expected)
//   - High Stuff+ → lower xFIP (negative coefficient expected)

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K_stuff;                     // stuff features
  int<lower=1> K_command;                    // command features
  int<lower=1> K_contact_mgmt;              // contact management features
  int<lower=1> K_arsenal;                    // arsenal diversity features
  int<lower=1> K_context;                    // contextual features

  array[N] int<lower=1, upper=P> player;
  vector[N] marcel;                          // Marcel xFIP baseline
  vector[N] y;                               // actual xFIP (target)

  matrix[N, K_stuff] X_stuff;
  matrix[N, K_command] X_command;
  matrix[N, K_contact_mgmt] X_contact_mgmt;
  matrix[N, K_arsenal] X_arsenal;
  matrix[N, K_context] X_context;

  vector[N] z_age;
  vector[N] z_age_sq;
  vector[N] lgb_delta;
  vector[N] cat_delta;
  vector[N] z_log_ip;                        // z-scored log(IP) for heteroscedasticity

  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=P> player_pred;
  vector[N_pred] marcel_pred;
  matrix[N_pred, K_stuff] X_stuff_pred;
  matrix[N_pred, K_command] X_command_pred;
  matrix[N_pred, K_contact_mgmt] X_contact_mgmt_pred;
  matrix[N_pred, K_arsenal] X_arsenal_pred;
  matrix[N_pred, K_context] X_context_pred;
  vector[N_pred] z_age_pred;
  vector[N_pred] z_age_sq_pred;
  vector[N_pred] lgb_delta_pred;
  vector[N_pred] cat_delta_pred;
  vector[N_pred] z_log_ip_pred;
}

parameters {
  // Player hierarchy
  real<lower=0> sigma_alpha;
  vector[P] z_alpha;

  // Stuff: K%, Stuff+, SwStr%, best_whiff, avg_whiff_weighted
  real<lower=0> tau_stuff;
  vector[K_stuff] z_beta_stuff;

  // Command: BB%, Location+, CSW%, K-BB%
  real<lower=0> tau_command;
  vector[K_command] z_beta_command;

  // Contact management: brl%, exit_velo, HardHit%, est_woba
  real<lower=0> tau_contact_mgmt;
  vector[K_contact_mgmt] z_beta_contact_mgmt;

  // Arsenal diversity: n_pitch_types, usage_entropy
  real<lower=0> tau_arsenal;
  vector[K_arsenal] z_beta_arsenal;

  // Context: park_factor, ip_rate, team_changed, g_change_rate
  real<lower=0> tau_context;
  vector[K_context] z_beta_context;

  // Aging (pitchers decline differently: velocity loss, injury risk)
  real beta_age;
  real beta_age2;

  // Stacking
  real beta_lgb;
  real beta_cat;

  // Noise
  real<lower=0> sigma_base;
  real gamma_ip;                             // IP scaling (more IP → less noise)
}

transformed parameters {
  vector[P] alpha = sigma_alpha * z_alpha;
  vector[K_stuff] beta_stuff = tau_stuff * z_beta_stuff;
  vector[K_command] beta_command = tau_command * z_beta_command;
  vector[K_contact_mgmt] beta_contact_mgmt = tau_contact_mgmt * z_beta_contact_mgmt;
  vector[K_arsenal] beta_arsenal = tau_arsenal * z_beta_arsenal;
  vector[K_context] beta_context = tau_context * z_beta_context;
}

model {
  // === Priors ===
  sigma_alpha ~ exponential(5);              // expect ~0.2 (xFIP has larger scale)
  z_alpha ~ std_normal();

  // Stuff has strongest effect on xFIP (K-rate drives the formula)
  tau_stuff ~ normal(0, 0.15);
  tau_command ~ normal(0, 0.12);
  tau_contact_mgmt ~ normal(0, 0.08);
  tau_arsenal ~ normal(0, 0.05);
  tau_context ~ normal(0, 0.05);

  z_beta_stuff ~ std_normal();
  z_beta_command ~ std_normal();
  z_beta_contact_mgmt ~ std_normal();
  z_beta_arsenal ~ std_normal();
  z_beta_context ~ std_normal();

  // Pitcher aging: velocity declines ~0.5 mph/year after 30 → xFIP rises
  beta_age ~ normal(0.006, 0.02);           // positive = xFIP worsens with age
  beta_age2 ~ normal(0.002, 0.01);          // accelerating decline

  // Stacking
  beta_lgb ~ normal(0.3, 0.2);
  beta_cat ~ normal(0.2, 0.2);

  // Noise (xFIP has larger variance than wOBA: ~0.5 vs ~0.05)
  sigma_base ~ exponential(2);               // expect ~0.5
  gamma_ip ~ normal(-0.1, 0.1);

  // === Likelihood ===
  {
    vector[N] mu = marcel
                   + alpha[player]
                   + X_stuff * beta_stuff
                   + X_command * beta_command
                   + X_contact_mgmt * beta_contact_mgmt
                   + X_arsenal * beta_arsenal
                   + X_context * beta_context
                   + beta_age * z_age
                   + beta_age2 * z_age_sq
                   + beta_lgb * lgb_delta
                   + beta_cat * cat_delta;

    vector[N] sigma;
    for (n in 1:N)
      sigma[n] = sigma_base * exp(fmin(gamma_ip * z_log_ip[n], 2.0));

    y ~ normal(mu, sigma);
  }
}

generated quantities {
  vector[N_pred] y_pred;
  vector[N] log_lik;

  for (i in 1:N_pred) {
    real mu_i = marcel_pred[i]
                + alpha[player_pred[i]]
                + X_stuff_pred[i] * beta_stuff
                + X_command_pred[i] * beta_command
                + X_contact_mgmt_pred[i] * beta_contact_mgmt
                + X_arsenal_pred[i] * beta_arsenal
                + X_context_pred[i] * beta_context
                + beta_age * z_age_pred[i]
                + beta_age2 * z_age_sq_pred[i]
                + beta_lgb * lgb_delta_pred[i]
                + beta_cat * cat_delta_pred[i];
    real sigma_i = sigma_base * exp(fmin(gamma_ip * z_log_ip_pred[i], 2.0));
    y_pred[i] = normal_rng(mu_i, sigma_i);
  }

  for (n in 1:N) {
    real mu_n = marcel[n]
                + alpha[player[n]]
                + X_stuff[n] * beta_stuff
                + X_command[n] * beta_command
                + X_contact_mgmt[n] * beta_contact_mgmt
                + X_arsenal[n] * beta_arsenal
                + X_context[n] * beta_context
                + beta_age * z_age[n]
                + beta_age2 * z_age_sq[n]
                + beta_lgb * lgb_delta[n]
                + beta_cat * cat_delta[n];
    real sigma_n = sigma_base * exp(fmin(gamma_ip * z_log_ip[n], 2.0));
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
