data {
  int<lower=0> N;			// Number of observation
  vector[N] y; 			// N observation for J tournaments
}
parameters {
  real mu; // common mean
  real<lower=0> sigma; // common std
}
model {
  y ~ normal(mu, sigma);	// Model for fitting data using tournament specific mu and common std
}
generated quantities {
  vector[N] log_lik;
  real ypred;
  
  ypred = normal_rng(mu, sigma);	//Prediction of tournament
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
}
