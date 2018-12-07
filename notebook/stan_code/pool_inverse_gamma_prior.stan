data {
  int<lower=0> N;			// Number of observation
  vector[N] y; 			// N observation for J tournaments
  real<lower=0.1> alpha; //Shape
  real<lower=0.1> beta; //Scale
}
parameters {
  real mu; // common mean
  real<lower=0> sigmaSq; // common var
}
transformed parameters {
  real<lower=0> sigma;
  sigma <- sqrt(sigmaSq);
}
model {
  sigmaSq ~ inv_gamma(alpha,beta);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  real ypred;
  
  ypred = normal_rng(mu, sigma);	// Prediction of tournament
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
}
