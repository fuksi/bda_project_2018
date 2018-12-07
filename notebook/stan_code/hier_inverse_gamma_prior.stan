// Gaussian hierarchical model with common std
data {
  int<lower=0> N; 			// Number of observations
  int<lower=0> J; 			// Number of machines
  matrix[N,J] y; 			// N measurements for J machines
  real<lower=0.1> alpha; //Shape
  real<lower=0.1> beta; //Scale
}
parameters {
  real mu0;				// Common mu for each J machine's mu
  real<lower=0> sigma0;			// Common std for each J machines mu
  real mu[J];			// Machine specific mu
  real<lower=0> sigmaSq; // common var
}
transformed parameters {
  real<lower=0> sigma;			// Common std
  sigma <- sqrt(sigmaSq);
}
model {
  sigmaSq ~ inv_gamma(alpha, beta);
  for (j in 1:J)
    mu[j] ~ normal(mu0, sigma0);	// Model for computing machine specific mu from common mu0 and sigma0
  for (j in 1:J)
    y[:,j] ~ normal(mu[j], sigma);	// Model for fitting data using machine specific mu and common std
}
generated quantities {
  matrix[N,J] log_lik;    
  real ypred[J];
  real mu_new;
  real ypred_new;
  
  for (j in 1:J)
     ypred[j] = normal_rng(mu[j], sigma);	// Predictive distibutions of all the machines
  mu_new = normal_rng(mu0, sigma0);	// Next posterior distribution from commonly learned mu0 and sigma0
  ypred_new = normal_rng(mu_new, sigma);	// Next predictive distibutions of 7th machine
  
  for (j in 1:J)
     for (n in 1:N)
        log_lik[n,j] = normal_lpdf(y[n,j] | mu[j], sigma);
}

