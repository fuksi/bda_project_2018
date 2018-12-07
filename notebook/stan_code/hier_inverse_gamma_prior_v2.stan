// Gaussian hierarchical model with common std
data {
  int<lower=0> N; 			// Number of observations
  int<lower=0> J; 			// Number of tournaments
  matrix[N,J] y; 			// N observations for J tournaments
  real<lower=0.1> alpha; //Shape
  real<lower=0.1> beta; //Scale
}
parameters {
  real mu0;				// Common mu for each J tournament's mu
  real<lower=0> sigma0;			// Common std for each J tournament's mu
  real mu_tilde[J];
  real<lower=0> sigmaSq; // common var
}
transformed parameters {
  real mu[J];			// Tournament specific mu
  real<lower=0> sigma;			// Common std
  for (j in 1:J)
    mu[j] = mu0 + sigma0 * mu_tilde[j];
  sigma <- sqrt(sigmaSq);
}
model {
  sigmaSq ~ inv_gamma(alpha, beta);
  for (j in 1:J) // Model for computing tournament specific mu from common mu0 and sigma0
    mu_tilde[j] ~ normal(0, 1);	// Implies mu[j] ~ noraml(mu0, sigma0)
  for (j in 1:J)
    y[:,j] ~ normal(mu[j], sigma);	// Model for fitting data using tournament specific mu and common std
}
generated quantities {
  matrix[N,J] log_lik;    
  real ypred[J];
  real mu_new;
  real ypred_new;
  
  for (j in 1:J)
     ypred[j] = normal_rng(mu[j], sigma);	// Predictive distibutions of all the tournaments
  mu_new = normal_rng(mu0, sigma0);	// Next posterior distribution from commonly learned mu0 and sigma0
  ypred_new = normal_rng(mu_new, sigma);	// Next predictive distibutions of new tournament
  
  for (j in 1:J)
     for (n in 1:N)
        log_lik[n,j] = normal_lpdf(y[n,j] | mu[j], sigma); //Log-likelihood
}

