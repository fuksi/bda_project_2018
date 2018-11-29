// Gaussian hierarchical model with common std
data {
  int<lower=0> N; 			// Number of observations
  int<lower=0> J; 			// Number of machines
  matrix[N,J] y; 			// N measurements for J machines
}
parameters {
  real theta0;				// Common theta for each J machine's theta
  real<lower=0> sigma0;			// Common std for each J machines theta
  real theta[J];			// Machine specific theta
  real<lower=0> sigma;			// Common std
}
model {
  for (j in 1:J)
    theta[j] ~ normal(theta0, sigma0);	// Model for computing machine specific theta from common theta0 and sigma0
  for (j in 1:J)
    y[:,j] ~ normal(theta[j], sigma);	// Model for fitting data using machine specific theta and common std
}
generated quantities {
  matrix[N,J] log_lik;    
  real ypred[J];
  real theta_new;
  real ypred_new;
  
  for (j in 1:J)
     ypred[j] = normal_rng(theta[j], sigma);	// Predictive distibutions of all the machines
  theta_new = normal_rng(theta0, sigma0);	// Next posterior distribution from commonly learned theta0 and sigma0
  ypred_new = normal_rng(theta_new, sigma);	// Next predictive distibutions of 7th machine
  
  for (j in 1:J)
     for (n in 1:N)
        log_lik[n,j] = normal_lpdf(y[n,j] | theta[j], sigma);
}

