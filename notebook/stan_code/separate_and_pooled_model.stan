// Gaussian separate and pooled model
data {
  int<lower=0> N;			// Number of data measurements
  int<lower=0> J; 			// Number of machines
  matrix[N,J] y; 			// N measurements for J machines
}
parameters {
  real mu[J];			// Machine specific mean
  real<lower=0> sigma[J];			// Machine specific standard deviation
}
model {
  for (j in 1:J)
    y[:,j] ~ normal(mu[j], sigma[j]);	// Model for fitting data using machine specific mu and common std
}
generated quantities {
  matrix[N,J] log_lik;
  real ypred;
  
  ypred = normal_rng(theta, sigma);	//Prediction of machine
  
  for (j in 1:J)
      for (n in 1:N)
         log_lik[n,j] = normal_lpdf(y[n,j] | mu[j], sigma[j]);
}
