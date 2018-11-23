// Binomial model with linear fit prior
data {
  int<lower=0> N;
  int n[N];  //Number of trials per spread
  row_vector[N] x; //Spreads
  int y[N];  //Number of successes per spread
}
parameters {
  real alpha; 
  real beta; 
}
transformed parameters {
  row_vector[N] p;
  p=inv_logit(alpha + beta*x);
}
model {
  y ~ binomial(n,p); //Note! Probabilities should be constrained to lie between 0 and 1.
}
generated quantities {
 vector[N] y_rep;
 for(i in 1:N){
  y_rep[i] <- binomial_rng(n[i],p[i]); //posterior draws to get posterior predictive checks
 }
}