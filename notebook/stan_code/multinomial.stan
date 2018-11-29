// Multinomial with with linear logistic prior
data {
  int r; // number of rows
  int c; // number of columns;
  int y[r, c];
}
parameters {
  simplex[c] theta;
}
model {
  for (i in 1:r) {
    y[1,] ~ multinomial(theta);
  }
}