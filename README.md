
# BWF Badminton tournaments
##### Bayesian Data Analysis Project, 09-Dec-2018
___

## 1 Introduction

**Goal**: analyze the distribution of outcomes in a badminton tournament.

**Approach**: apply Bayesian data analysis on historical data of badminton tournaments. The estimand of interest is the probability of a certain outcome. The modelling of a match outcome will be explained more in section 2.

**Implementation**:
* Start with some naive assumption of the estimand, in order to choose the model later
* Collect and preprocess data
* Decide on prior choices and models
* Do stan analysis on each models
* Model comparision (using PSIS-LOO)
* Do posterior predictive comparision between models
* Conclusion, possible improvements

## 2 Analysis problem

### 2.1 Discretizing the problem
In one tournament, there are 8 seed players and some unranked players. To discretize the ranking spread, we chose 12th as the rank for all unranked players. The spread is calculated as follows:

**Spread (from 1st player perspective) = Rank(2nd player) - Rank(1st player)**

E.g.

* Spread(from 1st rank player to 8th rank player) = 8 - 1 = 7
* Spread(from 2nd rank player to unrank player) = 12 - 2 = 10
* Spread(from unrank player to 3rd rank player) = 3 - 12 = -9

The discrete space for ranking spread is then **[-11,-10,-9,...,9,10,11]**

A match (which has at most 3 games) has 6 possible outcomes:
1. Lose Lose     -> Lose
2. Lose Win Lose -> Lose
3. Win Lose Lose -> Lose
4. Lose Win Win  -> Win
5. Win Lose Win  -> Win
6. Win Win       -> Win 

To discretize this parameter, we map the outcome of a match to **[1,2,3,4,5,6]** in terms of win degree (i.e. win degree 1 is the worst, and win degree 6 is the best).

For the match below, ranking spread is 11, win degree is 4

<img src="match_beautified.png" alt="match" style="width: 50%; margin-left: 0;"/>

### 2.2 Modeling the problem

Unless stated otherwise, all information in the data are from 1st player perspective  
An observation of a match includes 2 pieces of information:
* ranking spread
* win degree

To formulate the observations as a one dimensional space collection, we need to reduce the matrix 23x6 (23 different spreads, 6 different win degrees) of all possible raw observations. The intuition of the reduction is as follows:
* With the same ranking spread, higher win degree correlates to higher value (see arrow A in image below)
* With the same win degree, lower ranking spread correlates to higher value (see arrow B in image below)
* Step between value is 1

With the given constraint, we will have **28 possible values of observation from [1,28]**. The mapping is as follow (columns are win degrees and rows are ranking spreads)

<img src="mapping4.jpg" alt="mapping" style="width: 35%; margin-left: 0"/>

### 2.3 Analysing the problem

The problem analysis explores the distribution of the observations, especially concentrating on the predictive distribution of the new tournament. Furthermore, we will try to analyze the affect of the different models and prior choices. 

## 3 Dataset and data model

The dataset is collected from Badminton World Federation (BWF) tournament database using Scrapy crawler. After collecting, the data is preprocessed as stated in the previous section. In the end, the format of data is similar to the factory data assignments. A peek of the data:


```python
show_first_rows_of_data()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tournament 1</th>
      <th>Tournament 2</th>
      <th>Tournament 3</th>
      <th>Tournament 4</th>
      <th>Tournament 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Match 1</th>
      <td>6.0</td>
      <td>17.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Match 2</th>
      <td>15.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>Match 3</th>
      <td>16.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Match 4</th>
      <td>15.0</td>
      <td>13.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Match 5</th>
      <td>19.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_summary_of_data()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tournament 1</th>
      <th>Tournament 2</th>
      <th>Tournament 3</th>
      <th>Tournament 4</th>
      <th>Tournament 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>67.000000</td>
      <td>67.000000</td>
      <td>67.000000</td>
      <td>67.000000</td>
      <td>67.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.776119</td>
      <td>14.388060</td>
      <td>13.746269</td>
      <td>14.880597</td>
      <td>14.164179</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.746727</td>
      <td>5.635266</td>
      <td>5.329572</td>
      <td>5.878885</td>
      <td>5.703792</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15.000000</td>
      <td>15.000000</td>
      <td>15.000000</td>
      <td>16.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17.000000</td>
      <td>18.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>25.000000</td>
      <td>27.000000</td>
      <td>26.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 4 Prior choices

We decided to use two different priors: 
* **Inverse gamma** is chosen on variance because it is the conjugate prior to normal likelihood and it has a closed form solution for the outcome of the posterior
* **Uniform** is chosen as weak prior to observe how sensitive is outcome in regards the prior and the data input

## 5 Model

In normal distribution where $\mu$ is known and $\sigma^2$ is unknown, the marginal posterior distribution $p(\sigma^2|y)$ can be computed as described below. The posterior distribution is computed using two different priors, whereas the first is an uniformative (uniform) and the second an informative (inverse gamma) prior.

__Priors:__

Uniform prior

\begin{equation*}
p(\sigma^2) \propto Uniform(0, \infty)
\end{equation*}

Inverse gamma prior

\begin{equation*}  
p(\sigma^2) \propto (\sigma^2)^{-(a+1)}e^{-\beta/\sigma^2} \propto Inv-Gamma(\alpha, \beta) \\
\end{equation*}

where $\alpha=1$ and $\beta=1$ are the shape and scale parameters. Our prior assumption is that variance will be relatively small but we are not sure how small it is, and our guess is around [1,2]. The pdf of inverse gamma with $\alpha=1$ and $\beta=1$ is a good fit for our assumption.

__Likelihood:__
\begin{equation*}
p(y|\mu,\sigma^2) \propto \prod_{i=1}^{N} p(y_i | \mu, \sigma^2) \propto N(y | \mu, \sigma^2) 
\end{equation*}

where $\mu$ is known.

__Posterior:__

\begin{equation*}
p(\sigma^2 | y) \propto p(\sigma^2)p(y|\mu, \sigma^2)
\end{equation*}

Since one of the objective is to predict the distribution of a new tournament, we will use pooled and hierarchical model. The separate model is excluded because it handles the tournaments uniquely  without having any common parameters which could be used to predict the new tournament.

In the pooled model the mean and the variance is computed from the combined data of all the tournaments and there is no distinction between different tournaments. This means that also the new tournament will have similar distibution as the predictive distribution of the tournaments.  

In the hierarchical model each tournament is handled separately having own mean and common standard deviation. Furthermore, all the means are controlled by common hyperparameters ($\mu_0$ and $\sigma^2_0$) which means that the means are drawn from the common distribution described by these hyperparameters. The result of the new tournament can be predicted using the common hyperparameters: first draw the mean from the common distribution and use it to sample the predictive distribution. 

Then based on the prior choices, we have 4 different models:
* pooled with uniform prior
* pooled with inverse gamma prior for variance
* hierarchical with uniform prior
* hierarchical with inverse gamma prior for variance

## 6 Stan analysis of the models

For each model, we will show the Stan model and convergence diagnostic.

The Stan model is fitted using Stan's default parameters (4 chains, 1000 warmup iterations, 1000 sampling iteration, ending up to 4000 samples and 10 as maximum tree depth). In addition, to avoid false positive conclusion about divergences, the *adapt_delta* value is set to 0.9. This means that the fitting uses larger target acceptance probability and therefore all the divergences can be seen. If the resulting value is still 0 after this, we can verify that there are no divergences. If not, the divergences could be further analyzed by increasing *adapt_delta*.

Besides divergences, the convergence diagnostic includes a short discussion about $\hat R$ and n_eff. Generally, if the $\hat R$ values of the parameters are close to 1 and below 1.1, the fit has been good. The low $\hat R$ values combined with high effective sample size (n_eff) per transition informs that the Markov chains were mixed well. Note that discussion about depth tree and energy Bayesian fraction of missing information (E-BFMI) is left out because their results were same for all the models (depth tree 0 and E-BFMI did not give any information).

### 6.1 Pooled model with uniform prior

#### Stan model

The stan code of the model:
```
data {
  int<lower=0> N;		// Number of observations
  vector[N] y;           // N observations for J tournaments
}
parameters {
  real mu;              // Common mean
  real<lower=0> sigma;  // Common std
}
model {
  y ~ normal(mu, sigma);// Model for fitting data using tournament specific mu and common std
}
generated quantities {
  vector[N] log_lik;
  real ypred;
  
  ypred = normal_rng(mu, sigma);	              //Prediction of tournament
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);   //Log-likelihood
}
```
#### Convergence diagnostic
After compiling the model and fitting the combined data of the tournaments, the diagnostic of the fit was examined:

| Diagnostic | Result   |
|---|---|
| All $\hat R$ values close to 1 | OK |
| Low $\hat R$ values combined with high effective sample size (n_eff) | OK |
| Divergences is 0 | OK |

All the results for pooled uniform prior model fitting are good. The full fit can be found in the _Attachment 1_ and a shorter summary is shown below.


```python
# Fit pooled uniform model
pool_uni_df, pool_uni_fit = compute_model(r'stan_code/pool_uniform_prior.stan', pooled_data_model)
# Print summary of the fit
print_compact_fit(pool_uni_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>14.1935</td>
      <td>0.00603298</td>
      <td>0.314179</td>
      <td>13.5613</td>
      <td>13.9842</td>
      <td>14.2001</td>
      <td>14.409</td>
      <td>14.8204</td>
      <td>2712</td>
      <td>1.00109</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>5.66282</td>
      <td>0.00441324</td>
      <td>0.224035</td>
      <td>5.24378</td>
      <td>5.50642</td>
      <td>5.65253</td>
      <td>5.80839</td>
      <td>6.12739</td>
      <td>2577</td>
      <td>1.00018</td>
    </tr>
    <tr>
      <th>log_lik[0]</th>
      <td>-3.70522</td>
      <td>0.00177279</td>
      <td>0.091105</td>
      <td>-3.89356</td>
      <td>-3.76527</td>
      <td>-3.70164</td>
      <td>-3.64204</td>
      <td>-3.53086</td>
      <td>2641</td>
      <td>1.00078</td>
    </tr>
    <tr>
      <th>log_lik[1]</th>
      <td>-2.66381</td>
      <td>0.000781838</td>
      <td>0.0394576</td>
      <td>-2.7439</td>
      <td>-2.68967</td>
      <td>-2.66273</td>
      <td>-2.63632</td>
      <td>-2.58826</td>
      <td>2547</td>
      <td>1.00023</td>
    </tr>
    <tr>
      <th>log_lik[2]</th>
      <td>-2.70475</td>
      <td>0.000794411</td>
      <td>0.0395853</td>
      <td>-2.78604</td>
      <td>-2.73049</td>
      <td>-2.70434</td>
      <td>-2.67761</td>
      <td>-2.62968</td>
      <td>2483</td>
      <td>1.00032</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>log_lik[332]</th>
      <td>-2.70475</td>
      <td>0.000794411</td>
      <td>0.0395853</td>
      <td>-2.78604</td>
      <td>-2.73049</td>
      <td>-2.70434</td>
      <td>-2.67761</td>
      <td>-2.62968</td>
      <td>2483</td>
      <td>1.00032</td>
    </tr>
    <tr>
      <th>log_lik[333]</th>
      <td>-2.81336</td>
      <td>0.000768998</td>
      <td>0.0414903</td>
      <td>-2.89455</td>
      <td>-2.84142</td>
      <td>-2.81289</td>
      <td>-2.78428</td>
      <td>-2.73293</td>
      <td>2911</td>
      <td>1.00091</td>
    </tr>
    <tr>
      <th>log_lik[334]</th>
      <td>-3.46419</td>
      <td>0.001425</td>
      <td>0.0745643</td>
      <td>-3.61728</td>
      <td>-3.51306</td>
      <td>-3.46244</td>
      <td>-3.41316</td>
      <td>-3.32091</td>
      <td>2738</td>
      <td>1.00092</td>
    </tr>
    <tr>
      <th>ypred</th>
      <td>14.2868</td>
      <td>0.0901683</td>
      <td>5.70274</td>
      <td>3.42566</td>
      <td>10.3648</td>
      <td>14.2842</td>
      <td>18.1957</td>
      <td>25.5974</td>
      <td>4000</td>
      <td>0.999995</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-746.025</td>
      <td>0.0248809</td>
      <td>1.00668</td>
      <td>-748.748</td>
      <td>-746.425</td>
      <td>-745.714</td>
      <td>-745.295</td>
      <td>-745.018</td>
      <td>1637</td>
      <td>0.999565</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print additional checking of the fit
print_compact_fit_checking(pool_uni_fit, pool_uni_df)
```

    Maximum value of the Rhat: 
    1.0012218111524749
    
    Divergences:
    0.0 of 4000 iterations ended with a divergence (0.0%)
    
    

### 6.2 Pooled model with inverse gamma prior

#### Stan model
The stan code of the model:
```
data {
  int<lower=0> N;		 // Number of observations
  vector[N] y;            // N observations for J tournaments
  real<lower=0.1> alpha;  //Shape
  real<lower=0.1> beta;  //Scale
}
parameters {
  real mu;               // Common mean
  real<lower=0> sigmaSq; // Common var
}
transformed parameters {
  real<lower=0> sigma;
  sigma <- sqrt(sigmaSq);
}
model {
  sigmaSq ~ inv_gamma(alpha,beta);  // Prior
  y ~ normal(mu, sigma);            // Fitting of the model
}
generated quantities {
  vector[N] log_lik;
  real ypred;
  
  ypred = normal_rng(mu, sigma);	// Prediction of tournament
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | mu, sigma); //Log-likelihood
}

```

#### Convergence diagnostic
The same procedure is followed here (as in the previous section) and similar results were obtained:

| Diagnostic | Result   |
|---|---|
| All $\hat R$ values close to 1 | OK |
| Low $\hat R$ values combined with high effective sample size (n_eff) | OK |
| Divergences is 0 | OK |

All the results for pooled inverse gamma prior model fitting are good. The full fit can be found in the _Attachment 2_ and a shorter summary is shown below.


```python
# Fit pooled inverse gamma model
pool_inv_df, pool_inv_fit = compute_model(r'stan_code/pool_inverse_gamma_prior.stan', pooled_data_model)
# Print summary of the fit
print_compact_fit(pool_inv_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>14.1912</td>
      <td>0.00659562</td>
      <td>0.311324</td>
      <td>13.5848</td>
      <td>13.9808</td>
      <td>14.1944</td>
      <td>14.3957</td>
      <td>14.8102</td>
      <td>2228</td>
      <td>1.00237</td>
    </tr>
    <tr>
      <th>sigmaSq</th>
      <td>31.819</td>
      <td>0.0465451</td>
      <td>2.42705</td>
      <td>27.4953</td>
      <td>30.0994</td>
      <td>31.688</td>
      <td>33.3725</td>
      <td>36.8217</td>
      <td>2719</td>
      <td>1.00037</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>5.63676</td>
      <td>0.0040921</td>
      <td>0.21424</td>
      <td>5.2436</td>
      <td>5.48629</td>
      <td>5.62921</td>
      <td>5.77689</td>
      <td>6.06809</td>
      <td>2741</td>
      <td>1.00037</td>
    </tr>
    <tr>
      <th>log_lik[0]</th>
      <td>-3.70983</td>
      <td>0.00189007</td>
      <td>0.0933245</td>
      <td>-3.89964</td>
      <td>-3.77044</td>
      <td>-3.7058</td>
      <td>-3.64535</td>
      <td>-3.53664</td>
      <td>2438</td>
      <td>1.00087</td>
    </tr>
    <tr>
      <th>log_lik[1]</th>
      <td>-2.65936</td>
      <td>0.000735954</td>
      <td>0.0384743</td>
      <td>-2.73647</td>
      <td>-2.68499</td>
      <td>-2.65882</td>
      <td>-2.63316</td>
      <td>-2.58596</td>
      <td>2733</td>
      <td>0.999909</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>log_lik[332]</th>
      <td>-2.70068</td>
      <td>0.000759725</td>
      <td>0.039352</td>
      <td>-2.78049</td>
      <td>-2.72698</td>
      <td>-2.70021</td>
      <td>-2.67379</td>
      <td>-2.62523</td>
      <td>2683</td>
      <td>0.999668</td>
    </tr>
    <tr>
      <th>log_lik[333]</th>
      <td>-2.81015</td>
      <td>0.000869736</td>
      <td>0.0394174</td>
      <td>-2.88917</td>
      <td>-2.83559</td>
      <td>-2.80985</td>
      <td>-2.78294</td>
      <td>-2.7354</td>
      <td>2054</td>
      <td>1.00321</td>
    </tr>
    <tr>
      <th>log_lik[334]</th>
      <td>-3.46668</td>
      <td>0.0015697</td>
      <td>0.0761101</td>
      <td>-3.62117</td>
      <td>-3.51465</td>
      <td>-3.46372</td>
      <td>-3.41446</td>
      <td>-3.3255</td>
      <td>2351</td>
      <td>1.00139</td>
    </tr>
    <tr>
      <th>ypred</th>
      <td>14.2633</td>
      <td>0.08992</td>
      <td>5.61407</td>
      <td>3.23815</td>
      <td>10.4847</td>
      <td>14.328</td>
      <td>18.0029</td>
      <td>25.224</td>
      <td>3898</td>
      <td>0.999202</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-751.202</td>
      <td>0.0262385</td>
      <td>1.01486</td>
      <td>-753.917</td>
      <td>-751.573</td>
      <td>-750.898</td>
      <td>-750.491</td>
      <td>-750.235</td>
      <td>1496</td>
      <td>1.00201</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print additional checking of the fit
print_compact_fit_checking(pool_inv_fit, pool_inv_df)
```

    Maximum value of the Rhat: 
    1.00330613987631
    
    Divergences:
    0.0 of 4000 iterations ended with a divergence (0.0%)
    
    

### 6.3 Hierarchical model with uniform prior

### 6.3.1  Unsuccessful fitting

#### Stan model 
The stan code of the model:
```
data {
  int<lower=0> N;  // Number of observations
  int<lower=0> J;  // Number of tournaments
  matrix[N,J] y;  // N observations for J tournaments
}
parameters {
  real mu0;             // Common mu for each J tournament's mu
  real<lower=0> sigma0; // Common std for each J tournament's mu
  real<lower=0> sigma; // Common std between tournaments
  real mu_tilde[J];    // Tournament specific mu
}
model {
  for (j in 1:J)
    mu[j] ~ normal(mu0, sigma0);	   // Model for computing tournament specific mu from common mu0 and sigma0
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
  mu_new = normal_rng(mu0, sigma0);	      // Next posterior distribution from commonly learned mu0 and sigma0
  ypred_new = normal_rng(mu_new, sigma);	// Next predictive distibutions of new tournament
  
  for (j in 1:J)
     for (n in 1:N)
        log_lik[n,j] = normal_lpdf(y[n,j] | mu[j], sigma); //Log-likelihood
}
```

#### Convergence diagnostic attempt 1
The same procedure is followed here (as in the previous section) with minor change. The data used for fitting is a matrix where columns are the tournaments and rows are the matches in the tournaments. The diagnostic results are:

| Diagnostic | Result   |
|--|--|
| All $\hat R$ values close to 1 | __NO__ <br> *lp__* is 1.4 |
| Low $\hat R$ values combined with high effective sample size (n_eff) | __NO__ |
| Divergences is 0 | __NO__ <br> 8% of the target posterior was not explored|


Because none of the conditions were fulfilled, in the next step we will try to improve the results by reducing the accuracy of the simulations by increasing the value of the *adapt_delta* parameter.

The results of the attempt 1 can be seen below.


```python
# Fit hierarchical uniform model
hier_uni_df, hier_uni_fit = compute_model(r'stan_code/hier_uniform_prior.stan', hierarchical_data_model)
# Print the summary of the fit
print_compact_fit(hier_uni_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu0</th>
      <td>14.222</td>
      <td>0.0306316</td>
      <td>0.518028</td>
      <td>13.2403</td>
      <td>13.9695</td>
      <td>14.264</td>
      <td>14.4845</td>
      <td>15.0738</td>
      <td>286</td>
      <td>1.0193</td>
    </tr>
    <tr>
      <th>sigma0</th>
      <td>0.560246</td>
      <td>0.0731073</td>
      <td>0.759754</td>
      <td>0.034287</td>
      <td>0.172624</td>
      <td>0.381537</td>
      <td>0.709618</td>
      <td>2.13192</td>
      <td>108</td>
      <td>1.03272</td>
    </tr>
    <tr>
      <th>mu[0]</th>
      <td>14.1029</td>
      <td>0.0484593</td>
      <td>0.477268</td>
      <td>13.0482</td>
      <td>13.8036</td>
      <td>14.1453</td>
      <td>14.4149</td>
      <td>14.938</td>
      <td>97</td>
      <td>1.03988</td>
    </tr>
    <tr>
      <th>mu[1]</th>
      <td>14.2979</td>
      <td>0.0144251</td>
      <td>0.458663</td>
      <td>13.3891</td>
      <td>14.0025</td>
      <td>14.3344</td>
      <td>14.5531</td>
      <td>15.27</td>
      <td>1011</td>
      <td>1.0106</td>
    </tr>
    <tr>
      <th>mu[2]</th>
      <td>14.0964</td>
      <td>0.0545487</td>
      <td>0.496962</td>
      <td>13.0192</td>
      <td>13.7931</td>
      <td>14.1295</td>
      <td>14.4328</td>
      <td>14.9691</td>
      <td>83</td>
      <td>1.044</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ypred[3]</th>
      <td>14.4326</td>
      <td>0.0903994</td>
      <td>5.6887</td>
      <td>3.38663</td>
      <td>10.585</td>
      <td>14.451</td>
      <td>18.2019</td>
      <td>25.6342</td>
      <td>3960</td>
      <td>0.999456</td>
    </tr>
    <tr>
      <th>ypred[4]</th>
      <td>14.3471</td>
      <td>0.0899178</td>
      <td>5.6869</td>
      <td>3.39243</td>
      <td>10.5848</td>
      <td>14.3392</td>
      <td>18.2295</td>
      <td>25.5183</td>
      <td>4000</td>
      <td>1.00017</td>
    </tr>
    <tr>
      <th>mu_new</th>
      <td>14.208</td>
      <td>0.0255003</td>
      <td>1.1708</td>
      <td>12.3481</td>
      <td>13.8682</td>
      <td>14.2873</td>
      <td>14.5676</td>
      <td>15.8755</td>
      <td>2108</td>
      <td>1.00535</td>
    </tr>
    <tr>
      <th>ypred_new</th>
      <td>14.2926</td>
      <td>0.0919394</td>
      <td>5.81476</td>
      <td>2.84883</td>
      <td>10.4361</td>
      <td>14.4064</td>
      <td>18.2223</td>
      <td>25.4268</td>
      <td>4000</td>
      <td>1.00122</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-743.541</td>
      <td>1.47587</td>
      <td>4.89491</td>
      <td>-752.243</td>
      <td>-746.688</td>
      <td>-744.196</td>
      <td>-741.133</td>
      <td>-732.58</td>
      <td>11</td>
      <td>1.42308</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print additional checking of the fit
print_compact_fit_checking(hier_uni_fit, hier_uni_df)
```

    Maximum value of the Rhat: 
    1.423080255915649
    
    Divergences:
    308.0 of 4000 iterations ended with a divergence (7.7%)
    Try running with larger adapt_delta to remove the divergences
    
    

#### Convergence diagnostic attempt 2

After changing the accuracy of the simulations from 0.9 to 0.93, the data is re-fit and following results are gained:

| Diagnostic | Result   |
|--|--|
| All $\hat R$ values close to 1 | OK |
| Low $\hat R$ values combined with high effective sample size (n_eff) | OK |
| Divergences is 0 | __NO__ <br> still 2% of the target posterior was not explored|

With these results we can verify that the hierarchical uniform prior model fitting is almost successful. Although, this is not the desired result and therefore, in the next step the further improvement is discussed. 

The results of the attempt 2 is shown below.


```python
# Fit hierarchical uniform model
hier_uni_df, hier_uni_fit = compute_model(r'stan_code/hier_uniform_prior.stan', 
                                          hierarchical_data_model, adapt_delta=0.93)
# Print the summary of the fit
print_compact_fit(hier_uni_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu0</th>
      <td>14.1722</td>
      <td>0.0133645</td>
      <td>0.44123</td>
      <td>13.3144</td>
      <td>13.9179</td>
      <td>14.159</td>
      <td>14.4377</td>
      <td>15.0467</td>
      <td>1090</td>
      <td>1.00438</td>
    </tr>
    <tr>
      <th>sigma0</th>
      <td>0.563965</td>
      <td>0.0243783</td>
      <td>0.571721</td>
      <td>0.0510735</td>
      <td>0.200409</td>
      <td>0.40928</td>
      <td>0.731901</td>
      <td>2.12706</td>
      <td>550</td>
      <td>1.00114</td>
    </tr>
    <tr>
      <th>mu[0]</th>
      <td>14.0559</td>
      <td>0.0143581</td>
      <td>0.480942</td>
      <td>13.0217</td>
      <td>13.7558</td>
      <td>14.0839</td>
      <td>14.3739</td>
      <td>14.9692</td>
      <td>1122</td>
      <td>1.0034</td>
    </tr>
    <tr>
      <th>mu[1]</th>
      <td>14.2551</td>
      <td>0.013469</td>
      <td>0.481318</td>
      <td>13.3542</td>
      <td>13.9509</td>
      <td>14.2275</td>
      <td>14.5544</td>
      <td>15.2806</td>
      <td>1277</td>
      <td>1.00411</td>
    </tr>
    <tr>
      <th>mu[2]</th>
      <td>14.0422</td>
      <td>0.0138805</td>
      <td>0.481636</td>
      <td>12.9916</td>
      <td>13.76</td>
      <td>14.056</td>
      <td>14.3534</td>
      <td>14.9512</td>
      <td>1204</td>
      <td>1.00298</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ypred[3]</th>
      <td>14.3333</td>
      <td>0.0915444</td>
      <td>5.68976</td>
      <td>3.33591</td>
      <td>10.4447</td>
      <td>14.3595</td>
      <td>18.0902</td>
      <td>25.5877</td>
      <td>3863</td>
      <td>0.99995</td>
    </tr>
    <tr>
      <th>ypred[4]</th>
      <td>14.2131</td>
      <td>0.0884799</td>
      <td>5.58756</td>
      <td>3.2083</td>
      <td>10.4591</td>
      <td>14.2649</td>
      <td>18.0418</td>
      <td>24.8965</td>
      <td>3988</td>
      <td>1.00015</td>
    </tr>
    <tr>
      <th>mu_new</th>
      <td>14.1864</td>
      <td>0.0184552</td>
      <td>0.961623</td>
      <td>12.4116</td>
      <td>13.7922</td>
      <td>14.1716</td>
      <td>14.569</td>
      <td>16.0339</td>
      <td>2715</td>
      <td>1.00083</td>
    </tr>
    <tr>
      <th>ypred_new</th>
      <td>14.2098</td>
      <td>0.0896986</td>
      <td>5.67304</td>
      <td>2.94677</td>
      <td>10.4409</td>
      <td>14.1165</td>
      <td>17.8849</td>
      <td>25.5395</td>
      <td>4000</td>
      <td>0.999853</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-744.229</td>
      <td>0.245898</td>
      <td>4.09994</td>
      <td>-752.034</td>
      <td>-747.03</td>
      <td>-744.541</td>
      <td>-741.593</td>
      <td>-735.542</td>
      <td>278</td>
      <td>1.00407</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print additional checking of the fit
print_compact_fit_checking(hier_uni_fit, hier_uni_df)
```

    Maximum value of the Rhat: 
    1.0043815114485763
    
    Divergences:
    79.0 of 4000 iterations ended with a divergence (1.975%)
    Try running with larger adapt_delta to remove the divergences
    
    

### 6.3.2 Successful fitting

In this section, we will modify the Stan code and use exactly the same approach as described in pystan's workflow (http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html, A Successful Fit). Because the fit which uses the centered parametrization is not successful, we should change the Stan code using non-centered parametrization.

__Centered__ parametrization of parameter _mu_
```
parameters {
  ...
  real mu[J]; // Tournament specific mu
  ...
}
model {
  for (j in 1:J) // Model for computing tournament specific mu from common mu0 and sigma0
    mu[j] ~ normal(mu0, sigma0);
  ...
}
```

is converted __to non-centered__ parametrization 
```
parameters {
  ...
  real mu_tilde[J];
}
transformed parameters {
  real mu[J];// Tournament specific mu
  for (j in 1:J)
    mu[j] = mu0 + sigma0 * mu_tilde[j];
}
model {
  for (j in 1:J) // Model for computing tournament specific mu from common mu0 and sigma0
    mu_tilde[j] ~ normal(0, 1); // Implies mu[j] ~ normal(mu0,sigma0)
  ...
}
```
The full updated Stan code is shown in the next section.

#### Stan model 
The stan code of the model:
```
data {
  int<lower=0> N;  // Number of observations
  int<lower=0> J;  // Number of tournaments
  matrix[N,J] y;  // N observations for J tournaments
}
```

```
parameters {
  real mu0;             // Common mu for each J tournament's mu
  real<lower=0> sigma0; // Common std for each J tournament's mu
  real<lower=0> sigma; // Common std between tournaments
    real mu_tilde[J];
}
transformed parameters {
  real mu[J];          // Tournament specific mu
  for (j in 1:J)
    mu[j] = mu0 + sigma0 * mu_tilde[j];
}
model {
  for (j in 1:J)                   // Model for computing tournament specific mu from common mu0 and sigma0
    mu_tilde[j] ~ normal(0, 1);    // Implies mu[j] ~ normal(mu0,sigma0)
  for (j in 1:J)
    y[:,j] ~ normal(mu[j], sigma); // Model for fitting data using machine specific mu and common std
}
generated quantities {
  matrix[N,J] log_lik;    
  real ypred[J];
  real mu_new;
  real ypred_new;
  
  for (j in 1:J)
     ypred[j] = normal_rng(mu[j], sigma);	// Predictive distibutions of all the tournaments
  mu_new = normal_rng(mu0, sigma0);	      // Next posterior distribution from commonly learned mu0 and sigma0
  ypred_new = normal_rng(mu_new, sigma);	// Next predictive distibutions of new tournament
  
  for (j in 1:J)
     for (n in 1:N)
        log_lik[n,j] = normal_lpdf(y[n,j] | mu[j], sigma); //Log-likelihood
}
```

#### Convergence diagnostic attempt 3

The same procedure is followed (as in the previous section) and improved results were obtained:

| Diagnostic | Result   |
|--|--|
| All $\hat R$ values close to 1 | OK |
| Low $\hat R$ values combined with high effective sample size (n_eff) | OK |
| Divergences is 0 | GOOD ENOUGH <br> still 0.025% of the target posterior was not expolred <br> and increasing *adapt_delta* did not improve this result|

All of these verify that hierarchical uniform prior model fitting is successful enough. The full fit can be found in the _Attachment 3_ and a shorter summary is shown below.


```python
# Fit hierarchical uniform model
hier_uni_df, hier_uni_fit = compute_model(r'stan_code/hier_uniform_prior_v2.stan', hierarchical_data_model)
# Print the summary of the fit
print_compact_fit(hier_uni_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu0</th>
      <td>14.1961</td>
      <td>0.0141698</td>
      <td>0.458058</td>
      <td>13.2531</td>
      <td>13.9249</td>
      <td>14.1917</td>
      <td>14.464</td>
      <td>15.1017</td>
      <td>1045</td>
      <td>1.00285</td>
    </tr>
    <tr>
      <th>sigma0</th>
      <td>0.550535</td>
      <td>0.0179981</td>
      <td>0.576781</td>
      <td>0.0140076</td>
      <td>0.18154</td>
      <td>0.388406</td>
      <td>0.717958</td>
      <td>2.06309</td>
      <td>1027</td>
      <td>1.00309</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>5.66836</td>
      <td>0.00337786</td>
      <td>0.213634</td>
      <td>5.26576</td>
      <td>5.52314</td>
      <td>5.66113</td>
      <td>5.80597</td>
      <td>6.10786</td>
      <td>4000</td>
      <td>0.999766</td>
    </tr>
    <tr>
      <th>mu_tilde[0]</th>
      <td>-0.207582</td>
      <td>0.0159645</td>
      <td>0.883688</td>
      <td>-1.97272</td>
      <td>-0.795004</td>
      <td>-0.21698</td>
      <td>0.373531</td>
      <td>1.49234</td>
      <td>3064</td>
      <td>0.999875</td>
    </tr>
    <tr>
      <th>mu_tilde[1]</th>
      <td>0.10843</td>
      <td>0.0154156</td>
      <td>0.883542</td>
      <td>-1.68338</td>
      <td>-0.477808</td>
      <td>0.101281</td>
      <td>0.691408</td>
      <td>1.87599</td>
      <td>3285</td>
      <td>1.00029</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ypred[3]</th>
      <td>14.4115</td>
      <td>0.0933838</td>
      <td>5.77547</td>
      <td>2.91519</td>
      <td>10.6273</td>
      <td>14.3439</td>
      <td>18.2793</td>
      <td>25.8232</td>
      <td>3825</td>
      <td>0.999828</td>
    </tr>
    <tr>
      <th>ypred[4]</th>
      <td>14.1985</td>
      <td>0.0891236</td>
      <td>5.63667</td>
      <td>3.05899</td>
      <td>10.3145</td>
      <td>14.244</td>
      <td>18.0424</td>
      <td>24.8852</td>
      <td>4000</td>
      <td>0.999548</td>
    </tr>
    <tr>
      <th>mu_new</th>
      <td>14.2178</td>
      <td>0.0179958</td>
      <td>0.940442</td>
      <td>12.446</td>
      <td>13.8592</td>
      <td>14.2051</td>
      <td>14.5741</td>
      <td>16.0938</td>
      <td>2731</td>
      <td>0.999788</td>
    </tr>
    <tr>
      <th>ypred_new</th>
      <td>14.198</td>
      <td>0.0917411</td>
      <td>5.80221</td>
      <td>2.83785</td>
      <td>10.2555</td>
      <td>14.1872</td>
      <td>18.1655</td>
      <td>25.4919</td>
      <td>4000</td>
      <td>0.999715</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-749.31</td>
      <td>0.0670804</td>
      <td>2.27085</td>
      <td>-754.317</td>
      <td>-750.67</td>
      <td>-749.121</td>
      <td>-747.698</td>
      <td>-745.474</td>
      <td>1146</td>
      <td>1.00138</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_compact_fit_checking(hier_uni_fit, hier_uni_df)
```

    Maximum value of the Rhat: 
    1.0030853274280642
    
    Divergences:
    1.0 of 4000 iterations ended with a divergence (0.025%)
    Try running with larger adapt_delta to remove the divergences
    
    

### 6.4 Hierarchical model with inverse gamma prior

#### Stan model

The stan code of the model (using non-centered parametrization):
```
data {
  int<lower=0> N; 			// Number of observations
  int<lower=0> J; 			// Number of tournaments
  matrix[N,J] y; 			// N measurements for J tournaments
}
```

```
parameters {
  real mu0;				// Common mu for each J tournaments's mu
  real<lower=0> sigma0;			// Common std for each J tournament's mu
  real<lower=0> sigma;			// Common std
  real mu_tilde[J];
}
transformed parameters {
  real mu[J];			// Tournament specific mu
  for (j in 1:J)
    mu[j] = mu0 + sigma0 * mu_tilde[j];
}
model {
  for (j in 1:J) // Model for computing tournament specific mu from common mu0 and sigma0
    mu_tilde[j] ~ normal(0, 1); // Implies mu[j] ~ normal(mu0,sigma0)
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


```

#### Convergence diagnostic
The same procedure is followed (as in the previous section) and the diagnostic results are:


| Diagnostic | Result   |
|--|--|
| All $\hat R$ values close to 1 | OK |
| Low $\hat R$ values combined with high effective sample size (n_eff) | OK |
| Divergences is 0 | GOOD ENOUGH <br> still 0.025% of the target posterior was not expolred <br> and increasing *adapt_delta* did not improve this result|

All of these verify that hierarchical inverse gamma prior model fitting is successful enough. The full fit can be found in the _Attachment 4_ and a shorter summary is shown below.


```python
# Fit hierarchical uniform model
hier_inv_df, hier_inv_fit = compute_model(r'stan_code/hier_inverse_gamma_prior_v2.stan', hierarchical_data_model)
print_compact_fit(hier_inv_df)
```

    Using cached StanModel
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>se_mean</th>
      <th>sd</th>
      <th>2.5%</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>97.5%</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu0</th>
      <td>14.1742</td>
      <td>0.0106244</td>
      <td>0.429863</td>
      <td>13.3323</td>
      <td>13.9126</td>
      <td>14.1783</td>
      <td>14.4469</td>
      <td>15.0025</td>
      <td>1637</td>
      <td>1.00147</td>
    </tr>
    <tr>
      <th>sigma0</th>
      <td>0.55408</td>
      <td>0.0187816</td>
      <td>0.567502</td>
      <td>0.0227001</td>
      <td>0.193471</td>
      <td>0.403468</td>
      <td>0.728336</td>
      <td>2.04482</td>
      <td>913</td>
      <td>1.0023</td>
    </tr>
    <tr>
      <th>mu_tilde[0]</th>
      <td>-0.202636</td>
      <td>0.0156438</td>
      <td>0.883703</td>
      <td>-1.96254</td>
      <td>-0.760705</td>
      <td>-0.211218</td>
      <td>0.37144</td>
      <td>1.55705</td>
      <td>3191</td>
      <td>1.0007</td>
    </tr>
    <tr>
      <th>mu_tilde[1]</th>
      <td>0.115359</td>
      <td>0.0161837</td>
      <td>0.880342</td>
      <td>-1.68287</td>
      <td>-0.427827</td>
      <td>0.10454</td>
      <td>0.695337</td>
      <td>1.82562</td>
      <td>2959</td>
      <td>1.00076</td>
    </tr>
    <tr>
      <th>mu_tilde[2]</th>
      <td>-0.200715</td>
      <td>0.0146266</td>
      <td>0.857001</td>
      <td>-1.93516</td>
      <td>-0.749326</td>
      <td>-0.218871</td>
      <td>0.354451</td>
      <td>1.48706</td>
      <td>3433</td>
      <td>1.00081</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ypred[3]</th>
      <td>14.5288</td>
      <td>0.0921506</td>
      <td>5.67531</td>
      <td>3.5257</td>
      <td>10.7779</td>
      <td>14.5297</td>
      <td>18.2851</td>
      <td>26.0277</td>
      <td>3793</td>
      <td>0.999918</td>
    </tr>
    <tr>
      <th>ypred[4]</th>
      <td>14.1416</td>
      <td>0.0893992</td>
      <td>5.6541</td>
      <td>3.05447</td>
      <td>10.3961</td>
      <td>14.0928</td>
      <td>17.9503</td>
      <td>25.1413</td>
      <td>4000</td>
      <td>0.999683</td>
    </tr>
    <tr>
      <th>mu_new</th>
      <td>14.1745</td>
      <td>0.017315</td>
      <td>0.895872</td>
      <td>12.3202</td>
      <td>13.8097</td>
      <td>14.1975</td>
      <td>14.5734</td>
      <td>15.8751</td>
      <td>2677</td>
      <td>1.00043</td>
    </tr>
    <tr>
      <th>ypred_new</th>
      <td>14.2066</td>
      <td>0.0899841</td>
      <td>5.69109</td>
      <td>3.30853</td>
      <td>10.3106</td>
      <td>14.0458</td>
      <td>18.1636</td>
      <td>25.1357</td>
      <td>4000</td>
      <td>0.999641</td>
    </tr>
    <tr>
      <th>lp__</th>
      <td>-754.52</td>
      <td>0.0668638</td>
      <td>2.37343</td>
      <td>-759.858</td>
      <td>-756.017</td>
      <td>-754.246</td>
      <td>-752.799</td>
      <td>-750.641</td>
      <td>1260</td>
      <td>1.00191</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_compact_fit_checking(hier_inv_fit, hier_inv_df)
```

    Maximum value of the Rhat: 
    1.0022963849717355
    
    Divergences:
    1.0 of 4000 iterations ended with a divergence (0.025%)
    Try running with larger adapt_delta to remove the divergences
    
    

### 6.5 Conclusion

Based on the diagnostic results, all the four models can be used for further analysis (next section).

## 7 Model comparision with PSIS-LOO and $P_{LOO-CV}$ 


* Model selection according to the hightest LOO-CV sum
* Reliability based on the _k_ values: < 0.7 ok, < 0.5 good

The PSIS-LOO values of the models can be computed using provided _psisloo_ function. The function returns observation specific _k_-values and PSIS-LOO-CV values. In addition, it returns the sum of the PSIS-LOO-CV values, hence the sum of the LOO log desnities:

\begin{equation*}
lppd_{loo-cv} = \sum_{i=1}^{N} log \left( \frac{1}{S} \sum_{s=1}^{S} p(y_i|\theta^{is}) \right)
\end{equation*}

The estimated effective number of parameters ($P_{LOO-CV}$) in the model is computed as follows:

\begin{equation*}
p_{loo-cv} = lppd-lppd_{loo-cv} 
\end{equation*}

where $lppd$ is the sum of the log densities of the posterior draws:
\begin{equation*}
lppd = \sum_{i=1}^{N} log \left( \frac{1}{S} \sum_{s=1}^{S} p(y_i|\theta^{s}) \right)
\end{equation*}

### Comparision

All the PSIS-LOO values, estimated effective number of parameters and _k_-values are shown below.



```python
compare_psis_loo(fits=[pool_uni_fit, pool_inv_fit, hier_uni_fit, hier_inv_fit],  model_labels=[
	'Pooled model with uniform prior',
	'Pooled model with inverse gamma prior',
	'Hierarchical model with uniform prior',
	'Hierarchical model with inverse gamma prior'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Models</th>
      <th>Psisloo</th>
      <th>P_eff</th>
      <th>Max k value</th>
      <th>Min k value</th>
      <th>Mean k value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pooled model with uniform prior</td>
      <td>-1056.45</td>
      <td>1.67</td>
      <td>-0.07</td>
      <td>-0.25</td>
      <td>-0.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pooled model with inverse gamma prior</td>
      <td>-1056.38</td>
      <td>1.64</td>
      <td>-0.01</td>
      <td>-0.16</td>
      <td>-0.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hierarchical model with uniform prior</td>
      <td>-1057.25</td>
      <td>2.96</td>
      <td>0.10</td>
      <td>-0.14</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hierarchical model with inverse gamma prior</td>
      <td>-1057.39</td>
      <td>3.10</td>
      <td>0.11</td>
      <td>-0.18</td>
      <td>-0.04</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

All the models are considered very reliable with very low k-values. If we consider towards model
with best predictive accuracy, then **Pooled model with inverse gamma prior** should be selected, because its
PSIS-LOO value is the highest (in other words the sum of log predictive density is the highest)

## 8 Posterior predictive checking 

In this section, for all 4 models evaluated in the previous sessions, we will compare predictive distribution versus actual distribution for a new tournament

### Comparision




```python
compare_predictive_vs_actual(fits=[pool_uni_fit, pool_inv_fit, hier_uni_fit, hier_inv_fit], labels=[
	'Pooled model with uniform prior',
	'Pooled model with inverse gamma prior',
	'Hierarchical model with uniform prior',
	'Hierarchical model with inverse gamma prior'],
     ypred_accessors= ['ypred', 'ypred', 'ypred_new', 'ypred_new'], new_data=last)
```


![png](output_34_0.png)


### Observation

Posterior predictive distribution are almost identical among all models. This can be guessed already in session 3, where the data can be seen to by highly consistent throughout different tournaments. 

Errors are considerable between prediction and actual distribution. Albeit the posterior of the estimand seems to follow a normal distribution, the amount of data in only one new tournament is too small to construct a normal distribution, hence the difference between prediction and actual values.

## 9 Conclusion


**Problems**
* Data model cannot be used for direct inference of a single match
* The values of the divergences for the hierarchical models were 0.025% instead of exact 0% (0.025% of the target posterior was not explored). Because the value is quite small, we decided to use the hierarchical models in the further analysis.

**Potential improvements**
* Given the outcome of this report, binomial can be a good fit as well
* Data model can be improved so that the estimand is a joint distribution of some parameters (e.g. absolute ranking + win degree)
* Data model can be modified to fit with multinomial model
* Prior and model analysis could be improved by adding the sensitivity analysis
 
**Discussion**

From the badminton domain perspective, the result is satisfiable:
* There is a visible correlation between ranking spread and win degree
* Probability of extreme outcome (towards 1 or 28) are low, and not expected in the tournament
 
From the statistical inference perspective, the result is also satisfiable
* Given the domain knowledge, one would expect the distribution of the estimand to be a normal distribution
* Given the found posteriors, we can see the result is highly data-driven
* Given two models, pooled and hierarchical, we can see that hierarchical model ends up as pooled model

___

# Source code


```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import pystan
import stan_utility
import psis
import warnings
# For hiding warnings that do not effect the functionality of the code
warnings.filterwarnings('ignore')
```


```python
# Data is provided as files, 1 file contains all matches result of a tournament
# We will use data from all tournaments to fit model
# Except for the last one which will be used to evaluate prediction accuracy
data = []
filenames = os.listdir(r'./data/')
for idx, filename in enumerate(filenames):
    col = np.loadtxt(f'data//{filename}').tolist()
    if (idx == (len(filenames) - 1)):
        last = col
    else:
        data.append(col[0:67])
np_data = np.array(data)

# Show the data in dataframe and change the names of the rows and columns to more descriable
def show_first_rows_of_data():
    df = pd.DataFrame(np_data.T)
    df.columns=['Tournament '+str(i+1) for i in range(np_data.shape[0])]
    df = df.rename({i: 'Match '+str(i+1) for i in range(np_data.shape[1])}, axis='index')
    return df.head()

# Show the data summary: mean, min, max, ....
def show_summary_of_data():
    df = pd.DataFrame(np_data.T)
    df.columns=['Tournament '+str(i+1) for i in range(np_data.shape[0])]
    return df.describe()

# Pooled data and its model for Stan compiler
pooled_data = np_data.flatten()
pooled_data_model = dict(N=len(pooled_data), y=pooled_data, alpha=1, beta=1)
#pooled_inv_g_data_model = dict(N=len(pooled_data), y=pooled_data)

# Hierarchical data (np_data.T) and its model for Stan compiler
hierarchical_data_model = dict(N = np_data.shape[1], J= np_data.shape[0],y = np_data.T, alpha=1, beta=1)

# Compile the given model and fit the given data.
# Parameters:
#     file_path: ''
#         The path of the stan model code
#     data: numpy array
#         The data to be fitted
#     adapt_delta: 0...1
#        Effects to divergences, hence to the accuracy of the posterior. 
#        The smaller the value is the more strict the Stan model is in accepting sampels.
#        The bigger the value is the easier the Stan model accepts samples. 
# Returns the summary of the fit and the fit itself.
def compute_model(file_path, data, adapt_delta=0.9):
    
    # Compile model for both separated and pooled
    model = stan_utility.compile_model(file_path) 

    # Fit model: adapt_delta is used for divergences
    fit = model.sampling(data=data, seed=194838,control=dict(adapt_delta=adapt_delta))

    # get summary of the fit, use pandas data frame for layout
    summary = fit.summary()
    df = pd.DataFrame(summary['summary'], index=summary['summary_rownames'], columns=summary['summary_colnames'])
    
    return df, fit

# Show compact details of the fit instead of showing the whole fit
def print_compact_fit(fit_df, number_of_rows_head=5, number_of_rows_tail=5):
    df = fit_df.head(number_of_rows_head)
    df = df.append([{'mean':'...','se_mean':'...','sd':'...','2.5%':'...','25%':'...',
                   '50%':'...','75%':'...','97.5%':'...','n_eff':'...','Rhat':'...'}])
    df = df.rename({0: '...'}, axis='index')
    df = df.append(fit_df.tail(number_of_rows_tail))
    return df

# Show key details of the checking of the fit: rhat and divergences
def print_compact_fit_checking(fit, df):
    #Check the maximum value of the Rhat
    print("Maximum value of the Rhat: ")
    print(df.describe()['Rhat'][7])
    print("")
    
    # Check divergences
    print("Divergences:")
    stan_utility.check_div(fit)
    print("")
    
# Compare PSIS-LOO values
def compare_psis_loo(fits, model_labels):
    psis_loos, p_effs, k_max, k_min, k_mean = [], [], [], [], []
    for fit in fits:
        psis_loo, p_eff, ks = extract_psis_loo(fit)
        psis_loos.append(psis_loo)
        p_effs.append(p_eff)
        k_max.append(np.max(ks))
        k_min.append(np.min(ks))
        k_mean.append(np.mean(ks))
    
    df = pd.DataFrame({
        'Models': model_labels,
        'Psisloo': psis_loos,
        'P_eff': p_effs,
        'Max k value': k_max,
        'Min k value': k_min,
        'Mean k value': k_mean
    })
    
    return df.round(2)

# Get the effective sample size of the parameters
def get_p_eff(log_lik, lppd_loocv):    
    likelihoods = np.asarray([np.exp(log_likelihood.flatten()) for log_likelihood in log_lik])
    num_sim, num_obs = likelihoods.shape
    lppd = 0
    for obs in range(num_obs):
        lppd += np.log(np.sum(likelihoods[:, obs]) / num_sim)
    
    p_eff = lppd - lppd_loocv
    return p_eff

def extract_psis_loo(samples, plot_title=''):
    log_lik_matrix = np.asarray([single_sample.flatten() for single_sample in samples['log_lik']])
    loo, loos, ks = psis.psisloo(log_lik_matrix)

    # Calculate p_eff
    p_eff = get_p_eff(log_lik_matrix, loo)

    return loo, p_eff, ks

def compare_predictive_vs_actual(fits, labels, ypred_accessors, new_data):
    bar_width = 0.3
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,figsize=(16,12), subplot_kw=dict(aspect='auto'))
    plots = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    
    # Group new_data 
    # e.g. [1,2,3,2,2] => [1: 0.2, 2: 0.6, 3: 0.2]
    aggregated = dict()
    for i in new_data:
        if i in aggregated:
            aggregated[i] += 1
        else:
            aggregated[i] = 1

    # Dist of new_data
    x = np.array(list(aggregated.keys()))
    y = np.array(list(aggregated.values())) / len(new_data)
    
    for i in range(0, len(fits)):
        fit = fits[i]
        label = labels[i]
        ypred_accessor = ypred_accessors[i]
        ypred = fit.extract(permuted=True)[ypred_accessor]
        
        # Dist of ypred for new_data values
        y2_dist = stats.norm(np.mean(ypred), np.std(ypred))
        y2 = y2_dist.pdf(x)
        
        partial_plt = plots[i]
        partial_plt.bar(x,y,width=bar_width,label="Actual", color="C1")
        partial_plt.bar(x-bar_width,y2,width=bar_width,label="Prediction", color="C0")
        partial_plt.set_title(label)
        partial_plt.legend()
    
    plt.show()

def show_attachments():
    print("##########################################################################")
    print("########## Attachment 1: Fit of pooled model with uniform prior ##########")
    print("##########################################################################")
    print(""); print(pool_uni_fit); print("");
    print("##########################################################################")
    print("####### Attachment 2: Fit of pooled model with inverse gamma prior #######")
    print("##########################################################################")
    print(""); print(pool_inv_fit); print("");
    print("##########################################################################")
    print("####### Attachment 3: Fit of hierarchical model with uniform prior #######")
    print("##########################################################################")
    print(""); print(hier_uni_fit); print("")
    print("##########################################################################")
    print("#### Attachment 4: Fit of hierarchical model with inverse gamma prior ####")
    print("##########################################################################")
    print(""); print(hier_inv_fit); print("");
    
#show_attachments()
```

    ##########################################################################
    ########## Attachment 1: Fit of pooled model with uniform prior ##########
    ##########################################################################
    
    Inference for Stan model: anon_model_9b8e95f23a292c6baefb5978c5223890.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
    
                   mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu            14.19  6.0e-3   0.31  13.56  13.98   14.2  14.41  14.82   2712    1.0
    sigma          5.66  4.4e-3   0.22   5.24   5.51   5.65   5.81   6.13   2577    1.0
    log_lik[0]    -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[1]    -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[2]     -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[3]    -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[4]    -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[5]    -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[6]    -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[7]    -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[8]    -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[9]    -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[10]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[11]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[12]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[13]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[14]   -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[15]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[16]   -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[17]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[18]    -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[19]   -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[20]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[21]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[22]   -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[23]   -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[24]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[25]   -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[26]   -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[27]   -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[28]   -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[29]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[30]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[31]   -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[32]   -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[33]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[34]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[35]   -4.48  2.8e-3   0.15  -4.79  -4.58  -4.48  -4.38  -4.21   2940    1.0
    log_lik[36]   -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[37]   -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[38]    -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[39]   -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[40]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[41]   -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[42]   -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[43]   -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[44]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[45]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[46]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[47]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[48]   -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[49]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[50]   -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[51]   -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[52]   -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[53]   -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[54]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[55]    -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[56]   -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[57]   -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[58]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[59]   -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[60]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[61]   -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[62]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[63]   -3.46  1.4e-3   0.07  -3.62  -3.51  -3.46  -3.41  -3.32   2738    1.0
    log_lik[64]    -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[65]   -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[66]   -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[67]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[68]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[69]   -3.46  1.4e-3   0.07  -3.62  -3.51  -3.46  -3.41  -3.32   2738    1.0
    log_lik[70]   -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[71]   -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[72]   -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[73]    -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[74]   -5.22  3.8e-3   0.21  -5.65  -5.36  -5.22  -5.08  -4.85   2963    1.0
    log_lik[75]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[76]   -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[77]   -2.65  7.8e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   2561    1.0
    log_lik[78]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[79]   -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[80]   -3.46  1.4e-3   0.07  -3.62  -3.51  -3.46  -3.41  -3.32   2738    1.0
    log_lik[81]   -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[82]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[83]   -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[84]   -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[85]   -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[86]   -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[87]   -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[88]   -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[89]   -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[90]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[91]   -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[92]   -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[93]   -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[94]   -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[95]   -2.65  7.8e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   2561    1.0
    log_lik[96]   -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[97]   -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[98]   -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[99]   -4.62  3.1e-3   0.16  -4.95  -4.72  -4.61  -4.51   -4.3   2548    1.0
    log_lik[100]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[101]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[102]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[103]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[104]  -4.48  2.8e-3   0.15  -4.79  -4.58  -4.48  -4.38  -4.21   2940    1.0
    log_lik[105]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[106]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[107]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[108]  -3.46  1.4e-3   0.07  -3.62  -3.51  -3.46  -3.41  -3.32   2738    1.0
    log_lik[109]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[110]  -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[111]  -2.65  7.8e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   2561    1.0
    log_lik[112]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[113]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[114]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[115]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[116]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[117]  -4.48  2.8e-3   0.15  -4.79  -4.58  -4.48  -4.38  -4.21   2940    1.0
    log_lik[118]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[119]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[120]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[121]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[122]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[123]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[124]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[125]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[126]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[127]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[128]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[129]  -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[130]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[131]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[132]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[133]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[134]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[135]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[136]  -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[137]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[138]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[139]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[140]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[141]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[142]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[143]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[144]  -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[145]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[146]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[147]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[148]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[149]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[150]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[151]  -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[152]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[153]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[154]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[155]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[156]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[157]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[158]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[159]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[160]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[161]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[162]  -3.87  1.9e-3    0.1  -4.08  -3.94  -3.86   -3.8  -3.68   2892    1.0
    log_lik[163]  -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[164]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[165]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[166]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[167]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[168]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[169]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[170]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[171]  -4.84  3.3e-3   0.18   -5.2  -4.96  -4.83  -4.71  -4.52   2953    1.0
    log_lik[172]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[173]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[174]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[175]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[176]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[177]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[178]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[179]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[180]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[181]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[182]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[183]  -3.87  1.9e-3    0.1  -4.08  -3.94  -3.86   -3.8  -3.68   2892    1.0
    log_lik[184]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[185]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[186]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[187]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[188]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[189]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[190]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[191]  -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[192]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[193]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[194]  -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[195]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[196]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[197]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[198]  -2.65  7.8e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   2561    1.0
    log_lik[199]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[200]  -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[201]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[202]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[203]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[204]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[205]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[206]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[207]  -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[208]  -4.84  3.3e-3   0.18   -5.2  -4.96  -4.83  -4.71  -4.52   2953    1.0
    log_lik[209]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[210]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[211]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[212]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[213]  -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[214]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[215]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[216]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[217]  -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[218]  -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[219]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[220]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[221]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[222]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[223]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[224]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[225]  -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[226]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[227]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[228]  -3.02  9.3e-4   0.05  -3.12  -3.05  -3.01  -2.98  -2.92   2698    1.0
    log_lik[229]  -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[230]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[231]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[232]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[233]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[234]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[235]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[236]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[237]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[238]  -5.22  3.8e-3   0.21  -5.65  -5.36  -5.22  -5.08  -4.85   2963    1.0
    log_lik[239]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[240]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[241]  -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[242]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[243]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[244]  -4.84  3.3e-3   0.18   -5.2  -4.96  -4.83  -4.71  -4.52   2953    1.0
    log_lik[245]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[246]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[247]  -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[248]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[249]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[250]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[251]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[252]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[253]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[254]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[255]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[256]  -4.62  3.1e-3   0.16  -4.95  -4.72  -4.61  -4.51   -4.3   2548    1.0
    log_lik[257]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[258]  -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[259]  -4.16  2.3e-3   0.13  -4.42  -4.24  -4.15  -4.07  -3.94   2920    1.0
    log_lik[260]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[261]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[262]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[263]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[264]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[265]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[266]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[267]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[268]  -5.38  4.3e-3   0.22  -5.83  -5.52  -5.37  -5.23  -4.96   2533    1.0
    log_lik[269]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[270]  -3.98  2.2e-3   0.11  -4.21  -4.05  -3.97   -3.9  -3.76   2595    1.0
    log_lik[271]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[272]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[273]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[274]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[275]  -3.87  1.9e-3    0.1  -4.08  -3.94  -3.86   -3.8  -3.68   2892    1.0
    log_lik[276]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[277]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[278]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[279]  -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[280]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[281]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[282]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[283]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[284]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[285]  -3.18  1.1e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2746    1.0
    log_lik[286]  -5.22  3.8e-3   0.21  -5.65  -5.36  -5.22  -5.08  -4.85   2963    1.0
    log_lik[287]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[288]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[289]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[290]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[291]  -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[292]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[293]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[294]  -4.62  3.1e-3   0.16  -4.95  -4.72  -4.61  -4.51   -4.3   2548    1.0
    log_lik[295]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[296]  -3.08  9.7e-4   0.05  -3.18  -3.11  -3.08  -3.04  -2.98   2793    1.0
    log_lik[297]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[298]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[299]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[300]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[301]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[302]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[303]  -4.28  2.6e-3   0.13  -4.56  -4.37  -4.28  -4.19  -4.02   2565    1.0
    log_lik[304]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[305]  -2.68  7.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   2661    1.0
    log_lik[306]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[307]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[308]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[309]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[310]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[311]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[312]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[313]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[314]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[315]  -2.88  8.5e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2565    1.0
    log_lik[316]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[317]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[318]  -2.66  7.8e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   2547    1.0
    log_lik[319]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[320]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[321]  -3.71  1.8e-3   0.09  -3.89  -3.77   -3.7  -3.64  -3.53   2641    1.0
    log_lik[322]  -2.65  7.8e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   2561    1.0
    log_lik[323]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[324]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[325]  -2.73  7.4e-4   0.04  -2.81  -2.76  -2.73   -2.7  -2.65   2942    1.0
    log_lik[326]  -3.25  1.2e-3   0.06  -3.38   -3.3  -3.25  -3.21  -3.14   2754    1.0
    log_lik[327]  -2.93  8.4e-4   0.05  -3.02  -2.96  -2.93   -2.9  -2.84   2853    1.0
    log_lik[328]  -2.78  8.1e-4   0.04  -2.86   -2.8  -2.78  -2.75   -2.7   2484    1.0
    log_lik[329]  -4.84  3.3e-3   0.18   -5.2  -4.96  -4.83  -4.71  -4.52   2953    1.0
    log_lik[330]  -3.38  1.3e-3   0.07  -3.52  -3.43  -3.38  -3.33  -3.25   2803    1.0
    log_lik[331]  -3.61  1.6e-3   0.08  -3.78  -3.66   -3.6  -3.55  -3.46   2853    1.0
    log_lik[332]   -2.7  7.9e-4   0.04  -2.79  -2.73   -2.7  -2.68  -2.63   2483    1.0
    log_lik[333]  -2.81  7.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.73   2911    1.0
    log_lik[334]  -3.46  1.4e-3   0.07  -3.62  -3.51  -3.46  -3.41  -3.32   2738    1.0
    ypred         14.29    0.09    5.7   3.43  10.36  14.28   18.2   25.6   4000    1.0
    lp__         -746.0    0.02   1.01 -748.7 -746.4 -745.7 -745.3 -745.0   1637    1.0
    
    Samples were drawn using NUTS at Sun Dec  9 13:47:05 2018.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).
    
    ##########################################################################
    ####### Attachment 2: Fit of pooled model with inverse gamma prior #######
    ##########################################################################
    
    Inference for Stan model: anon_model_fa6169eb72725ff16af37d26935097d5.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
    
                   mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu            14.19  6.6e-3   0.31  13.58  13.98  14.19   14.4  14.81   2228    1.0
    sigmaSq       31.82    0.05   2.43   27.5   30.1  31.69  33.37  36.82   2719    1.0
    sigma          5.64  4.1e-3   0.21   5.24   5.49   5.63   5.78   6.07   2741    1.0
    log_lik[0]    -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[1]    -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[2]     -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[3]    -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[4]    -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[5]    -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[6]    -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[7]    -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[8]    -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[9]    -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[10]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[11]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[12]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[13]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[14]   -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[15]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[16]   -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[17]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[18]    -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[19]   -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[20]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[21]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[22]   -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[23]   -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[24]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[25]   -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[26]   -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[27]   -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[28]   -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[29]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[30]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[31]   -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[32]   -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[33]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[34]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[35]    -4.5  3.1e-3   0.14  -4.79  -4.59  -4.49   -4.4  -4.23   2112    1.0
    log_lik[36]   -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[37]   -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[38]    -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[39]   -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[40]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[41]   -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[42]   -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[43]   -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[44]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[45]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[46]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[47]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[48]   -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[49]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[50]   -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[51]   -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[52]   -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[53]   -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[54]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[55]    -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[56]   -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[57]   -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[58]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[59]   -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[60]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[61]   -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[62]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[63]   -3.47  1.6e-3   0.08  -3.62  -3.51  -3.46  -3.41  -3.33   2351    1.0
    log_lik[64]    -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[65]   -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[66]   -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[67]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[68]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[69]   -3.47  1.6e-3   0.08  -3.62  -3.51  -3.46  -3.41  -3.33   2351    1.0
    log_lik[70]   -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[71]   -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[72]   -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[73]    -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[74]   -5.24  4.2e-3    0.2  -5.63  -5.37  -5.24  -5.11  -4.88   2161    1.0
    log_lik[75]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[76]   -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[77]   -2.65  7.3e-4   0.04  -2.73  -2.67  -2.65  -2.62  -2.58   2690    1.0
    log_lik[78]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[79]   -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[80]   -3.47  1.6e-3   0.08  -3.62  -3.51  -3.46  -3.41  -3.33   2351    1.0
    log_lik[81]   -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[82]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[83]   -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[84]   -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[85]   -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[86]   -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[87]   -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[88]   -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[89]   -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[90]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[91]   -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[92]   -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[93]   -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[94]   -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[95]   -2.65  7.3e-4   0.04  -2.73  -2.67  -2.65  -2.62  -2.58   2690    1.0
    log_lik[96]   -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[97]   -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[98]   -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[99]   -4.63  3.2e-3   0.16  -4.96  -4.73  -4.62  -4.52  -4.33   2624    1.0
    log_lik[100]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[101]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[102]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[103]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[104]   -4.5  3.1e-3   0.14  -4.79  -4.59  -4.49   -4.4  -4.23   2112    1.0
    log_lik[105]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[106]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[107]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[108]  -3.47  1.6e-3   0.08  -3.62  -3.51  -3.46  -3.41  -3.33   2351    1.0
    log_lik[109]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[110]  -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[111]  -2.65  7.3e-4   0.04  -2.73  -2.67  -2.65  -2.62  -2.58   2690    1.0
    log_lik[112]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[113]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[114]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[115]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[116]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[117]   -4.5  3.1e-3   0.14  -4.79  -4.59  -4.49   -4.4  -4.23   2112    1.0
    log_lik[118]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[119]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[120]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[121]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[122]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[123]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[124]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[125]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[126]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[127]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[128]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[129]  -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[130]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[131]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[132]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[133]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[134]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[135]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[136]  -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[137]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[138]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[139]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[140]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[141]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[142]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[143]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[144]  -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[145]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[146]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[147]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[148]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[149]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[150]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[151]  -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[152]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[153]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[154]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[155]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[156]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[157]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[158]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[159]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[160]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[161]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[162]  -3.88  2.2e-3    0.1  -4.08  -3.94  -3.87  -3.81  -3.69   2077    1.0
    log_lik[163]  -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[164]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[165]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[166]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[167]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[168]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[169]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[170]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[171]  -4.85  3.6e-3   0.17  -5.19  -4.96  -4.85  -4.74  -4.54   2136    1.0
    log_lik[172]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[173]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[174]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[175]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[176]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[177]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[178]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[179]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[180]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[181]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[182]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[183]  -3.88  2.2e-3    0.1  -4.08  -3.94  -3.87  -3.81  -3.69   2077    1.0
    log_lik[184]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[185]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[186]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[187]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[188]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[189]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[190]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[191]  -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[192]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[193]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[194]  -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[195]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[196]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[197]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[198]  -2.65  7.3e-4   0.04  -2.73  -2.67  -2.65  -2.62  -2.58   2690    1.0
    log_lik[199]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[200]  -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[201]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[202]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[203]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[204]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[205]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[206]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[207]  -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[208]  -4.85  3.6e-3   0.17  -5.19  -4.96  -4.85  -4.74  -4.54   2136    1.0
    log_lik[209]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[210]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[211]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[212]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[213]  -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[214]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[215]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[216]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[217]  -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[218]  -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[219]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[220]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[221]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[222]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[223]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[224]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[225]  -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[226]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[227]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[228]  -3.01  1.0e-3   0.05  -3.11  -3.05  -3.01  -2.98  -2.92   2272    1.0
    log_lik[229]  -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[230]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[231]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[232]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[233]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[234]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[235]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[236]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[237]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[238]  -5.24  4.2e-3    0.2  -5.63  -5.37  -5.24  -5.11  -4.88   2161    1.0
    log_lik[239]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[240]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[241]  -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[242]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[243]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[244]  -4.85  3.6e-3   0.17  -5.19  -4.96  -4.85  -4.74  -4.54   2136    1.0
    log_lik[245]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[246]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[247]  -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[248]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[249]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[250]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[251]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[252]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[253]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[254]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[255]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[256]  -4.63  3.2e-3   0.16  -4.96  -4.73  -4.62  -4.52  -4.33   2624    1.0
    log_lik[257]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[258]  -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[259]  -4.17  2.6e-3   0.12  -4.41  -4.25  -4.16  -4.09  -3.95   2091    1.0
    log_lik[260]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[261]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[262]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[263]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[264]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[265]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[266]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[267]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[268]   -5.4  4.2e-3   0.22  -5.85  -5.55  -5.39  -5.25  -4.99   2694    1.0
    log_lik[269]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[270]  -3.98  2.3e-3   0.11  -4.22  -4.06  -3.98  -3.91  -3.78   2513    1.0
    log_lik[271]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[272]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[273]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[274]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[275]  -3.88  2.2e-3    0.1  -4.08  -3.94  -3.87  -3.81  -3.69   2077    1.0
    log_lik[276]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[277]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[278]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[279]  -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[280]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[281]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[282]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[283]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[284]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[285]  -3.18  1.2e-3   0.06   -3.3  -3.22  -3.18  -3.14  -3.07   2205    1.0
    log_lik[286]  -5.24  4.2e-3    0.2  -5.63  -5.37  -5.24  -5.11  -4.88   2161    1.0
    log_lik[287]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[288]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[289]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[290]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[291]  -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[292]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[293]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[294]  -4.63  3.2e-3   0.16  -4.96  -4.73  -4.62  -4.52  -4.33   2624    1.0
    log_lik[295]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[296]  -3.08  1.1e-3   0.05  -3.18  -3.11  -3.07  -3.04  -2.98   2186    1.0
    log_lik[297]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[298]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[299]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[300]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[301]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[302]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[303]  -4.29  2.7e-3   0.14  -4.57  -4.38  -4.28   -4.2  -4.04   2574    1.0
    log_lik[304]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[305]  -2.67  7.8e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   2296    1.0
    log_lik[306]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[307]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[308]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[309]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[310]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[311]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[312]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[313]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[314]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[315]  -2.88  8.9e-4   0.04  -2.97  -2.91  -2.88  -2.85   -2.8   2399    1.0
    log_lik[316]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[317]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[318]  -2.66  7.4e-4   0.04  -2.74  -2.68  -2.66  -2.63  -2.59   2733    1.0
    log_lik[319]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[320]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[321]  -3.71  1.9e-3   0.09   -3.9  -3.77  -3.71  -3.65  -3.54   2438    1.0
    log_lik[322]  -2.65  7.3e-4   0.04  -2.73  -2.67  -2.65  -2.62  -2.58   2690    1.0
    log_lik[323]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[324]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[325]  -2.73  8.1e-4   0.04   -2.8  -2.75  -2.72   -2.7  -2.65   2145    1.0
    log_lik[326]  -3.26  1.3e-3   0.06  -3.38  -3.29  -3.25  -3.21  -3.14   2263    1.0
    log_lik[327]  -2.93  9.6e-4   0.04  -3.02  -2.95  -2.93   -2.9  -2.85   2055    1.0
    log_lik[328]  -2.77  8.1e-4   0.04  -2.86   -2.8  -2.77  -2.75   -2.7   2555    1.0
    log_lik[329]  -4.85  3.6e-3   0.17  -5.19  -4.96  -4.85  -4.74  -4.54   2136    1.0
    log_lik[330]  -3.38  1.5e-3   0.07  -3.52  -3.42  -3.38  -3.34  -3.25   2112    1.0
    log_lik[331]  -3.61  1.8e-3   0.08  -3.77  -3.67  -3.61  -3.56  -3.46   2080    1.0
    log_lik[332]   -2.7  7.6e-4   0.04  -2.78  -2.73   -2.7  -2.67  -2.63   2683    1.0
    log_lik[333]  -2.81  8.7e-4   0.04  -2.89  -2.84  -2.81  -2.78  -2.74   2054    1.0
    log_lik[334]  -3.47  1.6e-3   0.08  -3.62  -3.51  -3.46  -3.41  -3.33   2351    1.0
    ypred         14.26    0.09   5.61   3.24  10.48  14.33   18.0  25.22   3898    1.0
    lp__         -751.2    0.03   1.01 -753.9 -751.5 -750.9 -750.4 -750.2   1496    1.0
    
    Samples were drawn using NUTS at Sun Dec  9 13:47:30 2018.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).
    
    ##########################################################################
    ####### Attachment 3: Fit of hierarchical model with uniform prior #######
    ##########################################################################
    
    Inference for Stan model: anon_model_c4c17a1f535ecd44756a69898a3750cd.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
    
                    mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu0             14.2    0.01   0.46  13.25  13.92  14.19  14.46   15.1   1045    1.0
    sigma0          0.55    0.02   0.58   0.01   0.18   0.39   0.72   2.06   1027    1.0
    sigma           5.67  3.4e-3   0.21   5.27   5.52   5.66   5.81   6.11   4000    1.0
    mu_tilde[0]    -0.21    0.02   0.88  -1.97   -0.8  -0.22   0.37   1.49   3064    1.0
    mu_tilde[1]     0.11    0.02   0.88  -1.68  -0.48    0.1   0.69   1.88   3285    1.0
    mu_tilde[2]    -0.21    0.02   0.88  -1.89   -0.8  -0.22   0.37   1.52   2993    1.0
    mu_tilde[3]     0.33    0.02   0.85  -1.43   -0.2   0.35    0.9   1.95   2605    1.0
    mu_tilde[4]    -0.03    0.01   0.86  -1.74   -0.6  -0.02   0.53   1.71   3305    1.0
    mu[0]          14.06  7.6e-3   0.48  12.98  13.79  14.09  14.37  14.95   4000    1.0
    mu[1]          14.25  7.4e-3   0.47  13.31  13.95  14.25  14.55  15.23   4000    1.0
    mu[2]          14.06  7.6e-3   0.48  13.03  13.77  14.08  14.38  14.95   4000    1.0
    mu[3]          14.41  7.8e-3    0.5  13.54  14.06  14.36   14.7   15.5   4000    1.0
    mu[4]          14.18  7.5e-3   0.47  13.22  13.89  14.19  14.47  15.12   4000    1.0
    log_lik[0,0]   -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[1,0]   -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[2,0]   -2.72  7.2e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[3,0]   -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[4,0]   -3.04  1.2e-3   0.08  -3.22  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[5,0]   -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[6,0]   -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[7,0]   -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[8,0]   -3.94  2.3e-3   0.15  -4.22  -4.03  -3.94  -3.85  -3.63   4000    1.0
    log_lik[9,0]   -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[10,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[11,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[12,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[13,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[14,0]  -3.94  2.3e-3   0.15  -4.22  -4.03  -3.94  -3.85  -3.63   4000    1.0
    log_lik[15,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[16,0]   -2.8  8.2e-4   0.05  -2.91  -2.84   -2.8  -2.77   -2.7   4000    1.0
    log_lik[17,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[18,0]  -2.72  7.2e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[19,0]  -3.41  1.7e-3   0.11  -3.66  -3.47   -3.4  -3.34  -3.22   4000    1.0
    log_lik[20,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[21,0]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[22,0]   -4.2  2.7e-3   0.17  -4.59   -4.3  -4.19  -4.09   -3.9   4000    1.0
    log_lik[23,0]  -2.91  9.9e-4   0.06  -3.04  -2.95  -2.92  -2.88  -2.78   4000    1.0
    log_lik[24,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[25,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.18  -3.04   4000    1.0
    log_lik[26,0]  -3.64  2.0e-3   0.13  -3.93  -3.71  -3.63  -3.56  -3.42   4000    1.0
    log_lik[27,0]  -2.72  7.0e-4   0.04  -2.81  -2.75  -2.72  -2.69  -2.64   4000    1.0
    log_lik[28,0]  -3.04  1.2e-3   0.08  -3.22  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[29,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[30,0]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[31,0]  -2.72  7.0e-4   0.04  -2.81  -2.75  -2.72  -2.69  -2.64   4000    1.0
    log_lik[32,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.18  -3.04   4000    1.0
    log_lik[33,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[34,0]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[35,0]  -4.53  3.1e-3   0.19  -4.96  -4.64  -4.51   -4.4  -4.18   4000    1.0
    log_lik[36,0]  -3.04  1.2e-3   0.08  -3.22  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[37,0]   -4.2  2.7e-3   0.17  -4.59   -4.3  -4.19  -4.09   -3.9   4000    1.0
    log_lik[38,0]  -2.72  7.2e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[39,0]  -3.94  2.3e-3   0.15  -4.22  -4.03  -3.94  -3.85  -3.63   4000    1.0
    log_lik[40,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[41,0]  -3.21  1.4e-3   0.09  -3.42  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[42,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.18  -3.04   4000    1.0
    log_lik[43,0]  -3.64  2.0e-3   0.13  -3.93  -3.71  -3.63  -3.56  -3.42   4000    1.0
    log_lik[44,0]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[45,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[46,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[47,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[48,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.18  -3.04   4000    1.0
    log_lik[49,0]  -2.79  8.5e-4   0.05  -2.92  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[50,0]  -3.41  1.7e-3   0.11  -3.66  -3.47   -3.4  -3.34  -3.22   4000    1.0
    log_lik[51,0]  -3.04  1.2e-3   0.08  -3.22  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[52,0]  -3.64  2.0e-3   0.13  -3.93  -3.71  -3.63  -3.56  -3.42   4000    1.0
    log_lik[53,0]  -4.24  2.7e-3   0.17  -4.56  -4.35  -4.24  -4.13  -3.88   4000    1.0
    log_lik[54,0]  -3.06  1.2e-3   0.08  -3.21   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[55,0]  -2.72  7.2e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[56,0]   -2.9  1.0e-3   0.06  -3.05  -2.93  -2.89  -2.86  -2.79   4000    1.0
    log_lik[57,0]  -2.72  7.0e-4   0.04  -2.81  -2.75  -2.72  -2.69  -2.64   4000    1.0
    log_lik[58,0]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64   -2.6   4000    1.0
    log_lik[59,0]  -2.91  9.9e-4   0.06  -3.04  -2.95  -2.92  -2.88  -2.78   4000    1.0
    log_lik[60,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[61,0]  -2.91  9.9e-4   0.06  -3.04  -2.95  -2.92  -2.88  -2.78   4000    1.0
    log_lik[62,0]  -3.67  2.0e-3   0.13  -3.91  -3.75  -3.67   -3.6  -3.41   4000    1.0
    log_lik[63,0]  -3.44  1.7e-3   0.11  -3.64   -3.5  -3.44  -3.37  -3.21   4000    1.0
    log_lik[64,0]  -2.72  7.2e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[65,0]  -3.21  1.4e-3   0.09  -3.42  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[66,0]  -3.64  2.0e-3   0.13  -3.93  -3.71  -3.63  -3.56  -3.42   4000    1.0
    log_lik[0,1]   -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[1,1]   -2.67  6.2e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[2,1]   -3.48  1.7e-3   0.11  -3.72  -3.54  -3.47  -3.41  -3.27   4000    1.0
    log_lik[3,1]   -2.68  6.4e-4   0.04  -2.76  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[4,1]   -3.17  1.3e-3   0.08  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[5,1]   -3.17  1.3e-3   0.08  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[6,1]    -2.7  6.8e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[7,1]    -5.2  3.7e-3   0.24  -5.68  -5.35  -5.19  -5.04  -4.74   4000    1.0
    log_lik[8,1]   -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[9,1]   -2.74  7.2e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.65   4000    1.0
    log_lik[10,1]  -2.66  6.0e-4   0.04  -2.73  -2.68  -2.66  -2.63  -2.58   4000    1.0
    log_lik[11,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[12,1]  -3.37  1.6e-3    0.1  -3.57  -3.43  -3.36   -3.3  -3.18   4000    1.0
    log_lik[13,1]  -3.48  1.7e-3   0.11  -3.72  -3.54  -3.47  -3.41  -3.27   4000    1.0
    log_lik[14,1]  -2.82  8.4e-4   0.05  -2.94  -2.85  -2.82  -2.78  -2.73   4000    1.0
    log_lik[15,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[16,1]  -2.68  6.4e-4   0.04  -2.76  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[17,1]  -3.27  1.5e-3   0.09  -3.47  -3.32  -3.26  -3.21   -3.1   4000    1.0
    log_lik[18,1]  -2.74  7.2e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.65   4000    1.0
    log_lik[19,1]  -3.01  1.1e-3   0.07  -3.16  -3.05  -3.01  -2.96  -2.88   4000    1.0
    log_lik[20,1]  -3.17  1.3e-3   0.08  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[21,1]  -2.88  9.3e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.77   4000    1.0
    log_lik[22,1]  -4.14  2.5e-3   0.16  -4.46  -4.24  -4.14  -4.04  -3.84   4000    1.0
    log_lik[23,1]  -3.09  1.2e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.95   4000    1.0
    log_lik[24,1]  -2.68  6.4e-4   0.04  -2.76  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[25,1]  -2.88  9.3e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.77   4000    1.0
    log_lik[26,1]  -3.37  1.6e-3    0.1  -3.57  -3.43  -3.36   -3.3  -3.18   4000    1.0
    log_lik[27,1]  -2.67  6.2e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[28,1]  -2.66  6.0e-4   0.04  -2.73  -2.68  -2.66  -2.63  -2.58   4000    1.0
    log_lik[29,1]  -3.09  1.2e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.95   4000    1.0
    log_lik[30,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[31,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[32,1]  -4.64  3.2e-3    0.2  -5.05  -4.76  -4.63   -4.5  -4.26   4000    1.0
    log_lik[33,1]  -2.67  6.2e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[34,1]   -2.7  6.8e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[35,1]  -3.17  1.3e-3   0.08  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[36,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[37,1]  -4.46  2.9e-3   0.18  -4.83  -4.58  -4.46  -4.34  -4.11   4000    1.0
    log_lik[38,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[39,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[40,1]   -2.7  6.8e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[41,1]  -3.48  1.7e-3   0.11  -3.72  -3.54  -3.47  -3.41  -3.27   4000    1.0
    log_lik[42,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[43,1]  -3.09  1.2e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.95   4000    1.0
    log_lik[44,1]  -2.66  6.0e-4   0.04  -2.73  -2.68  -2.66  -2.63  -2.58   4000    1.0
    log_lik[45,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[46,1]  -2.68  6.4e-4   0.04  -2.76  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[47,1]  -3.27  1.5e-3   0.09  -3.47  -3.32  -3.26  -3.21   -3.1   4000    1.0
    log_lik[48,1]  -2.94  1.0e-3   0.06  -3.08  -2.98  -2.94  -2.89  -2.82   4000    1.0
    log_lik[49,1]  -2.88  9.3e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.77   4000    1.0
    log_lik[50,1]  -4.46  2.9e-3   0.18  -4.83  -4.58  -4.46  -4.34  -4.11   4000    1.0
    log_lik[51,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[52,1]  -3.59  1.8e-3   0.12  -3.83  -3.67  -3.59  -3.52  -3.37   4000    1.0
    log_lik[53,1]  -2.94  1.0e-3   0.06  -3.08  -2.98  -2.94  -2.89  -2.82   4000    1.0
    log_lik[54,1]  -2.67  6.2e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[55,1]  -3.72  2.0e-3   0.13  -3.99   -3.8  -3.72  -3.63  -3.48   4000    1.0
    log_lik[56,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[57,1]  -2.82  8.4e-4   0.05  -2.94  -2.85  -2.82  -2.78  -2.73   4000    1.0
    log_lik[58,1]  -3.59  1.8e-3   0.12  -3.83  -3.67  -3.59  -3.52  -3.37   4000    1.0
    log_lik[59,1]  -2.88  9.3e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.77   4000    1.0
    log_lik[60,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[61,1]  -2.68  6.4e-4   0.04  -2.76  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[62,1]   -4.3  2.8e-3   0.17  -4.66  -4.41  -4.29  -4.18  -3.98   4000    1.0
    log_lik[63,1]  -2.94  1.0e-3   0.06  -3.08  -2.98  -2.94  -2.89  -2.82   4000    1.0
    log_lik[64,1]  -2.88  9.3e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.77   4000    1.0
    log_lik[65,1]  -3.17  1.3e-3   0.08  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[66,1]  -2.77  7.9e-4   0.05  -2.88  -2.81  -2.77  -2.74  -2.68   4000    1.0
    log_lik[0,2]   -2.72  7.1e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[1,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[2,2]   -4.24  2.7e-3   0.17  -4.58  -4.35  -4.23  -4.13   -3.9   4000    1.0
    log_lik[3,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[4,2]   -3.21  1.4e-3   0.09  -3.41  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[5,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[6,2]   -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[7,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[8,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[9,2]   -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[10,2]  -3.94  2.3e-3   0.15  -4.24  -4.04  -3.94  -3.85  -3.65   4000    1.0
    log_lik[11,2]  -2.72  7.1e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[12,2]  -2.72  7.1e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[13,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[14,2]  -2.72  7.1e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[15,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[16,2]   -2.8  8.3e-4   0.05  -2.91  -2.84   -2.8  -2.77   -2.7   4000    1.0
    log_lik[17,2]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01  -2.91   4000    1.0
    log_lik[18,2]  -2.91  9.9e-4   0.06  -3.04  -2.96  -2.92  -2.87  -2.79   4000    1.0
    log_lik[19,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[20,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[21,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[22,2]  -3.64  2.0e-3   0.13  -3.91  -3.72  -3.63  -3.56  -3.43   4000    1.0
    log_lik[23,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[24,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[25,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[26,2]  -3.64  2.0e-3   0.13  -3.91  -3.72  -3.63  -3.56  -3.43   4000    1.0
    log_lik[27,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[28,2]  -3.91  2.3e-3   0.15  -4.22  -3.99  -3.89  -3.81  -3.65   4000    1.0
    log_lik[29,2]  -3.04  1.2e-3   0.08  -3.21  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[30,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[31,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[32,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[33,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[34,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[35,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[36,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[37,2]  -4.88  3.5e-3   0.22  -5.36  -5.02  -4.87  -4.73  -4.49   4000    1.0
    log_lik[38,2]  -2.67  6.2e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[39,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[40,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[41,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[42,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[43,2]  -2.67  6.3e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[44,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[45,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[46,2]  -2.91  9.9e-4   0.06  -3.04  -2.96  -2.92  -2.87  -2.79   4000    1.0
    log_lik[47,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[48,2]   -2.8  8.3e-4   0.05  -2.91  -2.84   -2.8  -2.77   -2.7   4000    1.0
    log_lik[49,2]  -3.91  2.3e-3   0.15  -4.22  -3.99  -3.89  -3.81  -3.65   4000    1.0
    log_lik[50,2]  -3.21  1.4e-3   0.09  -3.41  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[51,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[52,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.42   4000    1.0
    log_lik[53,2]  -2.91  9.9e-4   0.06  -3.04  -2.96  -2.92  -2.87  -2.79   4000    1.0
    log_lik[54,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[55,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[56,2]  -2.72  7.1e-4   0.05  -2.81  -2.74  -2.71  -2.69  -2.63   4000    1.0
    log_lik[57,2]   -4.2  2.7e-3   0.17  -4.57  -4.31  -4.18  -4.09  -3.91   4000    1.0
    log_lik[58,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[59,2]  -2.79  8.4e-4   0.05  -2.91  -2.82  -2.79  -2.76   -2.7   4000    1.0
    log_lik[60,2]  -2.72  7.0e-4   0.04  -2.81  -2.75  -2.72  -2.69  -2.64   4000    1.0
    log_lik[61,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[62,2]  -2.67  6.2e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[63,2]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[64,2]  -2.66  6.0e-4   0.04  -2.73  -2.68  -2.66  -2.63  -2.58   4000    1.0
    log_lik[65,2]  -2.67  6.2e-4   0.04  -2.75   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[66,2]  -3.94  2.3e-3   0.15  -4.24  -4.04  -3.94  -3.85  -3.65   4000    1.0
    log_lik[0,3]   -3.76  2.2e-3   0.14  -4.09  -3.84  -3.75  -3.66  -3.53   4000    1.0
    log_lik[1,3]   -2.69  6.7e-4   0.04  -2.78  -2.72  -2.69  -2.66  -2.61   4000    1.0
    log_lik[2,3]   -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[3,3]   -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[4,3]   -2.66  6.1e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[5,3]   -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[6,3]   -4.09  2.5e-3   0.16   -4.4   -4.2   -4.1  -3.99  -3.77   4000    1.0
    log_lik[7,3]   -4.76  3.3e-3   0.21  -5.16   -4.9  -4.76  -4.62  -4.34   4000    1.0
    log_lik[8,3]   -2.96  1.1e-3   0.07  -3.13   -3.0  -2.95  -2.91  -2.85   4000    1.0
    log_lik[9,3]   -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[10,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[11,3]   -2.7  6.7e-4   0.04  -2.78  -2.72   -2.7  -2.67  -2.61   4000    1.0
    log_lik[12,3]  -4.04  2.6e-3   0.16  -4.41  -4.13  -4.02  -3.93  -3.77   4000    1.0
    log_lik[13,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[14,3]  -3.76  2.2e-3   0.14  -4.09  -3.84  -3.75  -3.66  -3.53   4000    1.0
    log_lik[15,3]  -2.66  6.1e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[16,3]  -3.11  1.4e-3   0.09  -3.31  -3.16   -3.1  -3.05  -2.98   4000    1.0
    log_lik[17,3]  -4.35  3.0e-3   0.19  -4.77  -4.46  -4.33  -4.22  -4.03   4000    1.0
    log_lik[18,3]  -2.96  1.1e-3   0.07  -3.13   -3.0  -2.95  -2.91  -2.85   4000    1.0
    log_lik[19,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[20,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[21,3]  -2.66  6.1e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[22,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[23,3]   -2.7  6.7e-4   0.04  -2.78  -2.72   -2.7  -2.67  -2.61   4000    1.0
    log_lik[24,3]  -3.34  1.6e-3    0.1  -3.52   -3.4  -3.34  -3.27  -3.13   4000    1.0
    log_lik[25,3]  -2.96  1.1e-3   0.07  -3.13   -3.0  -2.95  -2.91  -2.85   4000    1.0
    log_lik[26,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[27,3]  -2.99  1.1e-3   0.07  -3.12  -3.03  -2.99  -2.94  -2.84   4000    1.0
    log_lik[28,3]  -3.34  1.6e-3    0.1  -3.52   -3.4  -3.34  -3.27  -3.13   4000    1.0
    log_lik[29,3]  -2.66  6.1e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[30,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[31,3]   -2.7  6.7e-4   0.04  -2.78  -2.72   -2.7  -2.67  -2.61   4000    1.0
    log_lik[32,3]   -3.3  1.6e-3    0.1  -3.53  -3.35  -3.29  -3.23  -3.13   4000    1.0
    log_lik[33,3]  -2.96  1.1e-3   0.07  -3.13   -3.0  -2.95  -2.91  -2.85   4000    1.0
    log_lik[34,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[35,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[36,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[37,3]  -5.13  3.8e-3   0.24   -5.6   -5.3  -5.14  -4.98  -4.67   4000    1.0
    log_lik[38,3]  -3.76  2.2e-3   0.14  -4.09  -3.84  -3.75  -3.66  -3.53   4000    1.0
    log_lik[39,3]  -2.84  9.4e-4   0.06  -2.98  -2.87  -2.83   -2.8  -2.74   4000    1.0
    log_lik[40,3]  -3.11  1.4e-3   0.09  -3.31  -3.16   -3.1  -3.05  -2.98   4000    1.0
    log_lik[41,3]  -2.96  1.1e-3   0.07  -3.13   -3.0  -2.95  -2.91  -2.85   4000    1.0
    log_lik[42,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[43,3]  -4.76  3.3e-3   0.21  -5.16   -4.9  -4.76  -4.62  -4.34   4000    1.0
    log_lik[44,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[45,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[46,3]  -2.75  7.8e-4   0.05  -2.86  -2.78  -2.74  -2.71  -2.66   4000    1.0
    log_lik[47,3]  -2.84  9.4e-4   0.06  -2.98  -2.87  -2.83   -2.8  -2.74   4000    1.0
    log_lik[48,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[49,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[50,3]   -3.3  1.6e-3    0.1  -3.53  -3.35  -3.29  -3.23  -3.13   4000    1.0
    log_lik[51,3]   -2.7  6.7e-4   0.04  -2.78  -2.72   -2.7  -2.67  -2.61   4000    1.0
    log_lik[52,3]   -3.3  1.6e-3    0.1  -3.53  -3.35  -3.29  -3.23  -3.13   4000    1.0
    log_lik[53,3]  -3.76  2.2e-3   0.14  -4.09  -3.84  -3.75  -3.66  -3.53   4000    1.0
    log_lik[54,3]  -2.84  9.4e-4   0.06  -2.98  -2.87  -2.83   -2.8  -2.74   4000    1.0
    log_lik[55,3]  -4.69  3.4e-3   0.22  -5.18  -4.82  -4.67  -4.54  -4.32   4000    1.0
    log_lik[56,3]   -3.3  1.6e-3    0.1  -3.53  -3.35  -3.29  -3.23  -3.13   4000    1.0
    log_lik[57,3]  -3.34  1.6e-3    0.1  -3.52   -3.4  -3.34  -3.27  -3.13   4000    1.0
    log_lik[58,3]  -4.09  2.5e-3   0.16   -4.4   -4.2   -4.1  -3.99  -3.77   4000    1.0
    log_lik[59,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[60,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[61,3]  -2.84  9.4e-4   0.06  -2.98  -2.87  -2.83   -2.8  -2.74   4000    1.0
    log_lik[62,3]  -2.76  7.8e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[63,3]  -2.66  6.1e-4   0.04  -2.74  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[64,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[65,3]  -3.76  2.2e-3   0.14  -4.09  -3.84  -3.75  -3.66  -3.53   4000    1.0
    log_lik[66,3]  -3.56  1.9e-3   0.12  -3.78  -3.64  -3.56  -3.48  -3.31   4000    1.0
    log_lik[0,4]   -5.37  4.1e-3   0.26   -5.9  -5.54  -5.37   -5.2  -4.88   4000    1.0
    log_lik[1,4]   -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[2,4]   -3.98  2.4e-3   0.15  -4.28  -4.07  -3.97  -3.88  -3.69   4000    1.0
    log_lik[3,4]   -2.68  6.3e-4   0.04  -2.76   -2.7  -2.68  -2.65   -2.6   4000    1.0
    log_lik[4,4]   -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[5,4]   -3.18  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[6,4]   -3.18  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[7,4]   -3.87  2.2e-3   0.14  -4.17  -3.95  -3.86  -3.78  -3.61   4000    1.0
    log_lik[8,4]   -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[9,4]   -2.88  9.6e-4   0.06  -3.02  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[10,4]  -2.67  6.2e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[11,4]  -4.28  2.7e-3   0.17  -4.64  -4.39  -4.27  -4.16  -3.94   4000    1.0
    log_lik[12,4]  -2.82  8.4e-4   0.05  -2.93  -2.85  -2.81  -2.78  -2.71   4000    1.0
    log_lik[13,4]  -3.25  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[14,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[15,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[16,4]  -3.18  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[17,4]  -3.18  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[18,4]  -5.22  3.8e-3   0.24  -5.73  -5.37  -5.22  -5.06  -4.76   4000    1.0
    log_lik[19,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[20,4]  -2.68  6.3e-4   0.04  -2.76   -2.7  -2.68  -2.65   -2.6   4000    1.0
    log_lik[21,4]  -2.67  6.2e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[22,4]  -2.82  8.4e-4   0.05  -2.93  -2.85  -2.81  -2.78  -2.71   4000    1.0
    log_lik[23,4]  -2.73  7.2e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.65   4000    1.0
    log_lik[24,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[25,4]  -2.68  6.3e-4   0.04  -2.76   -2.7  -2.68  -2.65   -2.6   4000    1.0
    log_lik[26,4]  -4.61  3.1e-3    0.2  -5.03  -4.74  -4.61  -4.48  -4.23   4000    1.0
    log_lik[27,4]   -3.7  2.0e-3   0.13  -3.97  -3.78   -3.7  -3.62  -3.46   4000    1.0
    log_lik[28,4]  -3.08  1.2e-3   0.08  -3.24  -3.12  -3.07  -3.03  -2.93   4000    1.0
    log_lik[29,4]  -2.67  6.2e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[30,4]  -3.61  1.9e-3   0.12  -3.87  -3.68   -3.6  -3.54  -3.39   4000    1.0
    log_lik[31,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[32,4]  -2.88  9.6e-4   0.06  -3.02  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[33,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[34,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[35,4]  -4.28  2.7e-3   0.17  -4.64  -4.39  -4.27  -4.16  -3.94   4000    1.0
    log_lik[36,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[37,4]  -2.68  6.3e-4   0.04  -2.76   -2.7  -2.68  -2.65   -2.6   4000    1.0
    log_lik[38,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[39,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[40,4]  -2.88  9.6e-4   0.06  -3.02  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[41,4]   -3.7  2.0e-3   0.13  -3.97  -3.78   -3.7  -3.62  -3.46   4000    1.0
    log_lik[42,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[43,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[44,4]  -3.25  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[45,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[46,4]  -3.61  1.9e-3   0.12  -3.87  -3.68   -3.6  -3.54  -3.39   4000    1.0
    log_lik[47,4]  -2.88  9.6e-4   0.06  -3.02  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[48,4]  -3.61  1.9e-3   0.12  -3.87  -3.68   -3.6  -3.54  -3.39   4000    1.0
    log_lik[49,4]   -3.7  2.0e-3   0.13  -3.97  -3.78   -3.7  -3.62  -3.46   4000    1.0
    log_lik[50,4]  -2.67  6.2e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[51,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[52,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[53,4]   -3.7  2.0e-3   0.13  -3.97  -3.78   -3.7  -3.62  -3.46   4000    1.0
    log_lik[54,4]  -2.66  6.0e-4   0.04  -2.73  -2.68  -2.66  -2.63  -2.58   4000    1.0
    log_lik[55,4]  -2.82  8.4e-4   0.05  -2.93  -2.85  -2.81  -2.78  -2.71   4000    1.0
    log_lik[56,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[57,4]  -2.73  7.2e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.65   4000    1.0
    log_lik[58,4]  -3.25  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[59,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[60,4]  -2.78  8.0e-4   0.05  -2.89  -2.81  -2.78  -2.75  -2.69   4000    1.0
    log_lik[61,4]  -4.84  3.4e-3   0.21  -5.29  -4.97  -4.83   -4.7  -4.44   4000    1.0
    log_lik[62,4]  -3.38  1.6e-3    0.1  -3.61  -3.44  -3.38  -3.32  -3.19   4000    1.0
    log_lik[63,4]  -3.61  1.9e-3   0.12  -3.87  -3.68   -3.6  -3.54  -3.39   4000    1.0
    log_lik[64,4]  -2.71  6.9e-4   0.04   -2.8  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[65,4]  -2.82  8.4e-4   0.05  -2.93  -2.85  -2.81  -2.78  -2.71   4000    1.0
    log_lik[66,4]  -3.46  1.7e-3   0.11  -3.69  -3.53  -3.46  -3.39  -3.26   4000    1.0
    ypred[0]       14.17    0.09   5.63   2.93  10.34  14.11  17.91   25.5   3712    1.0
    ypred[1]       14.32    0.09   5.72   2.96  10.46  14.23  18.35  25.13   3657    1.0
    ypred[2]       14.22    0.09   5.69   2.71  10.53  14.23  18.08  25.33   4000    1.0
    ypred[3]       14.41    0.09   5.78   2.92  10.63  14.34  18.28  25.82   3825    1.0
    ypred[4]        14.2    0.09   5.64   3.06  10.31  14.24  18.04  24.89   4000    1.0
    mu_new         14.22    0.02   0.94  12.45  13.86  14.21  14.57  16.09   2731    1.0
    ypred_new       14.2    0.09    5.8   2.84  10.26  14.19  18.17  25.49   4000    1.0
    lp__          -749.3    0.07   2.27 -754.3 -750.6 -749.1 -747.7 -745.4   1146    1.0
    
    Samples were drawn using NUTS at Sun Dec  9 13:48:51 2018.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).
    
    ##########################################################################
    #### Attachment 4: Fit of hierarchical model with inverse gamma prior ####
    ##########################################################################
    
    Inference for Stan model: anon_model_48a5c6bedf159373816b9432a8388eb4.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
    
                    mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu0            14.17    0.01   0.43  13.33  13.91  14.18  14.45   15.0   1637    1.0
    sigma0          0.55    0.02   0.57   0.02   0.19    0.4   0.73   2.04    913    1.0
    mu_tilde[0]     -0.2    0.02   0.88  -1.96  -0.76  -0.21   0.37   1.56   3191    1.0
    mu_tilde[1]     0.12    0.02   0.88  -1.68  -0.43    0.1    0.7   1.83   2959    1.0
    mu_tilde[2]     -0.2    0.01   0.86  -1.94  -0.75  -0.22   0.35   1.49   3433    1.0
    mu_tilde[3]     0.37    0.02   0.85  -1.42  -0.16   0.38   0.93   2.05   3213    1.0
    mu_tilde[4]   3.1e-3    0.01   0.84  -1.68  -0.54-1.8e-3   0.53   1.75   3325    1.0
    sigmaSq        31.97    0.04   2.56  27.37   30.2  31.81  33.63  37.32   4000    1.0
    mu[0]          14.06  7.7e-3   0.48  13.02  13.77  14.09  14.39  14.96   4000    1.0
    mu[1]          14.25  7.5e-3   0.47  13.32  13.96  14.24  14.55  15.22   4000    1.0
    mu[2]          14.05  7.7e-3   0.49   13.0  13.75  14.08  14.36  14.94   4000    1.0
    mu[3]          14.41  7.8e-3   0.49  13.54  14.06  14.38  14.71  15.49   4000    1.0
    mu[4]          14.18  7.4e-3   0.47  13.23   13.9  14.19  14.49   15.1   4000    1.0
    sigma           5.65  3.6e-3   0.23   5.23    5.5   5.64    5.8   6.11   4000    1.0
    log_lik[0,0]   -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[1,0]   -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[2,0]   -2.71  7.3e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[3,0]   -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[4,0]   -3.04  1.2e-3   0.08  -3.21  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[5,0]   -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[6,0]   -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[7,0]   -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[8,0]   -3.95  2.4e-3   0.15  -4.24  -4.04  -3.95  -3.85  -3.64   4000    1.0
    log_lik[9,0]   -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[10,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[11,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[12,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[13,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[14,0]  -3.95  2.4e-3   0.15  -4.24  -4.04  -3.95  -3.85  -3.64   4000    1.0
    log_lik[15,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[16,0]   -2.8  8.6e-4   0.05  -2.91  -2.84   -2.8  -2.76   -2.7   4000    1.0
    log_lik[17,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[18,0]  -2.71  7.3e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[19,0]  -3.41  1.7e-3   0.11  -3.65  -3.47   -3.4  -3.34  -3.22   4000    1.0
    log_lik[20,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[21,0]  -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[22,0]  -4.21  2.8e-3   0.18   -4.6  -4.32  -4.19  -4.09  -3.89   4000    1.0
    log_lik[23,0]  -2.91  1.0e-3   0.06  -3.04  -2.96  -2.91  -2.87  -2.79   4000    1.0
    log_lik[24,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[25,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[26,0]  -3.65  2.0e-3   0.13  -3.93  -3.72  -3.63  -3.56  -3.42   4000    1.0
    log_lik[27,0]  -2.72  7.4e-4   0.05  -2.81  -2.75  -2.72  -2.69  -2.63   4000    1.0
    log_lik[28,0]  -3.04  1.2e-3   0.08  -3.21  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[29,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[30,0]  -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[31,0]  -2.72  7.4e-4   0.05  -2.81  -2.75  -2.72  -2.69  -2.63   4000    1.0
    log_lik[32,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[33,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[34,0]  -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[35,0]  -4.54  3.2e-3    0.2  -4.98  -4.66  -4.52  -4.39  -4.18   4000    1.0
    log_lik[36,0]  -3.04  1.2e-3   0.08  -3.21  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[37,0]  -4.21  2.8e-3   0.18   -4.6  -4.32  -4.19  -4.09  -3.89   4000    1.0
    log_lik[38,0]  -2.71  7.3e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[39,0]  -3.95  2.4e-3   0.15  -4.24  -4.04  -3.95  -3.85  -3.64   4000    1.0
    log_lik[40,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[41,0]  -3.21  1.5e-3   0.09  -3.41  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[42,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[43,0]  -3.65  2.0e-3   0.13  -3.93  -3.72  -3.63  -3.56  -3.42   4000    1.0
    log_lik[44,0]  -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[45,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[46,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[47,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[48,0]  -3.23  1.4e-3   0.09  -3.41  -3.29  -3.23  -3.17  -3.05   4000    1.0
    log_lik[49,0]  -2.79  8.5e-4   0.05  -2.91  -2.82  -2.79  -2.75  -2.69   4000    1.0
    log_lik[50,0]  -3.41  1.7e-3   0.11  -3.65  -3.47   -3.4  -3.34  -3.22   4000    1.0
    log_lik[51,0]  -3.04  1.2e-3   0.08  -3.21  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[52,0]  -3.65  2.0e-3   0.13  -3.93  -3.72  -3.63  -3.56  -3.42   4000    1.0
    log_lik[53,0]  -4.25  2.7e-3   0.17  -4.59  -4.36  -4.25  -4.13   -3.9   4000    1.0
    log_lik[54,0]  -3.06  1.2e-3   0.08  -3.21  -3.11  -3.06  -3.01   -2.9   4000    1.0
    log_lik[55,0]  -2.71  7.3e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[56,0]   -2.9  1.0e-3   0.06  -3.04  -2.93  -2.89  -2.86  -2.79   4000    1.0
    log_lik[57,0]  -2.72  7.4e-4   0.05  -2.81  -2.75  -2.72  -2.69  -2.63   4000    1.0
    log_lik[58,0]  -2.67  6.6e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[59,0]  -2.91  1.0e-3   0.06  -3.04  -2.96  -2.91  -2.87  -2.79   4000    1.0
    log_lik[60,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[61,0]  -2.91  1.0e-3   0.06  -3.04  -2.96  -2.91  -2.87  -2.79   4000    1.0
    log_lik[62,0]  -3.68  2.0e-3   0.13  -3.93  -3.76  -3.68  -3.59  -3.42   4000    1.0
    log_lik[63,0]  -3.44  1.7e-3   0.11  -3.65  -3.51  -3.44  -3.37  -3.22   4000    1.0
    log_lik[64,0]  -2.71  7.3e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[65,0]  -3.21  1.5e-3   0.09  -3.41  -3.26   -3.2  -3.15  -3.05   4000    1.0
    log_lik[66,0]  -3.65  2.0e-3   0.13  -3.93  -3.72  -3.63  -3.56  -3.42   4000    1.0
    log_lik[0,1]   -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[1,1]   -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[2,1]   -3.48  1.8e-3   0.11  -3.72  -3.55  -3.47  -3.41  -3.27   4000    1.0
    log_lik[3,1]   -2.68  6.8e-4   0.04  -2.77  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[4,1]   -3.17  1.3e-3   0.09  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[5,1]   -3.17  1.3e-3   0.09  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[6,1]    -2.7  7.0e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[7,1]   -5.21  4.0e-3   0.25  -5.73  -5.38   -5.2  -5.04  -4.73   4000    1.0
    log_lik[8,1]   -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[9,1]   -2.73  7.7e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.64   4000    1.0
    log_lik[10,1]  -2.65  6.4e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   4000    1.0
    log_lik[11,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[12,1]  -3.37  1.6e-3    0.1  -3.59  -3.43  -3.37   -3.3  -3.17   4000    1.0
    log_lik[13,1]  -3.48  1.8e-3   0.11  -3.72  -3.55  -3.47  -3.41  -3.27   4000    1.0
    log_lik[14,1]  -2.82  8.9e-4   0.06  -2.94  -2.85  -2.82  -2.78  -2.72   4000    1.0
    log_lik[15,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[16,1]  -2.68  6.8e-4   0.04  -2.77  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[17,1]  -3.27  1.5e-3   0.09  -3.48  -3.32  -3.26  -3.21   -3.1   4000    1.0
    log_lik[18,1]  -2.73  7.7e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.64   4000    1.0
    log_lik[19,1]  -3.01  1.1e-3   0.07  -3.16  -3.05  -3.01  -2.96  -2.87   4000    1.0
    log_lik[20,1]  -3.17  1.3e-3   0.09  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[21,1]  -2.87  9.4e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.76   4000    1.0
    log_lik[22,1]  -4.15  2.6e-3   0.17  -4.49  -4.25  -4.14  -4.04  -3.83   4000    1.0
    log_lik[23,1]  -3.09  1.3e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.94   4000    1.0
    log_lik[24,1]  -2.68  6.8e-4   0.04  -2.77  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[25,1]  -2.87  9.4e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.76   4000    1.0
    log_lik[26,1]  -3.37  1.6e-3    0.1  -3.59  -3.43  -3.37   -3.3  -3.17   4000    1.0
    log_lik[27,1]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[28,1]  -2.65  6.4e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   4000    1.0
    log_lik[29,1]  -3.09  1.3e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.94   4000    1.0
    log_lik[30,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[31,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[32,1]  -4.65  3.2e-3    0.2  -5.07  -4.77  -4.64  -4.51  -4.27   4000    1.0
    log_lik[33,1]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[34,1]   -2.7  7.0e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[35,1]  -3.17  1.3e-3   0.09  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[36,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[37,1]  -4.47  3.1e-3   0.19  -4.87  -4.59  -4.47  -4.34   -4.1   4000    1.0
    log_lik[38,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[39,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[40,1]   -2.7  7.0e-4   0.04  -2.79  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[41,1]  -3.48  1.8e-3   0.11  -3.72  -3.55  -3.47  -3.41  -3.27   4000    1.0
    log_lik[42,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[43,1]  -3.09  1.3e-3   0.08  -3.26  -3.13  -3.08  -3.04  -2.94   4000    1.0
    log_lik[44,1]  -2.65  6.4e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   4000    1.0
    log_lik[45,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[46,1]  -2.68  6.8e-4   0.04  -2.77  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[47,1]  -3.27  1.5e-3   0.09  -3.48  -3.32  -3.26  -3.21   -3.1   4000    1.0
    log_lik[48,1]  -2.94  1.1e-3   0.07  -3.08  -2.98  -2.93  -2.89  -2.82   4000    1.0
    log_lik[49,1]  -2.87  9.4e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.76   4000    1.0
    log_lik[50,1]  -4.47  3.1e-3   0.19  -4.87  -4.59  -4.47  -4.34   -4.1   4000    1.0
    log_lik[51,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[52,1]   -3.6  1.9e-3   0.12  -3.85  -3.67   -3.6  -3.52  -3.36   4000    1.0
    log_lik[53,1]  -2.94  1.1e-3   0.07  -3.08  -2.98  -2.93  -2.89  -2.82   4000    1.0
    log_lik[54,1]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[55,1]  -3.73  2.1e-3   0.13  -4.01   -3.8  -3.72  -3.64  -3.49   4000    1.0
    log_lik[56,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[57,1]  -2.82  8.9e-4   0.06  -2.94  -2.85  -2.82  -2.78  -2.72   4000    1.0
    log_lik[58,1]   -3.6  1.9e-3   0.12  -3.85  -3.67   -3.6  -3.52  -3.36   4000    1.0
    log_lik[59,1]  -2.87  9.4e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.76   4000    1.0
    log_lik[60,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[61,1]  -2.68  6.8e-4   0.04  -2.77  -2.71  -2.68  -2.65   -2.6   4000    1.0
    log_lik[62,1]  -4.31  2.8e-3   0.18  -4.68  -4.42   -4.3  -4.19  -3.98   4000    1.0
    log_lik[63,1]  -2.94  1.1e-3   0.07  -3.08  -2.98  -2.93  -2.89  -2.82   4000    1.0
    log_lik[64,1]  -2.87  9.4e-4   0.06   -3.0  -2.91  -2.87  -2.84  -2.76   4000    1.0
    log_lik[65,1]  -3.17  1.3e-3   0.09  -3.35  -3.23  -3.17  -3.12  -3.01   4000    1.0
    log_lik[66,1]  -2.77  7.9e-4   0.05  -2.88   -2.8  -2.77  -2.74  -2.68   4000    1.0
    log_lik[0,2]   -2.71  7.5e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[1,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[2,2]   -4.24  2.8e-3   0.17  -4.58  -4.36  -4.24  -4.13   -3.9   4000    1.0
    log_lik[3,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[4,2]   -3.21  1.5e-3   0.09  -3.42  -3.26   -3.2  -3.15  -3.06   4000    1.0
    log_lik[5,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[6,2]   -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[7,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[8,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[9,2]   -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[10,2]  -3.94  2.4e-3   0.15  -4.24  -4.04  -3.94  -3.84  -3.65   4000    1.0
    log_lik[11,2]  -2.71  7.5e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[12,2]  -2.71  7.5e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[13,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[14,2]  -2.71  7.5e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[15,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[16,2]   -2.8  8.5e-4   0.05  -2.91  -2.83   -2.8  -2.76  -2.69   4000    1.0
    log_lik[17,2]  -3.05  1.2e-3   0.08   -3.2   -3.1  -3.06  -3.01   -2.9   4000    1.0
    log_lik[18,2]  -2.91  1.0e-3   0.06  -3.04  -2.95  -2.91  -2.87  -2.78   4000    1.0
    log_lik[19,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[20,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[21,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[22,2]  -3.65  2.1e-3   0.13  -3.95  -3.72  -3.64  -3.56  -3.42   4000    1.0
    log_lik[23,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[24,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[25,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[26,2]  -3.65  2.1e-3   0.13  -3.95  -3.72  -3.64  -3.56  -3.42   4000    1.0
    log_lik[27,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[28,2]  -3.92  2.4e-3   0.15  -4.26   -4.0   -3.9  -3.81  -3.65   4000    1.0
    log_lik[29,2]  -3.04  1.2e-3   0.08  -3.22  -3.08  -3.03  -2.99  -2.91   4000    1.0
    log_lik[30,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[31,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[32,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[33,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[34,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[35,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[36,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[37,2]   -4.9  3.7e-3   0.23  -5.42  -5.04  -4.88  -4.74  -4.49   4000    1.0
    log_lik[38,2]  -2.67  6.6e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[39,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[40,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[41,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[42,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[43,2]  -2.67  6.7e-4   0.04  -2.75  -2.69  -2.67  -2.64  -2.59   4000    1.0
    log_lik[44,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[45,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[46,2]  -2.91  1.0e-3   0.06  -3.04  -2.95  -2.91  -2.87  -2.78   4000    1.0
    log_lik[47,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[48,2]   -2.8  8.5e-4   0.05  -2.91  -2.83   -2.8  -2.76  -2.69   4000    1.0
    log_lik[49,2]  -3.92  2.4e-3   0.15  -4.26   -4.0   -3.9  -3.81  -3.65   4000    1.0
    log_lik[50,2]  -3.21  1.5e-3   0.09  -3.42  -3.26   -3.2  -3.15  -3.06   4000    1.0
    log_lik[51,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[52,2]  -3.67  2.0e-3   0.13  -3.92  -3.75  -3.67  -3.59  -3.41   4000    1.0
    log_lik[53,2]  -2.91  1.0e-3   0.06  -3.04  -2.95  -2.91  -2.87  -2.78   4000    1.0
    log_lik[54,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[55,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[56,2]  -2.71  7.5e-4   0.05  -2.81  -2.74  -2.71  -2.68  -2.63   4000    1.0
    log_lik[57,2]  -4.21  2.8e-3   0.18  -4.61  -4.32   -4.2  -4.09  -3.91   4000    1.0
    log_lik[58,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[59,2]  -2.79  8.7e-4   0.05  -2.91  -2.82  -2.79  -2.75   -2.7   4000    1.0
    log_lik[60,2]  -2.72  7.3e-4   0.05  -2.81  -2.75  -2.72  -2.69  -2.63   4000    1.0
    log_lik[61,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[62,2]  -2.67  6.6e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[63,2]  -3.23  1.4e-3   0.09   -3.4  -3.29  -3.23  -3.17  -3.04   4000    1.0
    log_lik[64,2]  -2.65  6.4e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   4000    1.0
    log_lik[65,2]  -2.67  6.6e-4   0.04  -2.75   -2.7  -2.67  -2.64  -2.59   4000    1.0
    log_lik[66,2]  -3.94  2.4e-3   0.15  -4.24  -4.04  -3.94  -3.84  -3.65   4000    1.0
    log_lik[0,3]   -3.77  2.3e-3   0.14  -4.09  -3.85  -3.75  -3.67  -3.53   4000    1.0
    log_lik[1,3]   -2.68  7.0e-4   0.04  -2.78  -2.71  -2.68  -2.66   -2.6   4000    1.0
    log_lik[2,3]   -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[3,3]   -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[4,3]   -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[5,3]   -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[6,3]    -4.1  2.6e-3   0.17  -4.43  -4.21   -4.1  -3.99  -3.77   4000    1.0
    log_lik[7,3]   -4.77  3.5e-3   0.22  -5.21  -4.92  -4.77  -4.62  -4.34   4000    1.0
    log_lik[8,3]   -2.96  1.1e-3   0.07  -3.12   -3.0  -2.95  -2.91  -2.84   4000    1.0
    log_lik[9,3]   -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[10,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[11,3]  -2.69  6.9e-4   0.04  -2.78  -2.72  -2.69  -2.66  -2.61   4000    1.0
    log_lik[12,3]  -4.05  2.6e-3   0.17  -4.42  -4.14  -4.03  -3.93  -3.77   4000    1.0
    log_lik[13,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[14,3]  -3.77  2.3e-3   0.14  -4.09  -3.85  -3.75  -3.67  -3.53   4000    1.0
    log_lik[15,3]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[16,3]  -3.11  1.4e-3   0.09  -3.32  -3.16   -3.1  -3.05  -2.97   4000    1.0
    log_lik[17,3]  -4.36  3.0e-3   0.19  -4.79  -4.47  -4.34  -4.23  -4.03   4000    1.0
    log_lik[18,3]  -2.96  1.1e-3   0.07  -3.12   -3.0  -2.95  -2.91  -2.84   4000    1.0
    log_lik[19,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[20,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[21,3]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[22,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[23,3]  -2.69  6.9e-4   0.04  -2.78  -2.72  -2.69  -2.66  -2.61   4000    1.0
    log_lik[24,3]  -3.34  1.6e-3    0.1  -3.53  -3.41  -3.34  -3.27  -3.13   4000    1.0
    log_lik[25,3]  -2.96  1.1e-3   0.07  -3.12   -3.0  -2.95  -2.91  -2.84   4000    1.0
    log_lik[26,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[27,3]  -2.99  1.1e-3   0.07  -3.12  -3.03  -2.99  -2.94  -2.84   4000    1.0
    log_lik[28,3]  -3.34  1.6e-3    0.1  -3.53  -3.41  -3.34  -3.27  -3.13   4000    1.0
    log_lik[29,3]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[30,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[31,3]  -2.69  6.9e-4   0.04  -2.78  -2.72  -2.69  -2.66  -2.61   4000    1.0
    log_lik[32,3]   -3.3  1.6e-3    0.1  -3.54  -3.36  -3.29  -3.23  -3.13   4000    1.0
    log_lik[33,3]  -2.96  1.1e-3   0.07  -3.12   -3.0  -2.95  -2.91  -2.84   4000    1.0
    log_lik[34,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[35,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[36,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[37,3]  -5.15  4.0e-3   0.25  -5.66  -5.32  -5.14  -4.98  -4.67   4000    1.0
    log_lik[38,3]  -3.77  2.3e-3   0.14  -4.09  -3.85  -3.75  -3.67  -3.53   4000    1.0
    log_lik[39,3]  -2.84  9.6e-4   0.06  -2.97  -2.87  -2.83  -2.79  -2.73   4000    1.0
    log_lik[40,3]  -3.11  1.4e-3   0.09  -3.32  -3.16   -3.1  -3.05  -2.97   4000    1.0
    log_lik[41,3]  -2.96  1.1e-3   0.07  -3.12   -3.0  -2.95  -2.91  -2.84   4000    1.0
    log_lik[42,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[43,3]  -4.77  3.5e-3   0.22  -5.21  -4.92  -4.77  -4.62  -4.34   4000    1.0
    log_lik[44,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[45,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[46,3]  -2.74  8.1e-4   0.05  -2.85  -2.78  -2.74  -2.71  -2.65   4000    1.0
    log_lik[47,3]  -2.84  9.6e-4   0.06  -2.97  -2.87  -2.83  -2.79  -2.73   4000    1.0
    log_lik[48,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[49,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[50,3]   -3.3  1.6e-3    0.1  -3.54  -3.36  -3.29  -3.23  -3.13   4000    1.0
    log_lik[51,3]  -2.69  6.9e-4   0.04  -2.78  -2.72  -2.69  -2.66  -2.61   4000    1.0
    log_lik[52,3]   -3.3  1.6e-3    0.1  -3.54  -3.36  -3.29  -3.23  -3.13   4000    1.0
    log_lik[53,3]  -3.77  2.3e-3   0.14  -4.09  -3.85  -3.75  -3.67  -3.53   4000    1.0
    log_lik[54,3]  -2.84  9.6e-4   0.06  -2.97  -2.87  -2.83  -2.79  -2.73   4000    1.0
    log_lik[55,3]   -4.7  3.5e-3   0.22  -5.19  -4.83  -4.68  -4.55  -4.32   4000    1.0
    log_lik[56,3]   -3.3  1.6e-3    0.1  -3.54  -3.36  -3.29  -3.23  -3.13   4000    1.0
    log_lik[57,3]  -3.34  1.6e-3    0.1  -3.53  -3.41  -3.34  -3.27  -3.13   4000    1.0
    log_lik[58,3]   -4.1  2.6e-3   0.17  -4.43  -4.21   -4.1  -3.99  -3.77   4000    1.0
    log_lik[59,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[60,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[61,3]  -2.84  9.6e-4   0.06  -2.97  -2.87  -2.83  -2.79  -2.73   4000    1.0
    log_lik[62,3]  -2.76  8.0e-4   0.05  -2.86  -2.79  -2.76  -2.73  -2.66   4000    1.0
    log_lik[63,3]  -2.66  6.4e-4   0.04  -2.74  -2.69  -2.66  -2.63  -2.58   4000    1.0
    log_lik[64,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[65,3]  -3.77  2.3e-3   0.14  -4.09  -3.85  -3.75  -3.67  -3.53   4000    1.0
    log_lik[66,3]  -3.56  1.9e-3   0.12  -3.79  -3.64  -3.56  -3.48  -3.32   4000    1.0
    log_lik[0,4]   -5.39  4.1e-3   0.26  -5.92  -5.56  -5.38  -5.22  -4.89   4000    1.0
    log_lik[1,4]   -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[2,4]   -3.98  2.4e-3   0.15  -4.28  -4.07  -3.97  -3.88   -3.7   4000    1.0
    log_lik[3,4]   -2.68  6.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[4,4]   -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[5,4]   -3.19  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[6,4]   -3.19  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[7,4]   -3.88  2.3e-3   0.14  -4.18  -3.97  -3.87  -3.78  -3.61   4000    1.0
    log_lik[8,4]   -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[9,4]   -2.88  9.5e-4   0.06  -3.01  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[10,4]  -2.66  6.5e-4   0.04  -2.75  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[11,4]  -4.29  2.7e-3   0.17  -4.64  -4.39  -4.28  -4.17  -3.96   4000    1.0
    log_lik[12,4]  -2.81  8.6e-4   0.05  -2.92  -2.85  -2.81  -2.77  -2.71   4000    1.0
    log_lik[13,4]  -3.26  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[14,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[15,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[16,4]  -3.19  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[17,4]  -3.19  1.4e-3   0.09  -3.37  -3.24  -3.18  -3.13  -3.03   4000    1.0
    log_lik[18,4]  -5.24  4.0e-3   0.25  -5.76   -5.4  -5.23  -5.06  -4.77   4000    1.0
    log_lik[19,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[20,4]  -2.68  6.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[21,4]  -2.66  6.5e-4   0.04  -2.75  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[22,4]  -2.81  8.6e-4   0.05  -2.92  -2.85  -2.81  -2.77  -2.71   4000    1.0
    log_lik[23,4]  -2.73  7.5e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.64   4000    1.0
    log_lik[24,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[25,4]  -2.68  6.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[26,4]  -4.62  3.2e-3    0.2  -5.04  -4.75  -4.61  -4.49  -4.24   4000    1.0
    log_lik[27,4]  -3.71  2.0e-3   0.13  -3.97  -3.79   -3.7  -3.62  -3.47   4000    1.0
    log_lik[28,4]  -3.08  1.2e-3   0.08  -3.24  -3.12  -3.07  -3.03  -2.93   4000    1.0
    log_lik[29,4]  -2.66  6.5e-4   0.04  -2.75  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[30,4]  -3.61  1.9e-3   0.12  -3.87  -3.69  -3.61  -3.53  -3.39   4000    1.0
    log_lik[31,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[32,4]  -2.88  9.5e-4   0.06  -3.01  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[33,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[34,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[35,4]  -4.29  2.7e-3   0.17  -4.64  -4.39  -4.28  -4.17  -3.96   4000    1.0
    log_lik[36,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[37,4]  -2.68  6.7e-4   0.04  -2.76   -2.7  -2.67  -2.65   -2.6   4000    1.0
    log_lik[38,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[39,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[40,4]  -2.88  9.5e-4   0.06  -3.01  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[41,4]  -3.71  2.0e-3   0.13  -3.97  -3.79   -3.7  -3.62  -3.47   4000    1.0
    log_lik[42,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[43,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[44,4]  -3.26  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[45,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[46,4]  -3.61  1.9e-3   0.12  -3.87  -3.69  -3.61  -3.53  -3.39   4000    1.0
    log_lik[47,4]  -2.88  9.5e-4   0.06  -3.01  -2.92  -2.88  -2.84  -2.77   4000    1.0
    log_lik[48,4]  -3.61  1.9e-3   0.12  -3.87  -3.69  -3.61  -3.53  -3.39   4000    1.0
    log_lik[49,4]  -3.71  2.0e-3   0.13  -3.97  -3.79   -3.7  -3.62  -3.47   4000    1.0
    log_lik[50,4]  -2.66  6.5e-4   0.04  -2.75  -2.69  -2.66  -2.64  -2.59   4000    1.0
    log_lik[51,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[52,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[53,4]  -3.71  2.0e-3   0.13  -3.97  -3.79   -3.7  -3.62  -3.47   4000    1.0
    log_lik[54,4]  -2.65  6.4e-4   0.04  -2.73  -2.68  -2.65  -2.63  -2.58   4000    1.0
    log_lik[55,4]  -2.81  8.6e-4   0.05  -2.92  -2.85  -2.81  -2.77  -2.71   4000    1.0
    log_lik[56,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[57,4]  -2.73  7.5e-4   0.05  -2.83  -2.76  -2.73   -2.7  -2.64   4000    1.0
    log_lik[58,4]  -3.26  1.4e-3   0.09  -3.44  -3.31  -3.25   -3.2  -3.08   4000    1.0
    log_lik[59,4]  -2.93  1.0e-3   0.06  -3.06  -2.97  -2.93  -2.89  -2.81   4000    1.0
    log_lik[60,4]  -2.78  8.1e-4   0.05  -2.89  -2.81  -2.78  -2.74  -2.68   4000    1.0
    log_lik[61,4]  -4.85  3.5e-3   0.22  -5.31  -4.99  -4.84   -4.7  -4.44   4000    1.0
    log_lik[62,4]  -3.38  1.6e-3    0.1   -3.6  -3.45  -3.38  -3.32   -3.2   4000    1.0
    log_lik[63,4]  -3.61  1.9e-3   0.12  -3.87  -3.69  -3.61  -3.53  -3.39   4000    1.0
    log_lik[64,4]  -2.71  7.1e-4   0.04   -2.8  -2.73   -2.7  -2.67  -2.62   4000    1.0
    log_lik[65,4]  -2.81  8.6e-4   0.05  -2.92  -2.85  -2.81  -2.77  -2.71   4000    1.0
    log_lik[66,4]  -3.47  1.7e-3   0.11  -3.69  -3.53  -3.46   -3.4  -3.26   4000    1.0
    ypred[0]       14.08    0.09   5.72   2.95   10.2  14.09  17.87  25.31   4000    1.0
    ypred[1]       14.35    0.09   5.68   3.07  10.52  14.26   18.2  25.59   4000    1.0
    ypred[2]       13.98    0.09   5.56   3.09  10.25  14.05  17.71  24.85   4000    1.0
    ypred[3]       14.53    0.09   5.68   3.53  10.78  14.53  18.29  26.03   3793    1.0
    ypred[4]       14.14    0.09   5.65   3.05   10.4  14.09  17.95  25.14   4000    1.0
    mu_new         14.17    0.02    0.9  12.32  13.81   14.2  14.57  15.88   2677    1.0
    ypred_new      14.21    0.09   5.69   3.31  10.31  14.05  18.16  25.14   4000    1.0
    lp__          -754.5    0.07   2.37 -759.8 -756.0 -754.2 -752.8 -750.6   1260    1.0
    
    Samples were drawn using NUTS at Sun Dec  9 13:49:18 2018.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).
    
    
