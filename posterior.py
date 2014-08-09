#!/usr/bin/python

```
posterior inference for longitudinal Bayesian variable selection model
```

### import modules ###
import numpy as np

### functions ###

def mcmc_update(data, p, L, uni_days, uni_diet, G, N_id, N_g, id_g, n_i, W, X, Z):
	# set parameters
	N_sim = 10000
	
        # hyper parameters
        d1 =
        d2 =
        d3 =
        d4 =

	# initial values	
	alpha = np.zeros(p)
	beta = np.zeros([G, L])
	gamma = np.zeros([G, L])
	b = np.zeros(p)
	sigma2 = np.ones(1)
	lambda_D = np.ones(1)
	
	# updates
	alpha_now = alpha.copy().reshape(p, 1)
	beta_now = beta.copy()
	gamma_now = gamma.copy()
	b_now = b.copy().reshape(p, 1)
	sigma2_now = sigma2[0]
	lambda_D_now = lambda_D[0]
	
	# MCMC updates
	for iter in xrange(N_sim):
		# update alpha









