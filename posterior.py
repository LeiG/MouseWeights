#!/usr/bin/python

```
posterior inference for longitudinal Bayesian variable selection model
```

### import modules ###
import numpy as np
from numpy.linalg import inv

### functions ###
def XXsum(g, uni_diet, id_g, gamma, X):
	temp = np.zeros([np.sum(gamma[g]), np.sum(gamma[g])])
	for i in id_g[uni_diet[g]]:
		temp += np.dot(X[i][:,gamma[g]!=0].T, (X[i][:,gamma[g]!=0])
	return temp

def update_alpha(sigma2, beta, gamma, b, d3, d4, N_g, G, uni_diet, uni_id, W, X, Z, y):
	V1 = d4
	mean1 = d4.dot(d3)
	temp1 = np.zeros([p, p])
        temp2 = np.zeros([p, p])
	temp4 = np.zeros([p, p])
        temp7 = np.zeros([p, p])
	for g in range(G):
		if np.all(gamma[g] == 0): # check if all gamma's are 0
			for i in id_g[uni_diet[g]]:
	                        temp1 += np.dot(W[i].T, W[i])
				temp4 += np.dot(W[i].T, (y[i] - np.dot(Z[i], b[np.where(uni_id == i)[0][0]].reshape(p, 1))))
		else:
        		temp3 = np.zeros([p, p])
			temp5 = np.zeros([p, p])
			for i in id_g[uni_diet[g]]:
				temp1 += np.dot(W[i].T, W[i])
				temp3 += np.dot(X[i][:,gamma[g]!=0].T, W[i])
                                temp4 += np.dot(W[i].T, (y[i] - np.dot(X[i][:,gamma[g]!=0], beta[g][gamma[g]!=0].reshape(np.sum(gamma[g]), 1)) - np.dot(Z[i], b[np.where(uni_id == i)[0][0]].reshape(p, 1))))
				temp5 += np.dot(X[i][:,gamma[g]!=0].T, (y[i] - np.dot(X[i][:,gamma[g]!=0], beta[g][gamma[g]!=0].reshape(np.sum(gamma[g]), 1)) - np.dot(Z[i], b[np.where(uni_id == i)[0][0]].reshape(p, 1))))
		temp6 = XXsum(g, uni_diet, id_g, gamma, X)
		temp2 += temp3.T.dot(inv(temp6)).dot(temp3)/N_g[g]
		temp7 += temp4.T.dot(inv(temp6)).dot(temp5)/N_g[g]
	V1 += (temp1 + temp2)/sigma2
	mean1 += (temp4 + temp7)/sigma2
	mean1 = np.dot(inv(V1), mean1)
	
	#update
	alpha_new = np.random.multivariate_normal(mean1, inv(V1)).reshape(p, 1)
	return alpha_new

def update_b(i, b, gamma, lambda_D, N_g, sigma2, uni_id, W, X, Z):
	V2 = 


def mcmc_update(p, L, uni_days, uni_diet, uni_id, G, N_id, N_g, id_g, n_i, W, X, Z, y):
	# set parameters
	N_sim = 10000
	
        # hyper parameters
        d1 =
        d2 =
        d3 = np.ones([p, 1])
        d4 = np.identity(p)

	# initial values	
	alpha = np.zeros(p)
	beta = np.zeros([G, L])
	gamma = np.zeros([G, L])
	b = np.zeros([N_id, p])
	sigma2 = np.ones(1)
	lambda_D = np.ones(1)
	
	# updates
	alpha_now = alpha.copy().reshape(p, 1)
	beta_now = beta.copy()
	gamma_now = gamma.copy()
	b_now = b.copy()
	sigma2_now = sigma2[0]
	lambda_D_now = lambda_D[0]
	
	# MCMC updates
	for iter in xrange(N_sim):
		# update alpha
		alpha_now = update_alpha(sigma2_now, beta_now, gamma_now, b_now, d3, d4, N_g, G, uni_diet, uni_id, W, X, Z, y)

		# update b
		for i in xrange(N_id):
			b_now[i] = update_b(i, b_now[i], lambda_D_now, uni_id)






