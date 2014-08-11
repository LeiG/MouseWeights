#!/usr/bin/python

```
Longitudinal Bayesian variable selection model on Mouse Weights data

input: mouse_weights_nomiss.txt

```

### import modules ###
import numpy as np
import pandas as pd
import sys
import os

### main() ###
def main():
	# make new dir as input to store results
	if len(sys.argv) >= 2:
		dirname = sys.argv[1]
		os.mkdir(dirname)   # make new directory
    
	np.random.seed(3)   # set random seed
	
	# read data
	Data = pd.read_csv('mouse_weights_nomiss.txt', sep=" ")
	
	# parameters
	uni_days = np.unique(Data['days'])
	uni_diet = np.unique(Data['diet']) # include control group
	G = uni_diet.size
	uni_id = np.unique(Data['id'])
	N_id = uni_id.size
	
	N_g = {} # number of mouse within diet groups
	id_g = {} # set of id within diet groups
	for g in uni_diet:
		N_g.update({g: np.unique(Data['id'][Data['diet']==g]).size}) 
		id_g.update({g: np.unique(Data['id'][Data['diet']==g])})

	n_i = {} # number of time poinst for each mouse
	p = 3 # order
	L = 2 # polynomial
	y = {} # weights for each mouse
	W = {}
	X = {}
	for i in Data['id']:
		n_i.update({i: np.sum(Data['id'] == i)})
		y.update({i: Data['weight'][Data['id']==i].reshape(n_i[i], 1)})
		W.update({i: np.vstack([np.ones(n_i[i]),Data['days'][Data['id']==i],Data['days'][Data['id']==i]**2]).T})	
		X.update({i: np.vstack([Data['days'][Data['id']==i],Data['days'][Data['id']==i]**2]).T})		

	Z = W.copy()
		







