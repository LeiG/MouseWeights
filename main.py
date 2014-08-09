#!/usr/bin/python

```
Longitudinal Bayesian variable selection model on Mouse Weights data

input: mouse_weights_clean.txt

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
	Data = pd.read_csv('mouse_weights_clean.txt', sep=" ")
	
	# parameters
	uni_days = np.unique(Data['days'])
	uni_diet = np.unique(Data['diet'])
	G = uni_diet.size
	N_id = np.unique(Data['id']).size
	
	N_g = {} # number of mouse within diet groups
	id_g = {} # set of id within diet groups
	for g in uni_diet:
		N_g.update({g: np.unique(Data['id'][Data['diet']==g]).size}) 
		id_g.update({g: np.unique(Data['id'][Data['diet']==g])})

	n_i = {} # number of time poinst for each mouse
	p = 3 # order
	L = 2
	W = {}
	X = {}
	Z = {}
	for i in Data['id']:
		n_i.update({i: np.sum(Data['id'] == i)})
		W.update({i: np.vstack([np.ones(n_i[i]),Data['days'][Data['id']==i],Data['days'][Data['id']==i]**2]).T})	
		X.update({i: np.vstack([Data['days'][Data['id']==i],Data['days'][Data['id']==i]**2]).T})		
		Z.update({i: np.vstack([np.ones(n_i[i]),Data['days'][Data['id']==i],Data['days'][Data['id']==i]**2]).T})


		







