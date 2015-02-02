#!/usr/bin/python

"""
This program uses Bayesian variable selection(BVS) techniques on mixed
effects model to differentiate multiple group effects from a
longitudinal study.

The dataset is from Dr. Stephen Spindler's lab, where they studied the
effects of different diets on mouse weights. By using BVS techniques,
it is possible to identify multiple diets that have effects on mouse
weights simultaneously.
"""

import os
import sys
import numpy as np
from numpy.linalg import pinv
import pandas as pd
import postdist

class WeightsData:
    """
    Read raw data from .txt file and extract information related to modeling.

    Parameters
    ----------
    filename: string
        The name of the raw data file in the working directory.

    diest: list, optional (default=False)
        Select a subgroup of diets, default setting selects all diets.

    ctrlgrp: int, optional (default=99)
        Set the index for control group, default is index 99.

    Attributes
    ----------
    'rawdata': np.array
        The entire dataset with scaled days (divided by 365).

    'ctrlgrp': int
        Index set for the control group.

    'data': np.array
        A subset of raw datasets based on the selected diets.

    'unidays': np.array
        An array of unique days.

    'unidiets': np.array
        An array of selected diets.

    'uniids': np.array
        An array of unique ids of mouse.

    'grp': int
        Total number of selected diets.

    'ntot': int
        Total number of unique mouse.

    'grp_uniids': dic
        A dic of arrays of unique mouse for each diet group.

    'grp_ntot': dic
        A dic of total numbers of unique mouse for each diet group.

    'grp_dtot': dic
        A dic of total data points (id*time) for each diet group.
    """
    def __init__(self, filename, diets = False, ctrlgrp = 99):
        self.rawdata = pd.read_csv(filename, sep=" ")
        self.rawdata['days'] = self.rawdata['days']/365.0 - 1.0 # scale days

        # select subgroups
        if diets == False:  # select all diet groups
            self.data = self.rawdata
        else:
            self.data = self.rawdata[np.in1d(self.rawdata['diet'], diets)]

        # set parameters
        self.unidays = np.unique(self.data['days'])
        self.unidiets = np.unique(self.data['diet'])
        self.ctrlidx = np.where(self.unidiets == ctrlgrp)[0][0]
        self.uniids = np.unique(self.data['id'])
        self.grp = self.unidiets.size # total number of diets
        self.ntot = self.uniids.size # total number of mouse
        self.grp_uniids = {}
        self.grp_ntot = {}
        self.grp_dtot = {}
        for g in self.unidiets:
            temp = self.data['id'][self.data['diet']==g]
            self.grp_uniids.update({g: np.unique(temp)})
            self.grp_dtot.update({g: temp.size})
            # self.grp_dtot.update({g: self.grp_uniids[g].size})
            self.grp_ntot.update({g: self.grp_uniids[g].size})

    def setParams(self, p = 2, l = 1):
        """
        Set parameters to control the complexity of the model.

        Parameters
        ----------
        p: int, optional(default=2)
            The number of terms in the baseline model.

        l: int, optional(default=1)
            The number of terms in the model selection.

        Attributes
        ----------
        'id_ntot': dic
            A dic of total number of data points for each id.

        'id_y': dic
            A dic of arrays of the response vector for each id.

        'id_W': dic
            A dic of baseline design matrix W for each id.

        'id_X': dic
            A dic of design matrix X for each id.

        'id_Z': dic
            A dic of random design matrix Z for each id.

        Return
        ------
        None

        """
        self.p = p
        self.l = l
        self.id_ntot = {}
        self.id_y = {}
        self.id_W = {}
        self.id_X = {}
        for i in self.uniids:
            tracker = (self.data['id'] == i)
            self.id_ntot.update({i: np.sum(tracker)})
            self.id_y.update({i:
                self.data['weight'][tracker].reshape(np.sum(tracker), 1)})
            self.id_W.update({i: self._designMatrix_(p, tracker)})
            self.id_X.update({i: self._designMatrix_(l+1, tracker, is_X=True)})
        self.id_Z = self.id_W.copy()

    def _designMatrix_(self, p, tracker, is_X=False):
        """Build design matrix based on order p."""
        temp1 = np.zeros([1, np.sum(tracker)])
        for pv in range(p):
            temp2 = self.data['days'][tracker].reshape(1, np.sum(tracker))**pv
            temp1 = np.vstack([temp1, temp2])
        if is_X: # if it is the design matrix for X, removes intercept
            temp1 = temp1[2:, ]
        else:
            temp1 = temp1[1:, ]
        return temp1.T


class PriorParams:
    '''Hold the prior parameters.'''
    def __init__(self):
        pass

    def setD1(self, d1):
        self.d1 = d1

    def setD2(self, d2):
        self.d2 = d2

    def setD3(self, d3):
        self.d3 = d3

    def setD4(self, d4):
        self.d4 = d4

    def setPai(self, pai):
        self.pai = pai

def mcmcrun(data, priors, dirname):
    """
    Run Markov chain Monte Carlo (MCMC) simulations using Gibbs sampler.

    Parameters
    ----------
    data: WeightsData object
        Contains information summarized from raw data.

    dirname: string
        Create a folder using the given dirname to store simulation results.

    Return
    ------
    None

    """
    def initParams(data):
        '''Initialize parameters from fitted mixed effects model in R.'''
        params = postdist.ParamsHolder()
        params.setAlpha(np.array([45.50, -5.75]).reshape(data.p, 1))
        params.setBeta(np.zeros([data.grp, data.l]))
        params.setGamma(np.zeros([data.grp, data.l]))
        params.setLambdaD(np.array(0.086).reshape(1, 1))
        params.setB(np.random.normal(0, np.sqrt(1.0/params.lambdaD),
                    size = data.ntot*data.p).reshape(data.ntot, data.p))
        params.setSigma2(np.array(5.06).reshape(1, 1))
        return params

    # initialize parameters
    temp_params = initParams(data)
    params = temp_params.toArray(data.ntot, data.grp, data.p, data.l)

    # MCMC updates
    totSimulation = 1000
    counter = 0
    while(counter < totSimulation):
        counter += 1

        # update gamma
        gamma_pd = postdist.GammaPosterior(data, temp_params, priors)
        temp_params.gamma = gamma_pd.getUpdates()
        # print temp_params.gamma.shape

        # update beta
        beta_pd = postdist.BetaPosterior(data, temp_params)
        temp_params.beta = beta_pd.getUpdates()
        # print temp_params.beta.shape

        # update alpha
        alpha_pd = postdist.AlphaPosterior(data, temp_params, priors)
        temp_params.alpha = alpha_pd.getUpdates()
        # print temp_params.alpha.shape

        # update lambdaD
        lambdaD_pd = postdist.LambdaDPosterior(data, temp_params, priors)
        temp_params.lambdaD = lambdaD_pd.getUpdates()
        # print temp_params.lambdaD.shape

        # update sigma2
        sigma2_pd = postdist.Sigma2Posterior(data, temp_params)
        temp_params.sigma2 = sigma2_pd.getUpdates()
        # print temp_params.sigma2.shape

        # update b
        b_pd = postdist.BPosterior(data, temp_params)
        temp_params.b = b_pd.getUpdates()
        # print temp_params.b.shape

        # print "Mean is {0} and Cov is {1}".format(b_pd.mean, b_pd.cov)
        # print "New b's are {0}".format(temp_params.b)
        # raw_input("Press Enter to Continue ...")

        # store updates
        params = np.hstack([params,
                temp_params.toArray(data.ntot, data.grp, data.p, data.l)])

        # write to file
        np.savetxt(dirname+"/updates.txt", params, delimiter=',')
        # np.savetxt(dirname+"/counter.txt", counter, delimiter=',')


if __name__ == '__main__':
    # make new dir as input to store results
    if len(sys.argv) >= 2:
        dirname = sys.argv[1]
        datafile = sys.argv[2]
        try:
            os.mkdir(dirname)   #make new directory
        except OSError:
            print "\n" + dirname + "/\tALREADY EXISTS...\nFILES ARE WRITTEN..."

    np.random.seed(3)   #set random seed

    # mousediet = WeightsData('mouse_weights_nomiss.txt', diets = [99, 27])
    mousediet = WeightsData(datafile, diets = [99, 1])

    mousediet.setParams(p=2, l=1)

    # set priors
    priors = PriorParams()
    priors.setD1(75.95)
    priors.setD2(871.47)
    priors.setD3(np.array([45.50, -5.75]).reshape(mousediet.p, 1))
    priors.setD4(pinv(np.array([0.04, -0.02, -0.02,
                                0.06]).reshape(mousediet.p, mousediet.p)))
    priors.setPai(0.5*np.ones(mousediet.grp))

    mcmcrun(mousediet, priors, dirname)
