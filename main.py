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
import fwsr
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams

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
            # number of total number of measurements in a group
            self.grp_dtot.update({g: temp.size})
            # number of unique ids in a group
            self.grp_ntot.update({g: self.grp_uniids[g].size})
        self.id_dtot = {}
        for i in self.uniids:
            temp = self.data['days'][self.data['id']==i]
            # number of measurements for each ids
            self.id_dtot.update({i: temp.size})

    def setParams(self, p = 2):
        """
        Set parameters to control the complexity of the model.

        Parameters
        ----------
        p: int, optional(default=2)
            The number of terms in the baseline model.

        Attributes
        ----------
        l: int
            The number of terms in the model selection.

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
        self.l = p - 1
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
            self.id_X.update({i:
                            self._designMatrix_(self.l+1,tracker,is_X=True)})
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
        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.d4 = None
        self.pai = None
        self.sigma2 = None

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

    def setSigma2(self, sigma2):
        self.sigma2 = sigma2


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
        params.updateBeta(np.zeros([data.grp, data.l]))
        params.updateGamma(np.zeros([data.grp, data.l]))

        # linear
        # params.updateAlpha(priors.d3)
        # params.updateLambdaD(np.array(0.086).reshape(1, 1))

        # quadratic
        params.updateAlpha(np.array([45.50, -3.96, -1.41]).reshape(data.p, 1))
        params.updateLambdaD(np.array(0.047).reshape(1, 1))

        params.updateB(np.random.normal(0, np.sqrt(1.0/params.lambdaD),
                    size = data.ntot*data.p).reshape(data.ntot, data.p))
        params.updateSigma2(np.array(priors.sigma2).reshape(1, 1))
        return params

    # initialize parameters
    temp_params = initParams(data)
    temp_paramsArray = temp_params.toArray(data.ntot, data.grp,
                                           data.p, data.l)
    with open(dirname+"/updates.txt", 'w') as f_handle:
        np.savetxt(f_handle,
                   temp_paramsArray,
                   delimiter=',')

    stoppingRule = fwsr.StoppingRule(temp_paramsArray, dirname, eps = 0.2)

    # MCMC updates
    # totSimulation = 10000
    counter = 1
    while True:
        counter += 1

        # update gamma
        gamma_pd = postdist.GammaPosterior(data, temp_params, priors)
        temp_params.updateGamma(gamma_pd.getUpdates())

        # update beta
        beta_pd = postdist.BetaPosterior(data, temp_params)
        temp_params.updateBeta(beta_pd.getUpdates())

        # update lambdaD
        lambdaD_pd = postdist.LambdaDPosterior(data, temp_params, priors)
        temp_params.updateLambdaD(lambdaD_pd.getUpdates())

        # update sigma2
        sigma2_pd = postdist.Sigma2Posterior(data, temp_params)
        temp_params.updateSigma2(sigma2_pd.getUpdates())

        # update alpha
        alpha_pd = postdist.AlphaPosterior(data, temp_params, priors)
        temp_params.updateAlpha(alpha_pd.getUpdates())

        # update b
        b_pd = postdist.BPosterior(data, temp_params)
        temp_params.updateB(b_pd.getUpdates())

        # # print out results with gaps
        # if counter % 10 == 0:
        #     print "==== This is the {0}th iteration ====".format(counter)
        #     # raw_input("Press Enter to Continue ...")
        #     print "====beta (trt & ctrl): \n{0}".format(temp_params.beta)
        #     print "====alpha: \n{0}".format(temp_params.alpha)
        #     print "====lambdaD: \n{0}".format(temp_params.lambdaD)
        #     print "====sigma2: \n{0}".format(temp_params.sigma2)
        #     if int(raw_input("Do you want the results of b? Press 0/1:")):
        #         print "===b: {0} with Cov: {1}".format(temp_params.b,
        #                                                b_pd.cov[1])

        # store updates
        # params = np.hstack([params,
        #         temp_params.toArray(data.ntot, data.grp, data.p, data.l)])

        # write to file
        # np.savetxt(dirname+"/updates.txt", params, delimiter=',')

        temp_paramsArray = temp_params.toArray(data.ntot, data.grp,
                                               data.p, data.l)
        with open(dirname+"/updates.txt", 'a') as f_handle:
            np.savetxt(f_handle,
                       temp_paramsArray,
                       delimiter=',')

        np.savetxt(dirname+"/counter.txt", np.array([counter]))

        isTerminated = stoppingRule.update(temp_paramsArray)
        if isTerminated:
            break


if __name__ == '__main__':
    # make new dir as input to store results
    if len(sys.argv) >= 2:
        dirname = sys.argv[1]
        datafile = sys.argv[2]
        try:
            os.mkdir(dirname)   #make new directory
        except OSError:
            print "\n" + dirname + \
                            "/\tALREADY EXISTS...\nFILES ARE REWRITTEN..."

    np.random.seed(3)   #set random seed

    # mousediet = WeightsData(datafile, diets = [99, 1], ctrlgrp = 99)
    # mousediet = WeightsData(datafile, ctrlgrp = 99)
    mousediet = WeightsData(datafile,
                            diets = [99, 21, 22, 23, 24, 27, 28, 29,
                                     34, 35, 39, 42, 43, 44, 45, 48,
                                     53, 55, 63],
                            ctrlgrp = 99)

    # set priors
    priors = PriorParams()

    ## linear
    ## estimate priors from the control group
    # mousediet.setParams(p=2)
    #
    # data = mousediet.rawdata[mousediet.rawdata['diet'] == 99]
    # model = sm.MixedLM.from_formula('weight ~ days', data,
    #                                 re_formula='1 + days',
    #                                 groups=data['id'])
    # free = MixedLMParams(2, 2)
    # free.set_fe_params(np.ones(2))
    # free.set_cov_re(np.eye(2))
    # result = model.fit(free=free)
    #
    # # uninformative prior
    # priors.setD1(0.001)
    # priors.setD2(0.001)
    #
    # priors.setD3(result.fe_params.values.reshape(mousediet.p, 1))
    # priors.setD4(pinv(result.cov_params().iloc[:mousediet.p,
    #                                            :mousediet.p].values))
    # priors.setPai(0.5*np.ones(mousediet.grp))
    # priors.setSigma2(result.scale)


    ## quadratic
    mousediet.setParams(p=3)

    data = mousediet.rawdata[mousediet.rawdata['diet'] == 99]
    data['days2'] = data['days']**2
    model = sm.MixedLM.from_formula('weight ~ days + days2', data,
                                    re_formula='1 + days + days2',
                                    groups=data['id'])
    free = MixedLMParams(3, 3)
    free.set_fe_params(np.ones(3))
    free.set_cov_re(np.eye(3))
    result = model.fit(free=free)

    # uninformative prior
    priors.setD1(0.001)
    priors.setD2(0.001)

    priors.setD3(result.fe_params.values.reshape(mousediet.p, 1))
    priors.setD4(pinv(result.cov_params().iloc[:mousediet.p,
                                               :mousediet.p].values))
    priors.setPai(0.5*np.ones(mousediet.grp))
    priors.setSigma2(result.scale)


    mcmcrun(mousediet, priors, dirname)
