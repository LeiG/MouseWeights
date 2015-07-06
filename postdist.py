#!/usr/bin/python

'''
Calculates posterior distributions and performs posterior inference for the
longitudinal Bayesian variable selection model.
'''

from __future__ import division

import copy
import pdb
import numpy as np
from numpy.linalg import pinv, det
from scipy.stats import invgamma


class ParamsHolder:
    '''
    Temporay container for the parameters being updated.

    Arributes
    ---------
    updateAlpha
    updateBeta
    updateGamma
    updateLambdaD
    updateB
    updateSigma2

    'toArray': 1-d np.array
        Concatenate all updated parameters into a 1-d np.array.
    '''
    def __init__(self):
        pass

    def updateAlpha(self, alpha):
        self.alpha = alpha

    def updateBeta(self, beta):
        self.beta = beta

    def updateGamma(self, gamma):
        self.gamma = gamma

    def updateLambdaD(self, lambdaD):
        self.lambdaD = lambdaD

    def updateB(self, b):
        self.b = b

    def updateSigma2(self, sigma2):
        self.sigma2 = sigma2

    def toArray(self, ntot, grp, p, l):
        '''
        Parameters
        ----------
        ntot: int
            Total number of unique mouse.

        grp: int
            Total number of selected diets.

        p: int
            The number of terms in the baseline model.

        l: int
            The number of terms in the model selection.
        '''
        alpha_1d = self.alpha.reshape(p, 1)
        beta_1d = self.beta.reshape(grp*l, 1)
        gamma_1d = self.gamma.reshape(grp*l, 1)
        b_1d = self.b.reshape(ntot*p, 1)
        sigma2_1d = self.sigma2#.reshape(1, 1)
        lambdaD_1d = self.lambdaD#.reshape(1, 1)
        return np.concatenate([gamma_1d, alpha_1d, beta_1d,
                               lambdaD_1d, b_1d, sigma2_1d]).T
        # return gamma_1d # only look at gammas


class AlphaPosterior:
    '''
    Posterior distribution for alpha.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    priors: a PriorParams object
        Contains a set of prior parameters.

    Attributes
    ----------
    'mean': a 1-d array
        The mean of the posterior multivariate normal distribution.

    'cov': a 2-d array
        The covariance matrix of the posterior multivariate normal dist.

    'getUpdates': a function
        Generate a random sample from the posterior distribution.
    '''
    def __init__(self, data, params, priors):
        self._preCalculate_(data, params, priors)
        self._cov_(data, params, priors)
        self._mean_(data, params, priors)

    def _mean_(self, data, params, priors):
        '''Calculate mean of the normal posterior dist.'''
        temp1 = np.zeros([data.p, 1])
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            temp2 = np.zeros([data.p, 1])
            if nzro_gamma.any(): # not all 0's
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp4 = data.id_y[i] - \
                            np.dot(data.id_X[i][:, nzro_gamma],
                                   params.beta[gdx, nzro_gamma][:,np.newaxis])-\
                                   np.dot(data.id_Z[i],
                                   params.b[idx,:][:,np.newaxis])
                    temp2 += np.dot(data.id_W[i].T, temp4)
                temp1 += temp2
                temp1 += np.dot(self.__nxTw__[g].T,
                                np.dot(pinv(self.__nxTx__[g]),
                                       self.__nxTyxbeta__[g]))
            else: # all gammas are 0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp4 = data.id_y[i] -\
                                   np.dot(data.id_Z[i],
                                   params.b[idx,:][:,np.newaxis])
                    temp2 += np.dot(data.id_W[i].T, temp4)
                temp1 += temp2
        temp1 = temp1/params.sigma2 + np.dot(priors.d4, priors.d3)
        self.mean = np.dot(self.cov, temp1)[:, 0] # set to 1-d

    def _cov_(self, data, params, priors):
        '''Calculate covariance matrix of the normal posterior dist.'''
        V1 = np.zeros([data.p, data.p])
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            for i in data.grp_uniids[g]:
                V1 += np.dot(data.id_W[i].T, data.id_W[i])
            if nzro_gamma.any():
                V1 += np.dot(self.__nxTw__[g].T,
                             np.dot(pinv(self.__nxTx__[g]),
                                    self.__nxTw__[g]))
        V1 = V1/params.sigma2 + priors.d4
        self.cov = pinv(V1)

    def _preCalculate_(self, data, params, priors):
        '''Calculate 1/n*X^T*W, 1/n*x^Tx, 1/n*x^T*(y-x*beta).'''
        self.__nxTw__ = {}
        self.__nxTx__ = {}
        self.__nxTyxbeta__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any():
                tmp1 = np.zeros([np.sum(nzro_gamma), data.p])
                tmp2 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                tmp3 = np.zeros([np.sum(nzro_gamma), 1])
                for i in data.grp_uniids[g]:
                    tmp_x = data.id_X[i][:, nzro_gamma]
                    tmp1 += (np.dot(tmp_x.T,
                                    data.id_W[i]))/data.id_dtot[i]
                    tmp2 += (np.dot(tmp_x.T, tmp_x))/data.id_dtot[i]
                    tmp3 += (np.dot(tmp_x.T,
                            data.id_y[i] - np.dot(tmp_x,
                                params.beta[gdx, nzro_gamma][:, np.newaxis])))/data.id_dtot[i]
                self.__nxTw__.update({g: tmp1})
                self.__nxTx__.update({g: tmp2})
                self.__nxTyxbeta__.update({g: tmp3})

    def getUpdates(self):
        '''Generate random samples from the posterior dist.'''
        return np.random.multivariate_normal(self.mean,
                                             self.cov)[:, np.newaxis]


class BPosterior:
    '''
    Posterior distribution for b.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    Arributes
    ---------
    'mean': a dic
        The means of the posterior distribution for each individual.

    'cov': a dic
        The covariance matrix of the posterior for each individual.

    'getUpdates': a function
        Generate a random sample for each individual from the posterior.
    '''
    def __init__(self, data, params):
        self.__shape__ = [data.ntot, data.p]
        self.__uniids__ = data.uniids
        self._cov_(data, params)
        self._mean_(data, params)

    def _cov_(self, data, params):
        '''Calculate covariance matrix of the posteriors.'''
        self.cov = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            for i in data.grp_uniids[g]:
                V2 = np.dot(data.id_Z[i].T, data.id_Z[i])
                V2 = V2/params.sigma2
                np.fill_diagonal(V2, V2.diagonal() + params.lambdaD)
                self.cov.update({i: pinv(V2)})

    def _mean_(self, data, params):
        '''Calculate mean of the posteriors.'''
        self.mean = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any(): # not all 0's
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp1 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            np.dot(data.id_X[i][:, nzro_gamma],
                                   params.beta[gdx, nzro_gamma][:, np.newaxis])
                    temp1 = np.dot(data.id_Z[i].T, temp1)
                    self.mean.update({i: np.dot(self.cov[i],
                                                temp1/params.sigma2)})
            else: # all gammas are 0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp1 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)
                    temp1 = np.dot(data.id_Z[i].T, temp1)
                    self.mean.update({i: np.dot(self.cov[i],
                                                temp1/params.sigma2)})

    def getUpdates(self):
        '''Generate random samples from the posterior dist.'''
        samples = np.zeros(self.__shape__)
        for idx in range(self.__shape__[0]):
            i = self.__uniids__[idx]
            samples[idx, :] = \
                np.random.multivariate_normal(self.mean[i][:,0], self.cov[i])
        return samples


class BetaPosterior:
    '''
    Posterior distribution for beta.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    Arributes
    ---------
    'mean': a dic
        The means of the posterior distribution for each group.

    'cov': a dic
        The covariance matrix of the posterior for each group.

    'getUpdates': a function
        Generate a random sample for each group from the posterior.
    '''
    def __init__(self, data, params):
        self.__shape__ = [data.grp, data.l]
        self.__unidiets__ = data.unidiets
        self._preCalculate_(data, params)
        self._cov_(data, params)
        self._mean_(data, params)

    def _cov_(self, data, params):
        '''Calculate covariance matrix of the posteriors.'''
        self.cov = {}
        self.__nzro_gamma__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            self.__nzro_gamma__.update({g: nzro_gamma})
            if nzro_gamma.any(): # not all 0's
                V3 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                for i in data.grp_uniids[g]:
                    V3 += np.dot(data.id_X[i][:, nzro_gamma].T,
                         data.id_X[i][:, nzro_gamma])* \
                         (1.0 + 1.0/data.id_dtot[i])
                V3 = V3/params.sigma2
                self.cov.update({g: pinv(V3)})
            else: # all gammas are 0
                pass

    def _mean_(self, data, params):
        '''Calculate mean of the posteriors.'''
        self.mean = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = self.__nzro_gamma__[g]
            if nzro_gamma.any(): # not all 0's
                temp = np.zeros([np.sum(nzro_gamma), 1])
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp += np.dot(data.id_X[i][:, nzro_gamma].T,
                                   data.id_y[i] - \
                                   np.dot(data.id_W[i], params.alpha) - \
                                   np.dot(data.id_Z[i],
                                   params.b[idx,:][:, np.newaxis]))
                temp += np.dot(pinv(self.__nxTx__[g]), self.__nxTywalpha__[g])
                self.mean.update({g: np.dot(self.cov[g], temp/params.sigma2)})
            else: # all gammas are 0
                pass

    def _preCalculate_(self, data, params):
        '''Calculate 1/n*x^Tx, 1/n*x^T*(y-w*alpha).'''
        self.__nxTx__ = {}
        self.__nxTywalpha__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any():
                tmp1 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                tmp2 = np.zeros([np.sum(nzro_gamma), 1])
                for i in data.grp_uniids[g]:
                    tmp_x = data.id_X[i][:, nzro_gamma]
                    tmp1 += (np.dot(tmp_x.T, tmp_x))/data.id_dtot[i]
                    tmp2 += (np.dot(tmp_x.T,
                            data.id_y[i] - np.dot(data.id_W[i],
                                params.alpha)))/data.id_dtot[i]
                self.__nxTx__.update({g: tmp1})
                self.__nxTywalpha__.update({g: tmp2})

    def getUpdates(self):
        '''Generate random samples from the posterior dist.'''
        samples = np.zeros(self.__shape__)
        for gdx in range(self.__shape__[0]):
            g = self.__unidiets__[gdx]
            if self.__nzro_gamma__[g].any(): # not all 0's
                samples[gdx, self.__nzro_gamma__[g]] = \
                    np.random.multivariate_normal(self.mean[g][:,0],
                                                  self.cov[g])
        return samples


class LambdaDPosterior:
    '''
    Posterior distribution for lambdaD.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    priors: a PriorParams object
        Contains a set of prior parameters.

    Arributes
    ---------
    'k': a scalar > 0
        The 'k' parameter for a gamma distribution.

    'theta': a scalar > 0
        The 'theta' parameter for a gamma distribution.

    'getUpdates': a function
        Generate a random sample for each group from the posterior.
    '''
    def __init__(self, data, params, priors):
        self._k_(data, params, priors)
        self._theta_(data, params, priors)

    def _k_(self, data, params, priors):
        '''Calculate the 'k' parameter.'''
        self.k = priors.d1 + data.p*data.ntot/2.0

    def _theta_(self, data, params, priors):
        '''Calculate the 'theta' parameter.'''
        temp = 0
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            for i in data.grp_uniids[g]:
                idx = np.where(data.uniids == i)[0][0]
                temp += np.dot(params.b[idx,:][:, np.newaxis].T,
                               params.b[idx,:][:, np.newaxis])
        self.theta = 1.0/(temp/2.0 + priors.d2)

    def getUpdates(self):
        '''Genearte random sample from the posterior.'''
        return np.array([[np.random.gamma(self.k, self.theta)]])


class Sigma2Posterior:
    '''
    Posterior distribution for sigma2.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    Attributes
    ----------
    'a': a scalar > 0
        The shape parameter of the posterior distribution.

    'scale': a scaler > 0
        The scale parameter of the posterior distribution.

    'getUpdates': a function
        Generate a random sample from the posterior distribution.
    '''
    def __init__(self, data, params):
        self._preCalculate_(data, params)
        self._a_(data, params)
        self._scale_(data, params)

    def _a_(self, data, params):
        '''Calculate the shape parameter of the posterior.'''
        self.a = (np.sum(data.grp_dtot.values()) + np.sum(params.gamma))/2.0

    def _scale_(self, data, params):
        '''Calculate the scale parameter of the posterior.'''
        self.scale = 0.0
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any(): # not all 0's
                temp_beta = params.beta[gdx, nzro_gamma][:, np.newaxis]
                temp1 = 0.0
                temp2 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp_x = data.id_X[i][:, nzro_gamma]
                    temp3 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            np.dot(temp_x, temp_beta) - np.dot(data.id_Z[i],
                                                params.b[idx,:][:, np.newaxis])
                    temp1 += np.dot(temp3.T, temp3)
                temp2 = np.dot(pinv(self.__nxTx__[g]), self.__nxTywalpha__[g])
                self.scale += (temp1 + np.dot((temp_beta - temp2).T,
                                              np.dot(self.__nxTx__[g],
                                                     temp_beta - temp2)))
            else: # all gammas are 0
                temp1 = 0.0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp3 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            np.dot(data.id_Z[i], params.b[idx,:][:, np.newaxis])
                    temp1 += np.dot(temp3.T, temp3)
                self.scale += temp1
        self.scale = self.scale/2.0

    def _preCalculate_(self, data, params):
        '''Calculate 1/n*x^Tx, 1/n*x^T*(y -w*alpha).'''
        self.__nxTx__ = {}
        self.__nxTywalpha__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any():
                tmp1 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                tmp2 = np.zeros([np.sum(nzro_gamma), 1])
                for i in data.grp_uniids[g]:
                    tmp_x = data.id_X[i][:, nzro_gamma]
                    tmp1 += (np.dot(tmp_x.T, tmp_x))/data.id_dtot[i]
                    tmp2 += (np.dot(tmp_x.T,
                                    data.id_y[i] - np.dot(data.id_W[i],
                                                params.alpha)))/data.id_dtot[i]
                self.__nxTx__.update({g: tmp1})
                self.__nxTywalpha__.update({g: tmp2})

    def getUpdates(self):
        return invgamma.rvs(a = self.a, scale = self.scale, size = 1)


class GammaPosterior:
    '''
    Posterior distribution for gamma.

    Parameters
    ----------
    data: a WeightsData object
        Contains all information from the raw data set.

    params: a ParamsHolder object
        Contains a set of parameters to initialize the posterior distribution.

    priors: a PriorParams object
        Contains a set of prior parameters.

    Attributes
    ----------
    'getUpdates': a function
        Generate a random sample from the posterior distribution.
    '''
    def __init__(self, data, params, priors):
        self.data = data
        self.params = params
        self.priors = priors
        self.xTx = {}   # (1/n)xTx
        self.inv_xTx = {}  # inverse of 1/n*xTx
        self.inv_xTx1 = {}  # inverse of (1+1/n)xTx
        self.phiTphi = {}
        self.xTphi = {}
        self.xTywalpha = {}
        for gdx in range(self.data.grp):
            self._fixComponents_(gdx)
            self._varComponents_(gdx, self.params.gamma, True)

    def _fixComponents_(self, gdx):
        '''Calculate fixed components for each group.'''
        temp_phiTphi = 0.0
        g = self.data.unidiets[gdx]
        for i in self.data.grp_uniids[g]:
            idx = np.where(self.data.uniids == i)[0][0]
            temp_phi = self.data.id_y[i] - \
                       np.dot(self.data.id_W[i], self.params.alpha)-\
                       np.dot(self.data.id_Z[i],
                              self.params.b[idx,:][:, np.newaxis])
            temp_phiTphi += np.dot(temp_phi.T, temp_phi)
        self.phiTphi.update({g: temp_phiTphi})

    def _varComponents_(self, gdx, gamma, inline):
        '''Calculate variable components for each group.'''
        nzro_gamma = (gamma[gdx,:]!=0)
        temp_xTx = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
        temp_inv_xTx = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
        temp_inv_xTx1 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
        temp_xTphi = np.zeros([np.sum(nzro_gamma), 1])
        temp_xTywalpha = np.zeros([np.sum(nzro_gamma), 1])
        g = self.data.unidiets[gdx]
        if nzro_gamma.any(): # not all 0's
            for i in self.data.grp_uniids[g]:
                idx = np.where(self.data.uniids == i)[0][0]
                temp_x = self.data.id_X[i][:, nzro_gamma]
                temp_xTx += (np.dot(temp_x.T, temp_x))/self.data.id_dtot[i]
                temp_inv_xTx += (np.dot(temp_x.T, temp_x))/self.data.id_dtot[i]
                temp_inv_xTx1 += np.dot(temp_x.T, temp_x)*\
                                 (1.0 + 1.0/self.data.id_dtot[i])
                temp_phi = self.data.id_y[i] - \
                           np.dot(self.data.id_W[i], self.params.alpha)- \
                           np.dot(self.data.id_Z[i],
                                  self.params.b[idx,:][:, np.newaxis])
                temp_xTphi += np.dot(temp_x.T, temp_phi)
                temp_xTywalpha += np.dot(temp_x.T,
                    self.data.id_y[i] - np.dot(self.data.id_W[i],
                                self.params.alpha))/self.data.id_dtot[i]
            temp_inv_xTx = pinv(temp_inv_xTx)
            temp_inv_xTx1 = pinv(temp_inv_xTx1)

            if inline:
                self.xTx.update({g: temp_xTx})
                self.inv_xTx.update({g: temp_inv_xTx})
                self.inv_xTx1.update({g: temp_inv_xTx1})
                self.xTphi.update({g: temp_xTphi})
                self.xTywalpha.update({g: temp_xTywalpha})
            else:
                return temp_xTx, temp_inv_xTx, temp_inv_xTx1, \
                       temp_xTphi, temp_xTywalpha

    def _logProb_(self, gdx, l, gamma, *args):
        '''Calculate unnormailized log-probability.'''
        return np.log(1 - self.priors.pai[gdx])*(1 - gamma[gdx, l]) +\
               np.log(self.priors.pai[gdx])*(gamma[gdx, l]) +\
               self._logS_(gdx, l, gamma, *args)

    def _logS_(self, gdx, l, gamma, *args):
        '''Calculate the S function.'''
        nzro_gamma = (gamma[gdx,:]!=0)
        g = self.data.unidiets[gdx]
        if len(args):
            if nzro_gamma.any(): # not all 0's
                temp = self.phiTphi[g] + \
                       np.dot(args[0][4].T, np.dot(args[0][1], args[0][4])) - \
                       np.dot((args[0][3]+args[0][4]).T,
                              np.dot(args[0][2], args[0][3]+args[0][4]))
            else:
                temp = self.phiTphi[g]

            temp = -temp/(2*self.params.sigma2)

            if nzro_gamma.any():
                temp += (np.log(det(args[0][0])) + np.log(det(args[0][2])))*0.5
            return temp

        else:
            if nzro_gamma.any(): # not all 0's
                temp = self.phiTphi[g] + \
                       np.dot(self.xTywalpha[g].T,
                              np.dot(self.inv_xTx[g], self.xTywalpha[g])) - \
                       np.dot((self.xTphi[g]+self.xTywalpha[g]).T,
                              np.dot(self.inv_xTx1[g],
                                     self.xTphi[g]+self.xTywalpha[g]))
            else:
                temp = self.phiTphi[g]

            temp = -temp/(2*self.params.sigma2)

            if nzro_gamma.any():
                temp += (np.log(det(self.xTx[g])) + \
                         np.log(det(self.inv_xTx1[g])))*0.5
            return temp

    def getUpdates(self):
        for gdx in range(self.data.grp):
            if gdx != self.data.ctrlidx: # do not update control group
            # if 1: # update the control group as well
                g = self.data.unidiets[gdx]
                for l in range(self.data.l):
                    prob = self._logProb_(gdx, l, self.params.gamma)[0, 0]

                    # get the opposite value of the current gamma, e.g. 0->1
                    temp_gamma = copy.deepcopy(self.params.gamma)
                    temp_gamma[gdx, l] = abs(self.params.gamma[gdx, l] - 1)
                    temp = self._varComponents_(gdx, temp_gamma, False)
                    prob_temp = self._logProb_(gdx,l,temp_gamma,temp)[0, 0]

                    prob_temp = 1.0/(1.0 + np.exp(prob - prob_temp))
                    prob = 1.0 - prob_temp

                    # update gamma
                    if np.random.binomial(1, prob_temp, 1)[0]:
                        self.params.gamma[gdx, l] = temp_gamma[gdx, l]
                        try:
                            self.xTx.update({g: copy.deepcopy(temp[0])})
                            self.inv_xTx.update({g: copy.deepcopy(temp[1])})
                            self.inv_xTx1.update({g: copy.deepcopy(temp[2])})
                            self.xTphi.update({g: copy.deepcopy(temp[3])})
                            self.xTywalpha.update({g: copy.deepcopy(temp[4])})
                        except TypeError:
                            pass
        return self.params.gamma
