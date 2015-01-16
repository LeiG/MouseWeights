#!/usr/bin/python

'''
Calculates posterior distributions and performs posterior inference for the
longitudinal Bayesian variable selection model.
'''

from __future__ import division
import numpy as np
from numpy.linalg import pinv, det
from scipy.stats import invgamma

class ParamsHolder:
    '''
    Temporay container for the parameters being updated.

    Arributes
    ---------
    setAlpha
    setBeta
    setGamma
    setLambdaD
    setB
    setSigma2

    'toArray': 1-d np.array
        Concatenate all updated parameters into a 1-d np.array.
    '''
    def __init__(self):
        pass

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setBeta(self, beta):
        self.beta = beta

    def setGamma(self, gamma):
        self.gamma = gamma

    def setLambdaD(self, lambdaD):
        self.lambdaD = lambdaD

    def setB(self, b):
        self.b = b

    def setSigma2(self, sigma2):
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
                               lambdaD_1d, b_1d, sigma2_1d])
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
        self._partialInit_(data, params)
        self._cov_(data, params, priors)
        self._mean_(data, params, priors)

    def _partialInit_(self, data, params):
        '''Pre-calculate common components.'''
        self.__xwsum__ = {}
        self.__inv_xxsum__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            try:
                temp1 = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                temp2 = np.zeros([np.sum(nzro_gamma), data.p])
                for i in data.grp_uniids[g]:
                    temp1 += np.dot(data.id_X[i][:, nzro_gamma].T,
                                    data.id_X[i][:, nzro_gamma])
                    temp2 += np.dot(data.id_X[i][:, nzro_gamma].T,
                                    data.id_W[i])
                temp1 = pinv(temp1)
                self.__inv_xxsum__.update({g: temp1})
                self.__xwsum__.update({g: temp2})
            except:
                pass

    def _mean_(self, data, params, priors):
        '''Calculate mean of the normal posterior dist.'''
        temp1 = np.zeros([data.p, 1])
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            temp2 = np.zeros([data.p, 1])
            temp3 = np.zeros([data.l, 1])
            if nzro_gamma.any(): # not all 0's
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp4 = data.id_y[i] - \
                            np.dot(data.id_X[i][:, nzro_gamma],
                                   params.beta[gdx, nzro_gamma][:,np.newaxis])-\
                                   np.dot(data.id_Z[i],
                                   params.b[idx,:][:,np.newaxis])
                    temp2 += np.dot(data.id_W[i].T, temp4)
                    temp3 += np.dot(data.id_X[i][:, nzro_gamma].T, temp4)
                temp1 += temp2 + \
                         np.dot(self.__xwsum__[g].T,
                          np.dot(self.__inv_xxsum__[g], temp3))/data.grp_dtot[g]
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
            if nzro_gamma.any(): # not all 0's
                temp = np.dot(self.__xwsum__[g].T,
                              np.dot(self.__inv_xxsum__[g],
                                     self.__xwsum__[g]))/data.grp_dtot[g]
            else: # all gammas are 0
                temp = np.zeros([data.p, data.p])
            for i in data.grp_uniids[g]:
                temp += np.dot(data.id_W[i].T, data.id_W[i])
            V1 += temp
        V1 = V1/params.sigma2 + priors.d4
        self.cov = pinv(V1)

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
        self._partialInit_(data, params)
        self._cov_(data, params)
        self._mean_(data, params)

    def _partialInit_(self, data, params):
        '''Pre-calculate common components.'''
        self.__inv_xxsum__ = {}
        self.__xTz__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            try:
                temp = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                for i in data.grp_uniids[g]:
                    temp += np.dot(data.id_X[i][:, nzro_gamma].T,
                                   data.id_X[i][:, nzro_gamma])
                    self.__xTz__.update({i:
                                      np.dot(data.id_X[i][:, nzro_gamma].T,
                                      data.id_Z[i])})
                temp = pinv(temp)
                self.__inv_xxsum__.update({g: temp})
            except:
                pass

    def _cov_(self, data, params):
        '''Calculate covariance matrix of the posteriors.'''
        self.cov = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any(): # not all 0's
                for i in data.grp_uniids[g]:
                    V2 = np.dot(self.__xTz__[i].T,
                                np.dot(self.__inv_xxsum__[g], self.__xTz__[i]))
                    V2 = V2/data.grp_dtot[g] + \
                         np.dot(data.id_Z[i].T, data.id_Z[i])
                    V2 = V2/params.sigma2 + params.lambdaD
                    self.cov.update({i: pinv(V2)})
            else: # all gammas are 0
                for i in data.grp_uniids[g]:
                    V2 = np.dot(data.id_Z[i].T, data.id_Z[i])
                    V2 = V2/params.sigma2 + params.lambdaD
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
                    temp1 = np.zeros([np.sum(nzro_gamma), 1])
                    for j in data.grp_uniids[g]:
                        jdx = np.where(data.uniids == j)[0][0]
                        temp1 += np.dot(data.id_X[j][:, nzro_gamma].T,
                                        data.id_y[j] - \
                                        np.dot(data.id_W[j], params.alpha) - \
                                        np.dot(data.id_Z[j],
                                               params.b[jdx,:][:, np.newaxis]))
                    temp1 += np.dot(self.__xTz__[i],
                                    params.b[idx,:][:, np.newaxis])
                    temp1 = np.dot(self.__xTz__[i].T,
                                   np.dot(self.__inv_xxsum__[g], temp1))
                    temp1 = temp1/data.grp_dtot[g]
                    temp2 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            (1.0 + 1.0/data.grp_dtot[g])*\
                            np.dot(data.id_X[i][:, nzro_gamma],
                                   params.beta[gdx, nzro_gamma][:, np.newaxis])
                    temp2 = np.dot(data.id_Z[i].T, temp2)
                    self.mean.update({i: np.dot(self.cov[i],
                                                (temp1+temp2)/params.sigma2)})
            else: # all gammas are 0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp2 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)
                    temp2 = np.dot(data.id_Z[i].T, temp2)
                    self.mean.update({i: np.dot(self.cov[i],
                                                temp2/params.sigma2)})

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
        self._cov_(data, params)
        self._mean_(data, params)

    def _cov_(self, data, params):
        '''Calculate covariance matrix of the posteriors.'''
        self.cov = {}
        self.__nzro_gamma__ = {}
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = params.gamma[gdx,:]!=0
            self.__nzro_gamma__.update({g: nzro_gamma})
            if nzro_gamma.any(): # not all 0's
                temp = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
                for i in data.grp_uniids[g]:
                    temp += np.dot(data.id_X[i][:, nzro_gamma].T,
                                   data.id_X[i][:, nzro_gamma])
                temp = (data.grp_dtot[g] + 1.0/data.grp_dtot[g])*temp
                self.cov.update({g: pinv(temp)*params.sigma2})
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
                self.mean.update({g: np.dot(self.cov[g], temp)})
            else: # all gammas are 0
                pass

    def getUpdates(self):
        '''Generate random samples from the posterior dist.'''
        samples = np.zeros(self.__shape__)
        for gdx in range(self.__shape__[0]):
            g = self.__unidiets__[gdx]
            try:
                samples[gdx, self.__nzro_gamma__[g]] = \
                    np.random.multivariate_normal(self.mean[g][:,0],
                                                  self.cov[g])
            except:
                pass
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
        self._a_(data, params)
        self._scale_(data, params)

    def _a_(self, data, params):
        '''Calculate the shape parameter of the posterior.'''
        self.a = (np.sum(data.grp_dtot.values()) + np.sum(params.gamma))/2.0

    def _scale_(self, data, params):
        '''Calculate the scale parameter of the posterior.'''
        self.scale = 0
        for gdx in range(data.grp):
            g = data.unidiets[gdx]
            nzro_gamma = (params.gamma[gdx,:]!=0)
            if nzro_gamma.any(): # not all 0's
                temp_beta = params.beta[gdx, nzro_gamma][:, np.newaxis]
                temp1 = 0
                temp2 = 0
                temp3 = 0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp_x = data.id_X[i][:, nzro_gamma]
                    temp4 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            np.dot(data.id_Z[i],
                                   params.b[idx,:][:, np.newaxis])
                    temp1 += np.dot((temp4 - np.dot(temp_x, temp_beta)).T,
                                    (temp4 - np.dot(temp_x, temp_beta)))
                    temp2 += np.dot(temp_x.T, temp_x)
                    temp3 += np.dot(temp_x.T, temp4)
                temp5 = temp_beta - np.dot(pinv(temp2), temp3)
                self.scale += temp1 + np.dot(temp5.T,
                                      np.dot(temp2, temp5))/data.grp_dtot[g]
            else: # all gammas are 0
                temp1 = 0
                for i in data.grp_uniids[g]:
                    idx = np.where(data.uniids == i)[0][0]
                    temp4 = data.id_y[i] - np.dot(data.id_W[i], params.alpha)-\
                            np.dot(data.id_Z[i],
                                   params.b[idx,:][:, np.newaxis])
                    temp1 += np.dot(temp4.T, temp4)
                self.scale += temp1
        self.scale = self.scale/2.0

    def getUpdates(self):
        return invgamma.rvs(a = self.a, scale = self.scale)


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
        self.pai = priors.pai
        self.l = data.l
        self.y = data.id_y
        self.W = data.id_W
        self.Z = data.id_Z
        self.X = data.id_X
        self.grp = data.grp
        self.unidiets = data.unidiets
        self.grp_uniids = data.grp_uniids
        self.uniids = data.uniids
        self.alpha = params.alpha
        self.b = params.b
        self.gamma = params.gamma
        self.sigma2 = params.sigma2

    def _S_(self, gdx, temp = None):
        '''Calculate the S function.'''
        g = self.unidiets[gdx]
        if temp == None:
            nzro_gamma = (self.gamma[gdx,:]!=0)
            return self._SMain_(gdx, g, nzro_gamma)
        else:
            nzro_gamma = (temp!=0)
            return self._SMain_(gdx, g, nzro_gamma)

    def _SMain_(self, gdx, g, nzro_gamma):
        if nzro_gamma.any(): # not all 0's
            temp_xTx = np.zeros([np.sum(nzro_gamma), np.sum(nzro_gamma)])
            temp_xTphi = np.zeros([np.sum(nzro_gamma), 1])
            for i in self.grp_uniids[g]:
                idx = np.where(self.uniids == i)[0][0]
                temp1 = self.X[i][:, nzro_gamma]
                temp2 = self.y[i] - np.dot(self.W[i], self.alpha) - \
                        np.dot(self.Z[i], self.b[idx,:][:, np.newaxis])
                temp_xTx += np.dot(temp1.T, temp1)
                temp_xTphi += np.dot(temp1.T, temp2)
            return np.sqrt(1.0/det(temp_xTx))*np.exp(np.dot(temp_xTphi.T,
                   np.dot(pinv(temp_xTx), temp_xTphi))/(2.0*self.sigma2))
        else: # all gammas are 0
            return 0.0

    def getUpdates(self):
        for gdx in range(self.grp):
            for l in range(self.l):
                temp_gamma = self.gamma[gdx, :]
                temp_gamma[l] = np.random.binomial(1, self.pai[gdx])
                if temp_gamma[l] != self.gamma[gdx, l]:
                    S_temp = self._S_(gdx, temp = temp_gamma)
                    S = self._S_(gdx)
                    if S != 0:
                        hasting_ratio = (2*np.pi*self.sigma2)**\
                                    (0.5*(temp_gamma[l] - self.gamma[gdx, l]))*\
                                    S_temp/S
                    else:
                        # only denominator == 0
                        # note that denominator and nominator cannot be both 0,
                        # since temp_gamma != gamma
                        hasting_ratio = 1.0
                    u = np.random.uniform()
                    if hasting_ratio > u:
                        self.gamma[gdx, l] = temp_gamma[l]
        return self.gamma
