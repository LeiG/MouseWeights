#!/usr/bin/python

"""
Implement the relative standard deviation fixed width stopping rule.
"""

from __future__ import division
import copy
import pdb
import numpy as np

class StoppingRule:
    '''
    Parameters
    ----------
    paramsArray: a 1-d array
        Contains the updated parameters from MCMC simulation.
    eps: float (default = 0.1)
        Epsilon value in the stopping rule.
    thres: int (default = 2**14)
        The lower bound threshold to check the stopping rule.
    gap: int (default = 20)
        The gap between two checks of the stopping rule.

    Arributes
    ---------
    'update': Update batches with new paramsArray.
    '_mergeTank_': Merge nearby tanks by averaing.

    '''
    def __init__(self, paramsArray, dirname, eps=0.1, thres=2**14, gap=20):
        self.params = copy.deepcopy(paramsArray)[0, :]
        self.dirname = dirname
        self.d = len(self.params)
        self.eps = eps
        self.thres = thres
        self.gap = gap
        self.z = 1.96
        self.counter = 0
        self.bCounter = 0
        self.batchCounter = 0
        self.bSize = [np.sqrt(thres), np.sqrt(thres)]
        self.bRange = 2**np.arange(7, 15)
        self.batch = np.zeros([self.bSize[0], self.d])
        self.tankStd = np.zeros([3, self.d])

        self.batch[0, :] = copy.deepcopy(self.params)
        self.tankStd[0, :] = copy.deepcopy(self.params)

    def update(self, paramsArray):
        self.counter += 1
        self.bCounter += 1
        self.batch[self.bCounter, :] = copy.deepcopy(paramsArray)[0, :]

        # recursive standard deviation estimation
        self.tankStd[1,:] = (self.counter*self.tankStd[0,:] +\
                             self.batch[self.bCounter,:])/(self.counter+1)
        self.tankStd[2,:] = self.tankStd[2,:] + (self.counter+1)*\
                            (self.tankStd[1,:] -\
                             self.batch[self.bCounter,:])**2/self.counter
        self.tankStd[0,:] = self.tankStd[1,:]

        # check if batch is full
        if self.bCounter == (self.bSize[0] - 1):
            self.bCounter = 0
            if self.batchCounter == 0:
                self.tank = np.mean(self.batch, axis = 0)
                self.batchCounter += 1
            else:
                self.tank = np.vstack([self.tank,
                                       np.mean(self.batch, axis = 0)])
                self.batchCounter += 1

        # check between gap batches
        if (self.counter >= self.thres) and (self.batchCounter >= self.gap) and (not self.batch.shape[0]%2):
            self.batchCounter = 1
            self.bSize[1] = self.bSize[0]
            self.bSize[0] = min(
                        s for s in self.bRange if s >= np.sqrt(self.counter))

            # check if batch size changes
            if self.bSize[0] != self.bSize[1]:
                self.batch = np.zeros([self.bSize[0], self.d])
                self.tank = self._mergeTank_()

            self.tankMCSE = np.std(self.tank, axis = 0)*\
                            np.sqrt(self.bSize[0]/(self.counter+1))

            # self.cond = 2*self.z*self.tankMCSE + 1/(self.counter+1) -\
            #             self.eps*np.sqrt(self.tankStd[2,:]/self.counter)

            # remove 1/n to overcome parameters that stay 0's throughout
            self.cond = 2*self.z*self.tankMCSE + 1/(self.counter+1) -\
                        self.eps*np.sqrt(self.tankStd[2,:]/self.counter)

            np.savetxt(self.dirname+"/cond.txt", self.cond)
            if np.all(self.cond <= 0):
                np.savetxt(self.dirname+"/posterior_mean.txt",
                           np.mean(self.tank, axis = 0))
                return True

        return False

    def _mergeTank_(self):
        '''Merge nearby batches in tank.'''
        temp = np.zeros([self.tank.shape[0]/2, self.d])
        for i in range(int(self.tank.shape[0]/2)):
            temp[i,:] = np.mean(self.tank[(2*i):(2*(i+1)),:], axis = 0)
        return temp
