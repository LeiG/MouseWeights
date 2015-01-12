#!/usr/bin/python

'''
posterior inference for longitudinal Bayesian variable selection model
'''

import numpy as np
from numpy.linalg import pinv, det
from scipy.stats import invgamma
#import ipdb

def XXsum(g_index, uni_diet, id_g, gamma, X):
    temp = np.zeros([np.sum(gamma[g_index]), np.sum(gamma[g_index])])
    g = uni_diet[g_index]

    for i in id_g[g]:
        temp += np.dot(X[i][:,gamma[g_index]!=0].T,
                       (X[i][:,gamma[g_index]!=0]))
    return temp

def Sfunction(g, alpha, b, gamma, sigma2, p, id_g,
              uni_id, uni_diet, W, X, Z, y):
    g_index = np.where(uni_diet == g)[0][0]
    temp1 = XXsum(g_index, uni_diet, id_g, gamma, X)
    temp1 = pinv(temp1)
    temp2 = np.zeros([np.sum(gamma[g_index]), 1])

    for i in id_g[g]:
        i_index = np.where(uni_id == i)[0][0]
        temp2 += np.dot(X[i][:,gamma[g_index]!=0].T, y[i] - W[i].dot(alpha)
                        - np.dot(Z[i], b[i_index].reshape(p, 1)))

    S_out = np.sqrt(det(temp1))*np.exp(np.dot(temp2.T.dot(temp1), temp2)
                                       /(2.0*sigma2))
    return S_out


def update_gamma(g_index, l, alpha, b, gamma, sigma2, pi_g, p,
                 id_g, uni_id, uni_diet, W, X, Z, y):
    g = uni_diet[g_index]
    gamma_temp = gamma.copy()
    gamma_temp[g_index, l] = np.random.binomial(1, pi_g)

    if gamma_temp[g_index, l] != gamma[g_index, l]:
        if np.all(gamma[g_index] == 0):  #check if all gamma's are 0
            S = 1
            S_temp = 1
        else:  #if not all gamma's are 0
            S = Sfunction(g, alpha, b, gamma, sigma2, p, id_g,
                          uni_id, uni_diet, W, X, Z, y)
            if np.all(gamma_temp[g_index] == 0):  #if all gamma_temp's are 0
                S_temp = 0.0
            else:
                S_temp = Sfunction(g, alpha, b, gamma_temp, sigma2,
                                   p, id_g, uni_id, uni_diet, W, X, Z, y)
        #Hastings ratio
        H_ratio = np.power(2.0*np.pi*sigma2,
                           0.5*(gamma_temp[g_index, l]
                                - gamma[g_index, l]))*S_temp/S
        u = np.random.uniform()  #generate uniform r.v.
        if H_ratio > u:
            gamma[g_index, l] = gamma_temp[g_index, l]  #update gamma[v, j]

    return gamma[g_index, l]

def update_beta(g_index, beta, alpha, b, gamma, sigma2, p,
                uni_diet, uni_id, id_g, N_g, W, X, Z, y):
    g = uni_diet[g_index]
    if np.all(gamma[g_index] == 0):  #if all gamma's are 0
        return beta[g_index]  #no updates
    else:  #if not all gamma's are 0
        V3 = XXsum(g_index, uni_diet, id_g, gamma, X)
        V3 = V3*(N_g[g]+1.0/N_g[g])/sigma2
        V3_inv = pinv(V3)
        mean3 = np.zeros([np.sum(gamma[g_index]),1])
        for i in id_g[g]:
            i_index = np.where(uni_id == i)[0][0]
            mean3 += np.dot(X[i][:,gamma[g_index]!=0].T, y[i] - W[i].dot(alpha)
                            - np.dot(Z[i], b[i_index].reshape(p, 1)))
        mean3 = np.dot(V3_inv, mean3)

    #update
    beta[g_index][gamma[g_index]!=0] = np.random.multivariate_normal(
        mean3.reshape(np.sum(gamma[g_index]),),
        V3_inv.reshape(np.sum(gamma[g_index]), np.sum(gamma[g_index])))
    return beta[g_index]

def update_alpha(sigma2, beta, gamma, b, d3, d4, p, N_g,
                 G, id_g, uni_diet, uni_id, W, X, Z, y):
    V1 = d4
    mean1 = d4.dot(d3)
    temp1 = np.zeros([p, p])
    temp2 = np.zeros([p, p])
    temp4 = np.zeros([p, 1])
    temp7 = np.zeros([p, 1])

    for g_index in range(G):
        g = uni_diet[g_index]
        if np.all(gamma[g_index] == 0):  #check if all gamma's are 0
            for i in id_g[g]:
                i_index = np.where(uni_id == i)[0][0]
                temp1 += np.dot(W[i].T, W[i])
                temp4 += np.dot(W[i].T,
                                (y[i] - np.dot(Z[i],
                                               b[i_index].reshape(p, 1))))
        else:  #if not all gamma's are 0
            temp3 = np.zeros([np.sum(gamma[g_index]), p])
            temp5 = np.zeros([np.sum(gamma[g_index]), 1])
            for i in id_g[g]:
                i_index = np.where(uni_id == i)[0][0]
                temp1 += np.dot(W[i].T, W[i])
                temp3 += np.dot(X[i][:,gamma[g_index]!=0].T, W[i])
                temp4 += np.dot(W[i].T, (y[i] - np.dot(X[i][:,gamma[g_index]!=0],beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]), 1)) - np.dot(Z[i], b[i_index].reshape(p, 1))))
                temp5 += np.dot(X[i][:,gamma[g_index]!=0].T, (y[i] - np.dot(X[i][:,gamma[g_index]!=0], beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]), 1)) - np.dot(Z[i], b[i_index].reshape(p, 1))))
            temp6 = XXsum(g_index, uni_diet, id_g, gamma, X)
            temp6 = pinv(temp6)
            temp2 += np.dot(temp3.T, temp6).dot(temp3)/N_g[g]
            temp7 += np.dot(temp3.T, temp6).dot(temp5)/N_g[g]
        V1 += (temp1 + temp2)/sigma2
        mean1 += (temp4 + temp7)/sigma2

    V1_inv = pinv(V1)
    mean1 = np.dot(V1_inv, mean1).reshape(p, )

    #update
    alpha_new = np.random.multivariate_normal(mean1, V1_inv).reshape(p, 1)
    return alpha_new

def update_lambda_D(b, d1, d2, N_id, uni_id, uni_diet, id_g, p, G):
    temp1 = N_id*p/2.0 + d1
    temp2 = d2
    for g_index in range(G):
        g = uni_diet[g_index]
        for i in id_g[g]:
            i_index = np.where(uni_id == i)[0][0]
            temp2 += np.dot(b[i_index].reshape(p, 1).T,
                            b[i_index].reshape(p, 1))/2.0

    #update
    lambda_D_new = np.random.gamma(temp1, 1.0/temp2)
    return lambda_D_new

def update_b(i_index, b, alpha, beta, gamma, sigma2, lambda_D,
             N_g, uni_id, uni_diet, id_g, p, W, X, Z, y):
    i = uni_id[i_index]

    for g_search, i_search in id_g.iteritems():
        if np.any(i_search == i):
            g = g_search

    g_index = np.where(uni_diet == g)[0][0]

    if np.all(gamma[g_index] == 0):  #check if all gamma's are 0
        V2 = lambda_D + np.dot(Z[i].T, Z[i])/sigma2
        mean2 = np.dot(pinv(V2), np.dot(Z[i].T, y[i]-W[i].dot(alpha)))/sigma2
    else:
        V2 = lambda_D + np.dot(Z[i].T, Z[i])/sigma2
        temp1 = XXsum(g_index, uni_diet, id_g, gamma, X)
        temp1 = pinv(temp1)
        V2 = V2 + np.dot(np.dot(np.dot(Z[i].T, X[i][:,gamma[g_index]!=0]), temp1), (np.dot(X[i][:,gamma[g_index]!=0].T, Z[i])))/(sigma2*N_g[g])
        mean2 = np.dot(Z[i].T, y[i] - W[i].dot(alpha) - np.dot(X[i][:,gamma[g_index]!=0], beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]),1)))
        temp2 = np.dot(X[i][:,gamma[g_index]!=0].T, Z[i].dot(b[i_index].reshape(p, 1)))
        for j in id_g[g]:
            j_index = np.where(uni_id == j)[0][0]
            temp2 += np.dot(X[j][:,gamma[g_index]!=0].T, y[j] - W[j].dot(alpha) - Z[j].dot(b[j_index].reshape(p, 1)))
        mean2 = mean2 + np.dot(np.dot(Z[i].T.dot(X[i][:,gamma[g_index]!=0]), temp1), temp2)/N_g[g]
        mean2 = np.dot(pinv(V2), mean2)/sigma2

    #update
    b_new = np.random.multivariate_normal(mean2.reshape(p,), pinv(V2)).reshape(p, )
    return b_new

def update_sigma2(alpha, beta, gamma, b, G, n_i, uni_diet, id_g, uni_id, N_g, p, W, X, Z, y):
    temp1 = (np.sum(N_g.values()) + np.sum(gamma))/2.0
    temp3 = 0.0
    for g_index in xrange(G):
        g = uni_diet[g_index]
        temp2 = 0.0
        if np.all(gamma[g_index] == 0):  #check if all gamma's are 0
            for i in id_g[g]:
                i_index = np.where(uni_id == i)[0][0]
                temp2 += (y[i] - np.dot(W[i],alpha) - np.dot(Z[i], b[i_index].reshape(p, 1))).T.dot(y[i] - np.dot(W[i],alpha) - np.dot(Z[i], b[i_index].reshape(p, 1)))
            temp3 += temp2
        else:  #if not all gamma's are 0
            temp6 = XXsum(g_index, uni_diet, id_g, gamma, X)
            temp5 = np.zeros([np.sum(gamma[g_index]),1])
            for i in id_g[g]:
                i_index = np.where(uni_id == i)[0][0]
                temp2 += (y[i] - np.dot(W[i],alpha) - np.dot(X[i][:,gamma[g_index]!=0], beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]),1)) - np.dot(Z[i], b[i_index].reshape(p, 1))).T.dot(y[i] - np.dot(W[i],alpha) - np.dot(X[i][:,gamma[g_index]!=0], beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]),1)) - np.dot(Z[i], b[i_index].reshape(p, 1)))
                temp5 += np.dot(X[i][:,gamma[g_index]!=0].T, y[i] - np.dot(W[i], alpha) - np.dot(Z[i], b[i_index].reshape(p, 1)))
            temp4 = np.dot((beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]),1) - pinv(temp6).dot(temp5)).T, temp6).dot(beta[g_index][gamma[g_index]!=0].reshape(np.sum(gamma[g_index]),1) - pinv(temp6).dot(temp5))
            temp3 += temp2 + temp4/N_g[g]
    temp3 = temp3/2.0

    #update
    sigma2_new = invgamma.rvs(temp1, scale = temp3, size = 1)
    return sigma2_new


def mcmc_update(p, L, uni_days, uni_diet, uni_id, G,
                N_id, N_g, id_g, n_i, W, X, Z, y):
    # set parameters
    N_sim = 10000

    # hyper parameters
    d1 = 122.3124
    d2 = 2275.353
    d3 = np.array([50.979240, -5.563603]).reshape(p, 1)
    d4 = pinv(np.array([0.13397276, -0.07849482,
                        -0.07849482, 0.05860082]).reshape(p, p))
    pi_g = 0.5*np.ones(G)

    # initial values
    alpha = d3.copy()
    beta = np.zeros([G*L, 1])
    gamma = np.zeros([G*L, 1])
    lambda_D = np.array(0.05340017).reshape(1, 1)
    b = np.random.normal(0, 1.0/lambda_D, size = N_id*p).reshape(N_id*p, 1)
    sigma2 = np.array(2.248769**2).reshape(1, 1)


    # updates
    alpha_now = alpha.copy().reshape(p, 1)
    beta_now = beta.copy().reshape(G, L)
    gamma_now = gamma.copy().reshape(G, L)
    b_now = b.copy().reshape(N_id, p)
    sigma2_now = sigma2.copy().reshape(1, 1)
    lambda_D_now = lambda_D.copy().reshape(1, 1)

    # MCMC updates
    for iters in xrange(N_sim):
        # update gamma
#         print "Update gamma"
        for g_index in xrange(G):
            for l in xrange(L):
                gamma_now[g_index, l] = update_gamma(g_index, l, alpha_now,
                                                     b_now, gamma_now,
                                                     sigma2_now, pi_g[g_index],
                                                     p, id_g, uni_id, uni_diet,
                                                     W, X, Z, y)

        # update beta
#         print "Update beta"
        for g_index in xrange(G):
            beta_now[g_index] = update_beta(g_index, beta_now, alpha_now,
                                            b_now, gamma_now, sigma2_now,
                                            p, uni_diet, uni_id, id_g, N_g,
                                            W, X, Z, y)

        # update alpha
#         print "Update alpha"
        alpha_now = update_alpha(sigma2_now, beta_now, gamma_now, b_now,
                                 d3, d4, p, N_g, G, id_g, uni_diet,
                                 uni_id, W, X, Z, y)

        # update lambda_D
#         print "Update lambda_D"
        lambda_D_now = np.array(update_lambda_D(b_now, d1, d2, N_id,
                                                uni_id, uni_diet,
                                                id_g, p, G)).reshape(1, 1)

        # update b
#         print "Update b"
        for i_index in xrange(N_id):
            b_now[i_index] = update_b(i_index, b_now, alpha_now, beta_now,
                                      gamma_now, sigma2_now, lambda_D_now,
                                      N_g, uni_id, uni_diet, id_g, p, W,
                                      X, Z, y)

        # update sigma2
#         print "Update sigma2"
        sigma2_now = np.array(update_sigma2(alpha_now, beta_now, gamma_now,
                                            b_now, G, n_i, uni_diet,
                                            id_g, uni_id, N_g, p,
                                            W, X, Z, y)).reshape(1, 1)

        # store updates
        alpha = np.hstack([alpha, alpha_now])
        beta = np.hstack([beta, beta_now.reshape(G*L, 1)])
        gamma = np.hstack([gamma, gamma_now.reshape(G*L, 1)])
        b = np.hstack([b, b_now.reshape(N_id*p, 1)])
        sigma2 = np.hstack([sigma2, sigma2_now])
        lambda_D = np.hstack([lambda_D, lambda_D_now])

        # write to file
#         np.savetxt(dirname+'/alpha', alpha)
#         np.savetxt(dirname+'/beta', beta)
#         np.savetxt(dirname+'/gamma', gamma)
#         np.savetxt(dirname+'/b', b)
#         np.savetxt(dirname+'/sigma2', sigma2)
#         np.savetxt(dirname+'/lambda_D', lambda_D)
        np.savetxt('alpha.txt', alpha)
        np.savetxt('beta.txt', beta)
        np.savetxt('gamma.txt', gamma)
        np.savetxt('b.txt', b)
        np.savetxt('sigma2.txt', sigma2)
        np.savetxt('lambda_D.txt', lambda_D)
        print iters


    return 1
