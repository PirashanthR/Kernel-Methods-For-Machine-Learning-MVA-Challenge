'''
RBFKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains the implementation of the really basic RBF Kernel
'''

import numpy as np


def kernel_rbf(x, y,gamma):
    """ 
    Compute the basic RBF kernel function between points x and y given the parameter gamma
    Param: @x: (np.array) data 1 to use to feed the rbf rk
    @y: (np.array) data 2 to use to feed the rbf rk
    @gamma: (float) gamma parameter of the rbf kernel
    """
    return np.exp(- gamma * np.linalg.norm(x- y)**2)


def RBF_Gram(X,gamma,Y=[]):
    """ 
    Compute the gram matrix of X and Y (if Y is empty we compute the gram matrix 
    for training comparing all the elements of X) usign the rbf kernel
    Param: @X: (np.array)(nb_sample,nb_features) Training data
    @gamma: (float) gamma parameter of the rbf kernel
    @Y: (np.array)(nb_sample,nb_features) Testing data (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    """
    if len(Y)==0:
        len_X = X.shape[0]
        gram_matrix = np.zeros((len_X, len_X), dtype=np.float32)

        for i in range(len_X):
            for j in range(i,len_X):
                gram_matrix[i,j] = kernel_rbf(X[i],X[j],gamma)
                gram_matrix[j,i] = gram_matrix[i,j]
        
        return gram_matrix
    else:
        len_X = X.shape[0]
        len_Y = Y.shape[0]
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)
        
        for i in range(len_X):
            for j in range(len_Y):
                gram_matrix[i,j] = kernel_rbf(X[i],Y[j],gamma)
                
        return gram_matrix

                
        