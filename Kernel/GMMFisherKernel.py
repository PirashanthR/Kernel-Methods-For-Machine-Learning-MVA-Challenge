'''
GMM FisherKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the Fisher Kernel in the case of a gaussian mixtures model with
diagonal covariance matrice  (Perronnin and Dance, 2007) (slides 301-307 in the MVA Kernel methods in machine learning
class).
This method allows to aggregate visual words.
'''
import numpy as np
import sys

sys.path.append('../')

from Tools.Utils import compute_patch
from Classifiers.DiagonalGaussianMixture import DiagonalGaussianMixture

def compute_patches(X_train,X_test,k_gmm,nb_cut):
    list_of_patches = []
    X_train_patch = []
    X_test_patch = []
    for i in range(len(X_train)):
        X_train_patch+= [compute_patch(X_train[i],nb_cut)]
        list_of_patches.extend(X_train_patch[-1])
        
    for j in range(len(X_test)):
        X_test_patch+= [compute_patch(X_test_patch[j],nb_cut)]    
    
    X_train_patch = np.array(X_train_patch)
    X_test_patch = np.array(X_test_patch)
    
    return X_train_patch,X_test_patch,list_of_patches

def compute_linear_kernel(x1,x2):
    """ 
    Compute the scalar product between x1 and x2 (linear kernel in the given embedding)
    Param: @x1: (np.array) data 1 to use to feed the linear kernel computation
    @x2: (np.array) data 2 to use to feed the linear kernel computation
    """
    value= np.vdot(x1,x2)
    return value

def gmm_fisher_kernel(X1,k_gmm,nb_cut,X2=[]):
    '''
    This function computes the aggregation of visual words using fisher vectors gram matrix. 
    Param: @X1: (list) list of strings in the train set
    @S: k_gmm: number of cluster in the gaussian mixtures model
    @nb_cut: number of cuts to compute the patch within the sequence
    @X2: (list)  list of strings in the test  set (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    Return: Aggregation of visual words Gram matrix
    '''
    len_X2= len(X2)
    len_X1 = len(X1)
    X_train_patch,X_test_patch,list_of_patches = compute_patches(X1,X2,k_gmm,nb_cut)
    gmm = DiagonalGaussianMixture(nb_cluster=k_gmm)
    gmm.fit(np.array(list_of_patches),epsilon=1,verbose=1)
    if len_X2 ==0:
        # numpy array of Gram matrix
        list_fisher_vector_train = np.array(gmm.compute_fisher_vector(X_train_patch))

        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)

        for i in range(len_X1):
            for j in range(i, len_X1):
                gram_matrix[i, j] = compute_linear_kernel(list_fisher_vector_train[i], list_fisher_vector_train[j])
                #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix
    else:
        list_fisher_vector_train = np.array(gmm.compute_fisher_vector(X_train_patch))
        list_fisher_vector_test = np.array(gmm.compute_fisher_vector(X_test_patch))

        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        for i in range(len_X1):
            for j in range(len_X2):
                gram_matrix[i, j] = compute_linear_kernel(list_fisher_vector_train[i], list_fisher_vector_test[j])
        
        return gram_matrix