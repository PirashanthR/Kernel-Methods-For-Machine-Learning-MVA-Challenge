'''
PCA -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains our implementation of PCA for the challenge.
This function was really usefull in order to visualize the data but it is not used in the final submission.
'''
import numpy as np


                
class PCA:
    '''
    Class PCA: compute PCA of a given data matrix
    Attributes: d: nb_components for PCA
    U,mu: Projection and mean matrix that allows to go from the original space to the lower dimension one
    '''
    def __init__(self,nb_components):
        '''
        Basic constructor of the class
        '''
        self.U = 0
        self.mu = 0
        self.d = nb_components#nb of composents
        self.A=0
        
    def fit(self,data):
        '''
        Fit function that allows to learn the PCA embedding of the data matrix given the number of components
        '''
        self.mu = np.sum(data,axis=0)/data.shape[0]
        mu_rep = np.repeat(self.mu.reshape((1,-1)),data.shape[0],0)
        self.A = (data-mu_rep).T.dot(data-mu_rep)/(data.shape[0])
        
        e, v = np.linalg.eigh(self.A)
        self.U = v[:,-self.d:].T
        order = np.array(range(self.U.shape[0]))
        order[::-1].sort()
        self.U = self.U[order,:]
        
    def transform(self,data):
        '''
        Transform function that allows to transform the original space to the lower dimension one given
        the parameters learned by calling the fit function
        '''
        mu_rep = np.repeat(self.mu.reshape((1,-1)),data.shape[0],0)
        Y = (data-mu_rep).dot(self.U.T)
        return (Y)
        
        