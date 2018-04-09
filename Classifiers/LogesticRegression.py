'''
Logestic Regression -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains a basic logestic regression implementation
'''

import numpy as np 
import math

###########"Définition de fonctions simples mais utilisés fréquemment
sigmoid = lambda x: 1./(1+np.exp(-x)) # sigmoid
eta = lambda x,w: sigmoid(np.dot(w.transpose(),x)) #fonction eta du rapport
inverse_sigmoid = lambda x: math.log(1/x-1)  #inverse de la sigmoid
#####################################################################

def compute_D_eta(X_data,w):
    '''
    Intermed to compute the hessian that is required to compute the IRLS algorithm
    Paramètres: X_data :(np.array(nb_samples,nb_composante)) data matrix
                w : np.array(nb_composante,1) parameter of the log reg
    Retrun: La matrice D_eta evalue
    '''
    diag_compo = eta(X_data,w)
    diag_compo = diag_compo*(1-diag_compo) 
    return np.diag(diag_compo[0,:])

class LogisticRegression:
    '''
    Class LogisticRegression: create a classifier based on the logestic regression
    Attributes: - coef : parameters of the logestic regression
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        self.coef = 0
    
    def fit(self,data_raw,label,coef_old= np.zeros((51,1)),tolerance=0.01,lambda_regularisation=0):
        '''
        fit: Compute the IRLS algorithm in order to learn the parameters
        Paramaters: data_raw: (np.array(nb_samples,nb_composante)) data_raw data matrix
                    label: (np.array(nb_samples,)) true labels of the samples
                    coef_old:np.array(nb_composante,1) old coefficients before update
                    tolerance: const, stopping criteria
                    lambda_regularisation: 0=<const=<1 regularization parameter
        Return: The new learned parameters
        '''   
        ### matrice diagonale
        size_changed=False
        if label.ndim==1:
            label.resize([1,label.shape[0]])
            size_changed = True
        one_vector = np.ones((data_raw.shape[0],1))
        data = np.concatenate((data_raw,one_vector),axis=1)
        
        Diag = compute_D_eta(data.transpose(),coef_old)
        
        ### Hessienne et son inverse : pas de descente
        Hessian = np.dot(data.transpose(), np.dot(Diag,data)) + lambda_regularisation*np.eye(data.shape[1])
        Inv= np.linalg.inv(Hessian)
        Grad = np.dot((label-eta(data.transpose(),coef_old)),data).transpose() - lambda_regularisation*coef_old
        #### Terme de descente 
        D = np.dot(Inv,Grad)
        
        if size_changed:
            label.resize([label.shape[1],])
        #print(sum(label*np.log(eta(data.transpose(),coef_old))[0]) + sum((1-label)*np.log(eta(data.transpose(),-coef_old))[0]))
        print(np.linalg.norm(D))
        if (np.linalg.norm(D)<tolerance):
            self.coef = coef_old
        else:
            self.fit(data_raw,label,coef_old+D,tolerance,lambda_regularisation)
    
    def get_coef(self):
        '''
        get_coef: Return the coefficient of the class
        Paramaters: -
        Return: the model parameters
        '''
        return list(self.coef)

    def predict(self,data):
        '''
        predict: give the probability to obtain the label y=1 for the given data
        Paramaters: - data : (np.array(nb_samples,nb_composante)) data matrix
        Return: probabilities associated to each data in the data matrix as  a np.array
        '''
        if data.ndim == 1:
            data.resize([data.shape[0],1])
        one_vector = np.ones([1,data.shape[0]])
        return eta(np.concatenate((data.transpose(),one_vector)),self.coef)
    
    def predict_class(self,data):
        '''
        predict_class: Predict the label for a given set of data
        Paramaters: - data : (np.array(nb_samples,nb_composante)) data matrix
        Return: labels
        '''
        return np.array(self.predict(data)>=0.5,dtype=int)
    
