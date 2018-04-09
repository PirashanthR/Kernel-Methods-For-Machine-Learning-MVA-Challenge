'''
Kernel Logestic Regression -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains our implementation of the kernel logestic regression
'''

import numpy as np 
import math
#import seaborn as sns
#from Utils import *

###########"Définition de fonctions simples mais utilisés fréquemment
sigmoid = lambda x: 1./(1+np.exp(-x)) # sigmoid
eta = lambda x,w: sigmoid(np.dot(w.transpose(),x)) #fonction eta du rapport
inverse_sigmoid = lambda x: math.log(1/x-1)  #inverse de la sigmoid
#####################################################################

'''
All the following functions are intermed calculus that appears in the IRLS algorithm computation
One can see the slides about Kernel logestic Regression (slide 105-111 in the MVA Kernel Methods for Machine Learning
class)
'''
def compute_m(p_Kernel_Mat,p_alpha):
    return p_Kernel_Mat.dot(p_alpha)

def compute_P(p_y,p_m):
    return np.diag( -sigmoid(-p_y*p_m)[:,0])

def compute_W(p_y,p_m):
    diag_compo = sigmoid(p_y*p_m)
    diag_compo = diag_compo*(1-diag_compo) 
    return np.diag(diag_compo[:,0])

def compute_Z(p_m,p_y):
    z= p_m+p_y/sigmoid(-p_y*p_m)
    return z

def compute_alpha(p_Kernel,p_W,p_z,lambda_reg):
    W_12 = np.sqrt(p_W)
    n = p_Kernel.shape[0]
    to_inv = W_12.dot(p_Kernel).dot(W_12)+n*lambda_reg*np.eye(n)
    to_inv = np.linalg.inv(to_inv)
    alpha = W_12.dot(to_inv).dot(W_12).dot(p_z)
    return alpha

class KernelLogisticRegression:
    '''
    Class KernelLogisticRegression: create a kernel logestic regression binary classifier
    Attributes: - alpha_ : alpha parameter (that appears because of the representer theorem)
    '''
    
    def __init__(self,init_coef=0):
        '''
        Constructor
        '''
        if init_coef==0:
            self.alpha_ = 0
    
    def fit(self,kernel_train,label,alpha=None,tolerance=1,lambda_regularisation=0):
        '''
        Fonction fit: Compute the IRLS algorithm in order to learn the parameters
        Paramaters: kernel_train: (np.array(nb_samples,nb_samples)) gram training matrix
                    label: (np.array(nb_samples,)) true labels of the samples
                    coef_old:np.array(nb_composante,1) old coefficients before update
                    tolerance: const, stopping criteria
                    lambda_regularisation: 0=<const=<1 regularization parameter
        Return: The new learned parameters
        '''  
        ### matrice diagonale
        size_changed=False
        if label.ndim==1:
            label.resize([label.shape[0],1])
            size_changed = True
        
        if np.array((alpha==None)).any():
            alpha = np.random.rand(kernel_train.shape[0],1)
        
        old_alpha = np.array(alpha)
        m = compute_m(kernel_train,alpha)
        P = np.nan_to_num(compute_P(label,m))
        W = np.nan_to_num(compute_W(label,m))
        z = compute_Z(m,label)
        alpha = compute_alpha(kernel_train,W,z,lambda_regularisation)
        #print(alpha)
        if size_changed:
            label.resize([label.shape[0],])
        #print(sum(label*np.log(eta(data.transpose(),coef_old))[0]) + sum((1-label)*np.log(eta(data.transpose(),-coef_old))[0]))
        print(np.linalg.norm(alpha-old_alpha))
        if (np.linalg.norm(alpha-old_alpha)>tolerance):
            self.fit(kernel_train,label,alpha,tolerance,lambda_regularisation)
        else:
            self.alpha_=alpha
    
    def get_coef(self):
        '''
        get_coef: Return the alpha parameter of the class
        Paramaters: -
        Return: the model parameters
        '''
        return list(self.alpha_)

    def predict(self,kernel_test):
        '''
        predict: give the probability to obtain the label y=1 for the given data
        Paramaters: - kernel_test : (np.array(nb_samples_train,nb_sample_test)) test gram matrix
        Return: probabilities associated to each data in the data matrix as  a np.array
        '''
        prediction = ((self.alpha_.T.dot(kernel_test)).T).reshape(-1)
        prediction= sigmoid(prediction).reshape(-1)
        return prediction
    
    def predict_class(self,kernel_test):
        '''
        predict_class: Predict the label for a given set of data
        Paramaters: - kernel_test : (np.array(nb_samples_train,nb_sample_test)) test gram matrix
        Return: labels
        '''
        prediction = np.array(self.predict(kernel_test)>=0.5,dtype=int)
        prediction[prediction ==0]=-1
        return prediction


