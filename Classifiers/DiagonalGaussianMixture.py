'''
DiagonalGaussianMixture -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains our implementation of the Diagonal Gaussian Mixtures
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from Classifiers.Kmeans import KMeans


############### Fonction pour calculer les lois gaussiennes sans utiliser scipy#############
Gaussian_law_estimation_unidimensional = lambda x,sigma,mu: 1/math.sqrt(2*math.pi*sigma**2)*np.exp((x-mu)**2/(sigma**2))
compute_exponential_const_term = lambda x,Sigma,mu: (2*math.pi)**(Sigma.shape[0]/2)*math.sqrt(np.linalg.det(Sigma))
Gaussian_law_estimation_multidimensional = lambda x,Sigma,mu: 1/compute_exponential_const_term(x,Sigma,mu)*np.exp(-1/2*np.dot(np.dot((x-mu),np.linalg.inv(Sigma)),(x-mu).transpose()))
############### End Fonction pour calculer les lois gaussiennes sans utiliser scipy#############

    
class DiagonalGaussianMixture:
    '''
    Class DiagonalGaussianMixture: allows to compute the Diagonal Gaussian Mixture function
    Attributes:@k : nombre de cluster final, a fixer
               @Sigma_list : list(np.array) list of all the covariance matrices
               @mu_list :  list(np.array) list of all the mean matrices
               @pi_list :  list(float) prior probabilities
               @q_e_step : np.array (intermed EM)
    '''
    def __init__(self,nb_cluster=2):
        '''
        Constructor
        '''
        self.k = nb_cluster
        self.Sigma_list= 0
        self.mu_list =0 
        self.pi_list = 0
        self.q_e_step = 0
        
    def compute_E_step(self,data):
        '''
        Compute the E step of the EM algorithm
        Paramètres: data:(np.array(nb_samples,nb_composante)) data matrix
        '''
        for i in range(data.shape[0]):
            for k in range(self.k):
                self.q_e_step[i,k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            
            self.q_e_step[i,:] = self.q_e_step[i,:]/(np.sum(self.q_e_step[i,:]))
    
    def compute_M_step(self,data):
        '''
        Compute the M step of the EM algorithm
        Paramètres: data:(np.array(nb_samples,nb_composante)) data matrix
        '''
        for k in range(self.k):
            self.mu_list[k] = np.dot(data.transpose(),self.q_e_step[:,k])/(np.sum(self.q_e_step[:,k]))
            self.mu_list[k].resize((1,self.mu_list[k].shape[0]))
            self.pi_list[k] = np.sum(self.q_e_step[:,k])/np.sum(self.q_e_step)
            
            for j in range(data.shape[1]):
                sigma_square=0
                for i in range(data.shape[0]):
                    sigma_square += np.sum(self.q_e_step[i,k]*(data[i,j] -self.mu_list[k][0,j])**2)
                
                sigma_square = max(sigma_square/(np.sum(self.q_e_step[:,k])),0.0001)
                self.Sigma_list[k][j,j] = sigma_square            
    
    def init_q_with_kmeans(self,data):
        '''
        Initialization of the EM alorithm using Kmeans
        Paramètres: data:(np.array(nb_samples,nb_composante)) data matrix
        '''
        self.q_e_step = np.zeros([data.shape[0],self.k])
        km = KMeans(self.k)
        print('fit kmeans')
        km.fit(data)
        prediction = km.predict(data)
        for i in range(data.shape[0]):
            self.q_e_step[i,prediction[i]]=1
    
    def compute_log_likelihood_approx(self,data):
        '''
        Compute the approximation of the likelihood
        Paramètres: data:(np.array(nb_samples,nb_composante))  data matrix
        '''
        q = np.zeros([data.shape[0],self.k])
        current_log=0
        for i in range(data.shape[0]):
            for k in range(self.k):
                q[i,k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            q[i,:] = q[i,:]/(np.sum(q[i,:]))
            for k in range(self.k):
                current_log += self.q_e_step[i,k]*(math.log(self.pi_list[k]) + math.log(Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])))
        return current_log
    
    
    
    def compute_current_log_likelihood(self,data):
        '''
        Compute the likelihood
        Paramètres: data:(np.array(nb_samples,nb_composante)) data matrix
        '''
        
        current_log = 0
        for i in range(data.shape[0]):
            current_log_k=0
            for k in range(self.k):
                current_log_k += self.pi_list[k]*(Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k]))
            current_log += math.log(current_log_k)
        return current_log
    
    def fit(self,data,epsilon = 1e-5,verbose=0):
        '''
        fit: compute the EM algorithm in order to learn the parameters
        Paramètres: data: (np.array(nb_samples,nb_composante)) data matrix
                    epsilon: (float) convergence threshold
                    verbose: (0 ou 1) print outcome at each iterations or not 
        Return: -
        '''
        self.init_q_with_kmeans(data)
        
        self.mu_list = [None]*self.k
        self.pi_list = [None]*self.k
        self.Sigma_list = [np.zeros((data.shape[1],data.shape[1])) for l in range(self.k)]
        
        self.compute_M_step(data)
        old_lg_like = -float('Inf')
        lg_like = 0
        nb_iteration = 0
        
        while (abs(lg_like-old_lg_like)>epsilon):
            
            nb_iteration +=1
            old_lg_like = lg_like
            self.compute_E_step(data)
            self.compute_M_step(data)
            lg_like= self.compute_current_log_likelihood(data)

            if (verbose==1):
                print('Iteration ',nb_iteration,'Log likelihood ',lg_like)
    
    def predict(self,data):
        '''
        predict: Hard clustering of all the data
        Paramètres: - data : (np.array(nb_samples,nb_composante)) data matrix
        Return: list of labels
        '''
        q = np.zeros([self.k])
        label = []
        for i in range(data.shape[0]):
            for k in range(self.k):
                q[k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            
            label.append(np.argmax(q))
        return np.array(label)
            
    def compute_fisher_vector(self,data):
        #data dimension (n_sample,n_patch,n_features)
        '''
        compute_fisher_vector Compute the aggregation of visual words using fisher vectors (Perronin and Dance 2007)
        The equations used to compute the fisher vector are given in the slide of the class (301-307)
        Paramètres: - data : (np.array(nb_samples,nb_composante)) data matrix
        Return: new feature matrix
        '''
        q = np.zeros([data.shape[1],self.k])
        list_of_fisher_vectors= []
        for n in range(data.shape[0]):
            list_curfish = []
            for i in range(data.shape[1]):
                for j in range(self.k):
                    q[i,j] = self.pi_list[j]*Gaussian_law_estimation_multidimensional(data[n,i],self.Sigma_list[j],self.mu_list[j])
            
                q[i,:] = q[i,:]/np.sum(q[i,:])
                
            for j in range(self.k):
                constant = 1/(data.shape[1]*math.sqrt(self.pi_list[j]))
                phi_mu = sum([q[i,j]*(data[n,i] -self.mu_list[j]) for i in range(data.shape[1])])
                phi_mu = constant*np.array(phi_mu)/np.sqrt(np.diagonal(self.Sigma_list[j]))
                
                constant = constant*1/math.sqrt(2)
                phi_sigma = constant*sum([q[i,j]*((data[n,i] -self.mu_list[j])**2/np.diagonal(self.Sigma_list[j])-1) for i in range(data.shape[1])])
                list_curfish.extend(list(phi_mu.reshape(-1)) + list(phi_sigma.reshape(-1)))
                
            list_of_fisher_vectors.append(list_curfish)
            
        return np.array(list_of_fisher_vectors)
                
            