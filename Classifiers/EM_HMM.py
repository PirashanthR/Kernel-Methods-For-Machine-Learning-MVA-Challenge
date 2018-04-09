'''
EM_HMM -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains our implementation of the EM HMM that can be aligned on multiple sequences.
It allows to compute the so called fisher vectors (Jaakola 1999)
'''

#import pandas as pd
import numpy as np
import math
import matplotlib 
import matplotlib.pyplot as plt


#################Define Gaussian multi################""""""""
Gaussian_multi = lambda x,Sigma,mu: math.exp(-0.5*np.dot( np.transpose(x-mu),np.dot(np.linalg.inv(Sigma),x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(Sigma)))   

Proba_multi = lambda x,eta: eta[list(x).index(1)]


def LogSumExp(x):
    '''
    Compute the sum in the log domain (in order to avoid numerical issues)
    Attributs: x : elements to sum
    Return: Final sum
    '''
    n= len(x) 
    x_max = np.max(x)
    s=0
    for i in range(n) :
        s += np.exp(x[i]-x_max)
    return x_max + np.log(s) 

class EM_HMM:
    '''
    Class EM_HMM: HMM with underlying GM law
    Attributs: - k : number of clusters
               - Sigma_list : list(np.array) list of covariance matrices
               - mu_list :  list(np.array) list of means
               - pi_list :  list(float) list of prior probabilities for each cluster
               - q_e_step : np.array: probability that element i is in cluster k 
    '''
    def __init__(self,data,k=4):
        '''
        Constructor: Initialize the class
        '''

        self.k = k
        A0 = (1/self.k)*np.ones((self.k,self.k)) #Assume 4 hidden states

        self.eta = []
        
        for i in range(self.k):
            self.eta.append(list(np.random.dirichlet(np.ones(data.shape[2]),size=1)[0]))
             
        self.A = A0
        self.pi_0 = 1/self.k*np.ones((1,self.k))

        self.q_e_step = np.zeros([data.shape[0],data.shape[1],self.k]) 
        
    def compute_log_alpha(self, datas):
        '''
        Alpha recursion in logarithm domain given the observations and the GM law
        Parameter: np.array data : observations
        return: alphas for all the time steps
        '''
        alpha = []
        
        for data in datas:
            alpha_list = []
            alpha_0 = []
            for q in range(self.k): 
                alpha_0.append( np.log(self.pi_0[0,q]*Proba_multi(data[0],self.eta[q]))) 
            
            alpha_0 = np.array(alpha_0)    
            alpha_list.append(alpha_0)
            T=len(data)
            for t in range(1,T):
               alpha_prev = alpha_list[-1]
               alpha_t = []
        
               for z in range(self.k):
                   s=0
                   log_s=0
                   p = Proba_multi(data[t],self.eta[z])
                   s = np.log(self.A[z,:]) + alpha_prev[:]
                   log_s = LogSumExp(s)
                   alpha_t.append(log_s + np.log(p))
        
               alpha_t = np.array(alpha_t)
               alpha_list.append(alpha_t)
        
            alpha.append(alpha_list)
        
        return np.array(alpha)   
     
        
    
    
    def compute_log_beta(self,datas):
        '''
        Beta recursion in logarithm domain given the observations and the GM law
        Parameter: np.array data : observations
        return: betas for all the time steps
        '''
        beta = []
        for data in datas:
            beta_list = []
            beta_T = np.zeros((self.k,))
            beta_list.append(list(beta_T))
            T=len(data)
            for t in range(1,T):
            
                beta_prev = beta_list[-1]
                beta_t = []
            
                for z in range(self.k):
                    s=0
                    s = [np.log(self.A[q,z]*Proba_multi(data[-t],self.eta[q]))+beta_prev[q] for q in range(self.k)]
                    log_s = LogSumExp(s)
                    beta_t.append(log_s)
            
            
                beta_t = np.array(beta_t)
                beta_list.append(list(beta_t))
        
            beta_list.reverse()
            
            beta.append(beta_list)
        return np.array(beta)                     
    
        
    def compute_E_step(self,datas):
        '''
        Compute the E step for the HMM
        parameters: np.array data: observations
        '''
        alpha = self.compute_log_alpha(datas)
        beta = self.compute_log_beta(datas)
        for i in range(datas.shape[0]):
            for t in range(datas.shape[1]):
                for q in range(self.k):
                    self.q_e_step[i,t,q] = alpha[i,t,q] + beta[i,t,q] - LogSumExp(alpha[i,t,:] + beta[i,t,:])
                
            
            self.q_e_step[i] = np.exp(self.q_e_step[i])
        
    
    def compute_xi(self,datas):
        '''
        Compute p(q_t,q_{t+1}|y)
        parameters: np.array data: observations
        '''
        log_alpha = self.compute_log_alpha(datas)
        log_gamma = np.log(self.q_e_step)
        T = datas.shape[1]
        log_xi = np.zeros((datas.shape[0],T-1,self.k,self.k))
        for n,data in enumerate(datas):    
            for t in range(T-1): 
                for i in range(self.k):
                    for j in range(self.k) : 
                        
                        p = np.log(Proba_multi(data[t+1],self.eta[j]))
                        log_xi[n,t,i,j] = log_alpha[n,t,i] + p + log_gamma[n,t+1,j] + np.log(self.A[j,i]) - log_alpha[n,t+1,j]
            
        return np.exp(log_xi)
    
    
    def compute_M_step(self,datas):
        '''
        Compute the M step for the HMM
        parameters: np.array data: observations
        '''
        xi = self.compute_xi(datas)
        for i in range(self.k):
            self.pi_0[0,i] = np.mean(self.q_e_step[:,0,i])
            for j in range(self.k):
                self.A[i,j] = np.sum(xi[:,:,i,j])/np.sum(xi[:,:,i,:])
            self.eta[i] = np.zeros(np.shape(self.eta[i]))
            normalization =  0 
            for n,data in enumerate(datas):
                self.eta[i] += np.dot(data.transpose(),self.q_e_step[n,:,i])
                normalization+= (np.sum(np.dot(data.transpose(),self.q_e_step[n,:,i])))
            
            self.eta[i]= self.eta[i]/normalization
            
            #print(np.sum(self.q_e_step[:,i]))
    
    def compute_log_likelihood_approx(self,datas):
        '''
        Fonction qui calcule l'approximation utilisÃ© pour minorer la vraie log likehood des donnÃ©es avec le modÃ¨le de gaussian mixture utilisÃ©
        ParamÃ¨tres: data:(np.array(nb_samples,nb_composante)) Les Ã©chantillons sur lesquels sera calculÃ© la log likelihood
        '''
        xi = self.compute_xi(datas)
        current_log=0
        T=datas.shape[1]
        
        for n,data in enumerate(datas):
            for i in range(self.k) : 
                current_log+= self.q_e_step[n,0,i]*np.log(self.pi_0[i])
                for t in range(T-1):
                    for j in range(self.k):
                        current_log += xi[n,t,j,i]*np.log(self.A[i,j])
                    current_log += self.q_e_step[n,t,i]*np.log(Proba_multi(data[t],self.eta[i]))
                current_log += self.q_e_step[n,T-1,i]*np.log(Proba_multi(data[T-1],self.eta[i]))
            
            
        return current_log
        
       
    
    
    def compute_current_log_likelihood(self,datas):
        '''
        Fonction qui calcule le vrai log-likelihood des donnÃ©es avec le modÃ¨le de gaussian mixture utilisÃ©
        ParamÃ¨tres: data:(np.array(nb_samples,nb_composante)) Les Ã©chantillons sur lesquels sera calculÃ© la log likelihood
        '''
        
        log_alpha = self.compute_log_alpha(datas)
        log_beta = self.compute_log_beta(datas)
        ll= np.zeros((datas.shape[1],1))
        list_like = []
        for i,data in enumerate(datas):
            for t in range(data.shape[0]):
                ll[t,0] = LogSumExp(log_alpha[i,t,:]+log_beta[i,t,:])
            list_like.append(np.mean(ll))
            
        return np.mean(list_like), list_like
        
    
    def fit(self,data,epsilon = 1e-5,verbose=1,validation_set=None):
        '''
        Fonction fit: Compute the EM based learning to compute the parameters of the model
        ParamÃ¨tres: data: (np.array(nb_samples,nb_composante)) observations
                    epsilon: (float) stopping criterio
                    verbose: (0 ou 1) print verbose
        Return: Rien
        '''
        lg_like,ll = self.compute_current_log_likelihood(data)
        self.compute_E_step(data)
        old_lg_like = -float('Inf') #initialisation 
        likelihood = []
        likelihood.append(lg_like)
        if (validation_set!=None):
            likelihood_test = []
            ll0,ll1 = self.compute_current_log_likelihood(validation_set)
            likelihood_test.append(ll0)
        nb_iteration = 0
        print('Iteration 0','Log likelihood ',lg_like)
        while abs(lg_like-old_lg_like)>epsilon: #stopping criteria
            
            nb_iteration +=1
            old_lg_like = lg_like
            self.compute_M_step(data)
            lg_like,ll= self.compute_current_log_likelihood(data)
            likelihood.append(lg_like)
            if (validation_set!=None): #compute logloss for a validationset
                ll0,ll1 = self.compute_current_log_likelihood(validation_set)
                likelihood_test.append(ll0)
            if (verbose==1): 
                print('Iteration ',nb_iteration,'Log likelihood ',lg_like)
            self.compute_E_step(data)
        if (validation_set!=None):
            return likelihood,likelihood_test 
        else:
            return likelihood
    
    def predict_proba_observation(self,datas):
        ''' 
            A function to predict the probabilities of observation 
            Arguments : datas = (np.array(nb_samples,nb_composante)) observations
            return : p(Y)
        '''
        alpha = self.compute_log_alpha(datas)
        beta = self.compute_log_beta(datas)
        list_proba = []
        
        for n in range(datas.shape[0]):
            list_proba.append(np.exp(LogSumExp(alpha[n,1,:] + beta[n,1,:])))
        
        return list_proba
        
    def fisher_vectors(self,datas):
        ''' 
            Computes the first fisher vectors which are composed of the derivatives
            of the log likelihood with respect to the observation probabilities and 
            also the transition probabilities, evaluated in the parameters that give
            the maximum likelihood 
            
            Params : datas Observations
            
            Return : 
                d_obs : the fisher vectors that corresponds to the derivative w.r.t. to 
                        the observation probabilities
                d_tans : the fisher vectors that corresponds to the derivative w.r.t. to 
                        the translation probabilities
        '''
        log_alpha = self.compute_log_alpha(datas)
        log_alpha_t_1 = np.zeros(log_alpha.shape)
        log_alpha_t_1[0] = log_alpha[0]
        log_alpha_t_1[1:] = log_alpha[:-1]
        log_beta = self.compute_log_beta(datas)
        ll,log_likelihood = self.compute_current_log_likelihood(datas)
        Ltot = np.exp(log_likelihood)
        d_trans = np.zeros((datas.shape[0],self.k,self.k))
        d_obs = np.zeros((datas.shape[0],self.k,datas.shape[2]))
        for n in range(Ltot.shape[0]):
            for j in range(self.k):
                p=[]
                for t in range(datas.shape[1]):
                    p.append(Proba_multi(datas[n,t],self.eta[j]))
                p=np.array(p)
                for i in range(self.k):
                    d_trans[n,i,j] = (1/Ltot[n])*np.exp(LogSumExp(log_alpha_t_1[n,:,i]+log_beta[n,:,j]+np.exp(np.sum(np.log(p)))))
                for i in range(datas.shape[2]):
                    for t0 in range(datas.shape[1]):  
                        d_obs[n,j,i] += datas[n,t0,i]*(1/(Ltot[n]*p[t0]))*(np.exp(log_alpha[n,t0,j]+log_beta[n,t0,j]))
        return d_trans, d_obs
    
    def fisher_vectors2(self,datas):
        ''' 
            Computes the second fisher vectors which are composed of the derivatives
            of the log likelihood with respect to the emission probabilities only,
            and are evaluated in the parameters that give the maximum likelihood, these vectors are 
            
            Params : datas Observations
            
            Return : 
                d_obs : the fisher vectors that corresponds to the derivative w.r.t. to 
                        the emission probabilities
        '''
        
        alpha = self.compute_log_alpha(datas)
        beta = self.compute_log_beta(datas)
        gamma = np.zeros([datas.shape[0],datas.shape[1],self.k]) 
        for i in range(datas.shape[0]):
            for t in range(datas.shape[1]):
                for q in range(self.k):
                    gamma[i,t,q] = alpha[i,t,q] + beta[i,t,q] - LogSumExp(alpha[i,t,:] + beta[i,t,:])
            gamma[i] = np.exp(gamma[i])        #ll,log_likelihood = self.compute_current_log_likelihood(datas)
        #Ltot = np.exp(log_likelihood)
        d_obs = np.zeros((datas.shape[0],self.k,datas.shape[2]))
        for n in range(datas.shape[0]):
            for j in range(self.k):
                for i in range(datas.shape[2]):
                    d_obs[n,j,i] = (np.sum(datas[n,:,i]*gamma[n,:,j])/self.eta[j][i]) - np.sum(gamma[n,:,j])
        return d_obs


    
