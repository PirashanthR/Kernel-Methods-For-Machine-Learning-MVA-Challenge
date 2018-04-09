#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 02:06:06 2018

@author: sayemothmane
"""
import numpy as np 
import cvxopt
import cvxopt.solvers

'''
    Gradient projeté, à utiliser après (ou pas..) pour voir si ça tourne plus
    rapidement que le solver qp..
    
def gradient_projete(u,A, lambd, n):
    for i in range(len(u)):
        grad = 1 -  np.dot(A,u)[i]/(2*lambd)
        if (u[i]==0):
            g[i] = min(0,grad)
        else: 
            if u[i]==1/n : 
                g[i] = max(0,grad)
            else : 
                g[i] = grad
    return g
'''

class SVMC:

    def __init__(self, c=1,min_sv = 1e-4):
        self.alpha_ = None
        self.c = c #corresponds to (1/2*lambda)
        #if y_train is not None: self.C = float(self.C)
        self.min_sv = min_sv
            
    def fit(self,kernel_train,label):
        '''
        Solving C-SVM quadratic optimization problem : 
            min 1/2 u^T P u + q^T u
            s.t.  Au=b
                  Gu <=h 
        
        '''
        n = label.shape[0] 
        diag = np.zeros((n,n))
        np.fill_diagonal(diag, label)
        P = np.dot(diag, np.dot(kernel_train, diag))
        Pcvx = cvxopt.matrix(P)

        #Pcvx = cvxopt.matrix(np.outer(label,label) * kernel_train)
        qcvx = cvxopt.matrix(np.ones(n) * -1)
        
        if self.c is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            Ginf = np.diag(np.ones(n) * -1)
            Gsup = np.identity(n)
            G = cvxopt.matrix(np.vstack((Ginf, Gsup)))
            hinf = np.zeros(n)
            hsup = np.ones(n) *self.c
            h = cvxopt.matrix(np.hstack((hinf, hsup)))
        
        A = label.transpose()
        A=A.astype('double')
        Acvx = cvxopt.matrix(A)
        bcvx = cvxopt.matrix(0.0)
        
        # Solve QP problem using cvxopt solver for qp problems 
        u = cvxopt.solvers.qp(Pcvx, qcvx, G, h, Acvx, bcvx)
        
        #take Lagrange multipliers, and the solution of the dual problem
        alpha = np.ravel(u['x'])
        
        
        sv = alpha > self.min_sv
        ind = np.arange(len(alpha))[sv]
        
        self.alpha_ = alpha[sv]
        self.sv = np.argwhere(sv==True)
        self.sv_label = label[sv]
        print ("%d support vectors out of %d points" % (len(self.alpha_), n))

        # Bias value/intercept
        self.b = 0*1.0;
        #self.b = self.b.astype(np.float64)
        for i in range(len(self.alpha_)):
            self.b += self.sv_label[i]
            self.b -= np.sum(self.alpha_ * self.sv_label[:,0] * kernel_train[sv,ind[i]])
        self.b /= len(self.alpha_)
        
        
    def get_coef(self):
        '''
        Fonction get_coef: Récupère les attributs de la classe
        Paramètres: -
        Return: Les paramètres du modèle
        '''
        return list(self.alpha_)

    def predict(self,kernel_test):
        '''
        Fonction predict: Donne la probabilité d'obtenir le label y=1 pour d'échantillons données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les probabilités données par le modèle
        '''
        y_predict = np.zeros(kernel_test.shape[1])
        
        for i in range(kernel_test.shape[1]):
            y_predict[i] = sum(alpha * sv_label * kernel_test[sv,i] for alpha, sv, sv_label in zip(self.alpha_, self.sv, self.sv_label[:,0]))
        return y_predict + self.b

        prediction= np.sign(y_predict + self.b)
        
        return prediction
    
    def predict_class(self,kernel_test):
        '''
        Fonction predict_class: Donne le label évalué pour un ensemble d'échantillions données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels évalues
        '''
        prediction = np.array(self.predict(kernel_test)>=0,dtype=int)
        prediction[prediction ==0]=-1
        return prediction
