'''
HMMFisherKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the HMM based fisher Kernel (Jaakola et al., 1999)
'''


from Tools.EM_HMM import EM_HMM
import numpy as np
from Utils1 import X_train_matrix_0,X_train_matrix_1,X_train_matrix_2,\
X_train_0,X_train_1,X_train_2,Y_train_0,Y_train_1,Y_train_2,X_test_0\
,X_test_1,X_test_2,X_test_matrix_0,X_test_matrix_1,X_test_matrix_2
from Tools.Utils import transform_seq_into_spare_hot_vector_hmm
from Classifiers.KernelSVM import SVMC
from Tools.Utils import accuracy_score
from Tools.Utils import train_test_split




'''
#Train Kernel
datas1 = []
data_neg = []
for x_train,y_train in zip(X_train_0,Y_train_0):
    data = transform_seq_into_spare_hot_vector_hmm(x_train)
    if y_train==1:
        datas1.append(data)
    else:
        data_neg.append(data)

# Test Kernel
data_test = []
for x_train in X_test_0:
    data = transform_seq_into_spare_hot_vector_hmm(x_train)
    data_test.append(data)
data_test = np.array(data_test)

data_pos = np.array(datas1)
data_neg = np.array(data_neg)
datas = np.concatenate((data_pos,data_neg),axis=0)
'''

def fisher_HMM_train(datas,data_test,data_pos,val_size=0.1,k=3):
    ''' 
    Fonction qui entraine le modèle graphique HMM, en cherchant le maximum de vraissemblance
    et qui calcule en se basant sur les paramètres optimaux, les kernels de fisher pour les 
    données d'entrainement et les données de validation 
    
    Parametres : 
        - datas : observations, de dimension (n*t*m) avec n le nombre de séquences, t la longueur 
        de chaque séquence et m e nombre d'états (classes) possibles
        - data_pos : les observations du train, pour lesquels le label est égal à 1
        - data_test : les observations du test
        - val_size : pourcentage des données à prendre pour effectuer la validation
        - k : nombres d'états cachés du HMM
    Sorties  : 
        - K_fisher_train : le fisher kernel du train 
        - K_fisher_val : le fisher kernel du val 
        - y_val : les labels des observations utilisées pour valider le modele
        
    '''
    a = EM_HMM(data_pos,k)
    train_likelihood = a.fit(data_pos,epsilon=1e-2)

    # Training Kernel
    d_tran, d_obs = a.fisher_vectors(datas)
    y = np.concatenate((np.ones((1000,1)),-1*np.ones((1000,1))),axis=0)

    d_tran_train,d_tran_val,d_obs_train,d_obs_val,y_train,y_val = train_test_split(d_tran,d_obs,y,test_size=val_size) 


    fisher_tran_train = np.zeros((d_tran_train.shape[0],d_tran_train.shape[1]*d_tran_train.shape[1]))
    fisher_obs_train = np.zeros((d_obs_train.shape[0],d_obs_train.shape[1]*d_obs_train.shape[2]))

    for i in range(d_tran_train.shape[0]):
        fisher_tran_train[i,:] = d_tran_train[i,:,:].reshape(-1)
        fisher_obs_train[i,:] = d_obs_train[i,:,:].reshape(-1)

    fisher_train = np.concatenate((fisher_tran_train,fisher_obs_train),axis=1)
    #fisher_train = fisher_tran_train
    
    K_fisher_train = np.zeros((fisher_train.shape[0],fisher_train.shape[0]))
    for x in range(fisher_train.shape[0]):
        for y in range(K_fisher_train.shape[1]):
            K_fisher_train[x,y] = np.dot(fisher_train[x,:],fisher_train[y,:].transpose())
    
    # Validation Kernel 
    fisher_tran_val = np.zeros((d_tran_val.shape[0],d_tran_val.shape[1]*d_tran_val.shape[1]))
    fisher_obs_val = np.zeros((d_obs_val.shape[0],d_obs_val.shape[1]*d_obs_val.shape[2]))
    
    for i in range(d_tran_val.shape[0]):
        fisher_tran_val[i,:] = d_tran_val[i,:,:].reshape(-1)
        fisher_obs_val[i,:] = d_obs_val[i,:,:].reshape(-1)
    
    fisher_val = np.concatenate((fisher_tran_val,fisher_obs_val),axis=1)
    #fisher_val =fisher_tran_val
    K_fisher_val = np.zeros((fisher_train.shape[0],fisher_val.shape[0]))
    for x in range(fisher_train.shape[0]):
        for y in range(K_fisher_val.shape[1]):
            K_fisher_val[x,y] = np.dot(fisher_train[x,:],fisher_val[y,:].transpose())
    
    #test kernels
    d_tran_test, d_obs_test = a.fisher_vectors(data_test)
    fisher_tran_test = np.zeros((d_tran_test.shape[0],d_tran_test.shape[1]*d_tran_test.shape[1]))
    fisher_obs_test = np.zeros((d_obs_test.shape[0],d_obs_test.shape[1]*d_obs_test.shape[2]))
    
    for i in range(d_tran_test.shape[0]):
        fisher_tran_test[i,:] = d_tran_test[i,:,:].reshape(-1)
        fisher_obs_test[i,:] = d_obs_test[i,:,:].reshape(-1)
    
    fisher_test = np.concatenate((fisher_tran_test,fisher_obs_test),axis=1)
    K_fisher_test = np.zeros((fisher_test.shape[0],fisher_test.shape[0]))
    for x in range(fisher_train.shape[0]):
        for y in range(K_fisher_test.shape[1]):
            K_fisher_test[x,y] = np.dot(fisher_train[x,:],fisher_train[y,:].transpose())
         
    return K_fisher_train, K_fisher_val, K_fisher_test ,y_val

'''
# Validation accuracy : 
K_fisher_train, K_fisher_val,K_fisher_test,y_val = fisher_HMM_train(datas, data_pos,val_size=0.1,k=3)
svm_test = SVMC(c= 0.1,min_sv = 1e-4)
svm_test.fit(K_fisher_train,y_train)
y_train_pred = svm_test.predict_class(K_fisher_val)
print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))    '''    
