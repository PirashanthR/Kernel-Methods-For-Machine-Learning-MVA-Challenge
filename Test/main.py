'''
Main -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file allows to read and use the pickles given by the scripts that computes the different matrices
'''

import sys
sys.path.append('../')

from Classifiers.KernelLogisticRegression import KernelLogisticRegression
import numpy as np
from Tools.Utils import accuracy_score
from Tools.Utils import train_test_split,optimal_c_sv
from Classifiers.KernelSVM import SVMC

import pickle as pk

########PATH TO THE PICKLES TO COMPLETE########

with (open('./pickle/all_data_full_separated_mismatch701.p','rb')) as f:
    [gram_train_multi_proc6,y_train_split_06,gram_train_multi_proc_16,y_train_split_16,\
     gram_train_multi_proc_26,y_train_split_26]=pk.load(f) 

with (open('./pickle/all_test_full_mismatch701.p','rb')) as f:
    [gram_test_final_0,gram_test_final_1,gram_test_final_2] = pk.load(f)


c = 0.5
sv = 1e-4
lambda_log_reg = 1
tolerance = 0.001
list_of_prediction_test_0= []

for i in range(11):
    if (i==0)or(i==11):
        list_train = list(range(2000))
        list_val = []
    else:
        test_size = 1/10 
        list_train,list_val = train_test_split(list(range(2000)),test_size=test_size)
    gram_train_multi_proc = (gram_train_multi_proc6[list_train,:][:,list_train])
    gram_val_multi_proc = (gram_train_multi_proc6[list_train,:][:,list_val])
    y_train_split_0 = y_train_split_06[list_train]
    y_val_split_0 = y_train_split_06[list_val]
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test = SVMC(c= c,min_sv = sv)
        svm_test.fit(gram_train_multi_proc,y_train_split_0)
        y_train_pred = svm_test.predict_class(gram_val_multi_proc).reshape(-1)
        print('0 Precision SVM=',accuracy_score(y_val_split_0.reshape(-1),y_train_pred))
        y_test_pred_0 = svm_test.predict_class(gram_test_final_0[list_train,:])
        y_test_pred_0[y_test_pred_0==-1]=0

    else:
        log_reg = KernelLogisticRegression()
        log_reg.fit(gram_train_multi_proc,y_train_split_0,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
        y_train_pred = log_reg.predict_class(gram_val_multi_proc).reshape(-1)
        print('0 Precision Log Reg=',accuracy_score(y_val_split_0.reshape(-1),y_train_pred))
        y_test_pred_0 = log_reg.predict_class(gram_test_final_0[list_train,:])
        y_test_pred_0[y_test_pred_0==-1]=0
    list_of_prediction_test_0.append(y_test_pred_0)



y_pred_0 = np.array(np.array(list_of_prediction_test_0).mean(axis=0).reshape((-1,))>0.5,dtype=int) #average output of LGBM and XGB

c=0.4
list_of_prediction_test_1= []

for i in range(11):
    if (i==0)or(i==11):
        list_train = list(range(2000))
        list_val = []
    else:
        test_size = 1/10 
        list_train,list_val = train_test_split(list(range(2000)),test_size=test_size)
        
    y_train_split_1 = y_train_split_16[list_train]
    y_val_split_1 = y_train_split_16[list_val]
    
    gram_train_multi_proc_1 = (gram_train_multi_proc_16[list_train,:][:,list_train])
    gram_val_multi_proc1 = (gram_train_multi_proc_16[list_train,:][:,list_val])
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test_1 = SVMC(c= c,min_sv = sv)
        svm_test_1.fit(gram_train_multi_proc_1,y_train_split_1)
        
        y_train_pred = svm_test_1.predict_class(gram_val_multi_proc1).reshape(-1)
        print('1 Precision =',accuracy_score(y_val_split_1.reshape(-1),y_train_pred))
        y_test_pred_1 = svm_test_1.predict_class(gram_test_final_1[list_train,:])
        y_test_pred_1[y_test_pred_1==-1]=0
    else:
        log_reg_1 = KernelLogisticRegression()
        log_reg_1.fit(gram_train_multi_proc_1,y_train_split_1,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
        y_train_pred = log_reg_1.predict_class(gram_val_multi_proc1).reshape(-1)
        print('1 Precision Log Reg=',accuracy_score(y_val_split_1.reshape(-1),y_train_pred))
        y_test_pred_1 = log_reg_1.predict_class(gram_test_final_1[list_train,:])
        y_test_pred_1[y_test_pred_1==-1]=0
    list_of_prediction_test_1.append(y_test_pred_1)


y_pred_1 = np.array(np.array(list_of_prediction_test_1).mean(axis=0).reshape((-1,))>0.5,dtype=int) #average output of LGBM and XGB


c=0.5
sv =1e-4
list_of_prediction_test_2 = []

for i in range(11): 
    if (i==0)or(i==11):
        list_train = list(range(2000))
        list_val = []
    else:
        test_size = 1/10 
        list_train,list_val = train_test_split(list(range(2000)),test_size=test_size)
        
    y_train_split_2 = y_train_split_26[list_train]
    y_val_split_2 = y_train_split_26[list_val]
    
    
    gram_train_multi_proc_2 = (gram_train_multi_proc_26[list_train,:][:,list_train])
    gram_val_multi_proc2 = (gram_train_multi_proc_26[list_train,:][:,list_val])
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test_2 = SVMC(c=c,min_sv = sv)
        svm_test_2.fit(gram_train_multi_proc_2,y_train_split_2)
        
        y_train_pred = svm_test_2.predict_class(gram_val_multi_proc2).reshape(-1)
        print('2 Precision =',accuracy_score(y_val_split_2.reshape(-1),y_train_pred))
        y_test_pred_2 = svm_test_2.predict_class(gram_test_final_2[list_train,:])
        y_test_pred_2[y_test_pred_2==-1]=0

    else:
        log_reg_2 = KernelLogisticRegression()
        log_reg_2.fit(gram_train_multi_proc_2,y_train_split_2,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
        y_train_pred = log_reg_2.predict_class(gram_val_multi_proc2).reshape(-1)
        print('2 Precision Log Reg=',accuracy_score(y_val_split_2.reshape(-1),y_train_pred))
        y_test_pred_2 = log_reg_2.predict_class(gram_test_final_2[list_train,:])
        y_test_pred_2[y_test_pred_2==-1]=0
    list_of_prediction_test_2.append(y_test_pred_2)

y_pred_2 = np.array(np.array(list_of_prediction_test_2).mean(axis=0).reshape((-1,))>0.5,dtype=int) #average output of LGBM and XGB
   

#meilleur résultat possible : 65.5%


y_pred = list(y_pred_0)+list(y_pred_1)+list(y_pred_2)

with open("submission_bagging_mismatch7.csv", 'w') as f:
    f.write('Id,Bound\n')
    for i in range(len(y_pred)):
        f.write(str(i)+','+str(y_pred[i])+'\n')


