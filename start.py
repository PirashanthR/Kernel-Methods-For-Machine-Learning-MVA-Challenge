'''
Start -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
Required file that computes our final submission
'''

from Kernel.SpectrumKernel import AllPossibleCombinationlist
from Kernel.SpectrumKernel import CreateHistogramMismatchSeq,compute_idf
from Classifiers.KernelLogisticRegression import KernelLogisticRegression
from Kernel.SpectrumKernel import histogram_kernel
from Tools.Utils import X_train_matrix_0,X_train_matrix_1,X_train_matrix_2,\
X_train_0,X_train_1,X_train_2,Y_train_0,Y_train_1,Y_train_2,X_test_0\
,X_test_1,X_test_2,X_test_matrix_0,X_test_matrix_1,X_test_matrix_2
from Classifiers.KernelSVM import SVMC
import numpy as np
from Tools.Utils import accuracy_score
from Tools.Utils import train_test_split

nngram = 7 #param
list_all_combin_DNA = AllPossibleCombinationlist(['A','C','G','T'],nngram)
X_train_histo_0 = np.empty([len(X_train_0),len(list_all_combin_DNA)])
X_test_histo_0 = np.empty([len(X_test_0),len(list_all_combin_DNA)])
X_train_histo_1 = np.empty([len(X_train_1),len(list_all_combin_DNA)])
X_test_histo_1 = np.empty([len(X_test_1),len(list_all_combin_DNA)])
X_train_histo_2 = np.empty([len(X_train_2),len(list_all_combin_DNA)])
X_test_histo_2 = np.empty([len(X_test_2),len(list_all_combin_DNA)])

#################Read Data#######################

for i in range(len(X_train_0)):
    X_train_histo_0[i,:] = CreateHistogramMismatchSeq(X_train_0[i],list_all_combin_DNA,nngram)
    
for j in range(len(X_test_0)):
    X_test_histo_0[j,:] = CreateHistogramMismatchSeq(X_test_0[j],list_all_combin_DNA,nngram)

for i in range(len(X_train_1)):
    X_train_histo_1[i,:] = CreateHistogramMismatchSeq(X_train_1[i],list_all_combin_DNA,nngram)
    
for j in range(len(X_test_1)):
    X_test_histo_1[j,:] = CreateHistogramMismatchSeq(X_test_1[j],list_all_combin_DNA,nngram)



for i in range(len(X_train_2)):
    X_train_histo_2[i,:] = CreateHistogramMismatchSeq(X_train_2[i],list_all_combin_DNA,nngram)
    
for j in range(len(X_test_2)):
    X_test_histo_2[j,:] = CreateHistogramMismatchSeq(X_test_2[j],list_all_combin_DNA,nngram)


X_train_split_0 = X_train_histo_0
y_train_split_0 = Y_train_0
y_train_split_0[y_train_split_0==0]=-1

X_train_split_1 = X_train_histo_1
y_train_split_1 = Y_train_1
y_train_split_1[y_train_split_1==0]=-1 

X_train_split_2 = X_train_histo_2
y_train_split_2 = Y_train_2
y_train_split_2[y_train_split_2==0]=-1

#################Compute the train gram matrices#######################
gram_train_multi_proc = histogram_kernel(X_train_split_0,n_proc=1)          
gram_train_multi_proc_1 = histogram_kernel(X_train_split_1,n_proc=1)
gram_train_multi_proc_2 = histogram_kernel(X_train_split_2,n_proc=1)

#################Compute test gram matrices#######################
gram_test_final_0 =  histogram_kernel(X_train_split_0,X_test_histo_0,n_proc=1)
gram_test_final_1 =  histogram_kernel(X_train_split_1,X_test_histo_1,n_proc=1)
gram_test_final_2 =  histogram_kernel(X_train_split_2,X_test_histo_2,n_proc=1)


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
    gram_train_multi_proc_cur = (gram_train_multi_proc[list_train,:][:,list_train])
    gram_val_multi_proc = (gram_train_multi_proc[list_train,:][:,list_val])
    y_train_split_0_cur = y_train_split_0[list_train]
    y_val_split_0 = y_train_split_0[list_val]
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test = SVMC(c= c,min_sv = sv)
        svm_test.fit(gram_train_multi_proc_cur,y_train_split_0_cur)
        y_train_pred = svm_test.predict_class(gram_val_multi_proc).reshape(-1)
        print('0 Precision SVM=',accuracy_score(y_val_split_0.reshape(-1),y_train_pred))
        y_test_pred_0 = svm_test.predict_class(gram_test_final_0[list_train,:])
        y_test_pred_0[y_test_pred_0==-1]=0

    else:
        log_reg = KernelLogisticRegression()
        log_reg.fit(gram_train_multi_proc_cur,y_train_split_0_cur,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
        y_train_pred = log_reg.predict_class(gram_val_multi_proc).reshape(-1)
        print('0 Precision Log Reg=',accuracy_score(y_val_split_0.reshape(-1),y_train_pred))
        y_test_pred_0 = log_reg.predict_class(gram_test_final_0[list_train,:])
        y_test_pred_0[y_test_pred_0==-1]=0
    list_of_prediction_test_0.append(y_test_pred_0)



y_pred_0 = np.array(np.array(list_of_prediction_test_0).mean(axis=0).reshape((-1,))>0.5,dtype=int) #average output of LGBM and XGB

c=0.5
list_of_prediction_test_1= []

for i in range(11):
    if (i==0)or(i==11):
        list_train = list(range(2000))
        list_val = []
    else:
        test_size = 1/10 
        list_train,list_val = train_test_split(list(range(2000)),test_size=test_size)
        
    y_train_split_1_cur = y_train_split_1[list_train]
    y_val_split_1 = y_train_split_1[list_val]
    
    gram_train_multi_proc_1_cur = (gram_train_multi_proc_1[list_train,:][:,list_train])
    gram_val_multi_proc1 = (gram_train_multi_proc_1[list_train,:][:,list_val])
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test_1 = SVMC(c= c,min_sv = sv)
        svm_test_1.fit(gram_train_multi_proc_1_cur,y_train_split_1_cur)
        
        y_train_pred = svm_test_1.predict_class(gram_val_multi_proc1).reshape(-1)
        print('1 Precision =',accuracy_score(y_val_split_1.reshape(-1),y_train_pred))
        y_test_pred_1 = svm_test_1.predict_class(gram_test_final_1[list_train,:])
        y_test_pred_1[y_test_pred_1==-1]=0
    else:
        log_reg_1 = KernelLogisticRegression()
        log_reg_1.fit(gram_train_multi_proc_1_cur,y_train_split_1_cur,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
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
        
    y_train_split_2_cur = y_train_split_2[list_train]
    y_val_split_2 = y_train_split_2[list_val]
    
    
    gram_train_multi_proc_2_cur = (gram_train_multi_proc_2[list_train,:][:,list_train])
    gram_val_multi_proc2 = (gram_train_multi_proc_2[list_train,:][:,list_val])
    
    if i<100:
        print('opt c sv',c,sv)
        svm_test_2 = SVMC(c=c,min_sv = sv)
        svm_test_2.fit(gram_train_multi_proc_2_cur,y_train_split_2_cur)
        
        y_train_pred = svm_test_2.predict_class(gram_val_multi_proc2).reshape(-1)
        print('2 Precision =',accuracy_score(y_val_split_2.reshape(-1),y_train_pred))
        y_test_pred_2 = svm_test_2.predict_class(gram_test_final_2[list_train,:])
        y_test_pred_2[y_test_pred_2==-1]=0

    else:
        log_reg_2 = KernelLogisticRegression()
        log_reg_2.fit(gram_train_multi_proc_2_cur,y_train_split_2_cur,lambda_regularisation=lambda_log_reg,tolerance=tolerance)
        y_train_pred = log_reg_2.predict_class(gram_val_multi_proc2).reshape(-1)
        print('2 Precision Log Reg=',accuracy_score(y_val_split_2.reshape(-1),y_train_pred))
        y_test_pred_2 = log_reg_2.predict_class(gram_test_final_2[list_train,:])
        y_test_pred_2[y_test_pred_2==-1]=0
    list_of_prediction_test_2.append(y_test_pred_2)

y_pred_2 = np.array(np.array(list_of_prediction_test_2).mean(axis=0).reshape((-1,))>0.5,dtype=int) #average output of LGBM and XGB

y_pred = list(y_pred_0)+list(y_pred_1)+list(y_pred_2)

with open("Yte.csv", 'w') as f:
    f.write('Id,Bound\n')
    for i in range(len(y_pred)):
        f.write(str(i)+','+str(y_pred[i])+'\n')
