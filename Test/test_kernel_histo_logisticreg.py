"""
test_kernel_histo_logisticreg -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
Compute spectrum mismatch kernel and gives it in pickle files (to be read in the main)
"""

import sys

sys.path.append('../')

from Kernel.SpectrumKernel import AllPossibleCombinationlist
from Kernel.SpectrumKernel import CreateHistogramMismatchSeq,compute_idf
from Classifiers.KernelLogisticRegression import KernelLogisticRegression
from Kernel.SpectrumKernel import histogram_kernel
from Tools.Utils import X_train_matrix_0,X_train_matrix_1,X_train_matrix_2,\
X_train_0,X_train_1,X_train_2,Y_train_0,Y_train_1,Y_train_2,X_test_0\
,X_test_1,X_test_2,X_test_matrix_0,X_test_matrix_1,X_test_matrix_2

import numpy as np
from Tools.Utils import accuracy_score
from Tools.Utils import train_test_split


##########Create histograms for all the matrices#######
nngram = 7 #param
list_all_combin_DNA = AllPossibleCombinationlist(['A','C','G','T'],nngram)
X_train_histo_0 = np.empty([len(X_train_0),len(list_all_combin_DNA)])
X_test_histo_0 = np.empty([len(X_test_0),len(list_all_combin_DNA)])
X_train_histo_1 = np.empty([len(X_train_1),len(list_all_combin_DNA)])
X_test_histo_1 = np.empty([len(X_test_1),len(list_all_combin_DNA)])
X_train_histo_2 = np.empty([len(X_train_2),len(list_all_combin_DNA)])
X_test_histo_2 = np.empty([len(X_test_2),len(list_all_combin_DNA)])


for i in range(len(X_train_0)):
    if i%10==0:
        print('train',i)
    X_train_histo_0[i,:] = CreateHistogramMismatchSeq(X_train_0[i],list_all_combin_DNA,nngram)
    
for j in range(len(X_test_0)):
    if i%10==0:
        print('test',i)
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


#X_train_split_0,X_val_split_0,y_train_split_0,y_val_split_0 = train_test_split(X_train_histo_0,Y_train_0,test_size=0)

print('Compute Train Kernel 0')
gram_train_multi_proc = histogram_kernel(X_train_split_0,n_proc=1)          
gram_train_multi_proc_1 = histogram_kernel(X_train_split_1,n_proc=1)
gram_train_multi_proc_2 = histogram_kernel(X_train_split_2,n_proc=1)

#################COmpute test values
print('Compute Test Kernel 0')
gram_test_final_0 =  histogram_kernel(X_train_split_0,X_test_histo_0,n_proc=1)

print('Compute Test Kernel 1')
gram_test_final_1 =  histogram_kernel(X_train_split_1,X_test_histo_1,n_proc=1)

print('Compute Test Kernel 2')
gram_test_final_2 =  histogram_kernel(X_train_split_2,X_test_histo_2,n_proc=1)


import pickle as pk 

with (open('all_data_full_separated_mismatch702.p','wb')) as f:
    pk.dump([gram_train_multi_proc,y_train_split_0,gram_train_multi_proc_1,y_train_split_1,\
     gram_train_multi_proc_2,y_train_split_2],f)

import pickle as pk 

with (open('all_test_full_mismatch7stacking702.p','wb')) as f:
    pk.dump([gram_test_final_0,gram_test_final_1,gram_test_final_2],f)
