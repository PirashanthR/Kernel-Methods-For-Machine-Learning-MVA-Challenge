'''
StringKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the String Kernel (Lohdi et al., 2002)

Our implementation is largely inspired from the one that we can find here:
https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py
'''
'''
The following functions allows to compute the recursion given in the original article
'''

import numpy as np


def K(n, s, t,decay_param):
    smallest_size= min(len(s), len(t)) 
    print(s)
    if smallest_size< n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            print(j)
            if t[j] == s[-1]:
                part_sum += K1(n - 1, s[:-1], t[:j],decay_param)
        result = K(n, s[:-1], t,decay_param) + decay_param ** 2 * part_sum
        return result


def K1(n, s, t,decay_param):
    if n == 0:
        return 1
    elif min(len(s), len(t)) < n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            if t[j] == s[-1]:
                part_sum += K1(n - 1, s[:-1], t[:j],decay_param) * (decay_param ** (len(t) - (j + 1) + 2))
        result = decay_param * K1(n, s[:-1], t,decay_param) + part_sum
        return result

gram_matrix_elem = lambda str1,str2,sdkval1,sdkval2,subseq_length,decay_param: 1 if str1==str2 else \
K(subseq_length, str1, str2,decay_param) / (sdkval1 * sdkval2) ** 0.5

def string_kernel(X1,subseq_length,decay_param,X2=[]):
    '''
    This function computes the string kernel gram matrix. 
    Param: @X1: (list) list of strings in the train set
    @X2: (list)  list of strings in the test  set (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    Return: String Kernel Gram matrix
    '''
    len_X2= len(X2)
    len_X1 = len(X1)
    sim_docs_kernel_value = {}

    if len_X2 ==0:
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)

        #when lists of documents are identical
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            if i%1==0:
                print('self K',i)
            sim_docs_kernel_value[i] = K(subseq_length, X1[i], X1[i],decay_param)
            #calculate Gram matrix
        for i in range(len_X1):
            for j in range(i, len_X1):
                if i%100==0:
                    print(i)
                gram_matrix[i, j] = gram_matrix_elem(X1[i], X1[j], sim_docs_kernel_value[i],\
                           sim_docs_kernel_value[j],subseq_length,decay_param)
                #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix
    else:
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[1][i] = K(subseq_length, X1[i], X1[i],decay_param)
        for i in range(len_X2):
            sim_docs_kernel_value[2][i] = K(subseq_length, X2[i], X2[i],decay_param)
        
        for i in range(len_X1):
            for j in range(len_X2):
                gram_matrix[i, j] = gram_matrix_elem(X1[i], X2[j], sim_docs_kernel_value[1][i],sim_docs_kernel_value[2][j],subseq_length,decay_param)
        
        return gram_matrix