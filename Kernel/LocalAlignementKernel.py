'''
LocalAlignementKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the Local Alignement Kernel (Vert et al., 2004)
'''

import numpy as np
import math

def Blosum62_S(a,b):
    '''
    Compute the substitution score between string a and string b using the BLOSUM 62 score
    Param: @a (char) char between A,C,G,T
    @b (char) char between A,C,G,T
    Return: Blosum 62 score
    '''
    list_possible = ['A','C','G','T']
    '''score = np.array([[ 12.07228312,  -4.90676753,  -0.76592312,  -7.61492156],
                      [ -4.90676753,  12.84507877,  -9.07239427,   2.11392253],
                      [ -0.76592312,  -9.07239427,  10.66391564,  -0.68052441],
                      [ -7.61492156,   2.11392253,  -0.68052441,   6.48958972]])'''
    score = np.array([[3,-1,-1,-1],[-1,8,-4,-2],[-1,-4,5,-3],[-1,-2,-3,4]])

    return score[list_possible.index(a),list_possible.index(b)]

def freq_each_term(list_seq):
    '''
    Function that compute the frequence of each terms (A,C,G,T)
    Param: @list_seq: (list) list of DNA  sequence
    Return: (dict) all the frequencies of each term
    '''
    list_possible = ['A','C','G','T']
    dict_cur = {key: 0 for (key) in list_possible}
    total_elem = 0
    for seq in list_seq:
        for s in seq:
            dict_cur[s] +=1
            total_elem+=1
    
    for s in list_possible:
        dict_cur[s] = dict_cur[s]/total_elem
    
    return dict_cur


def count_all_pairs(column):
    '''
    Compute the number of appearance of each pairs in a given aligned column (we assume that our data are aligned)
    Param: @column one column in all the sequences (for instance all the letters in position i)
    Return: (dict) appearance of each term
    '''
    list_possible_pairs = [('A','C'),('A','G'),('A','T'),('A','A'),('C','C'),('C','G'),('C','T'),('G','G'),('G','T'),('T','T'),('total')]
    dict_cur = {key: 0 for (key) in list_possible_pairs}
    for ind,i in enumerate(column):
        for j in column[ind+1:]:
            if i>j:
                second_ind = i
                first_ind = j
            else:
                second_ind = j
                first_ind = i
            dict_cur[first_ind,second_ind]+=1
            dict_cur['total']+=1
    return dict_cur

def compute_substitution_matrix(list_seq):
    '''
    Compute the substitution matrix using the method used to compute BLOSUM matrix
    We assume that the list of sequences are aligned (that is not the case in general (one has
    to manually align the DNA sequences))
    Param: (list) list of str
    Return: The substitution matrix
    '''
    freq = freq_each_term(list_seq)
    aligned_array = np.array([list(s) for s in list_seq])
    all_colum = aligned_array.T
    all_colum = all_colum.tolist()
    list_possible_pairs = [('A','C'),('A','G'),('A','T'),('A','A'),('C','C'),('C','G'),('C','T'),('G','G'),('G','T'),('T','T')]
    dict_cur = {key: 0 for (key) in list_possible_pairs}
    compute_all_column_stat = [count_all_pairs(col) for col in all_colum]
    total_pairs= sum([d['total'] for d in compute_all_column_stat])
    for key in list_possible_pairs:
        dict_cur[key] = sum([d[key] for d in compute_all_column_stat])/total_pairs
        
    substitution_matrix = np.zeros((4,4))
    list_possible = ['A','C','G','T']

    for ind1,i in enumerate(list_possible):
        for ind2,j in enumerate(list_possible):
            if i>j:
                second_ind = i
                first_ind = j
            else:
                second_ind = j
                first_ind = i
            if i==j:
                e= freq[i]**2
            else:
                e = 2*freq[i]*freq[j]
            substitution_matrix[ind1,ind2] = 10000*math.log2(dict_cur[first_ind,second_ind]/e)
    return substitution_matrix

def compute_one_kernel(seq1, seq2,S):
    """ 
    Calculate the application of the local alignement rk on seq1 and seq2 given the substitution matrix S
    Param: @seq1: (string) data 1 to use to feed the kernel computation
    @seq2: (string) data 2 to use to feed the kernel computation
    @S: (string) Substitution matrix
    """
    n1, n2 = len(seq1), len(seq2)
    
    if n1>n2:
        seqA, seqB = seq2, seq1
    else:
        seqA, seqB = seq1, seq2
    
    nA, nB = len(seqA)+1, len(seqB)+1
    
    beta = 0.05
    e =11
    d=1
    M = np.zeros((nA, nB))
    X = np.zeros((nA,nB))
    Y = np.zeros((nA,nB))
    X2 = np.zeros((nA,nB))
    Y2 =  np.zeros((nA,nB))
    for i in range(1,nA):
        for j in range(1,nB):
            M[i,j] = np.exp(beta*(S(seqA[i-1],seqB[j-1])))*(1+X[i-1,j-1]+Y[i-1,j-1] + M[i-1,j-1])
            X[i,j] =  np.exp(beta*d)*M[i-1,j] + np.exp(beta*e)*X[i-1,j]
            Y[i,j] =  np.exp(beta*d)*(M[i,j-1]+X[i,j-1])+ np.exp(beta*e)*Y[i,j-1]
            X2[i,j] = M[i-1,j] + X2[i-1,j]
            Y2[i,j] = M[i,j-1] + X2[i,j-1] + Y2[i,j-1]
    return (1 + X2[-1,-1]+Y2[-1,-1]+M[-1,-1])

def normalize_kernel(seq1,seq2,S):
    """
    Normalized computation of one element of the local element kernel local alignement kernel
    """
    return compute_one_kernel(seq1,seq2,S)/(compute_one_kernel(seq1,seq1,S)*compute_one_kernel(seq2,seq2,S))**(1/2)


def local_alignement_kernel(X1,S,X2=[]):
    '''
    This function computes the string kernel gram matrix. 
    Param: @X1: (list) list of strings in the train set
    @S: Substitution matrix
    @X2: (list)  list of strings in the test  set (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    Return: String Kernel Gram matrix
    '''
    len_X2= len(X2)
    len_X1 = len(X1)

    if len_X2 ==0:
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)

        for i in range(len_X1):
            for j in range(i, len_X1):
                if i%100==0:
                    print(i)
                gram_matrix[i, j] = normalize_kernel(X1[i], X1[j], S)
                #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix
    else:
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        for i in range(len_X1):
            for j in range(len_X2):
                gram_matrix[i, j] = normalize_kernel(X1[i], X2[j], S)
        
        return gram_matrix
'''  
#Example of computaition to test the implementation using some sklearn tools
gramMat = np.zeros((1000,1000))
pair_seq = [X_train_0[i] for i in range(len(X_train_0)) if Y_train_0[i]==1]
a = compute_substitution_matrix(pair_seq)


for i in range(1000):
    print(i)
    for j in range(i,1000):
        print(i,j)
        gramMat[i,j] = normalize_kernel(X_train_0[i],X_train_0[j],a)
        gramMat[j,i] = gramMat[i,j]
        
list_train = list(range(0,80))
list_test = list(range(80,100))
gram_train = gramMat[list_train,:][:,list_train]
gram_test = gramMat[list_train,:][:,list_test]

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 

y_train= np.array(Y_train_0)[list_train]
y_val = np.array(Y_train_0)[list_test]

svm_model = SVC(kernel='precomputed') 
svm_model.fit(gram_train,y_train) 
y_train_pred = svm_model.predict(gram_test.T).reshape(-1)
print('Precision =',accuracy_score(y_val.reshape(-1),y_train_pred))
'''