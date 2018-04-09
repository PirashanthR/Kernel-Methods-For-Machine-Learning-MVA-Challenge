'''
SpectrumKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the Spectrum Kernel (Leslie et al., 2002)
and the Mismatch Kernel (Leslie et al., 2004)
'''


import numpy as np
from multiprocessing import Pool
from itertools import product

ngrams = lambda a, n: list(zip(*[a[i:] for i in range(n)])) #function that extract all the n grams in a given sequence

def AllPossibleCombinationlist(char_list,n):
    '''
    Compute all the possible ngrams that we can obtain from a list of char 
    This function will allow us to have a correspondance between all our histograms representing
    each sequences because the bin i will represent the same n gram (given by the i-th value of the list
    that we are returning) in all our histograms
    Param: char_list: (list) list of possible char
    n: (int) n in ngram - length of the subsequences considered
    '''
    #n corresponds to n in n gram
    return list(product(char_list,repeat=n))

def CreateHistogramSeq(Seq,AllCombinList,n):
    '''
    Create the embedding that allows to compute the spectrum kernel: histogram of all the subsequences of length n
    in the sequence
    Param: Seq: (str) DNA sequence containing only the letter A,C,G,T
    n: (int) length of the subsequences considered
    AllCombinList: (list) a list containing all the possible combination of length n that we can compute using the letters
    A C G T
    Return: value : np.array contains the representation of the sequence as an array
    '''
    decompose_seq= ngrams(Seq,n)
    value = np.zeros([len(AllCombinList),])
    for ngram in decompose_seq:
        index_ngram = AllCombinList.index(ngram)
        value[index_ngram] = value[index_ngram]+1
    return value

def CreateHistogramMismatchSeq(Seq,AllCombinList,n):
    '''
    Create the embedding that allows to compute the mismatch kernel: histogram of all the subsequences of length n
    in the sequence. This time allows one mismatch.
    Param: @Seq: (str) DNA sequence containing only the letter A,C,G,T
    @n: (int) length of the subsequences considered
    @AllCombinList: (list) a list containing all the possible combination of length n that we can compute using the letters
    A C G T
    Return: value : np.array contains the representation of the sequence as an array
    '''
    letters = ['A','C','G','T']
    decompose_seq= ngrams(Seq,n)
    value = np.zeros([len(AllCombinList),])
    for ngram in decompose_seq:
        index_ngram = AllCombinList.index(ngram)
        value[index_ngram] = value[index_ngram]+1
        copy_ngram = list(ngram)
        for ind,cur_letter in enumerate(copy_ngram):
            for letter in letters:
                if letter!=cur_letter:
                    new_ngram = list(copy_ngram)
                    new_ngram[ind]= letter
                    mismatch_ngram = tuple(new_ngram)
                    index_ngram = AllCombinList.index(mismatch_ngram)
                    value[index_ngram] = value[index_ngram]+0.1
    return value

def compute_idf(list_histograms):
    '''
    Compute the idf score for all the subsequences that appears in our computation.
    If a sequences appears rarely it will have a higher score than if it appears really frequently
    Param: @list_histograms: list of histograms that as been computed
    Return: (np.array) compute the idf score for all the bins in the histogram
    '''
    idf = 0.000001*np.ones((list_histograms.shape[1]))
    for sent in list_histograms:
        idf += np.array(sent)
    
    idf= np.maximum(1, np.log10(len(list_histograms) / (idf)))
    
    return idf


def compute_kernel_histogram(x1,x2):
    """ 
    Compute the scalar product between x1 and x2 (linear kernel in the embedding given in the Spectrum Kernel space)
    Param: @x1: (np.array) data 1 to use to feed the linear kernel computation
    @x2: (np.array) data 2 to use to feed the linear kernel computation
    """
    value= np.vdot(x1,x2)
    return value



'''
In order to allow the use of parallelization and the 
multiprocessing library we have computed some really basic functions using classes
The next few functions must be easy to understand
'''
compute_diag = lambda X,i: compute_kernel_histogram(X[i], X[i])
compute_element_kernel_square = lambda X1,sim_docs_kernel_value,i,j: compute_kernel_histogram(X1[i], X1[j])/(sim_docs_kernel_value[i] *sim_docs_kernel_value[j])**0.5
compute_element_kernel = lambda X1,X2,sim_docs_kernel_value,i,j: compute_kernel_histogram(X1[i], X2[j])/(sim_docs_kernel_value[1][i] *sim_docs_kernel_value[2][j])**0.5


class compute_diag_copy(object):
    def __init__(self, X):
        self.X = X
    def __call__(self, i):
        return compute_diag(self.X,i)

class compute_element_i(object):
    def __init__(self, X,sim_docs_kernel_value,i):
        self.X = X
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
    def __call__(self, j):
        return compute_element_kernel_square(self.X,self.sim_docs_kernel_value,self.i,j)

class compute_element_i_general(object):
    def __init__(self, X,X_p,sim_docs_kernel_value,i):
        self.X = X
        self.X_p = X_p
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
    def __call__(self, j):
        return compute_element_kernel(self.X,self.X_p,self.sim_docs_kernel_value,self.i,j)


def histogram_kernel(X1,X2=[],n_proc=1):
    '''
    This function computes the spectrum kernel gram matrix. (Because we assume that X1 and X2 are given
    in the rkhs space this kernel is equivalent to a basic linear kernel)
    Param: @X1: (np.array)(nb_sample,nb_features) Training data
    @X2: (np.array)(nb_sample,nb_features) Testing data (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    @n_proc: (int) allows to use more processor in order to compute the gram matrix quickly
    Return: Spectrum Kernel Gram matrix
    '''
    if n_proc>1: #multiprocessing
        pool = Pool(processes= n_proc)
        
        len_X2= len(X2)
        len_X1 = len(X1)
        sim_docs_kernel_value = {}
    
        if len_X2 ==0:
            # numpy array of Gram matrix
            gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)
            
            sim_docs_kernel_value = pool.map(compute_diag_copy(X1),range(len_X1))
                        
            for i in range(len_X1):
                    
                    gram_matrix[i, i:len_X1] = pool.map(compute_element_i(X1,sim_docs_kernel_value,i),range(i,len_X1))
    
            gram_matrix = gram_matrix+gram_matrix.T - np.diag(np.diag(gram_matrix))
            
            pool.close()
            #calculate Gram matrix
            return gram_matrix
        else:
            gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
    
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            #store K(s,s) values in dictionary to avoid recalculations
            sim_docs_kernel_value[1] = pool.map(compute_diag_copy(X1),range(len_X1))
            sim_docs_kernel_value[2] = pool.map(compute_diag_copy(X2),range(len_X2))
    
            for i in range(len_X1):
                    gram_matrix[i, :] = pool.map(compute_element_i_general(X1,X2,sim_docs_kernel_value,i),range(len_X2))
            
            pool.close()
    
            return gram_matrix
    else:#without multiprocessing
        len_X2= len(X2)
        len_X1 = len(X1)
        sim_docs_kernel_value = {}
        if len_X2 ==0:
            gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)
            for i in range(len_X1):
                sim_docs_kernel_value[i] = compute_diag_copy(X1)(i)
                            
            for i in range(len_X1):
                for j in range(i,len_X1):
                    
                    gram_matrix[i, j]= compute_element_i(X1,sim_docs_kernel_value,i)(j)
                    gram_matrix[j, i] = gram_matrix[i, j]
            #calculate Gram matrix
            return gram_matrix
        else:
            gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
    
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = compute_diag_copy(X1)(i)
            for j in range(len_X2):
                sim_docs_kernel_value[2][j] = compute_diag_copy(X2)(j)
    
            for i in range(len_X1):
                for j in range(len_X2):
                    gram_matrix[i, j] = compute_element_i_general(X1,X2,sim_docs_kernel_value,i)(j)    
            return gram_matrix