'''
EditDistanceKernel -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the tools in order to use the Edit Distance Kernel (Neuhaus et al., 2006)
'''


import numpy as np
from multiprocessing import Pool

def LevenshteinDistance(str1,str2):
    '''
    Compute the edit distance between str1 and str2
    Param: @(str1): (str) string 1 for the comparison
    @(str2): (str) string 2 for the comparison
    Return (int) distance
    '''
    len_s1 = len(str1) +1
    len_s2 = len(str2) +1
    m = np.zeros((len_s1,len_s2))
    for i in range(len_s1):
        m[i,0] = i
    
    for j in range(len_s2):
        m[0,j] = j
    
    for i in range(1,len_s1):
        for j in range(1,len_s2):
            if str1[i-1]==str2[j-1]:
                m[i,j]= min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1])
            else:
                m[i,j] =min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]+1)
    return m[-1,-1]
    




def compute_kernel_edit(str1,str2,str0):
    """ 
    Calculate the application of the edit distance rk on str1 and str1 given the root str0
    Param: @str1: (string) data 1 to use to feed the kernel computation
    @str2: (string) data 2 to use to feed the kernel computation
    @str0: (string) root used to compute the kernel function
    """
    d_xx0 = LevenshteinDistance(str1,str0)
    d_xxp = LevenshteinDistance(str1,str2)
    d_x0xp = LevenshteinDistance(str0,str2)
    
    return 0.5*(d_xx0**2+d_x0xp**2 - d_xxp**2)

compute_diag_lev = lambda X,x_0,i: compute_kernel_edit(X[i], X[i],x_0)
compute_element_kernel_square_lev = lambda X1,x_0,sim_docs_kernel_value,i,j: compute_kernel_edit(X1[i],X1[j],x_0)/(sim_docs_kernel_value[i] *sim_docs_kernel_value[j])**0.5
compute_element_kernel_lev = lambda X1,X2,x_0,sim_docs_kernel_value,i,j: compute_kernel_edit(X1[i], X2[j],x_0)/(sim_docs_kernel_value[1][i] *sim_docs_kernel_value[2][j])**0.5



'''
In order to allow the use of parallelization and the 
multiprocessing library we have computed some really basic functions using classes
The next few functions must be easy to understand
'''
class compute_diag_lev_copy(object):
    def __init__(self, X,x_0):
        self.X = X
        self.x_0 = x_0
        
    def __call__(self, i):
        return compute_diag_lev(self.X,self.x_0,i)

class compute_element_i_lev(object):
    def __init__(self, X,x_0,sim_docs_kernel_value,i):
        self.X = X
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
        self.x_0 = x_0
    def __call__(self, j):
        return compute_element_kernel_square_lev(self.X,self.x_0,self.sim_docs_kernel_value,self.i,j)



class compute_element_i_general_lev(object):
    def __init__(self, X,X_p,x_0,sim_docs_kernel_value,i):
        self.X = X
        self.X_p = X_p
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
        self.x_0 = x_0
    def __call__(self, j):
        return compute_element_kernel_lev(self.X,self.X_p,self.x_0,self.sim_docs_kernel_value,self.i,j)


def kernel_Edit_distance(X1,x_0,X2=[],n_proc=4):
    '''
    This function computes the edit distance kernel gram matrix. 
    Param: @X1: (list) list of strings in the train set
    @X2: (list)  list of strings in the test  set (if empty compute the gram matrix for training else compute
    the gram matrix for testing)
    @n_proc: (int) allows to use more processor in order to compute the gram matrix quickly
    Return: Edit distance Gram matrix
    '''
    pool = Pool(processes= n_proc)
    
    len_X2= len(X2)
    len_X1 = len(X1)
    sim_docs_kernel_value = {}
    if len_X2 ==0:
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)
        sim_docs_kernel_value = pool.map(compute_diag_lev_copy(X1,x_0),range(len_X1))
        for i in range(len_X1):
                gram_matrix[i, i:len_X1] = pool.map(compute_element_i_lev(X1,x_0,sim_docs_kernel_value,i),range(i,len_X1))

        gram_matrix = gram_matrix+gram_matrix.T - np.diag(np.diag(gram_matrix))
        
        pool.close()
        #calculate Gram matrix
        return gram_matrix
    else:
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
        sim_docs_kernel_value[1] = pool.map(compute_diag_lev_copy(X1,x_0),range(len_X1))
        sim_docs_kernel_value[2] = pool.map(compute_diag_lev_copy(X2,x_0),range(len_X2))

        for i in range(len_X1):
                gram_matrix[i, :] = pool.map(compute_element_i_general_lev(X1,X2,x_0,sim_docs_kernel_value,i),range(len_X2))
        
        pool.close()

        return gram_matrix
