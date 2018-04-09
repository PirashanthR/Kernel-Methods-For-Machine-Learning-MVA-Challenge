'''
Utils -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains all the basic tools in order to work with our datasets of DNA sequences.
It also contains some tools that provide the accuracy score and perform various basic tasks.
'''
import pandas as pd
import numpy as np
import random
####read data
import os
from Classifiers.KernelSVM import SVMC

list_dir = os.listdir('.')
if 'data' in list_dir:
    root = '.'
else:
    root='..'
'''
Read all the datasets
'''
X_train_0 = (pd.read_csv(root+ r'/data/Xtr0.csv',header=None).values).tolist()
X_train_1 = (pd.read_csv(root+r'/data/Xtr1.csv',header=None).values).tolist()
X_train_2 = (pd.read_csv(root+r'/data/Xtr2.csv',header=None).values).tolist()

X_train_matrix_0 = (pd.read_csv(root+r'/data/Xtr0_mat50.csv',sep=' ',header=None).values)
X_train_matrix_1 = (pd.read_csv(root+r'/data/Xtr1_mat50.csv',sep=' ',header=None).values)
X_train_matrix_2 = (pd.read_csv(root+r'/data/Xtr2_mat50.csv',sep=' ',header=None).values)

Y_train_0 = (pd.read_csv(root+r'/data/Ytr0.csv',sep=',',index_col=0).values)
Y_train_1 = (pd.read_csv(root+r'/data/Ytr1.csv',sep=',',index_col=0).values)
Y_train_2 = (pd.read_csv(root+r'/data/Ytr2.csv',sep=',',index_col=0).values)

X_test_0 = (pd.read_csv(root+r'/data/Xte0.csv',header=None).values).tolist()
X_test_1 = (pd.read_csv(root+r'/data/Xte1.csv',header=None).values).tolist()
X_test_2 = (pd.read_csv(root+r'/data/Xte2.csv',header=None).values).tolist()

X_test_matrix_0 = (pd.read_csv(root+r'/data/Xte0_mat50.csv',sep=' ',header=None).values)
X_test_matrix_1 = (pd.read_csv(root+r'/data/Xte1_mat50.csv',sep=' ',header=None).values)
X_test_matrix_2 = (pd.read_csv(root+r'/data/Xte2_mat50.csv',sep=' ',header=None).values)


X_train_0 = (np.array(X_train_0)[:,0]).tolist()
X_train_1 = np.array(X_train_1)[:,0].tolist()
X_train_2 = np.array(X_train_2)[:,0].tolist()

X_test_0 = (np.array(X_test_0)[:,0]).tolist()
X_test_1 = np.array(X_test_1)[:,0].tolist()
X_test_2 = np.array(X_test_2)[:,0].tolist()


 #A=(1, 0, 0, 0), C=(0, 1, 0, 0), G=(0, 0, 1, 0), T=(0, 0, 0, 1)

def optimal_c_sv(Gram_train,Gram_val,y_train,y_val,list_c,list_sv):
    opt_c = 1
    opt_sv = 1e-4
    loss_opt = 0
    for c in list_c:
        for sv in list_sv:
            svm_test = SVMC(c= c,min_sv = sv)
            svm_test.fit(Gram_train,y_train)
            y_train_pred = svm_test.predict_class(Gram_val).reshape(-1)
            score = accuracy_score(y_val.reshape(-1),y_train_pred)
            if score> loss_opt:
                opt_c = c
                opt_sv =sv
                loss_opt= score
    return opt_c,opt_sv
            

def transform_letter_in_one_hot_vector(letter):
    '''
    Compute the function that transform a letter into its one hot vector equivalent vector
    Param: @letter : (str) a letter within A,C,G,T 
    Return: (list) with the one hot embedding representation of the letter given as input
    '''
    if letter =='A':
        return [1,0,0,0]
    elif letter =='C':
        return [0,1,0,0]
    elif letter=='G':
        return [0,0,1,0]
    elif letter =='T':
        return [0,0,0,1]
    
def transform_seq_into_spare_hot_vector(sequence):
    '''
    Transform a all sequence into its one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @sequence : (str) a sequence of strings within A,C,G,T 
    Return: (list) representation of the sequence
    '''
    vector = [transform_letter_in_one_hot_vector(letter) for letter in sequence]
    vector = np.array(vector).reshape(-1)
    return vector.tolist()

def transform_seq_into_spare_hot_vector_hmm(sequence): #almost the same as the previously described function
    '''
    Transform a all sequence into its one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @sequence : (str) a sequence of strings within A,C,G,T 
    Return: (np.array) representation of the sequence
    '''
    vector = [transform_letter_in_one_hot_vector(letter) for letter in sequence]
    return np.array(vector)

def transform_data_into_sparse_hot_vector(data_seq_matrix):
    '''
    Transform a full list of sequences into their one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @data_seq_matrix : (list) list of sequences
    Return: (np.array)representation of each sequences
    '''
    matrix = [transform_seq_into_spare_hot_vector(seq[0]) for seq in data_seq_matrix]
    matrix = np.array(matrix)
    return matrix

def transform_seq_into_label_encode(sequence):
    '''
    Label encoder for our DNA sequences 
    '''
    transform = lambda letter: 0 if letter=='A' else 1 if letter=='C' else 2 if letter=='G' else 3 
    vector =  [transform(letter) for letter in sequence]
    return np.array(vector)

def compute_patch(sequence,nb_cut):
    '''
    Function that computes multiple patches representing the sequence
    Param: @sequence: (str) a DNA sequence of strings within A,C,G,T 
    @nb_cut: (int) number of patches 
    '''
    one_hot_vect = transform_seq_into_spare_hot_vector_hmm(sequence).reshape(-1)
    len_one_patch = int(len(one_hot_vect)/nb_cut)
    if len_one_patch*nb_cut != len(one_hot_vect):
        nb_rest = len(one_hot_vect)%(len_one_patch*nb_cut)
        one_hot_vect = np.concatenate((one_hot_vect,np.zeros(nb_rest)))
        len_one_patch = int(len(one_hot_vect)/nb_cut)
    patches = [one_hot_vect[i*len_one_patch:(i+1)*len_one_patch] for i in range(nb_cut)]
    return patches

def accuracy_score(y_true,y_pred):
    '''
    Function that computes the accuracy score for our prediction
    Param: @y_true: true label
    @y_pred: prediction to evaluate
    Return: value of the accuracy score
    '''
    return max(np.sum(np.array(np.array(y_true)==np.array(y_pred),dtype=np.int)),np.sum(np.array(np.array(y_true)!=np.array(y_pred),dtype=np.int)))/len(y_true)

def train_test_split(*arrays,test_size=0.5):
    '''
    Function that split arrays and list in two sets 
    Param: *arrays: arrays to split
    test_size: (float) size of the test dataset (between 0 and 1)
    '''
    list_to_return =  []
    shape_data = len(arrays[0])
    list_indice_shuffle= list(range(shape_data))
    random.shuffle(list_indice_shuffle)
    list_train, list_test = list_indice_shuffle[:int(len(list_indice_shuffle)*(1-test_size))],list_indice_shuffle[int(len(list_indice_shuffle)*(1-test_size)):]
    for array in arrays:
        if isinstance(array,list):
            list_to_return.extend([list(np.array(array)[list_train]),list(np.array(array)[list_test])])
        else:
            list_to_return.extend([(np.array(array)[list_train]),(np.array(array)[list_test])])
    return list_to_return    



