# Kernel methods for machine learning inclass data challenge 2017 - 2018  

This repository contains the handing for the MVA master's class:  Kernel methods for machine learning: http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2018mva/index.html

This work has been done in a team of two: RATNAMOGAN Pirashanth SAYEM Othmane.

Kaggle Team Name : Kernel InShape

The challenge was a classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

This challenge had the specificity that it requires to compute from scratch all the algorithms and the methods that we are using (no sklearn, libsvm, ...).

All our implementation has been done in python 3.6.

Hence in order to run our codes one will only needs few libraries:
numpy, pandas,multiprocessing, cvxopt, pickles, igraph,networkx,itertools, scipy, collections, copy and math essentially

The folder **Classifiers** contains all the classifiers that we have implemented:
* Kmeans -- for initialization of the Gaussian Mixtures
* DiagonalGaussianMixture -- in order to compute the fisher vectors (aggregation of visual words)
* EM_HMM -- in order to compute the HMM based fisher vectors (Jaakkola)
* LogesticRegression -- basic logestic regression (used in the baseline)
* KernelLogisticRegression -- Kernel logestic regression (takes as input gram matrices)
* KernelSVM -- Support Vector Machines (takes as input gram matrices)


The folder **Kernel** contains the functions that allows to compute the various gram matrices:
* EditDistanceKernel -- computes the edit distance kernel ( Neuhaus ‎2006)
* GMMFFisherKernel -- computes the aggregation of visual words kernel
* GoW_Kernels -- computes the graph based kernel
* HMMFisherKernel -- computes the hmm based fisher kernel (Jaakola et al 1999)
* LocalAlignementKernel -- computes the local alignement kernel (JP Vert 2004)
* RBFKernel -- computes the basic rbf kernel
* SpectrumKernel -- functions that allows to compute the Spectrum and the Mismatch Kernel (Leslie et al. 2004)
* StringKernel -- computes the string kernel (Lodhi ‎2002)

The folder Test contains some **raw draft functions** in order to use the various Kernel functions implemented in the folder Kernel (some more clean tests have to be added in the future):
* test_kernel_histo_logisticreg - compute the pickles for the mismatch kernel
* baseline - use the basic logestic regression 


The folder tools contains some basic tools that we have used during our implementation:
* PCA -- a basic implementation of the PCA 
* Utils -- contains some basic functions that allows for instance to compute scores or to split some lists in a train and a test set

In order to run the code one has first to set properly to path to the datasets in the script Tools/Utils.
The script start allows to produce our final submission.


