#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
KMeans

Test de l'utilisation de la KMeans sur le jeu d'entrainement que l'on a,
affichage du résultat en couleur, et affichage de la log-vraisemblance.
"""



import numpy as np
import random   
import matplotlib.pyplot as plt



def compute_distance_between_two_vectors(np_vector_1,np_vector_2):
    '''
    Fonction qui calcule la distance entre deux vecteurs ou un vecteur et un ensemble de vecteurs
    '''
    return np.sum((np_vector_2-np_vector_1)**2,axis=1)

                 
class KMeans:
    '''
    Class KMeans: permet de créer classifieur basé sur un modèle type KMeans
    Attributs: - k : nombre de cluster final, a fixer
               - centroids: centre des différents clusters
    '''
    
    def __init__(self,k=0):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.centroids = 0 
        self.k = k 
        
    def closest_centroid_index(self,data_to_evaluate):
        '''
        Fonction qui évalue le centre le plus proche de chaque données
        Paramètres: data_to_evaluate:(np.array(nb_samples,nb_composante)) Les échantillons 

        '''
        return np.argmin(compute_distance_between_two_vectors(data_to_evaluate,self.centroids),axis=0)
    
    def compute_new_centroid(self,full_data):
        '''
        Fonction qui calcule les nouveaux centroides.
        '''
        centroids = np.zeros(self.centroids.shape)
        nb_data_per_cluster = np.zeros(self.k,dtype=int)
        for i_data in full_data:
            label = self.closest_centroid_index(i_data)
            centroids[label,:] =  centroids[label,:]*nb_data_per_cluster[label]+i_data
            nb_data_per_cluster[label] +=1
            centroids[label,:] = centroids[label,:]/nb_data_per_cluster[label]
        return centroids
    
    def fit(self,data):
        '''
        Fonction fit: Permet de calculer les paramètres du modèle en utilisant EM
        Paramètres: data: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    epsilon: (float) seuil de convergence de l'algorithme
                    verbose: (0 ou 1) afficher les calculs de log vraisemblance a chaque iteration ou non
        Return: Rien
        '''
        self.centroids = data[random.sample(range(0,data.shape[0]),self.k)]
        prev_centroids = 10*self.centroids+10
        while not(((self.centroids-prev_centroids)==0).all()):
            prev_centroids = np.array(self.centroids) #deepcopy
            self.centroids = self.compute_new_centroid(data)
            
    def predict(self,data):
        '''
        Fonction predict: Hard clustering de toutes les données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels associés à chaque cluster
        '''
        label = [self.closest_centroid_index(data[i]) for i in range(data.shape[0])]
        return np.array(label)
    
    def compute_distortion(self,data):
        '''
        Fonction that compute the distortion for a given data
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        '''
        prediction = self.predict(data)
        distortion=0
        for i in range(data.shape[0]):
            distortion += np.linalg.norm((data[i,:]-self.centroids[prediction[i],:]))
        return distortion
            