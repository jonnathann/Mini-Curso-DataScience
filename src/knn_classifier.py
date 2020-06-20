#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:26:25 2020

@author: jonnathann
"""
import numpy as np

def accuracy(y_test, predict):
    size = len(y_test)
    count_true = 0
    for i in range(size):
        if y_test[i] == predict[i]:
            count_true += 1
    return (count_true/size) * 100

def probability(structure, classe):
    list_majoritaty_distances = []
    for i in range(len(structure)):
        if classe == structure[i, 1]:
            list_majoritaty_distances.append(structure[i, 0])
    return np.mean(list_majoritaty_distances)
        
def votation(list_classes):
    classes_uniques = np.unique(list_classes)
    new_list = []
    for cl in classes_uniques:
        new_list.append([list_classes.count(cl), cl])
    votation = max(new_list)
    return votation[1]

def KNeighborsClassifier(x_train, y_train, x_test, y_test, k = 1):
    
    if (k % 2) == 0:
        k = k - 1
    
    elif (k <= 0):
        print('Não é possível excutar o algoritmo com valores zeros ou menores do que zero')
        print('O valor de k será setado para o valor 1')
        k = 1
    
    classes_labels = []
    classes_prob = []
    for i in range(len(x_test)):
        
        list_distance_individual_instance = []
        
        for j in range(len(x_train)):
            
            #calc euclidean distance
            euclidean_distance = np.sqrt(np.sum(np.power(x_test[i] - x_train[j], 2)))
            list_distance_individual_instance.append([euclidean_distance, y_train[j]])
        
        #ordering distance values
        list_distance_individual_instance.sort()
        
        #defining k neighbors 
        k_neighbors = list_distance_individual_instance[0:k]
        k_neighbors = np.array(k_neighbors)
        print("Strucuture:", k_neighbors)
        k_neighbors_class = list(k_neighbors[:, 1])
        
        #majoritary votation
        classe = votation(k_neighbors_class)
        classes_labels.append(classe)
        
        #majoritary probability
        prob = probability(k_neighbors, classe)
        classes_prob.append(prob)
    return classes_labels, classes_prob