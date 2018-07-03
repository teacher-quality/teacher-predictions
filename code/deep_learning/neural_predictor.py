# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:27:27 2018

@author: Franco
"""


import numpy as np

def neural_predictor(X, parameters):

#    X = X_train[:,1:100].reshape(X_train.shape[0],99).astype('float32')
#    X = X_train[:,0].reshape(X_train.shape[1],1).astype('float32')
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = np.dot(W1, X) + b1                      # Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2                      # Z2 = np.dot(W2, a1) + b2
    A2 = 1 / (1 + np.exp(-Z2))                                # A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3                     # Z3 = np.dot(W3,Z2) + b3
    
    Pr_menu = np.exp(Z3[:-1,:])/np.sum(np.exp(Z3[:-1,:]),0)     #Compute probabilities without outside option
    
    return Pr_menu