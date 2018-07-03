# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:02:08 2018

@author: Franco
"""
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

def initialize_parameters(n_x,f_1,f_2,n_class):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [f_1, n_x]
                        b1 : [f_1, 1]
                        W2 : [f_2, f_1]
                        b2 : [f_2, 1]
                        W3 : [n_class, f_2]
                        b3 : [n_class, 1]    
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    
    W1 = tf.get_variable("W1", [f_1, n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [f_1, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [f_2, f_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [f_2, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_class, f_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [n_class, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters