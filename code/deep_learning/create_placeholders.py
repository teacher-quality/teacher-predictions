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

def create_placeholders(n_x,n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, number of input features
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """
    
    X = tf.placeholder(tf.float32, [n_x, None], name = "X_inputs")
    Y = tf.placeholder(tf.float32, [n_y, None], name = "Y_slate")
    
    return X, Y