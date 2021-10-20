############################################################################## UTILS

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
'''from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1


def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir


def optimizers(optimizer='adam', learning_rate=0.001, **kwargs):

    if optimizer == 'adam':
        if 'beta_1' in kwargs.keys():
            beta_1 = kwargs['beta_1']
        else:
            beta_1 = 0.9
        if 'beta_2' in kwargs.keys():
            beta_2 = kwargs['beta_2']
        else:
            beta_2 = 0.999
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    elif optimizer == 'sgd':
        if 'momentum' in kwargs.keys():
            momentum = kwargs['momentum']
        else:
            momentum = 0.0
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer


def activation_fn(activation):
    
    if activation == 'exp_relu':
        return exp_relu
    elif activation == 'shift_scale_tanh':
        return shift_scale_tanh
    elif activation == 'shift_scale_relu':
        return shift_scale_relu
    elif activation == 'shift_scale_sigmoid':
        return shift_scale_sigmoid
    elif activation == 'shift_relu':
        return shift_relu
    elif activation == 'shift_sigmoid':
        return shift_sigmoid
    elif activation == 'shift_tanh':
        return shift_tanh
    elif activation == 'scale_relu':
        return scale_relu
    elif activation == 'scale_sigmoid':
        return scale_sigmoid
    elif activation == 'scale_tanh':
        return scale_tanh
    elif activation == 'log_relu':
        return log_relu
    elif activation == 'log':
        return log
    elif activation == 'exp':
        return 'exponential'
    else:
        return activation
        
def exp_relu(x, beta=0.001):
    return K.relu(K.exp(.1*x)-1)

def log(x):
    return K.log(K.abs(x) + 1e-10)

def log_relu(x):
    return K.relu(K.log(K.abs(x) + 1e-10))

def shift_scale_tanh(x):
    return K.tanh(x-6.0)*500 + 500

def shift_scale_sigmoid(x):
    return K.sigmoid(x-8.0)*4000

def shift_scale_relu(x):
    return K.relu(K.pow(x-0.2, 3))

def shift_tanh(x):
    return K.tanh(x-6.0)

def shift_sigmoid(x):
    return K.sigmoid(x-8.0)

def shift_relu(x):
    return K.relu(x-0.2)

def scale_tanh(x):
    return K.tanh(x)*500 + 500

def scale_sigmoid(x):
    return K.sigmoid(x)*4000

def scale_relu(x):
    return K.relu((x)**3)'''
