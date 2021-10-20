################################################################################## EXPLAIN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1



def saliency(model, X, class_index=0, layer=-2, batch_size=256):
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]
    sess = K1.get_session()

    N = len(X)
    num_batches = int(np.floor(N/batch_size))

    attr_score = []
    for i in range(num_batches):
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[i*batch_size:(i+1)*batch_size]}))
    if num_batches*batch_size < N:
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[num_batches*batch_size:N]}))

    return np.concatenate(attr_score, axis=0)

 

def integrated_grad(model, X, class_index=0, layer=-2, num_background=10, num_steps=20, reference='shuffle'):

    def linear_path_sequences(x, num_background, num_steps, reference):
        def linear_interpolate(x, base, num_steps=20):
            x_interp = np.zeros(tuple([num_steps] +[i for i in x.shape]))
            for s in range(num_steps):
                x_interp[s] = base + (x - base)*(s*1.0/num_steps)
            return x_interp

        L, A = x.shape 
        seq = []
        for i in range(num_background):
            if reference == 'shuffle':
                shuffle = np.random.permutation(L)
                background = x[shuffle, :]
            else: 
                background = np.zeros(x.shape)        
            seq.append(linear_interpolate(x, background, num_steps))
        return np.concatenate(seq, axis=0)

    # setup op to get gradients from class-specific outputs to inputs
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]

    # start session
    sess = K1.get_session()

    attr_score = []
    for x in X:
        # generate num_background reference sequences that follow linear path towards x in num_steps
        seq = linear_path_sequences(x, num_background, num_steps, reference)
       
        # average/"integrate" the saliencies along path -- average across different references
        attr_score.append([np.mean(sess.run(saliency, {model.inputs[0]: seq}), axis=0)])
    attr_score = np.concatenate(attr_score, axis=0)

    return attr_score


    
def attribution_score(model, X, method='saliency', norm='times_input', class_index=0,  layer=-2, **kwargs):   #The method can be changed! 

    N, L, A = X.shape 
    if method == 'saliency':
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size=256
        
        attr_score = saliency(model, X, class_index, layer, batch_size)

        
    elif method == 'mutagenesis':
        
        attr_score = mutagenesis(model, X, class_index, layer)
        
    elif method == 'deepshap':
        if 'num_background' in kwargs:
            num_background = kwargs['num_background']
        else:
            num_background = 5
        if 'reference' in kwargs:
            reference = kwargs['reference']
        else:
            reference = 'shuffle'
    
        attr_score = deepshap(model, X, class_index, num_background, reference)

        
    elif method == 'integrated_grad':
        if 'num_background' in kwargs:
            num_background = kwargs['num_background']
        else:
            num_background = 10
        if 'num_steps' in kwargs:
            num_steps = kwargs['num_steps']
        else:
            num_steps = 20
        if 'reference' in kwargs:
            reference = kwargs['reference']
        else:
            reference = 'shuffle'
        
        attr_score = integrated_grad(model, X, class_index, layer, num_background, num_steps, reference)

    if norm == 'l2norm':
        attr_score = np.sqrt(np.sum(np.squeeze(attr_score)**2, axis=2, keepdims=True) + 1e-10)
        attr_score =  X * np.matmul(attr_score, np.ones((1, X.shape[-1])))
        
    elif norm == 'times_input':
        attr_score *= X

    return attr_score
