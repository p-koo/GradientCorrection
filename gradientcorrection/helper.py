################################################################################# HELPER 
import os, sys
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from tensorflow import keras


def load_data(file_path, reverse_compliment=False):

    # load dataset
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.float32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_synthetic_models(filepath, dataset='test'):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')
    if dataset == 'train':
        return np.array(trainmat['model_train']).astype(np.float32)
    elif dataset == 'valid':
        return np.array(trainmat['model_valid']).astype(np.float32)
    elif dataset == 'test':
        return np.array(trainmat['model_test']).astype(np.float32)


def load_model(model_name, activation='relu', input_shape=200):

    if model_name == 'cnn-50':
        from model_zoo.py import cnn_model
        model = cnn_model.model([50, 2], activation, input_shape)

    elif model_name == 'cnn-2':
        from model_zoo.py import cnn_model
        model = cnn_model.model([2, 50], activation, input_shape)

    elif model_name == 'cnn-deep':
        from model_zoo.py import cnn_deep
        model = cnn_deep.model(activation, input_shape)

    elif model_name == 'cnn-local':
        from model_zoo.py import cnn_local
        model = cnn_local_model(activation)

    elif model_name == 'cnn-dist':
        from model_zoo.py import cnn_dist
        model = cnn_dist_model(activation)

    elif model_name == 'basset':
        from model_zoo.py import basset
        model = basset.model(activation)

    elif model_name == 'residualbind':
        from model_zoo.py import residualbind
        model = residualbind.model(activation)

    return model

        
def interpretability_performance(X, score, X_model):

    score = np.sum(score, axis=2)
    pr_score = []
    roc_score = []
    gt_info_score = []
    for j, gs in enumerate(score):

        # calculate information of ground truth
        gt_info = np.log2(4) + np.sum(X_model[j]*np.log2(X_model[j]+1e-10),axis=0)

        # set label if information is greater than 0
        label = np.zeros(gt_info.shape)
        label[gt_info > 0.1] = 1  #Antonio change, was 0.01

        # precision recall metric
        precision, recall, thresholds = precision_recall_curve(label, gs)
        pr_score.append(auc(recall, precision))

        # roc curve
        fpr, tpr, thresholds = roc_curve(label, gs)
        roc_score.append(auc(fpr, tpr))

        #Antonio
        gt_info_score.append(gt_info)

    roc_score = np.array(roc_score)
    pr_score = np.array(pr_score)

    return roc_score, pr_score , gt_info_score
    


def get_callbacks(monitor='val_auroc', patience=20, decay_patience=5, decay_factor=0.2):
    es_callback = keras.callbacks.EarlyStopping(monitor=monitor, 
                                                patience=patience, 
                                                verbose=1, 
                                                mode='max', 
                                                restore_best_weights=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, 
                                                  factor=decay_factor,
                                                  patience=decay_patience, 
                                                  min_lr=1e-7,
                                                  mode='max',
                                                  verbose=1) 

    return [es_callback, reduce_lr]



def compile_model(model):

    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', auroc, aupr])
