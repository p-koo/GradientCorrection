"""
Train models on synthetic data.
""" 
import os, h5py
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, model_zoo

#------------------------------------------------------------------------

num_trials = 10  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  

results_path = helper.make_directory('../results', 'chipseq')  
params_path = helper.make_directory(results_path, 'model_params')  

#------------------------------------------------------------------------

# load data
experiment ='FOXK2'  # Include list of 10 proteins here. 
filename = experiment + '_200.h5'
data_path = '../data/' 
file_path = os.path.join(data_path, filename)
dataset = h5py.File(file_path, 'r') 
x_train = np.array(dataset['x_train']).astype(np.float32).transpose([0,2,1])  
y_train = np.array(dataset['y_train']).astype(np.float32)
x_valid = np.array(dataset['x_valid']).astype(np.float32).transpose([0,2,1])
y_valid = np.array(dataset['y_valid']).astype(np.float32)
x_test = np.array(dataset['x_test']).astype(np.float32).transpose([0,2,1])
y_test = np.array(dataset['y_test']).astype(np.float32)

# get shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1]

#Ensemble version - varied filter size
kernel_size=[3, 5, 7, 9,11,13,15,17,19,21]

#------------------------------------------------------------------------

for model_name in model_names:
    for activation in activations:
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            if model_name == 'cnn_deep':
                model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation, first_layer_kernel_size=kernel_size[trial])
            elif model_name == 'cnn_shallow':
                model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation, first_layer_kernel_size=kernel_size[trial])

            name = model_name+'_'+activation+'_'+str(trial)
            print('model: ' + name)

            # set up optimizer/metrics and compile model
            auroc = keras.metrics.AUC(curve='ROC', name='auroc')
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            loss = keras.losses.BinaryCrossentropy(from_logits=False)
            model.compile(optimizer=optimizer, loss=loss, metrics=[auroc])
            print(model.summary())
            # setup callbacks
            es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc', 
                                                        patience=10, 
                                                        verbose=1, 
                                                        mode='max', 
                                                        restore_best_weights=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                          factor=0.2,
                                                          patience=3, 
                                                          min_lr=1e-7,
                                                          mode='max',
                                                          verbose=1) 

            # fit model
            history = model.fit(x_train, y_train, 
                                epochs=100,
                                batch_size=100, 
                                shuffle=True,
                                validation_data=(x_valid, y_valid), 
                                callbacks=[es_callback, reduce_lr])

            # save model
            weights_path = os.path.join(params_path, name+'.h5')
            model.save_weights(weights_path)