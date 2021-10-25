"""
Train models on synthetic data.
""" 
import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, model_zoo

#------------------------------------------------------------------------

num_trials = 50  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  

results_path = helper.make_directory('../results', 'synthetic')  
params_path = helper.make_directory(results_path, 'model_params')  

#------------------------------------------------------------------------

# load data
data_path = '../data/synthetic_code_dataset.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = helper.load_data(data_path)  

# get shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1]

#------------------------------------------------------------------------

for model_name in model_names:
    for activation in activations:
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            model = helper.load_model(model_name, activation=activation)  
            if model_name == 'cnn_deep':
                model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation)
            elif model_name == 'cnn_shallow':
                model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation)

            name = model_name+'_'+activation+'_'+str(trial)
            print('model: ' + name)

            # set up optimizer/metrics and compile model
            auroc = keras.metrics.AUC(curve='ROC', name='auroc')
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            loss = keras.losses.BinaryCrossentropy(from_logits=False)
            model.compile(optimizer=optimizer, loss=loss, metrics=[auroc])

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
                                callbacks=callbacks)

            # save model
            weights_path = os.path.join(params_path, name+'.h5')
            model.save_weights(weights_path)


