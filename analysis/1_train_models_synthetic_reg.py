"""
Train models on synthetic data.
""" 
import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, model_zoo, regul 

#------------------------------------------------------------------------

num_trials = 10  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  

results_path = helper.make_directory('../results', 'synthetic_regul')  
params_path = helper.make_directory(results_path, 'model_params')  

#------------------------------------------------------------------------

# load data
data_path = '../data/synthetic_code_dataset.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = helper.load_data(data_path)  

# get shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1]

#Regularization details 
reg_factor=2e-2   #  [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

#------------------------------------------------------------------------


for model_name in model_names:
    for activation in activations:
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            if model_name == 'cnn_deep':
                model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation)
            elif model_name == 'cnn_shallow':
                model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation)

            name = model_name+'_'+activation+'_'+str(trial)
            print('model: ' + name)

            # set up optimizer/metrics and compile model
            optimizer = keras.optimizers.Adam(learning_rate=0.001) 
            loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
            history, trainer = regul.fit_attr_prior(model, loss, optimizer, x_train, y_train, validation_data=(x_valid, y_valid), verbose=True,  
                                 metrics=['auroc', 'aupr'], num_epochs=100, batch_size=100, shuffle=True, reg_factor=reg_factor,
                                  es_patience=10, es_metric='auroc', es_criterion='max',
                                  lr_decay=0.2, lr_patience=3, lr_metric='auroc', lr_criterion='max')

            # save model
            base_name = model_name+'_'+activation + '_' + str(reg_factor)
            name = base_name+'_'+str(trial)
            weights_path = os.path.join(params_path, name+'.hdf5')
            model.save_weights(weights_path)
            
