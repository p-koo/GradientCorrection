"""
Evaluate models on synthetic data: classification and interpretability performance.
""" 
import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, explain, model_zoo, geomath
import tfomics

#------------------------------------------------------------------------

num_trials = 20  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  
attr_methods = ['saliency'] #['saliency', 'smoothgrad', 'intgrad', 'expintgrad']

results_path = os.path.join('../results', 'synthetic_regul')  
params_path = os.path.join(results_path, 'model_params')  

#------------------------------------------------------------------------

# load data
data_path = '../data/synthetic_code_dataset.h5'
x_train, y_train, x_valid, y_valid, x_test, y_test = helper.load_data(data_path)  

# get shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1]

# load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]  
X_model = test_model[true_index][:500]  
X_model_centered =  X_model - 0.25

#Regularization details 
reg_factors = [2e-2, 1e-2, 3e-3, 1e-3, 1e-4, 2e-5, 1e-5, 1e-6, 1e-7]

#------------------------------------------------------------------------

for reg_factor in reg_factors:  
    for model_name in model_names:
        for activation in activations:
            base_name = model_name + '_' + activation

            # set up results dictionary of metrics to track
            results = {}
            results['auc'] = []
            for method in attr_methods:
                results[method] = {}
                results[method]['scores'] = []
                results[method]['cos_dist'] = []
                results[method]['angles_std'] = []
            

            # loop through trials and evaluate model
            for trial in range(num_trials):
                keras.backend.clear_session()
            
                # load model
                #model = helper.load_model(model_name, activation=activation)  #Antonio
                if model_name == 'cnn_deep':
                    model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation)
                elif model_name == 'cnn_shallow':
                    model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation)

                name = base_name+'_'+ str(reg_factor) + '_'+str(trial)
                print('model: ' + name)

                # set up optimizer/metrics and compile model
                auroc = keras.metrics.AUC(curve='ROC', name='auroc')
                optimizer = keras.optimizers.Adam(learning_rate=0.001)
                loss = keras.losses.BinaryCrossentropy(from_logits=False)
                model.compile(optimizer=optimizer, loss=loss, metrics=[auroc])

                # load model
                weights_path = os.path.join(params_path, name+'.h5')
                model.load_weights(weights_path)

                # classification performance evaluation
                _, auroc = model.evaluate(x_test, y_test)  
                results['auc'].append(auroc)

                # interpretability performance evaluation
                explainer = tfomics.explain.Explainer(model, class_index=0)

                # calculate attribution maps
                for method in attr_methods:
                    print('  attr method: ' + method)
                    if method == 'saliency':
                        scores = explainer.saliency_maps(X)

                    elif method == 'smoothgrad':
                        scores = explainer.smoothgrad(X, num_samples=50, mean=0.0, stddev=0.1)

                    elif method == 'intgrad':
                        scores = explainer.integrated_grad(X, baseline_type='random')

                    elif method == 'expintgrad':
                        scores = explainer.expected_integrated_grad(X, num_baseline=20, baseline_type='random', num_steps=20)
                    
                    # calculate cosine similarity interpretability performance
                    cos_dist = geomath.cosine_similarity(scores, X_model_centered)

                    # calculate angles
                    angles = geomath.calculate_angles(scores)

                    # store metrics in results dict
                    results[method]['scores'].append(scores)
                    results[method]['cos_dist'].append(cos_dist)
                    results[method]['angles_std'].append(np.std(angles))

            # save results dictionary
            filename = os.path.join(results_path, base_name+'_'+ str(reg_factor) + '_results.pickle')
            print('Saving results to: ' + filename)
            with open(filename, 'wb') as f:
                cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)






