"""
Evaluate models on synthetic data: classification and interpretability performance.
""" 
import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, explain, model_zoo, geomath
import tfomics #Antonio 

#------------------------------------------------------------------------

num_trials = 50  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  
attr_methods = ['saliency', 'intgrad']  # ['saliency', 'intgrad', 'smoothgrad', 'expintgrad']

results_path = os.path.join('../results', 'synthetic')  
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
X = x_test[true_index] 
X_model = test_model[true_index] 
X_model_centered =  X_model - 0.25

#------------------------------------------------------------------------

for model_name in model_names:
    for activation in activations:
        base_name = model_name + '_' + activation

        # set up results dictionary of metrics to track
        results = {}
        results['auc'] = []
        for method in attr_methods:
            results[method] = {}
            results[method]['scores'] = []
            results[method]['adj_scores'] = []
            results[method]['scores_times_input'] = []
            results[method]['adj_scores_times_input'] = []
            results[method]['auroc_scores'] = []
            results[method]['auroc_adj_scores'] = []
            results[method]['aupr_scores'] = []
            results[method]['aupr_adj_scores'] = []
            results[method]['cos_dist'] = []
            results[method]['adj_cos_dist'] = []
            results[method]['angles'] = []
            results[method]['improvement'] = [] 
            

        # loop through trials and evaluate model
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            #model = helper.load_model(model_name, activation=activation)  #Antonio
            if model_name == 'cnn_deep':
                model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation)
            elif model_name == 'cnn_shallow':
                model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation)

            name = base_name+'_'+str(trial)
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
            _, auroc = model.evaluate(x_test, y_test)   #model, instead of models
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
                    scores = explainer.integrated_grad(X, baseline_type='zeros')

                elif method == 'expintgrad':
                    scores = explainer.expected_integrated_grad(X, num_baseline=20, baseline_type='random', num_steps=20)
                    
                # calculate attribution correction
                adj_scores = geomath.attribution_correction(scores)

                # quantify interpretability performance for original attribution map
                scores_times_input = np.sum(scores * X, axis=2)
                auroc_scores, aupr_scores, gt_info_score = helper.interpretability_performance(scores_times_input, X_model)

                # quantify interpretability performance for corrected attribution map
                adj_scores_times_input = np.sum(adj_scores * X, axis=2)
                auroc_adj_scores, aupr_adj_scores, _ = helper.interpretability_performance(adj_scores_times_input, X_model)
                            
                # calculate cosine similarity interpretability performance
                cos_dist = geomath.cosine_similarity(scores, X_model_centered)
                adj_cos_dist = geomath.cosine_similarity(adj_scores, X_model_centered)

                # calculate angles
                angles = geomath.calculate_angles(scores)

                # improvement in attribution scores
                improvement = geomath.cosine_similarity_individual_nucleotides(adj_scores, X_model_centered) - geomath.cosine_similarity_individual_nucleotides(scores, X_model_centered)

                # store metrics in results dict
                results[method]['scores'].append(scores)
                results[method]['adj_scores'].append(adj_scores)
                results[method]['scores_times_input'].append(scores_times_input)
                results[method]['adj_scores_times_input'].append(adj_scores_times_input)
                results[method]['auroc_scores'].append(auroc_scores)
                results[method]['auroc_adj_scores'].append(auroc_adj_scores)
                results[method]['aupr_scores'].append(aupr_scores)
                results[method]['aupr_adj_scores'].append(aupr_adj_scores)
                results[method]['cos_dist'].append(cos_dist)
                results[method]['adj_cos_dist'].append(adj_cos_dist)
                results[method]['angles'].append(angles)
                results[method]['improvement'].append(improvement)

        # save results dictionary
        filename = os.path.join(results_path, base_name+'_results.pickle')
        print('Saving results to: ' + filename)
        with open(filename, 'wb') as f:
            cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)





