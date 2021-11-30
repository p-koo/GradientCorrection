"""
Evaluate models on synthetic data: classification and interpretability performance.
""" 
import os, h5py
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from gradient_correction import helper, explain, model_zoo, geomath
import tfomics

#------------------------------------------------------------------------

num_trials = 10  
model_names = ['cnn_deep', 'cnn_shallow'] 
activations = ['relu', 'exponential']  
attr_methods = ['saliency'] #['saliency', 'smoothgrad', 'intgrad', 'expintgrad']

results_path = os.path.join('../results', 'chipseq')  
params_path = os.path.join(results_path, 'model_params')  

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

#Subsample positive sequences only 
class_index = 0
# get positive (bind) sequences only 
pos_index = np.where(y_test[:, class_index] == 1)[0]
X = x_test[pos_index[:]]

# get shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1]

# load ground truth values - no ground truth 

#Ensemble version - varied filter size or number of filters 
num_filters=[12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  #kernel_size=[3, 5, 7, 9,11,13,15,17,19,21]

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
            results[method]['angles'] = []
        results['saliency']['ensemble_scores'] = [] 
        results['saliency']['ensemble_angles'] = []
        results['saliency']['dispersion'] = []
        results['saliency']['adj_dispersion'] = []
            
        # loop through trials and evaluate model
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            #model = helper.load_model(model_name, activation=activation)  #Antonio
            if model_name == 'cnn_deep':
                model = model_zoo.cnn_deep(input_shape, output_shape, activation=activation, num_filters=num_filters[trial])
            elif model_name == 'cnn_shallow':
                model = model_zoo.cnn_shallow(input_shape, output_shape, activation=activation, num_filters=num_filters[trial] )

            name = experiment +'_'+ base_name+'_'+str(trial)
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
            _, auroc = model.evaluate(x_test, y_test)  #model, instead of models
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
                    
                # calculate attribution correction
                adj_scores = geomath.attribution_correction(scores)

                # calculate angles
                angles = geomath.calculate_angles(scores)

                # improvement in attribution scores
                #improvement = geomath.

                # store metrics in results dict
                results[method]['scores'].append(scores)
                results[method]['adj_scores'].append(adj_scores)
                results[method]['angles'].append(angles)
        results['saliency']['ensemble_scores']=np.average(results['saliency']['scores'], axis=0)
        results['saliency']['ensemble_angles']=geomath.calculate_angles(results['saliency']['ensemble_scores'])
        
        #Dispersion
        results['saliency']['dispersion'] = geomath.dispersion(results['saliency']['scores'], results['saliency']['ensemble_scores']) 
        results['saliency']['adj_dispersion'] = geomath.dispersion(results['saliency']['adj_scores'], results['saliency']['ensemble_scores']) 
        
        # save results dictionary
        filename = os.path.join(results_path, base_name+'_results.pickle')
        print('Saving results to: ' + filename)
        with open(filename, 'wb') as f:
            cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)





