import numpy as np

def attribution_correction(attr_score):
    return attr_score - np.mean(attr_score, axis=2, keepdims=True)

def cosine_similarity(x1, x2):
	scalar_product = np.average(np.sum(np.sum(np.multiply(x1, x2), axis=2), axis=1)/(np.sqrt(np.sum(np.sum(np.multiply(x1,x1), axis=2), axis=1)) * np.sqrt(np.sum(np.sum(np.multiply(x2,x2), axis=2), axis=1))))
	return scalar_product

def cosine_similarity_individual_nucleotides(x1, x2):
	scalar_product =   np.sum(np.multiply(x1, x2), axis=-1)   /   (  np.sqrt(np.sum(np.multiply(x1,x1), axis=-1)) * np.sqrt(np.sum(np.multiply(x2,x2), axis=-1))) 
	return scalar_product

def scalar_product_set(x1, x2):
	scalar_product = np.sum(np.sum(np.multiply(x1, x2), axis=2), axis=1)/(np.sqrt(np.sum(np.sum(np.multiply(x1,x1), axis=2), axis=1)) * np.sqrt(np.sum(np.sum(np.multiply(x2,x2), axis=2), axis=1)))  
	return scalar_product   


def saliency_correction(saliency_score, axis=-1):
  num_dim = saliency_score.shape[axis]
  return  saliency_score - np.sum(saliency_score, axis=axis, keepdims=True)/num_dim 


def calculate_angles(saliency_score):
  orthogonal_residual = np.sum(saliency_score, axis=-1)
  L2_norm = np.sqrt(np.sum(np.square(saliency_score), axis=-1))
  sine = 1/2 * orthogonal_residual / L2_norm 
  sine = np.arcsin(sine) * (180/3.1416) 
  return sine


def L2(saliency_score):
    L2_norm = np.sqrt(np.sum(np.square(saliency_score), axis=-1))
    return L2_norm


def dispersion(saliency_score, saliency_score_ensemble):
  return L2(saliency_score - saliency_score_ensemble) 


def dispersion_bins_func(experiments, dispersion, sine_flattened):
    dispersion_bins={}
    dispersion_bins_corrected={}
    for experiment in experiments:  
        dispersion_flat = dispersion[experiment].reshape(-1) 
        sine_bins = np.arange(-90,91,5).astype(float)
        dispersion_bins_sum =  (sine_bins * 0).astype(float)
        dispersion_bins_count = (sine_bins * 0).astype(float)
        for i in range (0, len(sine_flattened[experiment])):
            bin = ((sine_flattened[experiment][i]+90)/180 * len(sine_bins)).astype(int)
            if(not np.isnan(sine_flattened[experiment][i]) ): # removing problematic resuls from NA and similar. 
                dispersion_bins_sum[bin]+=dispersion_flat[i] 
                dispersion_bins_count[bin]+=1
        dispersion_bins[experiment] = dispersion_bins_sum / dispersion_bins_count
    return dispersion_bins 


def count_large_angles(sine, threshold=30): 
    count_large_angles = np.zeros((len(sine), len(sine[0])))
    for z in range(len(sine)):
        for i in range(len(sine[0])):
            count = 0
            for j in range (len(sine[0,0])):
                if np.abs(sine[z,i,j]) > threshold: count+=1  
            count_large_angles[z,i] = count
    count_large_angles = np.array(count_large_angles.reshape(len(count_large_angles)*len(count_large_angles[0]),))  
    count_large_angles = 100 * count_large_angles/len(sine[0,0])
    return count_large_angles

