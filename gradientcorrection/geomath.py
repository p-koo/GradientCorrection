import numpy as np

def scalar_product(x1, x2):
	scalar_product = np.average(np.sum(np.sum(np.multiply(x1, x2), axis=2), axis=1)/(np.sqrt(np.sum(np.sum(np.multiply(x1,x1), axis=2), axis=1)) * np.sqrt(np.sum(np.sum(np.multiply(x2,x2), axis=2), axis=1))))
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



def count_large_angles(sine): 
    def frac_larger_angles(sine, threshold=30): 
        # N, L = sine.shape
       
        count_large_angles = np.zeros((len(sine), len(sine[0])))
        for z in range(len(sine)):
            for i in range(len(sine[0])):
                count = 0
                for j in range (len(sine[0,0])):
                    if np.abs(sine[z,i,j]) > large_angle: count+=1  
                count_large_angles[z,i] = count
        count_large_angles = np.array(count_large_angles.reshape(len(count_large_angles)*len(count_large_angles[0]),))  
        count_large_angles = 100 * count_large_angles/len(sine[0,0])
        return count_large_angles

    count_large_angles30 = frac_larger_angles(sine, threshold=30)
    count_large_angles45 = frac_larger_angles(sine, threshold=45)
    count_large_angles60 = frac_larger_angles(sine, threshold=60)
 
    return count_large_angles30, count_large_angles45, count_large_angles60

