import numpy as np

def Scalar_product(attr_score_copy, X_model_normalized_copy):
	scalar_product = np.average ( np.sum(np.sum(np.multiply(attr_score_copy , X_model_normalized_copy), axis=2), axis=1)/(   np.sqrt(np.sum(np.sum(np.multiply(attr_score_copy,attr_score_copy), axis=2), axis=1) )  * np.sqrt(np.sum(np.sum(np.multiply(X_model_normalized_copy,X_model_normalized_copy), axis=2), axis=1) ) )  )
	return scalar_product

def Scalar_product_set(attr_score_copy, X_model_normalized_copy):
	scalar_product = np.sum(np.sum(np.multiply(attr_score_copy , X_model_normalized_copy), axis=2), axis=1)/(   np.sqrt(np.sum(np.sum(np.multiply(attr_score_copy,attr_score_copy), axis=2), axis=1) )  * np.sqrt(np.sum(np.sum(np.multiply(X_model_normalized_copy,X_model_normalized_copy), axis=2), axis=1) ) )  
	return scalar_product   


def calculate_angles(saliency_score):
  orthogonal_residual = np.sum(saliency_score, axis=-1)
  L2_norm = np.sqrt(np.sum(np.square(saliency_score), axis=-1))
  sine = 1/2 * orthogonal_residual / L2_norm 
  sine = np.arcsin(sine) * (180/3.1416) 
  return sine

def L2_A(saliency_score):
    L2_norm = np.sqrt(np.sum(np.square(saliency_score), axis=-1))
    return L2_norm

def disperson_A(saliency_score, saliency_score_ensemble):
  return L2_A(saliency_score - saliency_score_ensemble) 
  #return cosine_similarity(saliency_score, saliency_score_ensemble)   
