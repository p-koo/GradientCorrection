
import numpy as np

def calculate_angles_A(saliency_score):
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