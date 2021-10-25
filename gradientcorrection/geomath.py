import numpy as np

def Scalar_product(x1, x2):
	scalar_product = np.average ( np.sum(np.sum(np.multiply(x1 , x2), axis=2), axis=1)/(   np.sqrt(np.sum(np.sum(np.multiply(x1,x1), axis=2), axis=1) )  * np.sqrt(np.sum(np.sum(np.multiply(x2,x2), axis=2), axis=1) ) )  )
	return scalar_product

def Scalar_product_set(x1, x2):
	scalar_product = np.sum(np.sum(np.multiply(x1 , x2), axis=2), axis=1)/(   np.sqrt(np.sum(np.sum(np.multiply(x1,x1), axis=2), axis=1) )  * np.sqrt(np.sum(np.sum(np.multiply(x2,x2), axis=2), axis=1) ) )  
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


#### COUNT LARGE ANGLES
def count_large_angles(sine): 

    #count lare angles
    large_angle = 30
    count_large_angles30 = np.zeros((len(sine),len(sine[0])))
    for z in range (len(sine)):
        for i in range (len(sine[0])):
            count=0
            for j in range (len(sine[0,0])):
                if(np.abs(sine[z,i,j])>large_angle): count+=1  
            count_large_angles30[z,i]=count
    count_large_angles30=np.array(count_large_angles30.reshape(len(count_large_angles30)*len(count_large_angles30[0]),))  
    count_large_angles30=100* count_large_angles30/len(sine[0,0])

    #count lare angles
    large_angle = 45
    count_large_angles45 = np.zeros((len(sine),len(sine[0])))
    for z in range (len(sine)):
        for i in range (len(sine[0])):
            count=0
            for j in range (len(sine[0,0])):
                if(np.abs(sine[z,i,j])>large_angle): count+=1  
            count_large_angles45[z,i]=count
    count_large_angles45=np.array(count_large_angles45.reshape(len(count_large_angles45)*len(count_large_angles45[0]),))  
    count_large_angles45=100* count_large_angles45/len(sine[0,0])

    #count lare angles
    large_angle = 60
    count_large_angles60 = np.zeros((len(sine),len(sine[0])))
    for z in range (len(sine)):
        for i in range (len(sine[0])):
            count=0
            for j in range (len(sine[0,0])):
                if(np.abs(sine[z,i,j])>large_angle): count+=1  
            count_large_angles60[z,i]=count
    count_large_angles60=np.array(count_large_angles60.reshape(len(count_large_angles60)*len(count_large_angles60[0]),))  
    count_large_angles60=100* count_large_angles60/len(sine[0,0])
 
    return count_large_angles30, count_large_angles45, count_large_angles60
