import numpy as np

## -- SIMILARITY MEASURES --
def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.linalg.norm((u - v),ord=1)

def chi2_distance(u,v):    
    return np.sum(np.nan_to_num(np.divide(np.power(u-v,2),(u+v))))

def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))

def hellinger_kernel(u,v):
    return np.sum(np.sqrt(np.multiply(u,v)))


##Inliers
def inliers_bounds(u):

    q1 = np.quantile(u,0.25) # First quantile
    q3 = np.quantile(u,0.75) # Second quantile
    q_inter = q3 - q1 # Interquantile interval
    
    #Inliers bounds
    upper_bound = q3 + 1.5*q_inter
    bottom_bound = q1 - 1.5*q_inter
    
    return upper_bound,bottom_bound


def inliers(u):
    
    # Detected border's must be close to each other
    upper_bound,bottom_bound = inliers_bounds(np.extract(u!=-1,u))    
    
    #Inliers
    inliers = u
    inliers[u > upper_bound] = -1
    inliers[u < bottom_bound] = -1
    
    return inliers
