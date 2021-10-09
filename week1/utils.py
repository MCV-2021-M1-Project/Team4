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
    
    u = np.extract(u!=0,u)    
    """ print(u) """

    q1 = np.quantile(u,0.25)
    q3 = np.quantile(u,0.75)
    
    upper_bound = q3 + 1.5*(q3-q1)
    bottom_bound = q1 - 1.5*(q3-q1)
    
    return upper_bound,bottom_bound


def borders(u,max_initial):
    upper_bound,bottom_bound = inliers_bounds(u)    
    
    aux = u
    aux[u > upper_bound] = 0
    aux[u < bottom_bound] = 0
    
    edges = np.argwhere(aux != 0)
    
    return aux[edges.min()],edges.min(),aux[edges.max()],edges.max()
