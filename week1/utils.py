import numpy as np

## -- SIMILARITY MEASURES --
def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.sum(np.absolute(u - v))

def chi2_distance(u,v):
    return np.sum(np.divide(np.power(u-v,2),(u+v)))

def histogram_intersection(u,v):
    retusrn np.sum(np.minimum(u,v))

def hellinger_kernel(u,v):
	return np.sum(np.sqrt(u,v))
