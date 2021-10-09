import numpy as np
import matplotlib.pyplot as plt

def evaluation(predicted,truth):
    
    tp = np.zeros(predicted.shape)
    fp = np.zeros(predicted.shape)
    fn = np.zeros(predicted.shape)
    
    tp[(predicted[:,:] == 1) & (truth[:,:] == 1)] = 1
    fp[(predicted[:,:] == 1) & (truth[:,:] == 0)] = 1
    fn[(predicted[:,:] == 0) & (truth[:,:] == 1)] = 1
    
    """ plt.subplot(131)
    plt.imshow(tp,cmap='gray')
    plt.subplot(132)
    plt.imshow(fp,cmap='gray')
    plt.subplot(133)
    plt.imshow(fn,cmap='gray')
    plt.show() """
    
    p = precision(tp,fp)
    r = recall(tp,fn)
    f1 = f1_measure(p,r)
    
    return p,r,f1
    
    
def precision(tp,fp):
    return np.sum(tp)/(np.sum(tp) + np.sum(fp))
    
    
def recall(tp,fn):
    return np.sum(tp)/(np.sum(tp) + np.sum(fn))
    
def f1_measure(p,r):
    return 2*p*r/(p + r)
    
