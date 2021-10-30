import cv2
import numpy as np
import  matplotlib.pyplot as plt
# FILE WITH ALL THE FUNCTIONS WHICH DEAL WITH COMPUTING THE SIMILARITIES BETWEEN DESCRIPTORS

def hellingerDistance(hist1, hist2):
    # plt.subplot(121)
    # plt.plot(hist1)
    #
    # plt.subplot(122)
    # plt.plot(hist2)
    # plt.show()

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.linalg.norm((u - v), ord=1)

def chi2_distance(u,v, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(u, v)])

def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))
