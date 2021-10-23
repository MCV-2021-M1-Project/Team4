
import glob
import numpy as np
import cv2
import os

# -- SIMILARITY MEASURES --

def euclidean_distance(u,v):
    return np.linalg.norm(u - v)

def l1_distance(u,v):
    return np.linalg.norm((u - v),ord=1)

def chi2_distance(u,v, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(u, v)])

def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))
    #return cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)

def hellinger_kernel(u,v):
    # return np.sum(np.sqrt(np.multiply(u,v)))
    n = len(u)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(u[i]) - np.sqrt(v[i])) ** 2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result
    # return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)

# -- IMAGE RETRIEVAL FUNCTIONS --

def computeHistImage(image, color_space, mask=None):
    if color_space == "GRAY":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image_color], [0], mask, [16], [0, 256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "RGB":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Already BGR
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "HSV":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [16,16,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "YCrCb":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)
    elif color_space == "CIELab":
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([image_color], [0,1,2], mask, [8,16,16], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)

    # plt.plot(image_hist)
    # plt.show()

    return hist.flatten()

def computeSimilarity(hist1, hist2, similarity_measure):
    if similarity_measure == 'euclidean':
        return euclidean_distance(hist1, hist2)
    elif similarity_measure == 'intersec':
        return histogram_intersection(hist1, hist2)
    elif similarity_measure == 'l1':
        return l1_distance(hist1, hist2)
    elif similarity_measure == 'chi2':
        return chi2_distance(hist1, hist2)
    elif similarity_measure == 'hellinger':
        return hellinger_kernel(hist1, hist2)
    elif similarity_measure == 'all':
        return euclidean_distance(hist1, hist2), histogram_intersection(hist1, hist2), l1_distance(hist1, hist2), chi2_distance(hist1, hist2), hellinger_kernel(hist1, hist2)

