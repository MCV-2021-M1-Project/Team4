import cv2
import numpy as np
import  matplotlib.pyplot as plt
# FILE WITH ALL THE FUNCTIONS WHICH DEAL WITH COMPUTING THE SIMILARITIES BETWEEN DESCRIPTORS

def hellingerDistance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def euclidean_distance(u,v):
    return np.linalg.norm(u - v)


def l1_distance(u,v):
    return np.linalg.norm((u - v), ord=1)


def chi2_distance(u,v, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(u, v)])


def histogram_intersection(u,v):
    return np.sum(np.minimum(u,v))


def feature_distance(descriptors_query, descriptors_bbdd, type='sift'):
    if type == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_query, descriptors_bbdd, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        if len(good) == 0:
            return 100000
        else:
            return 1 / len(good)

    elif type == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descriptors_query, descriptors_bbdd, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        if len(good) == 0:
            return 100000
        else:
            return 1 / len(good)