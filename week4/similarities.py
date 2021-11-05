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


def feature_distance(descriptors_query, descriptors_bbdd):
    total_distance = 0
    matches_ok = []
    threshold = 3000  # Threshold of L1 distance

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # Compute matches, discard some of them (with high distances)
    matches = bf.match(descriptors_query, descriptors_bbdd)
    for match in matches:
        if match.distance < threshold:
            total_distance += match.distance
            matches_ok.append(match)
    total_distance /= len(matches)

    return total_distance