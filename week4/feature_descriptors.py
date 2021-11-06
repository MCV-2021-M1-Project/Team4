import cv2
import numpy as np


def feature_descriptor(img, type='sift', mask=None):
    if type == 'sift':
        sift = cv2.SIFT_create(800)
        keypoints, descriptors = sift.detectAndCompute(img, mask)

    # Este no va muy bien...
    elif type == 'orb':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img, mask)

    return descriptors



