import cv2
import numpy as np

def feature_descriptor(img, type='sift', mask=None):
    if type == 'sift':
        sift = cv2.SIFT_create()
        # De momento probamos sin mask
        keypoints, descriptors = sift.detectAndCompute(img, mask)

        return descriptors

