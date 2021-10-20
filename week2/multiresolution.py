import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import computeHistImage

NUM_BLOCKS = 2

def multiresolution(image, color_space, mask=None):

    # If background has been applied to the image, compute the centroids of the mask
    if mask is not None:
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    # If the image is from the BBDD set (in other words, the image is cropped right in the frame),
    # find the center of the image
    else:
        # h: height, w: width, c: channels(not used)
        h, w, c = image.shape
        cX = int(h/NUM_BLOCKS)
        cY = int(h/NUM_BLOCKS)

    # List in which all the masks are stored
    masks = []
    pos = 0

    # From the centroid (either from the BBDD image or from the mask) divide the image in 4 different masks
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            masks.append(np.zeros((image.shape[0], image.shape[1]), dtype="uint8"))
            masks[pos][cY * i: cY * (i + 1), cX * j: cX * (j + 1)] = 1
            pos = pos + 1

    # Compute the intersection of the original mask and all the multiresolution masks
    if mask is not None:
        for i in range(len(masks)):
            masks[i] = masks[i] * mask

    # Compute the histogram of the original whole image and store it in a variable in which all the
    # histograms will be concatenated
    histograms = np.concatenate(computeHistImage(image, color_space=color_space, mask=mask)[:,np.newaxis])

    # Compute the multiresolution histograms and concatenate them
    for i in range(len(masks)):
        histograms = np.concatenate((histograms, computeHistImage(image, color_space=color_space, mask=masks[i])))

    return histograms

    
    
    
    
