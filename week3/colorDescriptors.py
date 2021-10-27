import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorDescriptors(image, block=1, mask = None):
    """
    Compute the color (HSV) descriptors of the image. The function has the possibility of computing the histogram
    of the entire image or doing it by blocks of 4 (block==2) and 16 (block==3)
    :param image: image in BGR
    :param block: level of resolution. 1- Entire image. 2- 4 blocks. 3- 16 blocks.
    :param mask: mask of the image if the backgorund ot text boxes have been substracted
    :return: histograms concatenated
    """
    # If block == 1 compute and return the histogram of the entire image
    if block == 1:
        return computeHistogram(image=image, mask=mask)

    # If block greater or equal than 2, compute the first level of block histograms
    elif block >= 2:
        histogram, masks = blockHistogram(image, mask=mask)

        #  If block == 2 return the histograms of level 2
        if block == 2:
            return histogram

        # If block == 3, compute and return the 16 histograms concatenated from the level 2 masks.
        else:
            histograms = np.array([])
            for submask in masks:
                blockhists, subsubmasks = blockHistogram(image, mask=submask)
                histograms = np.concatenate((histograms, blockhists))

            return histograms

# -- AUXILIARY FUNCTIONS --

def computeHistogram(image, mask = None):
    """
    Compute the 3D HSV histogram of the input image taking into account the input mask.
    :param image: image in BGR
    :param mask: mask of the image if the background or text box have been substracted. If not, None.
    :return: histogram
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0,1,2], mask, [16,16,8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


def blockHistogram(image, mask=None):

    # If background has been applied to the image, compute the centroids of the mask
    if mask is not None:
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    # If the image is from the BBDD set (in other words, the image is cropped right in the frame),
    # find the center of the image
    else:
        # h: height, w: width
        h, w, _ = image.shape
        cX = int(w / 2)
        cY = int(h / 2)

    # List in which all the submasks are stored
    masks = []
    pos = 0

    # From the centroid (either from the BBDD image or from the mask) divide the image in 4 different masks
    for i in range(2):
        for j in range(2):
            masks.append(np.zeros((image.shape[0], image.shape[1]), dtype="uint8"))
            masks[pos][cY * i: cY * (i + 1), cX * j: cX * (j + 1)] = 1
            pos = pos + 1

    # Compute the intersection of the original mask and all the multiresolution masks
    if mask is not None:
        for i in range(len(masks)):
            masks[i] = masks[i] * mask

    blockHists = np.array([])
    for i in range(len(masks)):
        blockHists = np.concatenate((blockHists, computeHistogram(image, mask=masks[i])))

    return blockHists, masks